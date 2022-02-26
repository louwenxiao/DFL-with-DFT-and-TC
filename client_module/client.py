import os
import time
import copy
import argparse
import asyncio
import concurrent.futures

import numpy as np
from numpy import linalg
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from config import ClientConfig
from client_comm_utils import *
from training_utils import *
import utils
import datasets, models_client

parser = argparse.ArgumentParser(description='Distributed Client')
parser.add_argument('--idx', type=str, default="0",
                    help='index of worker')
parser.add_argument('--master_ip', type=str, default="127.0.0.1",
                    help='IP address for controller or ps')
parser.add_argument('--master_port', type=int, default=58000, metavar='N',
                    help='')
parser.add_argument('--dataset_type', type=str, default='CIFAR10')
parser.add_argument('--model_type', type=str, default='VGG')
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--min_lr', type=float, default=0.001)
parser.add_argument('--ratio', type=float, default=0.2)
parser.add_argument('--step_size', type=float, default=1.0)
parser.add_argument('--decay_rate', type=float, default=0.97)
parser.add_argument('--weight_decay', type=float, default=5e-4)
parser.add_argument('--epoch', type=int, default=5)
parser.add_argument('--use_cuda', action="store_false", default=True)
parser.add_argument('--visible_cuda', type=str, default='-1')
parser.add_argument('--algorithm', type=str, default='proposed')
parser.add_argument('--num_data', type=int, default=4)


args = parser.parse_args()


if args.visible_cuda == '-1':
    os.environ['CUDA_VISIBLE_DEVICES'] = str((int(args.idx) +0)% 4)
else:
    os.environ['CUDA_VISIBLE_DEVICES'] = args.visible_cuda
device = torch.device("cuda" if args.use_cuda and torch.cuda.is_available() else "cpu")

def main():
    client_config = ClientConfig(
        idx=args.idx,
        master_ip=args.master_ip,
        master_port=args.master_port
    )
    utils.create_dir("logs")
    recorder = SummaryWriter("logs/log_"+str(args.idx))
    # receive config
    master_socket = connect_send_socket(args.master_ip, args.master_port)
    config_received = get_data_socket(master_socket)
    for k, v in config_received.__dict__.items():
        setattr(client_config, k, v)

    for arg in vars(args):
        print(arg, ":", getattr(args, arg))

    # create modekl
    local_model = models_client.create_model_instance(args.dataset_type, args.model_type)
    local_model.load_state_dict(torch.load("/data/wxlou/nonIID/My/fedgkt_cifar10/save_model/model.pt"))
    para_nums = torch.nn.utils.parameters_to_vector(local_model.parameters()).nelement()

    # create dataset
    print("train data len : {}\n".format(len(client_config.custom["train_data_idxes"])))
    train_dataset, test_dataset = datasets.load_datasets(args.dataset_type)
    train_loader = datasets.create_dataloaders(train_dataset, batch_size=args.batch_size, selected_idxs=client_config.custom["train_data_idxes"])
    print("train dataset:")
    data_distribution = utils.count_dataset(train_loader)
    test_loader = datasets.create_dataloaders(test_dataset, batch_size=32, selected_idxs=client_config.custom["test_data_idxes"], shuffle=False)
    print("test dataset:")
    _ = utils.count_dataset(test_loader)
    print("我的数据分布为：",data_distribution)
    
    # create p2p communication socket
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=20,)
    tasks = []
    for _, (neighbor_ip, send_port, listen_port, _) in client_config.custom["neighbor_info"].items():
        tasks.append(loop.run_in_executor(executor, connect_send_socket, neighbor_ip, send_port))
        tasks.append(loop.run_in_executor(executor, connect_get_socket, client_config.client_ip, listen_port))
    loop.run_until_complete(asyncio.wait(tasks))

    # save socket for later communication
    for task_idx, neighbor_idx in enumerate(client_config.custom["neighbor_info"].keys()):
        client_config.send_socket_dict[neighbor_idx] = tasks[task_idx*2].result()
        client_config.get_socket_dict[neighbor_idx] = tasks[task_idx*2+1].result()
    loop.close()

    epoch_lr = args.lr
    # diatance = []
    
    for epoch in range(1, 1+args.epoch):
        start_time = time.time()
        print("--**--")
        if epoch > 1 and epoch % 1 == 0:
            epoch_lr = max((args.decay_rate * epoch_lr, args.min_lr))
        print("epoch-{} lr: {}".format(epoch, epoch_lr))

        local_steps, comm_neighbors,num_data =  get_data_socket(master_socket)
        
        # 本地训练，提取特征，提取分类器参数
        # old_para = torch.nn.utils.parameters_to_vector(local_model.parameters()).clone().detach()
        optimizer = optim.SGD(local_model.parameters(), lr=epoch_lr, momentum=0.9, weight_decay=args.weight_decay)
        tensor_data,train_loss,model_dict = train(local_model, train_loader, optimizer, local_iters=local_steps, 
                                           device=device, model_type=args.model_type,num_data=num_data)
        print("train_loss: ",train_loss)

        local_model_dict = local_model.state_dict()
        neighbors_distribution = dict()
        if len(comm_neighbors) > 0:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            executor = concurrent.futures.ThreadPoolExecutor(max_workers=20,)
            tasks = []
            for neighbor_idx, _ in comm_neighbors:
                print("neighbor : {}\n".format(neighbor_idx))
                tasks.append(loop.run_in_executor(executor, send_data_socket, (tensor_data,model_dict),
                                                    client_config.send_socket_dict[neighbor_idx]))
                tasks.append(loop.run_in_executor(executor, get_compressed_model, client_config, 
                                                    neighbor_idx, para_nums))
            loop.run_until_complete(asyncio.wait(tasks))
            loop.close()
            print("compress, send and get time: ", time.time() - start_time)

            # aggregate parameters
            local_data,local_para,neighbors_distribution = aggregate_model(local_model_dict, comm_neighbors, client_config)
            
            local_model.load_state_dict(local_para)
            train_loss = train_neighbor_data(local_model, local_data, optimizer, device=device)
                
        cos_dis_vector = []
        for i in range(16):
            if i not in neighbors_distribution.keys():
                cos_dis_vector.append(0)
            else:
                print(data_distribution)
                print(neighbors_distribution)
                dis = torch.nn.functional.cosine_similarity(torch.tensor([data_distribution]), torch.tensor([neighbors_distribution[i]]))
                cos_dis_vector.append(dis.item())
                print("与{}之间的距离为：{}".format(i,dis))
        print(cos_dis_vector)
        
        test_loss, acc = test(local_model, test_loader, device, model_type=args.model_type)        
        send_data_socket((test_loss, cos_dis_vector, acc), master_socket)
        print("***")
        
        # 下面内容不要修改
        recorder.add_scalar('acc_worker-' + str(args.idx), acc, epoch)
        recorder.add_scalar('test_loss_worker-' + str(args.idx), test_loss, epoch)
        recorder.add_scalar('train_loss_worker-' + str(args.idx), train_loss, epoch)
        print("epoch: {}, train loss: {}, test loss: {}, test accuracy: {}".format(epoch, train_loss, test_loss, acc))
        print("\n\n")
        
        
    torch.save(local_model.state_dict(), './logs/model_'+str(args.idx)+'.pkl')
    # close socket
    for _, conn in client_config.send_socket_dict.items():
        conn.shutdown(2)
        conn.close()
    for _, conn in client_config.get_socket_dict.items():
        conn.shutdown(2)
        conn.close()
    master_socket.shutdown(2)
    master_socket.close()


# data_distribution计算邻居数据分布
def aggregate_model(local_para, comm_neighbors, client_config):
    data_distribution = dict()
    client_data = []
    local_model_para = []
    with torch.no_grad():
        for neighbor_idx, _ in comm_neighbors:
            client_data.append(client_config.neighbor_paras[neighbor_idx][0])
            v = [0 for i in range(10)]
            for data in client_config.neighbor_paras[neighbor_idx][0]:       # [1]表示标签
                for i in data[1]:
                    # print(i)
                    v[i] = v[i] + 1.0/(len(data[1])*len(client_config.neighbor_paras[neighbor_idx][0]))
            data_distribution[neighbor_idx] = v
            
            local_model_para.append(client_config.neighbor_paras[neighbor_idx][1])
    model_para = copy.deepcopy(local_para)
    for para in local_para.keys():
    
        key = ['features.0.weight', 'features.0.bias', 'features.3.weight', 'features.3.bias']
        # key = ['features.0.weight', 'features.0.bias', 'features.3.weight', 'features.3.bias', 'features.6.weight', 
        #    'features.6.bias', 'features.8.weight', 'features.8.bias', 'features.10.weight', 'features.10.bias']
        if para not in key:
            continue
        else:
            for p in local_model_para:
                model_para[para] = model_para[para] + p[para]
            model_para[para] = model_para[para] / (len(local_model_para)+1.0)
            
    return client_data,model_para,data_distribution



def get_compressed_model(config, name, nelement):
    start_time = time.time()
    received_para = get_data_socket(config.get_socket_dict[name])
    config.neighbor_paras[name] = received_para
    print("get time: ", time.time() - start_time)



if __name__ == '__main__':
    main()

