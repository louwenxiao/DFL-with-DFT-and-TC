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
parser.add_argument('--min_lr', type=float, default=0.0005)
parser.add_argument('--ratio', type=float, default=0.2)
parser.add_argument('--step_size', type=float, default=1.0)
parser.add_argument('--decay_rate', type=float, default=0.97)
parser.add_argument('--weight_decay', type=float, default=5e-4)
parser.add_argument('--epoch', type=int, default=5)
parser.add_argument('--use_cuda', action="store_false", default=True)
parser.add_argument('--visible_cuda', type=str, default='-1')
parser.add_argument('--algorithm', type=str, default='proposed')


args = parser.parse_args()


if args.visible_cuda == '-1':
    os.environ['CUDA_VISIBLE_DEVICES'] = str((int(args.idx) +2)% 4)
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

    local_model = models_client.create_model_instance(args.dataset_type, args.model_type)
    initial_para = client_config.para.clone().to(device)
    torch.nn.utils.vector_to_parameters(client_config.para, local_model.parameters())
    local_model.to(device)
    para_nums = torch.nn.utils.parameters_to_vector(local_model.parameters()).nelement()

    # create dataset
    print("train data len : {}\n".format(len(client_config.custom["train_data_idxes"])))
    train_dataset, test_dataset = datasets.load_datasets(args.dataset_type)
    train_loader = datasets.create_dataloaders(train_dataset, batch_size=args.batch_size, selected_idxs=client_config.custom["train_data_idxes"])
    print("train dataset:")
    utils.count_dataset(train_loader)
    test_loader = datasets.create_dataloaders(test_dataset, batch_size=128, selected_idxs=client_config.custom["test_data_idxes"], shuffle=False)
    print("test dataset:")
    utils.count_dataset(test_loader)
    
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
    for epoch in range(1, 1+args.epoch):
        epoch_steps = 0
        epoch_train_loss = 0.0
        print("--**--")
        epoch_start_time = time.time()
        if epoch > 1 and epoch % 1 == 0:
            epoch_lr = max((args.decay_rate * epoch_lr, args.min_lr))
        print("epoch-{} lr: {}".format(epoch, epoch_lr))

        local_steps, comm_neighbors, compre_ratio =  get_data_socket(master_socket)
        num_data = 5

        # 本地训练，提取特征，提取分类器参数
        while local_steps > 0:
            epoch_steps += local_steps
            print("Compression Ratio: ", compre_ratio)
            old_para = torch.nn.utils.parameters_to_vector(local_model.parameters()).clone().detach()

            optimizer = optim.SGD(local_model.parameters(), lr=epoch_lr, momentum=0.9, weight_decay=args.weight_decay)
            start_time = time.time()
            tensor_data,train_loss,model_dict = train(local_model, train_loader, optimizer, local_iters=local_steps, 
                                           device=device, model_type=args.model_type,num_data=num_data)
            print("train_lossL: ",train_loss)
            print("train time: ", time.time() - start_time)
            epoch_train_loss += (train_loss * local_steps)
            start_time = time.time()
            local_para = torch.nn.utils.parameters_to_vector(local_model.parameters()).clone().detach()
            local_model_dict = local_model.state_dict()
            local_update_norm = torch.norm(local_para - old_para).item() / local_steps
            # para_nums = tensor_data.nelement()

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
                local_data,local_para, local_cos = aggregate_model(local_model_dict, comm_neighbors, client_config, args.step_size, initial_para)
            
            # torch.nn.utils.vector_to_parameters(local_para, local_model.parameters())
            local_model.load_state_dict(local_para)
            train_loss = train_neighbor_data(local_model, local_data, optimizer, device=device)

            neighbors_consensus_distance = dict()
            for neighbor_idx, _ in comm_neighbors:
                neighbors_consensus_distance[neighbor_idx] = client_config.estimated_consensus_distance[neighbor_idx]
            send_data_socket((train_loss, neighbors_consensus_distance, local_update_norm, local_cos), master_socket)
            local_steps, comm_neighbors, compre_ratio =  get_data_socket(master_socket)
            print("***")
        start_time = time.time()
        test_loss, acc = test(local_model, test_loader, device, model_type=args.model_type)
        
        send_data_socket((epoch, time.time() - epoch_start_time, acc, test_loss, epoch_train_loss/epoch_steps), master_socket)
        print("test time: ", time.time() - start_time)
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

def aggregate_model(local_para, comm_neighbors, client_config, step_size, initial_para):
    client_data = []
    local_model_para = []
    weight_cosine_similarity = np.zeros((len(comm_neighbors), ))
    with torch.no_grad():
        for neighbor_idx, _ in comm_neighbors:
            client_data.append(client_config.neighbor_paras[neighbor_idx][0])
            local_model_para.append(client_config.neighbor_paras[neighbor_idx][1])
            client_config.estimated_consensus_distance[neighbor_idx] = 0.9
    model_para = copy.deepcopy(local_para)
    for para in local_para.keys():
        if para not in ['conv_layer.0.weight', 'conv_layer.0.bias', 'conv_layer.2.weight', 'conv_layer.2.bias']:
            continue
        else:
            for p in local_model_para:
                model_para[para] = model_para[para] + p[para]
            model_para[para] = model_para[para] / (len(local_model_para)+1)
    return client_data,model_para, weight_cosine_similarity


def compress_model(local_para, ratio):
    start_time = time.time()
    with torch.no_grad():
        send_para = local_para.detach()
        select_n = int(send_para.nelement() * ratio)
        rd_seed = np.random.randint(0, np.iinfo(np.uint32).max)
        rng = np.random.RandomState(rd_seed)
        indices = rng.choice(send_para.nelement(), size=select_n, replace=False)
        select_para = send_para[indices]
    print("compressed time: ", time.time() - start_time)

    return (select_para, select_n, rd_seed)

def get_compressed_model(config, name, nelement):
    start_time = time.time()
    received_para = get_data_socket(config.get_socket_dict[name])
    config.neighbor_paras[name] = received_para
    

def compress_model_top(local_para, ratio):
    start_time = time.time()
    with torch.no_grad():
        send_para = local_para.detach()
        topk = int(send_para.nelement() * ratio)
        _, indices = torch.topk(local_para.abs(), topk, largest=True, sorted=False)
        select_para = send_para[indices]
    print("compressed time: ", time.time() - start_time)

    return (select_para, indices)

def get_compressed_model_top(config, name, nelement):
    start_time = time.time()
    received_para, indices = get_data_socket(config.get_socket_dict[name])
    received_para.to(device)
    print("get time: ", time.time() - start_time)

    restored_model = torch.zeros(nelement).to(device)
    
    restored_model[indices] = received_para

    config.neighbor_paras[name] = restored_model.data
    config.neighbor_indices[name] = indices

if __name__ == '__main__':
    main()
