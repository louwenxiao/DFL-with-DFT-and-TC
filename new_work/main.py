import time
import datetime
import asyncio
import concurrent.futures
from data_and_model.models import create_model_instance
from data_and_model import datasets
from clients_and_server.clients import client
import copy
import torch
import argparse
import os
import psutil
import sys
import numpy as np
import pandas as pd
import gc

def main(a):
    
    args = copy.deepcopy(a)
    print('Initialize Topology and Dataset...')
    topology = get_topology(args.topology,args.num_workers)
    train_data_partition, test_data_partition = partition_data(args.dataset,worker_num=args.num_workers)
    train_dataset, test_dataset = datasets.load_datasets(args.dataset)

    device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else "cpu")
    print("device: ",device)
    model = create_model_instance(args.dataset)
    torch.save(model.state_dict(),"cache\model.pt")
    init_para = init_para = torch.nn.utils.parameters_to_vector(model.parameters())
    model_size = init_para.nelement() * 4 / 1024 / 1024
    model_size = 0.9 + 1.0*args.num_data

    del init_para,model
    gc.collect()
    print(u'当前进程的内存使用：%.4f GB' % (psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 / 1024))
    clients = []
    for i in range(args.num_workers):
        train_loader = datasets.create_dataloaders(train_dataset, batch_size=args.batch_size,
                                                   selected_idxs=train_data_partition.use(i))
        test_loader = datasets.create_dataloaders(test_dataset, batch_size=128,
                                    selected_idxs=test_data_partition.use(i), shuffle=False)
        clients.append(client(i=i, device=device, args=args, data=(train_loader, test_loader)))
        print(u'当前进程的内存使用：%.4f GB' % (psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 / 1024))
        print("---------------{}---------------".format(i))

    del train_data_partition, test_data_partition,train_loader,test_loader
    gc.collect()            # 删除上述没用的信息
    RESULT = [[0], [0], [0], [1]]  # 分别用来保存：带宽MB，时间s，精度，损失
    print(u'当前进程的内存使用：%.4f GB' % (psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 / 1024))

    start_time = datetime.datetime.now().strftime("%m-%d_%H-%M")
    print(start_time)
    lr = args.lr
    for epoch in range(1,args.epoch+1):
        time1 = time.time()
        lr = max((args.decay_rate * lr, args.min_lr))
        print("epoch-{} lr: {}".format(epoch, lr))

        model_dir = []
        data_dir = []
        print("worker 开始训练")
        for i in range(args.num_workers):
            model,data_feature = clients[i].local_train(lr)
            model_dir.append(model)
            data_dir.append(data_feature)

        for i in range(args.num_workers):
            # print("worker {} 训练邻居数据".format(i))
            clients[i].aggregate_model_and_data(topology,model_dir,data_dir,lr)

        time0 = time.time()
        RESULT[2].append(0)
        RESULT[3].append(0)
        res = []
        for i in range(args.num_workers):
            # print("worker {} 测试数据".format(i))
            loss, accuracy=clients[i].test_model()
            res.append(accuracy)
            RESULT[2][epoch] += accuracy/args.num_workers
            RESULT[3][epoch] += loss/args.num_workers
        print(res)

        print("测试时间：", time.time() - time0)

        worker_data_size = 0.0
        for worker in topology:
            worker_data_size += sum(worker) * model_size * args.prob
        RESULT[0].append(worker_data_size+RESULT[0][epoch-1])
        RESULT[1].append(0)
        print("通信：",RESULT[0][epoch])
        print("精度：",RESULT[2][epoch])
        print("损失：",RESULT[3][epoch])
        print("训练一轮时间：",time.time()-time1)
        print("\n")

        pd.DataFrame(RESULT).to_csv(
            'result/{}_Squeeze_{}_{}_data_pattern{}_lr{}_batch{}_localstep{}_rate{}_numiter{}_workers16.csv'.format(start_time,
                args.topology, args.dataset,args.data_pattern, args.lr, args.batch_size, args.local_updates, args.prob,
                args.epoch))            # 数据集，数据分布，学习率，batch，τ


def get_topology(topo,worker_num=16):
    topology = np.zeros((worker_num, worker_num), dtype=np.int)
    for worker_idx in range(worker_num):
        topology[worker_idx][worker_idx-1] = 1
        topology[worker_idx-1][worker_idx] = 1
    return topology

def partition_data(dataset_type, worker_num=16):  # 使用6个用户训练，需要添加一个IID
    train_dataset, test_dataset = datasets.load_datasets(dataset_type)

    test_partition_sizes = np.ones((100, worker_num)) * (1.0 / worker_num)
    partition_sizes = np.ones((100, worker_num)) * (1.0 / worker_num)

    train_data_partition = datasets.LabelwisePartitioner(train_dataset, partition_sizes=partition_sizes)
    test_data_partition = datasets.LabelwisePartitioner(test_dataset,
                                                        partition_sizes=test_partition_sizes)  # 训练集和测试集没有保持一致

    return train_data_partition, test_data_partition

def train(clients,i,lr,prob,comppressed_model):
    comppressed_model[i] = clients[i].local_train(lr, prob)

def test(clients,i,result):
    loss, accuracy = clients[i].test_model()
    # loss, accuracy = 0,0
    result[0][i] = loss
    result[1][i] = accuracy

def average_model(clients,i,topology):
    clients[i].aggregate_save_model(topology)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Distributed Client')
    parser.add_argument('--model_type', type=str, default='ale')
    parser.add_argument('--dataset', type=str, default='CIFAR100')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--data_pattern', type=int, default=0)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--algorithm', type=str, default='proposed')
    parser.add_argument('--mode', type=str, default='adaptive')
    parser.add_argument('--topology', type=str, default='ring')
    parser.add_argument('--prob', type=float, default=1.0)  # 压缩率
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--min_lr', type=float, default=0.001)
    parser.add_argument('--step_size', type=float, default=1.0)
    parser.add_argument('--decay_rate', type=float, default=0.99)
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--local_updates', type=int, default=60)
    parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--num_data', type=int, default=10)
    args = parser.parse_args()

    
    for arg in vars(args):
        print(arg, ":", getattr(args, arg))
    main(args)
