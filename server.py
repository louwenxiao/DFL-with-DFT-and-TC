import os
import sys
import time
import argparse
import socket
import pickle
import asyncio
import concurrent.futures
import json
import random
import math
import copy
import datetime
# import setproctitle

import numpy as np
from numpy import linalg
import torch
import pandas as pd

from config import *
from communication_module.comm_utils import *
from training_module import datasets, models_client, utils
from control_algorithm import *

parser = argparse.ArgumentParser(description='Distributed Client')
parser.add_argument('--model_type', type=str, default='A')
parser.add_argument('--dataset_type', type=str, default='CIFAR10')
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--data_pattern', type=int, default=11)
parser.add_argument('--weight_decay', type=float, default=5e-4)
parser.add_argument('--alpha', type=float, default=200)
parser.add_argument('--algorithm', type=str, default='proposed')
parser.add_argument('--mode', type=str, default='adaptive')
parser.add_argument('--topology', type=str, default='ring')
parser.add_argument('--prob', type=float, default=1.0)
parser.add_argument('--lr', type=float, default=0.02)
parser.add_argument('--step_size', type=float, default=1.0)
parser.add_argument('--decay_rate', type=float, default=0.98)
parser.add_argument('--epoch', type=int, default=1200)            # 注意，这里修改了
parser.add_argument('--local_updates', type=int, default=30)
parser.add_argument('--num_data', type=int, default=8)
parser.add_argument('--xigema', type=int, default=0)

args = parser.parse_args()

SERVER_IP = "127.0.0.1"
RESULT = [[0],[0],[0],[4]]      # 分别用来保存：带宽MB，时间s，精度，损失
model_size = 0.1 + 2.0 * args.num_data

def main():
    result = []
    for arg in vars(args):
        print(arg, ":", getattr(args, arg))
    # init config
    common_config = CommonConfig()
    common_config.master_listen_port_base += random.randint(0, 20) * 21
    common_config.model_type = args.model_type
    common_config.dataset_type = args.dataset_type
    common_config.batch_size = args.batch_size
    common_config.ratio = args.prob
    common_config.epoch = args.epoch
    common_config.lr = args.lr
    common_config.decay_rate = args.decay_rate
    common_config.weight_decay = args.weight_decay

    with open("worker_config.json") as json_file:
        workers_config = json.load(json_file)

    # init p2p topology
    
    workers_config['worker_config_list'] = workers_config['worker_config_list'][:16]
    worker_num = len(workers_config['worker_config_list'])
    adjacency_matrix = get_topology(worker_num)
    
    topology_prob = get_topology(worker_num)
    topology_prob = np.array(topology_prob)*1.0
    topology = get_topology(worker_num)
    train_num_topo = get_topology(worker_num)
    for topo in topology_prob:
        print(topo)
    # 设置为全连接的拓扑

    for worker_idx in range(worker_num):
        adjacency_matrix[worker_idx][worker_idx] = 0

    p2p_port = np.zeros_like(adjacency_matrix)
    curr_port = common_config.p2p_listen_port_base + random.randint(10, 20) * 192
    for idx_row in range(len(adjacency_matrix)):
        for idx_col in range(len(adjacency_matrix[0])):
            if adjacency_matrix[idx_row][idx_col] != 0:
                curr_port += 1
                p2p_port[idx_row][idx_col] = curr_port

    # create workers
    for worker_idx, worker_config in enumerate(workers_config['worker_config_list']):
        custom = dict()
        custom["neighbor_info"] = dict()

        custom["bandwidth"] = worker_config["bandwidth"]

        common_config.worker_list.append(
            Worker(config=ClientConfig(idx=worker_idx,
                                       client_ip=worker_config['ip_address'],
                                       master_ip=SERVER_IP,
                                       master_port=common_config.master_listen_port_base+worker_idx,
                                       custom=custom),
                   common_config=common_config, 
                   user_name=worker_config['user_name'],
                   pass_wd=worker_config['pass_wd'],
                   local_scripts_path=workers_config['scripts_path']['local'],
                   remote_scripts_path=workers_config['scripts_path']['remote'],
                   location='local'
                   )
        )
    
    # init workers' config
    for worker_idx, worker_config in enumerate(workers_config['worker_config_list']):
        for neighbor_idx, link in enumerate(adjacency_matrix[worker_idx]):
            if link == 1:
                neighbor_config = common_config.worker_list[neighbor_idx].config
                neighbor_ip = neighbor_config.client_ip
                neighbor_bandwidth = neighbor_config.custom["bandwidth"]

                # neighbor ip, send_port, listen_port
                common_config.worker_list[worker_idx].config.custom["neighbor_info"][neighbor_idx] = \
                        (neighbor_ip, p2p_port[worker_idx][neighbor_idx], p2p_port[neighbor_idx][worker_idx], neighbor_bandwidth)


    feature_extraction = models_client.create_model_instance(common_config.dataset_type, common_config.model_type)
    torch.save(feature_extraction.state_dict(),"/data/wxlou/nonIID/My/fedgkt_cifar10/save_model/model.pt")
    model_size = 0.5 + 1.5*args.num_data
    print("Model Size: {} MB".format(model_size))
    
    # partition dataset
    train_data_partition, test_data_partition = partition_data(common_config.dataset_type, args.data_pattern,worker_num=worker_num)
 
    for worker_idx, worker in enumerate(common_config.worker_list):
        worker.config.para = None
        worker.config.custom["train_data_idxes"] = train_data_partition.use(worker_idx)
        worker.config.custom["test_data_idxes"] = test_data_partition.use(worker_idx)

    # connect socket and send init config
    communication_parallel(common_config.worker_list, action="init")

    training_recorder = TrainingRecorder(common_config.worker_list, common_config.recoder)

    if args.local_updates > 0:
        local_steps = args.local_updates
    else:
        local_steps = int(np.ceil(50000 / worker_num / 64.0))
    print("local steps: {}".format(local_steps))
    
    # bandwidth = np.zeros((2, 10))
    # bandwidth[0] = np.random.rand(10) * 2.375 + 0.125 # 0.125MB/s ~ 2.5MB/s(1Mb/s ~ 20Mb/s)
    # bandwidth[0] = np.array([1, 3, 5, 7, 9, 11, 13, 15, 17, 19]) / 8
    # bandwidth[0] = np.array([1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000]) / 8
    # bandwidth[1] = bandwidth[0].copy() / 2.0
    start_time = datetime.datetime.now().strftime("%m-%d_%H-%M")
    for epoch_num in range(1, 1+common_config.epoch):
        print("\n--**--\nEpoch: {}".format(epoch_num))
        time1 = time.time()
        
        for i in range(len(topology_prob[0])):
            for j in range(i,len(topology_prob[0])):
                if random.random() < topology_prob[i][j]:
                    topology[i][j] = 1
                    topology[j][i] = 1
                else:
                    topology[i][j] = 0
                    topology[j][i] = 0
        # topology = get_connected_graph(topology,topology_prob)
        for topo in topology:
            print(topo)
        
        total_transfer_size = 0
        for worker in common_config.worker_list:
            worker_data_size = np.sum(topology[worker.idx]) * model_size
            common_config.recoder.add_scalar('Data Size-'+str(worker.idx), worker_data_size, epoch_num)
            total_transfer_size += worker_data_size
            neighbors_list = list()
            for neighbor_idx, link in enumerate(topology[worker.idx]):
                if link == 1:
                    neighbors_list.append((neighbor_idx, 1.0 / (np.max([np.sum(topology[worker.idx]), np.sum(topology[neighbor_idx])])+1)))
            send_data_socket((local_steps, neighbors_list,args.num_data), worker.socket)

        common_config.recoder.add_scalar('Data Size', total_transfer_size, epoch_num)
        common_config.recoder.add_scalar('Num of Links', np.sum(topology), epoch_num)
        common_config.recoder.add_scalar('Avg Rate', np.average(1.0), epoch_num)
        test_loss, cos_dis_vector, test_acc = training_recorder.get_train_info()
        
        print("Epoch: {}, average accuracy: {}, average test loss: {}".format(
            epoch_num, test_acc, test_loss))
        time2 = time.time()

        # topology_prob,train_num_topo = update_topo(topology_prob,train_num_topo,cos_dis_vector)
        
        RESULT[0].append(RESULT[0][epoch_num-1]+total_transfer_size)
        RESULT[1].append(RESULT[1][epoch_num-1]+0)
        RESULT[2].append(test_acc)
        RESULT[3].append(test_loss)

        pd.DataFrame(RESULT).to_csv('/data/wxlou/nonIID/My/result/{}_My_nonIID_CIFAR10_{}_lr{}_batch{}_localstep{}_numdata{}_workers16.csv'.format(start_time,
                            args.dataset_type,args.lr,args.batch_size,args.local_updates,args.num_data))      

    
    # close socket
    for worker in common_config.worker_list:
        worker.socket.shutdown(2)
        worker.socket.close()

class TrainingRecorder(object):
    def __init__(self, worker_list, recorder, beta=0.95):
        self.worker_list = worker_list
        self.worker_num = len(worker_list)
        self.beta = beta
        self.moving_consensus_distance = np.ones((self.worker_num, self.worker_num)) * 1e-6
        self.avg_update_norm = 0
        self.round = 0
        self.epoch = 0
        self.recorder = recorder
        self.total_time = 0

        for i in range(self.worker_num):
            self.moving_consensus_distance[i][i] = 0

    def get_train_info(self):
        self.round += 1
        communication_parallel(self.worker_list, action="get")

        test_loss = 0.0
        test_acc = 0.0
        cos_distance = []
        for worker in self.worker_list:
            loss, cos_dis_vector, acc = worker.train_info[-1]
            
            test_loss = test_loss + loss
            cos_distance.append(cos_dis_vector)
            test_acc = test_acc + acc
        test_loss = test_loss/len(self.worker_list)
        test_acc = test_acc/len(self.worker_list)

        return test_loss, cos_distance, test_acc

    def get_test_info(self):
        self.epoch += 1
        communication_parallel(self.worker_list, action="get")
        avg_acc = 0.0
        avg_test_loss = 0.0
        epoch_time = 0.0
        for worker in self.worker_list:
            _, worker_time, acc, loss, train_loss = worker.train_info[-1]
            self.recorder.add_scalar('Accuracy/worker_' + str(worker.idx), acc, self.round)
            self.recorder.add_scalar('Test_loss/worker_' + str(worker.idx), loss, self.round)
            self.recorder.add_scalar('Time/worker_' + str(worker.idx), worker_time, self.epoch)

            avg_acc += acc
            avg_test_loss += loss
            epoch_time = max(epoch_time, worker_time)
        
        avg_acc /= self.worker_num
        avg_test_loss /= self.worker_num
        self.total_time += epoch_time
        self.recorder.add_scalar('Time/total', epoch_time, self.epoch)
        self.recorder.add_scalar('Accuracy/average', avg_acc, self.epoch)
        self.recorder.add_scalar('Test_loss/average', avg_test_loss, self.epoch)
        self.recorder.add_scalar('Accuracy/round_average', avg_acc, self.round)
        self.recorder.add_scalar('Test_loss/round_average', avg_test_loss, self.round)
        print("Epoch: {}, time: {}, average accuracy: {}, average test loss: {}, average train loss: {}".format(self.epoch, self.total_time, avg_acc, avg_test_loss, train_loss))


def communication_parallel(worker_list, action, data=None):
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=len(worker_list),)
        tasks = []
        for worker in worker_list:
            if action == "init":
                tasks.append(loop.run_in_executor(executor, worker.send_init_config))
            elif action == "get":
                tasks.append(loop.run_in_executor(executor, worker.get_config))
            elif action == "send":
                tasks.append(loop.run_in_executor(executor, worker.send_data, data))
        loop.run_until_complete(asyncio.wait(tasks))
        loop.close()
    except:
        sys.exit(0)


def partition_data(dataset_type, data_pattern, worker_num=16):      # 使用6个用户训练，需要添加一个IID
    train_dataset, test_dataset = datasets.load_datasets(dataset_type)

    # partition_sizes = np.ones((100, worker_num)) * (1.0 / worker_num)
    partition_sizes = non_iid_partition(data_pattern,worker_num=worker_num)
    test_partition_sizes= np.ones((10, worker_num)) * (1.0 / worker_num)
    print(test_partition_sizes)

    train_data_partition = datasets.LabelwisePartitioner(train_dataset, partition_sizes=partition_sizes)
    test_data_partition = datasets.LabelwisePartitioner(test_dataset,partition_sizes=test_partition_sizes)
    return train_data_partition, test_data_partition
    

def non_iid_partition(ratio, worker_num=16):
    partition_sizes = np.ones((10, worker_num)) * ((1 - ratio) / (worker_num-2))
    for worker_idx in range(8):
        partition_sizes[worker_idx][worker_idx*2] = ratio / 2
        partition_sizes[worker_idx][worker_idx*2+1] = ratio / 2
    return partition_sizes

def get_topology(worker_num=16):
    topology = np.ones((worker_num, worker_num), dtype=np.int)
    for worker_idx in range(worker_num):
        topology[worker_idx][worker_idx] = 0
        
    return topology

def update_topo(topology_prob,train_num_topo,cos_dis_vector):
    train_num = copy.deepcopy(train_num_topo)
    topology = copy.deepcopy(topology_prob)
    for i in range(len(topology_prob)):
        for j in range(i,len(topology_prob)):
            if cos_dis_vector[i][j] == 0:
                continue
            else:
                prob = math.exp(-1.0 * args.xigema * (cos_dis_vector[i][j]+cos_dis_vector[j][i])/2.0)
                mid_prob = train_num_topo[i][j]*topology_prob[i][j] + prob
                topology[j][i] = mid_prob / (1.0*train_num_topo[i][j]+1.0)
                topology[i][j] = mid_prob / (1.0*train_num_topo[i][j]+1.0)
                train_num[i][j] = train_num[j][i] = train_num_topo[i][j] + 1
    return topology,train_num


def get_connected_graph(topology,topology_prob):
    pass


if __name__ == "__main__":
    main()
