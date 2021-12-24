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
import copy
import setproctitle

import numpy as np
from numpy import linalg
import torch

from config import *
from communication_module.comm_utils import *
from training_module import datasets, models_client, utils
from control_algorithm import *

parser = argparse.ArgumentParser(description='Distributed Client')
parser.add_argument('--model_type', type=str, default='VGG')
parser.add_argument('--dataset_type', type=str, default='CIFAR10')
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--data_pattern', type=int, default=11)
parser.add_argument('--weight_decay', type=float, default=5e-4)
parser.add_argument('--alpha', type=float, default=200)
parser.add_argument('--algorithm', type=str, default='proposed')
parser.add_argument('--mode', type=str, default='adaptive')
parser.add_argument('--topology', type=str, default='ring')
parser.add_argument('--prob', type=float, default=1.0)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--step_size', type=float, default=1.0)
parser.add_argument('--decay_rate', type=float, default=0.98)
parser.add_argument('--epoch', type=int, default=200)
parser.add_argument('--local_updates', type=int, default=40)

args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
pid_name = "FedGKT"
setproctitle.setproctitle(pid_name)
SERVER_IP = "127.0.0.1"

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
    workers_config['worker_config_list'] = workers_config['worker_config_list'][:6]
    worker_num = len(workers_config['worker_config_list'])
    adjacency_matrix = np.ones((worker_num, worker_num), dtype=np.int)

    for worker_idx in range(worker_num):
        adjacency_matrix[worker_idx][worker_idx] = 0

    p2p_port = np.zeros_like(adjacency_matrix)
    curr_port = common_config.p2p_listen_port_base + random.randint(0, 20) * 200
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
    
    init_para = torch.nn.utils.parameters_to_vector(feature_extraction.parameters())
    model_size = init_para.nelement() * 4 / 1024 / 1024
    print("Model Size: {} MB".format(model_size))
    
    # partition dataset
    train_data_partition, test_data_partition = partition_data(common_config.dataset_type, args.data_pattern)

    for worker_idx, worker in enumerate(common_config.worker_list):
        worker.config.para = init_para
        worker.config.custom["train_data_idxes"] = train_data_partition.use(worker_idx)
        worker.config.custom["test_data_idxes"] = test_data_partition.use(worker_idx)

    # connect socket and send init config
    communication_parallel(common_config.worker_list, action="init")

    curr_steps = 0    
    test_steps = int(len(train_data_partition) / common_config.batch_size)
    
    training_recorder = TrainingRecorder(common_config.worker_list, common_config.recoder)
    consensus_distance = np.ones((worker_num, worker_num)) * 1e-6

    if args.local_updates > 0:
        local_steps = args.local_updates
    else:
        local_steps = int(np.ceil(50000 / worker_num / 64.0))
    print("local steps: {}".format(local_steps))
    avg_update_norm = 4.0
    total_round = 0
    distance_factor = args.alpha
    bandwidth = np.zeros((2, 10))
    # bandwidth[0] = np.random.rand(10) * 2.375 + 0.125 # 0.125MB/s ~ 2.5MB/s(1Mb/s ~ 20Mb/s)
    # bandwidth[0] = np.array([1, 3, 5, 7, 9, 11, 13, 15, 17, 19]) / 8
    bandwidth[0] = np.array([10, 10, 10, 10, 10, 10, 10, 10, 10, 10]) / 8
    bandwidth[1] = bandwidth[0].copy() / 2.0
    server_cos_prob = np.ones((worker_num, 4))
    for epoch_num in range(1, 1+common_config.epoch):
        # distance_factor = distance_factor * 0.99
        if distance_factor > 50:
            distance_factor = distance_factor * 0.99
        else:
            distance_factor = 50
        print("\n--**--\nEpoch: {}".format(epoch_num))
        # simulated_epoch_time = 0
        while True:
            total_round += 1
            
            bandwidth_delta = np.random.rand(10) *4 - 2
            bandwidth[0] = np.clip(bandwidth[0] + bandwidth_delta, 0.125, 2.5)
            bandwidth[1] = bandwidth[0].copy() / 2.0
            
            topology, ratios = ring_topo(server_cos_prob)
           
            total_transfer_size = 0
            for worker in common_config.worker_list:
                worker_data_size = np.sum(topology[worker.idx]) * ratios[worker.idx] * model_size
                common_config.recoder.add_scalar('Data Size-'+str(worker.idx), worker_data_size, total_round)
                total_transfer_size += worker_data_size
                neighbors_list = list()
                for neighbor_idx, link in enumerate(topology[worker.idx]):
                    if link == 1:
                        neighbors_list.append((neighbor_idx, 1.0 / (np.max([np.sum(topology[worker.idx]), np.sum(topology[neighbor_idx])])+1)))
                # print(local_steps, neighbors_list, ratios[worker.idx])
                send_data_socket((local_steps, neighbors_list, ratios[worker.idx]), worker.socket)

                curr_steps += local_steps

            common_config.recoder.add_scalar('Data Size', total_transfer_size, total_round)
            common_config.recoder.add_scalar('Num of Links', np.sum(topology), total_round)
            common_config.recoder.add_scalar('Avg Rate', np.average(ratios), total_round)
            consensus_distance, avg_update_norm, local_prob_list = training_recorder.get_train_info()
            # server_cos_prob = speed_cos_prob(local_prob_list, topology, server_cos_prob)
            
            if curr_steps > test_steps:
                curr_steps = curr_steps % test_steps
                break
        
        # common_config.recoder.add_scalar('Simulated Epoch time', simulated_epoch_time, epoch_num)


        # start_time = time.time()
        # testing signal
        communication_parallel(common_config.worker_list, action="send", data=(-1, None, None))
        # print("info parallesim send time {}".format(time.time() - start_time))
        training_recorder.get_test_info()
    
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
        avg_train_loss = 0.0
        round_consensus_distance = np.ones_like(self.moving_consensus_distance) * -1
        round_update_norm = np.zeros(self.worker_num)
        # local_para_list = list()
        local_prob_list = list()
        for worker in self.worker_list:
            train_loss, neighbors_consensus_distance, local_update_norm, local_cos = worker.train_info[-1]
            # local_para_list.append(local_para)
            local_prob_list.append(1-local_cos)
            for neighbor_idx in neighbors_consensus_distance.keys():
                round_consensus_distance[worker.idx][neighbor_idx] = neighbors_consensus_distance[neighbor_idx]
            round_update_norm[worker.idx] = local_update_norm
            avg_train_loss += train_loss
        # probablity =  calculate_cosine(local_cos_list)
        # probablity = 1 - local_cos
        if self.round == 1:
            self.avg_update_norm = np.average(round_update_norm)
        else:
            self.avg_update_norm = self.beta * self.avg_update_norm + (1 - self.beta) * np.average(round_update_norm)
        # print("received distance:")
        # print(round_consensus_distance)
        # print("update norm", round_update_norm)
        for worker_idx in range(self.worker_num):
            for neighbor_idx in range(worker_idx+1, self.worker_num):
                round_consensus_distance[worker_idx][neighbor_idx] = (round_consensus_distance[worker_idx][neighbor_idx]
                                                                + round_consensus_distance[neighbor_idx][worker_idx]) / 2
                round_consensus_distance[neighbor_idx][worker_idx] = round_consensus_distance[worker_idx][neighbor_idx]
        
        backup_distance = round_consensus_distance.copy()
        for k in range(self.worker_num):
            for worker_idx in range(self.worker_num):
                for neighbor_idx in range(worker_idx+1, self.worker_num):
                    if round_consensus_distance[worker_idx][k] >= 0 and round_consensus_distance[neighbor_idx][k] >=0:
                        tmp = round_consensus_distance[worker_idx][k] + round_consensus_distance[neighbor_idx][k]
                        if round_consensus_distance[worker_idx][neighbor_idx] < 0:
                            round_consensus_distance[worker_idx][neighbor_idx] = tmp
                        else:    
                            round_consensus_distance[worker_idx][neighbor_idx] = np.min([round_consensus_distance[worker_idx][neighbor_idx], tmp])
                        round_consensus_distance[neighbor_idx][worker_idx] = round_consensus_distance[worker_idx][neighbor_idx]
        
        for worker_idx in range(self.worker_num):
            round_consensus_distance[worker_idx][worker_idx] = 0

        if self.round == 1:
            self.moving_consensus_distance = round_consensus_distance
        else:
            self.moving_consensus_distance = self.beta * self.moving_consensus_distance + (1 - self.beta) * round_consensus_distance

        for worker_idx in range(self.worker_num):
            for neighbor_idx in range(worker_idx+1, self.worker_num):
                if backup_distance[worker_idx][neighbor_idx] >= 0:
                    self.moving_consensus_distance[worker_idx][neighbor_idx] = backup_distance[worker_idx][neighbor_idx]
                    self.moving_consensus_distance[neighbor_idx][worker_idx] = backup_distance[worker_idx][neighbor_idx]
        # print("moving distance:")
        # print(self.moving_consensus_distance)

        avg_train_loss = avg_train_loss / self.worker_num
        self.recorder.add_scalar('Train_loss/train', avg_train_loss, self.round)
        self.recorder.add_scalar('Distance', np.average(self.moving_consensus_distance), self.round)

        print("communication round {}, train loss: {}".format(self.round, avg_train_loss))

        return self.moving_consensus_distance, self.avg_update_norm, local_prob_list

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



def get_topology_rates(mode, topology, consensus_distance, bandwidth, target_distance,  comp_time, model_size, prob):
    if mode == "static":
        if topology == "ring":
            topology, ratios = ring_topo(prob)
        elif topology == "ring_4":
            topology, ratios = ring_topo_4(prob)
        elif topology == "ring_6":
            topology, ratios = ring_topo_6(prob)
        elif topology == "topo_3":
            topology, ratios = topo_3(prob)
        elif topology == "topo_6_test":
            topology, ratios = topo_6_test(prob)
        elif topology == "ring_test":
            topology, ratios = ring_test(prob)

    return topology, ratios

def ring_topo(prob, worker_num=6):
    topology = np.zeros((worker_num, worker_num), dtype=np.int)

    for worker_idx in range(worker_num):
        topology[worker_idx][worker_idx-1] = 1
        topology[worker_idx-1][worker_idx] = 1
    
    ratios = np.ones(worker_num)
    return topology, ratios

def ring_topo_6(prob, worker_num=20):
    topology = np.zeros((worker_num, worker_num), dtype=np.int)

    for worker_idx in range(worker_num):
        topology[worker_idx][worker_idx-3] = 1
        topology[worker_idx][worker_idx-6] = 1
        topology[worker_idx][worker_idx-9] = 1

        topology[worker_idx-3][worker_idx] = 1
        topology[worker_idx-6][worker_idx] = 1
        topology[worker_idx-9][worker_idx] = 1
    
    ratios = np.ones(worker_num)
    return topology, ratios

def ring_topo_4(local_prob_list, worker_num=20):
    topology = np.zeros((worker_num, worker_num), dtype=np.int)

    for worker_idx in range(worker_num):
        topology[worker_idx][worker_idx-3] = 1
        topology[worker_idx][worker_idx-6] = 1

        topology[worker_idx-3][worker_idx] = 1
        topology[worker_idx-6][worker_idx] = 1

    ratios = np.ones(worker_num)
    return topology, ratios

def speed_cos_prob(local_prob_list, topology, server_cos_prob, worker_num=6):
    base_topology = np.zeros((worker_num, worker_num), dtype=np.int)

    for worker_idx in range(worker_num):
        base_topology[worker_idx][worker_idx-3] = 1
        base_topology[worker_idx][worker_idx-6] = 1

        base_topology[worker_idx-3][worker_idx] = 1
        base_topology[worker_idx-6][worker_idx] = 1

    for worker_idx in range(worker_num):
        tmp_idx = 0
        tmp_neigh_idx = 0
        for neigh_idx in range(worker_num):
            if base_topology[worker_idx][neigh_idx] == 1:
                if topology[worker_idx][neigh_idx] == 1:
                    server_cos_prob[worker_idx][tmp_neigh_idx] = 0.5 * np.max([0.1, np.random.rand()]) + 0.5 * local_prob_list[worker_idx][tmp_idx]
                    tmp_idx += 1
                tmp_neigh_idx += 1
    
    print("prob:")
    print(server_cos_prob)
    return server_cos_prob


def topo_3(prob):
    topology = [[0, 1, 1, 0, 0, 0, 0, 0, 0, 1],
                [1, 0, 1, 1, 0, 0, 0, 0, 0, 0],
                [1, 1, 0, 1, 0, 0, 0, 0, 0, 0],
                [0, 1, 1, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 1, 1, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 1, 0, 1, 0],
                [0, 0, 0, 0, 1, 1, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0, 1, 1],
                [0, 0, 0, 0, 0, 1, 0, 1, 0, 1],
                [1, 0, 0, 0, 0, 0, 0, 1, 1, 0]]

    topology = np.array(topology, dtype=np.int)
    for worker_idx in range(10):
        for neighbor_idx in range(0, worker_idx):
            if topology[worker_idx][neighbor_idx] == 1 and np.random.rand() > prob:
                topology[worker_idx][neighbor_idx] = 0
                topology[neighbor_idx][worker_idx] = 0

    ratios = np.ones(10)
    return topology, ratios

def topo_6_test(prob):
    if prob == 1.0:
        prob = [[0, 1.0, 1.0, 0.0, 1.0, 1.0],
                [0, 0.0, 1.0, 0.0, 0.0, 1.0],
                [0, 0.0, 0.0, 1.0, 1.0, 0.0],
                [0, 0.0, 0.0, 0.0, 1.0, 0.0],
                [0, 0.0, 0.0, 0.0, 0.0, 1.0],
                [0, 0, 0, 0, 0, 0]
            ]
    elif prob == 0.5:
        prob = [[0, 0.5, 0.5, 0.0, 0.5, 0.5],
                [0, 0.0, 0.5, 0.0, 0.0, 0.5],
                [0, 0.0, 0.0, 0.5, 0.5, 0.0],
                [0, 0.0, 0.0, 0.0, 0.5, 0.0],
                [0, 0.0, 0.0, 0.0, 0.0, 0.5],
                [0, 0, 0, 0, 0, 0]
            ]
    elif prob == 0.1:
        prob = [[0, 0.1, 1.0, 0.0, 1.0, 1.0],
                [0, 0.0, 1.0, 0.0, 0.0, 1.0],
                [0, 0.0, 0.0, 0.1, 1.0, 0.0],
                [0, 0.0, 0.0, 0.0, 1.0, 0.0],
                [0, 0.0, 0.0, 0.0, 0.0, 0.1],
                [0, 0, 0, 0, 0, 0]
            ]
    elif prob == 0.0:
        prob = [[0, 1.0, 0.0, 0.0, 0.0, 1.0],
                [0, 0.0, 1.0, 0.0, 0.0, 0.0],
                [0, 0.0, 0.0, 1.0, 0.0, 0.0],
                [0, 0.0, 0.0, 0.0, 1.0, 0.0],
                [0, 0.0, 0.0, 0.0, 0.0, 1.0],
                [0, 0, 0, 0, 0, 0]
            ]

    prob = np.array(prob)
    topology = np.zeros((6, 6), dtype=np.int)
    for worker_idx in range(6):
        for neighbor_idx in range(worker_idx+1, 6):
            if np.random.rand() < prob[worker_idx][neighbor_idx]:
                topology[worker_idx][neighbor_idx] = 1
                topology[neighbor_idx][worker_idx] = 1

    ratios = np.ones(6)
    return topology, ratios

def ring_test(prob):
    if prob == 1:
        prob_1 = 1.0
        prob_2 = 1.0
    elif prob == 2:
        prob_1 = 0.5
        prob_2 = 0.5
    elif prob == 3:
        prob_1 = 0.25
        prob_2 = 0.75
    elif prob == 4:
        prob_1 = 0.25
        prob_2 = 1.0
    elif prob == 5:
        prob_1 = 0.1
        prob_2 = 1.0
    
    topology = [[0, 1, 1, 0, 0, 0, 0, 0, 1, 1],
                [1, 0, 1, 1, 0, 0, 0, 0, 0, 1],
                [1, 1, 0, 1, 1, 0, 0, 0, 0, 0],
                [0, 1, 1, 0, 1, 1, 0, 0, 0, 0],
                [0, 0, 1, 1, 0, 1, 1, 0, 0, 0],
                [0, 0, 0, 1, 1, 0, 1, 1, 0, 0],
                [0, 0, 0, 0, 1, 1, 0, 1, 1, 0],
                [0, 0, 0, 0, 0, 1, 1, 0, 1, 1],
                [1, 0, 0, 0, 0, 0, 1, 1, 0, 1],
                [1, 1, 0, 0, 0, 0, 0, 1, 1, 0],
                ]
    prob = np.zeros((10, 10))
    for worker_idx in range(10):
        for neighbor_idx in range(worker_idx+1, 10):
            if topology[worker_idx][neighbor_idx] == 1:
                if (worker_idx < 5 and neighbor_idx < 5) or (worker_idx >= 5 and neighbor_idx >= 5):
                    prob[worker_idx][neighbor_idx] = prob_1
                else:
                    prob[worker_idx][neighbor_idx] = prob_2
    print(prob)

    topology = np.zeros((10, 10), dtype=np.int)
    for worker_idx in range(10):
        for neighbor_idx in range(worker_idx+1, 10):
            if np.random.rand() < prob[worker_idx][neighbor_idx]:
                topology[worker_idx][neighbor_idx] = 1
                topology[neighbor_idx][worker_idx] = 1

    ratios = np.ones(10)
    return topology, ratios

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

def non_iid_partition(ratio, worker_num=20):
    partition_sizes = np.ones((10, worker_num)) * ((1 - ratio) / (worker_num-2))

    for worker_idx in range(10):
        partition_sizes[worker_idx][worker_idx*2] = ratio / 2
        partition_sizes[worker_idx][worker_idx*2+1] = ratio / 2

    return partition_sizes


def partition_data(dataset_type, data_pattern, worker_num=6):      
    train_dataset, test_dataset = datasets.load_datasets(dataset_type)

    if data_pattern == 11:
        test_partition_sizes = np.ones((10, 6)) * (1 / 6)          
        
        partition_sizes = [ [0.33, 0.00, 0.34, 0.00, 0.33, 0.00], 
                            [0.33, 0.00, 0.34, 0.00, 0.33, 0.00],
                            [0.33, 0.00, 0.34, 0.00, 0.33, 0.00],
                            [0.33, 0.00, 0.34, 0.00, 0.33, 0.00],
                            [0.33, 0.00, 0.34, 0.00, 0.33, 0.00],
                            [0.00, 0.33, 0.00, 0.34, 0.00, 0.33],
                            [0.00, 0.33, 0.00, 0.34, 0.00, 0.33],
                            [0.00, 0.33, 0.00, 0.34, 0.00, 0.33],
                            [0.00, 0.33, 0.00, 0.34, 0.00, 0.33],
                            [0.00, 0.33, 0.00, 0.34, 0.00, 0.33],
                        ]
        partition_sizes = np.array(partition_sizes)
    elif data_pattern == 12:
        test_partition_sizes = np.ones((10, 6)) * (1 / 6)           
        partition_sizes = [ [0.3, 0.3, 0.3, 0.03, 0.03, 0.04], 
                            [0.3, 0.3, 0.3, 0.03, 0.03, 0.04], 
                            [0.3, 0.3, 0.3, 0.03, 0.03, 0.04], 
                            [0.3, 0.3, 0.3, 0.03, 0.03, 0.04], 
                            [0.3, 0.3, 0.3, 0.03, 0.03, 0.04], 
                            [0.03, 0.03, 0.04, 0.3, 0.3, 0.3],
                            [0.03, 0.03, 0.04, 0.3, 0.3, 0.3],
                            [0.03, 0.03, 0.04, 0.3, 0.3, 0.3],
                            [0.03, 0.03, 0.04, 0.3, 0.3, 0.3],
                            [0.03, 0.03, 0.04, 0.3, 0.3, 0.3],
                        ]
    elif data_pattern == 13:
        test_partition_sizes = np.ones((10, 6)) * (1 / 6)          
        partition_sizes = [ [0.5, 0.5, 0.0, 0.00, 0.00, 0.00], 
                            [0.5, 0.5, 0.0, 0.00, 0.00, 0.00],
                            [0.5, 0.5, 0.0, 0.00, 0.00, 0.00],
                            [0.0, 0.0, 0.5, 0.50, 0.00, 0.00],
                            [0.0, 0.0, 0.5, 0.50, 0.00, 0.00],
                            [0.0, 0.0, 0.5, 0.50, 0.00, 0.00],
                            [0.0, 0.0, 0.0, 0.00, 0.50, 0.50],
                            [0.0, 0.0, 0.0, 0.00, 0.50, 0.50],
                            [0.0, 0.0, 0.0, 0.00, 0.50, 0.50],
                            [0.0, 0.0, 0.0, 0.00, 0.50, 0.50],
                        ]
        partition_sizes = np.array(partition_sizes)
    elif data_pattern == 14:
        test_partition_sizes = np.ones((10, 6)) * (1 / 6)          
        partition_sizes = [ [0.250, 0.250, 0.125, 0.125, 0.125, 0.125],
                            [0.250, 0.250, 0.125, 0.125, 0.125, 0.125],
                            [0.250, 0.250, 0.125, 0.125, 0.125, 0.125],
                            [0.125, 0.125, 0.250, 0.250, 0.125, 0.125],
                            [0.125, 0.125, 0.250, 0.250, 0.125, 0.125],
                            [0.125, 0.125, 0.250, 0.250, 0.125, 0.125],
                            [0.125, 0.125, 0.125, 0.125, 0.250, 0.250],
                            [0.125, 0.125, 0.125, 0.125, 0.250, 0.250],
                            [0.125, 0.125, 0.125, 0.125, 0.250, 0.250],
                            [0.125, 0.125, 0.125, 0.125, 0.250, 0.250],
                        ]
        partition_sizes = np.array(partition_sizes)
    elif data_pattern == 15:
        test_partition_sizes = np.ones((10, 6)) * (1 / 6)           
        partition_sizes = [ [0.40, 0.40, 0.05, 0.05, 0.05, 0.05],
                            [0.40, 0.40, 0.05, 0.05, 0.05, 0.05],
                            [0.40, 0.40, 0.05, 0.05, 0.05, 0.05],
                            [0.05, 0.05, 0.40, 0.40, 0.05, 0.05],
                            [0.05, 0.05, 0.40, 0.40, 0.05, 0.05],
                            [0.05, 0.05, 0.40, 0.40, 0.05, 0.05],
                            [0.05, 0.05, 0.05, 0.05, 0.40, 0.40],
                            [0.05, 0.05, 0.05, 0.05, 0.40, 0.40],
                            [0.05, 0.05, 0.05, 0.05, 0.40, 0.40],
                            [0.05, 0.05, 0.05, 0.05, 0.40, 0.40]
                        ]
        partition_sizes = np.array(partition_sizes)
    else:                                                              
        test_partition_sizes = np.ones((10, 6)) * (1 / 6)  
        partition_sizes = np.ones((10, 6)) * (1 / 6)  

    # if dataset_type == "CIFAR100":
    #     test_partition_sizes = np.ones((100, worker_num)) * (1 / worker_num)
    #     partition_sizes = np.ones((100, worker_num)) * (1 / worker_num)
    # elif dataset_type == "CIFAR10":
    #     test_partition_sizes = np.ones((10, worker_num)) * (1 / worker_num)
    #     if data_pattern == 0:
    #         partition_sizes = np.ones((10, worker_num)) * (1.0 / worker_num)
    #     else:
    #         partition_sizes = non_iid_partition(data_pattern*0.1,worker_num=worker_num)

    train_data_partition = datasets.LabelwisePartitioner(train_dataset, partition_sizes=partition_sizes)
    # test_data_partition = datasets.LabelwisePartitioner(test_dataset, partition_sizes=partition_sizes)            
    test_data_partition = datasets.LabelwisePartitioner(test_dataset, partition_sizes=test_partition_sizes)     
    
    return train_data_partition, test_data_partition

if __name__ == "__main__":
    main()
