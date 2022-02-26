import os
import sys
import time
from time import sleep
import shutil
import subprocess

# python_path = '/opt/anaconda3/envs/edge/bin/python'
python_path = '/data/yxu/software/Anaconda/envs/torch1.6/bin/python'
# python_path = '/home/edge/anaconda3/envs/torch1.6/bin/python'
source_code_path = '/data/lwang/distirbuted-model-training-p-vgg-alg'
kill_server_cmd = """ pkill -f "server\.py --batch_size.*" """
kill_client_cmd = """ pkill -f "client\.py --master_ip.*" """

print(os.getcwd())

def create_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

# create exp dir and backup codes
def backup_codes(source_path, exp_path):
    print(exp_path)
    if os.path.exists(exp_path):
        print("exp dir exist!")
        return False

    for root, dirs, files in os.walk(source_path):
        if '/.' in root:
            continue

        for fl in files:
            fl_type = os.path.splitext(fl)[-1]
            if fl_type == '.py' or fl_type == '.json':
                dst_dir = root.replace(source_path, exp_path)
                create_dir(dst_dir)
                src_file = os.path.join(root, fl)
                dst_file = os.path.join(dst_dir, fl)
                shutil.copy(src_file, dst_file)
    
    return True

def excute_func(model_types, dataset_types, modes, prob, lrs, decay_rates, weight_decays, data_pattern, topologys, alphas, epoch, batch_size=[32], local_iters=[-1]):
    for mode in modes:
        for bs in batch_size:
            for model_type in model_types:
                for alpha in alphas:
                    for rt in prob:
                        for topo in topologys:
                            for lr in lrs:
                                for decay_rate in decay_rates:
                                    for wd in weight_decays:
                                        for local_iter in local_iters:
                                            for dataset_type in dataset_types:
                                                for dt in data_pattern:
                                                        complete = False
                                                        repeat = 0
                                                        while not complete:
                                                            os.system(kill_server_cmd)
                                                            os.system(kill_client_cmd)
                                                            exp_result_path = '/data/lwang/experiment_result_p/simulated_0514_ring_baseline/'\
                                                                    'mode-{}_topo-{}_modeltype{}_datatype{}_datapattern{}_lr{}_decayrate{}weightdecay{}_localsteps{}_'\
                                                                    'batchsize{}_prob{}_alpha{}'.format(mode, topo, model_type, dataset_type, dt, lr, decay_rate, wd, local_iter, bs, rt, alpha)
                                                            if backup_codes(source_code_path, exp_result_path):
                                                                cmd = 'cd ' + exp_result_path + ";" + python_path + ' -u server.py --batch_size ' + str(bs) \
                                                                        + ' --model_type ' + model_type +  ' --dataset_type ' + dataset_type + ' --prob ' + str(rt) + ' --lr ' + str(lr) \
                                                                        + ' --decay_rate ' + str(decay_rate) + ' --alpha ' + str(alpha) + ' --topology ' + topo\
                                                                        + ' --weight_decay ' + str(wd) + ' --local_updates ' + str(local_iter) + ' --data_pattern '\
                                                                        + str(dt) + ' --mode ' + mode + ' --epoch ' + str(epoch) + ' > resluts.txt'
                                                                time_start = time.time()
                                                                result = subprocess.call(cmd, shell=True)
                                                                
                                                                # sys.exit()
                                                                if time.time() - time_start < 300:
                                                                    shutil.rmtree(exp_result_path)
                                                                else:
                                                                    complete = True
                                                            else:
                                                                complete = True
                                                            repeat += 1
                                                            if repeat > 4:
                                                                break

model_types = ["VGG"] # "VGG", "ResNet"
dataset_types = ["CIFAR10"] # "CIFAR10", "CIFAR100"
modes = ["static"] # "adaptive", "rate-adaptive", "topo-adaptive", "static"
# prob = [0.75, 0.5, 0.25, 1.0]
# prob = [3, 1, 2, 4, 5]
prob = [0.0]
lrs = [0.05]
decay_rates = [0.98]
weight_decays = [5e-4]
data_pattern = [2, 4, 6, 8, 1]
topologys = ["ring"]
alphas = [0]
epoch = 600
batch_size = [32]
local_iters = [-1]
excute_func(model_types, dataset_types, modes, prob, lrs, decay_rates, weight_decays, data_pattern, topologys, alphas, epoch)

# model_types = ["VGG"] # "VGG", "ResNet"
# dataset_types = ["CIFAR10"] # "CIFAR10", "CIFAR100"
# modes = ["static"] # "adaptive", "rate-adaptive", "topo-adaptive", "static"
# # prob = [0.75, 0.5, 0.25, 1.0]
# # prob = [3, 1, 2, 4, 5]
# prob = [0.0]
# lrs = [0.05]
# decay_rates = [0.98]
# weight_decays = [5e-4]
# data_pattern = [2, 3, 4, 5, 1, 6, 7, 8, 9]
# topologys = ["ring_6"]
# alphas = [0]
# epoch = 600
# batch_size = [32]
# local_iters = [-1]
# excute_func(model_types, dataset_types, modes, prob, lrs, decay_rates, weight_decays, data_pattern, topologys, alphas, epoch)
