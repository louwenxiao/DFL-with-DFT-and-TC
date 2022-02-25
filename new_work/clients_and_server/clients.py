import torch
from torch import nn
from torch import optim
from data_and_model import datasets
import numpy as np
import math
import random
import copy
import gc
import time
from data_and_model.models import create_model_instance



class client(object):
    def __init__(self,i,device,args,data):
        self.id = i
        self.num_data = args.num_data
        self.device = device
        self.device_cpu = torch.device("cpu")
        self.args = args
        self.train_loader, self.test_loader = data
        self.local_model = create_model_instance(args.dataset)
        self.local_model.load_state_dict(torch.load("cache\model.pt"))

    def local_train(self,lr):
        self.local_model.to(self.device)
        optimizer = optim.SGD(self.local_model.parameters(), lr=lr, momentum=0.9, weight_decay=self.args.weight_decay)
        tensor_data, train_loss = train(self.local_model, self.train_loader, optimizer,
                                local_iters=self.args.local_updates,device=self.device, num_data=self.num_data)
        self.local_model.to(self.device_cpu)

        key = return_key()
        model_dict = dict()
        for para in self.local_model.state_dict().keys():
            if para in key:
                model_dict[para] = copy.deepcopy(self.local_model.state_dict()[para])


        return model_dict,tensor_data


    # #模型聚合

    def aggregate_model_and_data(self,topo,model_dir,data_dir,lr):
        model_para = copy.deepcopy(self.local_model.state_dict())
        client_data = []
        # 下面聚合数据
        for i in range(len(topo[self.id])):
            if topo[self.id][i] == 1:
                client_data.extend(data_dir[i])
        # 聚合模型
        key = return_key()
        for para in model_para.keys():
            if para not in key:
                continue
            else:
                for i in range(len(topo[0])):
                    if topo[self.id][i] == 1:
                        model_para[para] = model_para[para] + model_dir[i][para]
                # model_para[para] = model_para[para] / (len(topo[0]) + 1)
                model_para[para] = model_para[para] / 3.0
        self.local_model.load_state_dict(model_para)
        np.random.shuffle(client_data)
        optimizer = optim.SGD(self.local_model.parameters(), lr=lr, momentum=0.9, weight_decay=self.args.weight_decay)
        self.local_model.to(self.device)
        train_loss = train_neighbor_data(self.local_model, client_data, optimizer, device=self.device)
        self.local_model.to(self.device_cpu)

    # #模型测试
    def test_model(self):
        self.local_model.to(self.device)
        test_loss, test_accuracy = test(self.local_model, self.test_loader, device=self.device)
        self.local_model.to(self.device_cpu)

        return test_loss, test_accuracy




def count_dataset(loader):
    counts = np.zeros(len(loader.loader.dataset.classes))
    for _, target in loader.loader:
        labels = target.view(-1).numpy()
        for label in labels:
            counts[label] += 1
    print("class counts: ", counts)
    print("total data count: ", np.sum(counts))

def train(model, data_loader, optimizer, local_iters=None, device=torch.device("cpu"),
          model_type=None, num_data=0):
    CE_loss = nn.CrossEntropyLoss()
    model.train()

    num = list(range(local_iters))
    sample = random.sample(num, num_data)

    train_loss = 0.0
    samples_num = 0
    tensor_data = []
    for iter_idx in range(local_iters):
        data, target = next(data_loader)
        data, target = data.to(device), target.to(device)
        x_data, output = model(data)

        optimizer.zero_grad()
        loss = CE_loss(output, target)
        loss.backward()
        optimizer.step()

        train_loss += (loss.item() * data.size(0))
        samples_num += data.size(0)
        x_data, target = x_data.to(torch.device("cpu")), target.to(torch.device("cpu"))
        if iter_idx in sample:
            tensor_data.append((x_data, target))
    if samples_num != 0:
        train_loss /= samples_num

    return tensor_data, train_loss

def train_neighbor_data(model, data_loader, optimizer, device=torch.device("cpu")):
    model.train()
    CE_loss = nn.CrossEntropyLoss()
    train_loss = 0.0
    samples_num = 0

    for x_data, label in data_loader:
        data = x_data.detach().numpy()
        x_data = torch.tensor(data, requires_grad=False)
        x_data, label = x_data.to(device), label.to(device)
        _, output = model(x_data)

        loss = CE_loss(output, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    if samples_num != 0:
        train_loss /= samples_num
    return train_loss

def test(model, data_loader, device=torch.device("cpu"), model_type=None):
    model.eval()
    data_loader = data_loader.loader
    test_loss = 0.0
    test_accuracy = 0.0

    correct = 0

    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            _,output = model(data)

            test_loss += nn.CrossEntropyLoss()(output, target).item()
            # get the index of the max log-probability
            pred = output.argmax(1, keepdim=True)
            batch_correct = pred.eq(target.view_as(pred)).sum().item()

            correct += batch_correct

    test_loss /= len(data_loader.dataset)
    test_accuracy = np.float(1.0 * correct / len(data_loader.dataset))
    return test_loss, test_accuracy




def return_key():
    key = ['features1.0.weight', 'features1.0.bias', 'features1.1.weight', 'features1.1.bias',
           'features1.1.running_mean', 'features1.1.running_var', 'features1.1.num_batches_tracked',
           'features1.3.weight', 'features1.3.bias', 'features1.4.weight', 'features1.4.bias',
           'features1.4.running_mean', 'features1.4.running_var', 'features1.4.num_batches_tracked',
           'features1.7.weight', 'features1.7.bias', 'features1.8.weight', 'features1.8.bias',
           'features1.8.running_mean', 'features1.8.running_var', 'features1.8.num_batches_tracked',
           'features1.10.weight', 'features1.10.bias', 'features1.11.weight', 'features1.11.bias',
           'features1.11.running_mean', 'features1.11.running_var', 'features1.11.num_batches_tracked',
           'features1.14.weight', 'features1.14.bias', 'features1.15.weight', 'features1.15.bias',
           'features1.15.running_mean', 'features1.15.running_var', 'features1.15.num_batches_tracked',
           'features1.17.weight', 'features1.17.bias', 'features1.18.weight', 'features1.18.bias',
           'features1.18.running_mean', 'features1.18.running_var', 'features1.18.num_batches_tracked',
           'features1.20.weight', 'features1.20.bias', 'features1.21.weight', 'features1.21.bias',
           'features1.21.running_mean', 'features1.21.running_var', 'features1.21.num_batches_tracked']

    key2 = ['features1.0.weight', 'features1.0.bias', 'features1.1.weight', 'features1.1.bias',
            'features1.1.running_mean', 'features1.1.running_var', 'features1.1.num_batches_tracked',
            'features1.3.weight', 'features1.3.bias', 'features1.4.weight', 'features1.4.bias',
            'features1.4.running_mean', 'features1.4.running_var', 'features1.4.num_batches_tracked',
            'features1.7.weight', 'features1.7.bias', 'features1.8.weight', 'features1.8.bias',
            'features1.8.running_mean', 'features1.8.running_var', 'features1.8.num_batches_tracked',
            'features1.10.weight', 'features1.10.bias', 'features1.11.weight', 'features1.11.bias',
            'features1.11.running_mean', 'features1.11.running_var', 'features1.11.num_batches_tracked']

    return key2