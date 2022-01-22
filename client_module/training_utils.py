import sys
import time
import math
import re
import gc
import random
import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


CE_loss = nn.CrossEntropyLoss()
KL_loss = nn.KLDivLoss(reduction='mean')
LogSoftmax = nn.LogSoftmax(dim=1)
Softmax = nn.Softmax(dim=1)

def train(model, data_loader, optimizer, local_iters=None, device=torch.device("cpu"), 
          model_type=None,num_data=0):
    model.train()
    if local_iters is None:
        local_iters = math.ceil(len(data_loader.loader.dataset) / data_loader.loader.batch_size)
    print("local_iters: ", local_iters)

    train_loss = 0.0
    samples_num = 0
    tensor_data = []
    for iter_idx in range(local_iters):
        data, target = next(data_loader)
        
        data, target = data.to(device), target.to(device)
        x_data,output = model(data)

        optimizer.zero_grad()
        loss = CE_loss(output, target)
        loss.backward()
        optimizer.step()

        train_loss += (loss.item() * data.size(0))
        samples_num += data.size(0)
        x_data, output, target = x_data.to(torch.device("cpu")),output.to(torch.device("cpu")), target.to(torch.device("cpu"))
        if len(tensor_data) < num_data:
            if len(tensor_data) == 0:tensor_data.append((x_data, output, target))
            elif random.random() < 0.5:tensor_data.append((x_data, output, target))
    
    key = ['features.0.weight', 'features.0.bias', 'features.3.weight', 'features.3.bias', 'features.6.weight', 
           'features.6.bias', 'features.8.weight', 'features.8.bias', 'features.10.weight', 'features.10.bias']
    model_dict = dict()
    for para in model.state_dict().keys():
        if para in key:
            model_dict[para] = copy.deepcopy(model.state_dict()[para])

    if samples_num != 0:
        train_loss /= samples_num
    
    return tensor_data,train_loss,model_dict

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

            test_loss += CE_loss(output, target).item()
            # get the index of the max log-probability
            pred = output.argmax(1, keepdim=True)
            batch_correct = pred.eq(target.view_as(pred)).sum().item()

            correct += batch_correct

    test_loss /= len(data_loader.dataset)
    test_accuracy = np.float(1.0 * correct / len(data_loader.dataset))

    return test_loss, test_accuracy

def train_neighbor_data(model, data_loader, optimizer, device=torch.device("cpu")):
    model.train()
    kl_loss = KL_Loss()

    train_loss = 0.0
    samples_num = 0

    raw_data = []
    for d in data_loader:
        raw_data.extend(d)
    np.random.shuffle(raw_data)
    for i in range(1):
        for x_data, logit, label in raw_data:

    # for data in data_loader:
    #     for x_data, logit, label in data:
        
            x_data, logit, label = x_data.to(device), logit.to(device), label.to(device)
            # print(x_data)
            # print(label)
            
            _,output = model(x_data)

            optimizer.zero_grad()
        
            alpha = 1
            whether_distill_on_the_server = 0
            
            if whether_distill_on_the_server == 1:
                loss_kd = kl_loss(output, logit).to(device)
                loss_true = CE_loss(output, label).to(device)
                loss = loss_kd + alpha * loss_true
            else:
                loss_true = CE_loss(output, label).to(device)
                loss = loss_true
            loss.backward()
            optimizer.step()
        # print(label)

    if samples_num != 0:
        train_loss /= samples_num
    
    return train_loss

# KL损失函数
class KL_Loss(nn.Module):
    def __init__(self, temperature=3.0):
        super(KL_Loss, self).__init__()
        self.T = temperature

    def forward(self, output_batch, teacher_outputs):
        output_batch = F.log_softmax(output_batch / self.T, dim=1)
        teacher_outputs = F.softmax(teacher_outputs / self.T, dim=1) + 10 ** (-7)

        loss = self.T * self.T * nn.KLDivLoss(reduction='batchmean')(output_batch, teacher_outputs)

        return loss


