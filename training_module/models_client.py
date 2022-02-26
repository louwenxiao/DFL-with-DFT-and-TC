Skip to content
Search or jump to…
Pull requests
Issues
Marketplace
Explore
 
@louwenxiao 
louwenxiao
/
P2P_KD
Public
Code
Issues
Pull requests
Actions
Projects
Wiki
Security
Insights
Settings
P2P_KD/client_module/models_client.py /
@louwenxiao
louwenxiao Update models_client.py
Latest commit 06b345e in 26 seconds
 History
 1 contributor
102 lines (85 sloc)  3.35 KB

import torch
import torch.nn.functional as F
import torch.nn as nn


class AlexNet(nn.Module):
    def __init__(self, class_num=10):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 192, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.f2= nn.Sequential(
            nn.Conv2d(192, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 4 * 4, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, class_num),
        )

    def forward(self, x):
        e = x
        if x.size()[1] != 192:       # 如果输入参数为特征，那么不在使用conv_layer，而是直接从下一步计算
            x = self.features(x)
            e = x
        x = self.f2(x)
        x = x.view(x.size(0), 256 * 4 * 4)
        x = self.classifier(x)
        return e,x
    
def create_model_instance(dataset_type, model_type):
    return AlexNet()




© 2021 GitHub, Inc.
Terms
Privacy
Security
Status
Docs
Contact GitHub
Pricing
API
Training
Blog
About
