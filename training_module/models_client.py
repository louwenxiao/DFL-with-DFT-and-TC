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


class classify_VGG9(nn.Module):
    def __init__(self, num_classes=10):
        super(classify_VGG9, self).__init__()
        self.conv_layer = nn.Sequential(
            # Conv Layer block 1
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.conv2 = nn.Sequential(
            # Conv Layer block 2
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.05),

            # Conv Layer block 3
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),)

        self.fc_layer = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(4096, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        # print(x.size()[1])
        e = x
        if x.size()[1] != 64:       # 如果输入参数为特征，那么不在使用conv_layer，而是直接从下一步计算
            x = self.conv_layer(x)
            e = x
        print(x.size())
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layer(x)
        return e,x
    

    
class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.features1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            
        )
        self.features2 = nn.Sequential(
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(256 * 2 * 2, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 10),
        )

    def forward(self, x,z=None):
            
        e = x
        if x.size()[1] != 64:       # 如果输入参数为特征，那么不在使用conv_layer，而是直接从下一步计算
            x = self.features1(x)
            e = x
        x = self.features2(x)
        x = x.view(-1, 256 * 2 * 2)
        x = self.classifier(x)
        return e,x
        

    
def create_model_instance(dataset_type, model_type):
    return classify_VGG9()

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
