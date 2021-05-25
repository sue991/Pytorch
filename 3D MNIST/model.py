import torch.nn as nn
import torch

def layer_block(in_f, out_f, *args, **kwargs):
    return nn.Sequential(
        nn.Conv2d(in_f, out_f, *args, **kwargs),
        nn.ReLU(),
        nn.BatchNorm2d(out_f, eps=1e-05, momentum=0.1)
    )



class ModelClass(nn.Module):
    def __init__(self):
        super(ModelClass,self).__init__()

        self.layer1 = nn.Sequential(
            layer_block(3,32,kernel_size=5,padding=2), #28
            layer_block(32,32,kernel_size=5,padding=2), #28
            nn.MaxPool2d(kernel_size=2, stride=2) #14
        )
    
        self.layer2 = nn.Sequential(
            layer_block(32,64,kernel_size=3,padding=1), #14
            layer_block(64,64,kernel_size=3,padding=1), #14
            layer_block(64,64,kernel_size=3,stride = 2,padding=1), #7
            
            nn.Dropout(p=0.25),
            nn.MaxPool2d(kernel_size=2, stride=2) #3
        )
        
        self.layer3 = nn.Sequential(
            layer_block(64,128,kernel_size=3,padding=1), 
            layer_block(128,128,kernel_size=3,padding=1),#3
            layer_block(128,128,kernel_size=3,stride = 2,padding=1),
            nn.Dropout(p=0.25)
        )
        
        self.layer4 = nn.Sequential(
            layer_block(128,128,kernel_size=1), 
            layer_block(128,128,kernel_size=3,padding=1), 
            
            nn.Dropout(p=0.25),
            nn.AvgPool2d(kernel_size=2, stride=2)  #1
        )

        self.fc3 = nn.Linear(128,64, bias=True)
        torch.nn.init.xavier_uniform_(self.fc3.weight)    
        self.layer5 = nn.Sequential( 
            self.fc3, 
            nn.ReLU(),
            nn.Dropout(p= 0.5))

        self.fc5 = nn.Linear(64,10)
        
    
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = out.contiguous().view(out.size(0), -1)

        out = self.layer5(out)
        
        out = self.fc5(out)
        return out
        
