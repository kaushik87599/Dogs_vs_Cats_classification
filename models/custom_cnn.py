import torch
from torch import nn    
from models.basic_cnn import basic_cnn
from models.attention_block import attention_cnn


## adds or rather extends the basic cnn with attention block cnn
class custom_attention_cnn(torch.nn.Module):
    
    def __init__(self):
        super(custom_attention_cnn, self).__init__()
        
        self.base_cnn = basic_cnn()
        self.attention_block = attention_cnn()
        self.residual = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Flatten(),
            nn.Linear(in_features=512, out_features=2)
        )
        

    def forward(self, x):
        x = self.base_cnn(x)
        x = self.attention_block(x)
        x= self.residual(x)
        
        return x
    
# basic cnn 
class custom_basic_cnn(torch.nn.Module):
    
    def __init__(self):
        super(custom_basic_cnn, self).__init__()
        self.base_cnn = basic_cnn()
        self.residual = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Flatten(),
            nn.Linear(in_features=512, out_features=2)
        )
        
    def forward(self, x):
        x = self.base_cnn(x)
        x= self.residual(x)
        return x
    
def get_models():
    '''Returns attention and basic cnn models respectively'''
    
    return custom_attention_cnn(), custom_basic_cnn()