import torch
from torch import nn    


## adds or rather extends the basic cnn with attention block cnn
class custom_attention_cnn(torch.nn.Module):
    def __init__(self, base_cnn, attention_block):
        super(custom_attention_cnn, self).__init__()
        self.base_cnn = base_cnn
        self.attention_block = attention_block
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
    
