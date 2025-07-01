import torch


## adds or rather extends the basic cnn with attention block cnn
class CustomCNN(torch.nn.Module):
    def __init__(self, base_cnn, attention_block):
        super(CustomCNN, self).__init__()
        self.base_cnn = base_cnn
        self.attention_block = attention_block

    def forward(self, x):
        x = self.base_cnn(x)
        x = self.attention_block(x)
        x= x.dropout(p=0.5)
        return x
    
