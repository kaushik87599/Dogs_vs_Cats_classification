# A baseline CNN with 3-4 convolutional layers, max-pooling, and dropout.
from torch import nn
class Layer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Layer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(num_features=out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.maxpool(x)
        return x

def basic_cnn():
    ''' Basic CNN with 4 convolutional layers, max-pooling, dropout(0.5), flatten and linear (in=512,out=2). '''
    model = nn.Sequential(
        Layer(in_channels=3, out_channels=32),
        Layer(in_channels=32, out_channels=64),
        Layer(in_channels=64, out_channels=128),
        Layer(in_channels=128, out_channels=256),
        nn.Dropout(p=0.5),
        nn.Flatten(),
        nn.Linear(in_features=512, out_features=2)
    )
    return model
        
  
    