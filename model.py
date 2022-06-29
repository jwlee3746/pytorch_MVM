import math
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import grad, Variable

def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)

    
# G(z)
class Generator28(nn.Module):
    # initializers
    def __init__(self, input_size=128, out_dim = 28*28):
        super(Generator28, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(self.fc1.out_features, 512)
        self.fc3 = nn.Linear(self.fc2.out_features, 1024)
        self.fc4 = nn.Linear(self.fc3.out_features, out_dim)

    # forward method
    def forward(self, input):
        x = F.leaky_relu(self.fc1(input), 0.2)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = F.leaky_relu(self.fc3(x), 0.2)
        x = torch.tanh(self.fc4(x))
        x = x.view(-1,1,28,28)

        return x
    
class Generator64(nn.Module):
    def __init__(self, channel):
        super(Generator64, self).__init__()
        self.nz = 128
        self.n = 128
        n = self.n
        
        self.linear1 = nn.Linear(self.nz, n * 4 * 4)
        self.model = nn.Sequential(  
            ResidualBlock(n, n, 3, resample='up'),
            ResidualBlock(n, n, 3, resample='up'),
            ResidualBlock(n, n, 3, resample='up'),
            ResidualBlock(n, n, 3, resample='up'),
            nn.BatchNorm2d(n),
            nn.ReLU(inplace=True),
            nn.Conv2d(n, channel, 3, padding=(3-1)//2),  
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.view(-1, self.nz)
        x = self.linear1(x).view(-1, self.n, 4, 4)        
        img = self.model(x)
        return img
    
class discriminator(nn.Module):
    # initializers
    def __init__(self, input_size=28*28, out_dim=10):
        super(ML28, self).__init__()
        self.fc1 = nn.Linear(input_size, 1024)
        self.fc2 = nn.Linear(self.fc1.out_features, 512)
        self.fc3 = nn.Linear(self.fc2.out_features, 256)
        self.fc4 = nn.Linear(self.fc3.out_features, out_dim)

    # forward method
    def forward(self, input):
        x = F.leaky_relu(self.fc1(input), 0.2)
        x = F.dropout(x, 0.3)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = F.dropout(x, 0.3)
        x = F.leaky_relu(self.fc3(x), 0.2)
        x = F.dropout(x, 0.3)
        x = torch.sigmoid(self.fc4(x))

        return x

class ML28(nn.Module):
    def __init__(self, out_dim=10, channel=1):
        super(ML28, self).__init__()
        dim = 32

        self.image_to_features = nn.Sequential(
            nn.Conv2d(channel, dim, 2, 2, 1), #in_channels, out_channels, kernel_size, stride=1, padding=0,
            nn.LeakyReLU(0.2),
            nn.Conv2d(dim, 2 * dim, 2, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(2 * dim, 4 * dim, 2, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(4 * dim, 8 * dim, 2, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(8 * dim, 16 * dim, 2, 2, 1),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.features_to_prob = nn.Linear(16 * dim, out_dim)


    def forward(self, x):
        batch_size = x.size()[0]
        x = self.image_to_features(x)
        x = x.view(batch_size, -1)
        return self.features_to_prob(x)
    
class ML64(nn.Module):
    def __init__(self, out_dim=10, channel=3):
        super(ML64, self).__init__()
        dim = 32

        self.image_to_features = nn.Sequential(
            nn.Conv2d(channel, dim, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(dim, 2 * dim, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(2 * dim, 4 * dim, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(4 * dim, 8 * dim, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(8 * dim, 16 * dim, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.features_to_prob = nn.Linear(16 * dim, out_dim)


    def forward(self, x):
        batch_size = x.size()[0]
        x = self.image_to_features(x)
        x = x.view(batch_size, -1)
        return self.features_to_prob(x)
    


class MeanPoolConv(nn.Module):
    def __init__(self, n_input, n_output, k_size):
        super(MeanPoolConv, self).__init__()
        conv1 = nn.Conv2d(n_input, n_output, k_size, stride=1, padding=(k_size-1)//2, bias=True)
        self.model = nn.Sequential(conv1)
    def forward(self, x):
        out = (x[:,:,::2,::2] + x[:,:,1::2,::2] + x[:,:,::2,1::2] + x[:,:,1::2,1::2]) / 4.0
        out = self.model(out)
        return out

class ConvMeanPool(nn.Module):
    def __init__(self, n_input, n_output, k_size):
        super(ConvMeanPool, self).__init__()
        conv1 = nn.Conv2d(n_input, n_output, k_size, stride=1, padding=(k_size-1)//2, bias=True)
        self.model = nn.Sequential(conv1)
    def forward(self, x):
        out = self.model(x)
        out = (out[:,:,::2,::2] + out[:,:,1::2,::2] + out[:,:,::2,1::2] + out[:,:,1::2,1::2]) / 4.0
        return out

class UpsampleConv(nn.Module):
    def __init__(self, n_input, n_output, k_size):
        super(UpsampleConv, self).__init__()

        self.model = nn.Sequential(
            nn.PixelShuffle(2),
            nn.Conv2d(n_input, n_output, k_size, stride=1, padding=(k_size-1)//2, bias=True)
        )
    def forward(self, x):
        x = x.repeat((1, 4, 1, 1))
        out = self.model(x)
        return out

class ResidualBlock(nn.Module):
    def __init__(self, n_input, n_output, k_size, resample='up', bn=True, spatial_dim=None):
        super(ResidualBlock, self).__init__()

        self.resample = resample

        if resample == 'up':
            self.conv1 = UpsampleConv(n_input, n_output, k_size)
            self.conv2 = nn.Conv2d(n_output, n_output, k_size, padding=(k_size-1)//2)
            self.conv_shortcut = UpsampleConv(n_input, n_output, k_size)
            self.out_dim = n_output
        elif resample == 'down':
            self.conv1 = nn.Conv2d(n_input, n_input, k_size, padding=(k_size-1)//2)
            self.conv2 = ConvMeanPool(n_input, n_output, k_size)
            self.conv_shortcut = ConvMeanPool(n_input, n_output, k_size)
            self.out_dim = n_output
            self.ln_dims = [n_input, spatial_dim, spatial_dim] # Define the dimensions for layer normalization.
        else:
            self.conv1 = nn.Conv2d(n_input, n_input, k_size, padding=(k_size-1)//2)
            self.conv2 = nn.Conv2d(n_input, n_input, k_size, padding=(k_size-1)//2)
            self.conv_shortcut = None # Identity
            self.out_dim = n_input
            self.ln_dims = [n_input, spatial_dim, spatial_dim]

        self.model = nn.Sequential(
            nn.BatchNorm2d(n_input) if bn else nn.LayerNorm(self.ln_dims),
            nn.ReLU(inplace=True),
            self.conv1,
            nn.BatchNorm2d(self.out_dim) if bn else nn.LayerNorm(self.ln_dims),
            nn.ReLU(inplace=True),
            self.conv2,
        )

    def forward(self, x):
        if self.conv_shortcut is None:
            return x + self.model(x)
        else:
            return self.conv_shortcut(x) + self.model(x)

class DiscBlock1(nn.Module):
    def __init__(self, n_output):
        super(DiscBlock1, self).__init__()

        self.conv1 = nn.Conv2d(3, n_output, 3, padding=(3-1)//2)
        self.conv2 = ConvMeanPool(n_output, n_output, 1)
        self.conv_shortcut = MeanPoolConv(3, n_output, 1)

        self.model = nn.Sequential(
            self.conv1,
            nn.ReLU(inplace=True),
            self.conv2
        )

    def forward(self, x):
        return self.conv_shortcut(x) + self.model(x)
    
"""
from torchsummary import summary as summary_
device = torch.device('cuda')
model1 = Generator64(channel=3).to(device)
model2 = ML64(out_dim=10, channel=3).to(device)
print(summary_(model2, (1,64,64)))
print(summary_(model1, (128,1,1)))
print(model2(torch.rand(1,1,64,64).to(device)).shape)
print(model1(torch.rand(1,128,1,1).to(device)).shape)
"""

"""
from torchsummary import summary as summary_
device = torch.device('cuda')
model1 = Generator28().to(device)
model2 = ML28().to(device)
print(summary_(model2, (1, 28, 28)))
print(summary_(model1, (1, 128)))
print(model2(torch.rand(1, 1, 28, 28).to(device)).shape)
print(model1(torch.rand(1, 128).to(device)).shape)
"""
