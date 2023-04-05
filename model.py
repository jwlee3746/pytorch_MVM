import math
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import grad, Variable

def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
    
  
##################################### Gen model ######################################    
class Generator28_linear(nn.Module):
    # initializers
    def __init__(self, input_size=128, out_dim = 28*28):
        super(Generator28_linear, self).__init__()
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
    
class Generator32(nn.Module):
    def __init__(self, channel=3):
        super(Generator32, self).__init__()
        self.nz = 128
        self.n = 128
        n = self.n
        
        self.linear1 = nn.Linear(self.nz, n * 1 * 1)
        self.model = nn.Sequential(  
            nn.ConvTranspose2d(n, n * 4, 4, 1, 0, bias=False),
            nn.BatchNorm2d(n*4),
            nn.ReLU(True),
            nn.ConvTranspose2d(n * 4, n * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(n*2),
            nn.ReLU(True),
            nn.ConvTranspose2d(n * 2, n, 4, 2, 1, bias=False),
            nn.BatchNorm2d(n),
            nn.ReLU(True),
            nn.ConvTranspose2d(n, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )
    def forward(self, x):
        x = x.view(-1, self.nz)
        x = self.linear1(x).view(-1, self.n, 1, 1)      
        img = self.model(x)
        return img

    
class Generator64(nn.Module):
    def __init__(self, nz = 128, channel=3):
        super(Generator64, self).__init__()
        self.nz = nz
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

class Generator128(nn.Module):
    def __init__(self, nz = 128, channel=3):
        super(Generator128, self).__init__()
        self.nz = nz
        self.n = 128
        n = self.n
        
        self.linear1 = nn.Linear(self.nz, n * 8 * 8)
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
        x = self.linear1(x).view(-1, self.n, 8, 8)        
        img = self.model(x)
        return img

class Generator512(nn.Module):
    def __init__(self, nz = 128, channel=3):
        super(Generator512, self).__init__()
        self.nz = nz
        self.n = 128
        n = self.n
        
        self.linear1 = nn.Linear(self.nz, n * 8 * 8)
        self.model = nn.Sequential(  
            ResidualBlock(n, n, 3, resample='up'),
            ResidualBlock(n, n, 3, resample='up'),
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
        x = self.linear1(x).view(-1, self.n, 8, 8)        
        img = self.model(x)
        return img
    
##################################### ML model ######################################    
class ML28(nn.Module):
    def __init__(self, out_dim, channel=1):
        super(ML28, self).__init__()
        dim = 32

        self.image_to_features = nn.Sequential(
            nn.Conv2d(channel, dim, 3, 1, 1), #in_channels, out_channels, kernel_size, stride=1, padding=0,
            nn.LeakyReLU(0.2),
            nn.Conv2d(dim, 2 * dim, 3, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(2 * dim, 2 * dim, 3, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(2 * dim, 2 * dim, 3, 1, 1),
            nn.LeakyReLU(0.2)
        )

        self.features_to_prob = nn.Linear(3136, out_dim)


    def forward(self, x):
        batch_size = x.size()[0]
        x = self.image_to_features(x)
        x = x.view(batch_size, -1)
        return self.features_to_prob(x)

class ML32(nn.Module):
    def __init__(self, out_dim, channel=3):
        super(ML32, self).__init__()
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
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.features_to_prob = nn.Linear(8 * dim, out_dim)


    def forward(self, x):
        batch_size = x.size()[0]
        x = self.image_to_features(x)
        x = x.view(batch_size, -1)
        return self.features_to_prob(x)
    
class ML64(nn.Module):
    def __init__(self, out_dim, channel=3):
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
        '''
        self.last = hypnn.ToPoincare(
            c=0.1,
            ball_dim=out_dim,
            riemannian=False,
            clip_r=None,
        )
        self.features_to_prob = nn.Sequential(nn.Linear(16 * dim, out_dim), self.last)
        '''
        self.features_to_prob = nn.Linear(16 * dim, out_dim)

    def forward(self, x):
        batch_size = x.size()[0]
        x = self.image_to_features(x)
        x = x.view(batch_size, -1)
        return self.features_to_prob(x)

    
class ML128(nn.Module):
    def __init__(self, out_dim, channel=3):
        super(ML128, self).__init__()
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
            nn.Conv2d(16 * dim, 32 * dim, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.features_to_prob = nn.Linear(32 * dim, out_dim)


    def forward(self, x):
        batch_size = x.size()[0]
        x = self.image_to_features(x)
        x = x.view(batch_size, -1)
        return self.features_to_prob(x)


class ML512(nn.Module):
    def __init__(self, out_dim, channel=3):
        super(ML512, self).__init__()
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
            nn.Conv2d(16 * dim, 32 * dim, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32 * dim, 64 * dim, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64 * dim, 128 * dim, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.features_to_prob = nn.Linear(128 * dim, out_dim)


    def forward(self, x):
        batch_size = x.size()[0]
        x = self.image_to_features(x)
        x = x.view(batch_size, -1)
        return self.features_to_prob(x)
    
##################################### utils  ######################################    
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

##################################### Mapping network  ###################################### 
class MappingNetwork(nn.Module):
    """Create mapping network that project latent space to style space"""

    def __init__(self, features, n_layers):
        super().__init__()
        layers = []
        for i in range(n_layers):
            layers.append(EqualizedLinear(features, features))
            layers.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))
        self.net = nn.Sequential(*layers)

    def forward(self, z):
        z = F.normalize(z, dim=1)
        return self.net(z)

class EqualizedLinear(nn.Module):
    """
    This applies the equalized weight method for a linear layer
    input:
      input features
      output features
      bias
    return:
      y = wx+b
    """

    def __init__(self, in_features, out_features, bias=0.0):
        super().__init__()
        self.weight = EqualizedWeight([out_features, in_features])
        self.bias = nn.Parameter(torch.ones(out_features) * bias)

    def forward(self, x):
        return F.linear(x, self.weight(), bias=self.bias)

class EqualizedWeight(nn.Module):
    """
    Introduced in https://arxiv.org/pdf/1710.10196.pdf
    Instead of draw weight from a normal distribution of (0,c),
    It draws from (0,1) instead, and times the constant c.
    so that when using optimizer like Adam which will normalize through
    all gradients, which may be a problem if the dynamic range of the weight
    is too large.
    input:
    shape: [in_features,out_features]
    return:
    Randomized weight for corresponding layer
    """

    def __init__(self, shape):
        super().__init__()
        self.c = 1 / math.sqrt(np.prod(shape[1:]))
        self.weight = nn.Parameter(torch.randn(shape))

    def forward(self):
        return self.weight * self.c

##################################### DCGAN  ###################################### 
# G(z)
class gen64_DCGAN(nn.Module):
    # initializers
    def __init__(self, nz=100, channel=3, d=128):
        super(gen64_DCGAN, self).__init__()
        self.deconv1 = nn.ConvTranspose2d(nz, d*8, 4, 1, 0)
        self.deconv1_bn = nn.BatchNorm2d(d*8)
        self.deconv2 = nn.ConvTranspose2d(d*8, d*4, 4, 2, 1)
        self.deconv2_bn = nn.BatchNorm2d(d*4)
        self.deconv3 = nn.ConvTranspose2d(d*4, d*2, 4, 2, 1)
        self.deconv3_bn = nn.BatchNorm2d(d*2)
        self.deconv4 = nn.ConvTranspose2d(d*2, d, 4, 2, 1)
        self.deconv4_bn = nn.BatchNorm2d(d)
        self.deconv5 = nn.ConvTranspose2d(d, channel, 4, 2, 1)

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, input):
        # x = F.relu(self.deconv1(input))
        x = F.relu(self.deconv1_bn(self.deconv1(input)))
        x = F.relu(self.deconv2_bn(self.deconv2(x)))
        x = F.relu(self.deconv3_bn(self.deconv3(x)))
        x = F.relu(self.deconv4_bn(self.deconv4(x)))
        x = F.tanh(self.deconv5(x))

        return x
    
class ML64_DCGAN(nn.Module):
    # initializers
    def __init__(self, out_dim, channel=3, d=128):
        super(ML64_DCGAN, self).__init__()
        
        self.conv1 = nn.Conv2d(channel, d, 4, 2, 1)
        self.conv2 = nn.Conv2d(d, d*2, 4, 2, 1)
        self.conv2_bn = nn.BatchNorm2d(d*2)
        self.conv3 = nn.Conv2d(d*2, d*4, 4, 2, 1)
        self.conv3_bn = nn.BatchNorm2d(d*4)
        self.conv4 = nn.Conv2d(d*4, d*8, 4, 2, 1)
        self.conv4_bn = nn.BatchNorm2d(d*8)
        self.conv5 = nn.Conv2d(d*8, out_dim, 4, 1, 0)

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, input):
        batch_size = input.size()[0]
        x = F.leaky_relu(self.conv1(input), 0.2)
        x = F.leaky_relu(self.conv2_bn(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.conv3_bn(self.conv3(x)), 0.2)
        x = F.leaky_relu(self.conv4_bn(self.conv4(x)), 0.2)
        x = F.sigmoid(self.conv5(x))
        x = x.view(batch_size, -1)
        return x

class discriminator(nn.Module):
    
    # initializers
    def __init__(self, input_dim, n_class=1):
        super(discriminator, self).__init__()
        self.fc1 = nn.Linear(input_dim, n_class)

    # forward method
    def forward(self, input):
        x = F.sigmoid(self.fc1(input))

        return x
    '''
    def __init__(self, input_dim, n_layers, n_class=1):
        super(discriminator, self).__init__()
        layers = []
        for i in range(n_layers):
            layers.append(nn.Linear(input_dim, input_dim))
            layers.append(nn.LeakyReLU(0.2))
        layers.append(nn.Linear(input_dim, n_class))
        self.net = nn.Sequential(*layers)

    def forward(self, input):
        return F.sigmoid(self.net(input))
    '''
'''
from torchsummary import summary as summary_
device = torch.device('cuda')
model1 = Generator32_DCGAN().to(device)
model2 = ML32(out_dim=10, channel=3).to(device)
print(summary_(model1, (128,1,1)))
print(summary_(model2, (3,32,32)))
print(model1(torch.rand(1,128,1,1).to(device)).shape)
print(model2(torch.rand(1,3,32,32).to(device)).shape)
'''

'''
from torchsummary import summary as summary_
device = torch.device('cuda')
model1 = Generator64().to(device)
model2 = ML64(out_dim=100).to(device)
print(summary_(model1, (128, 1, 1)))
print(model1(torch.rand(10, 128, 1, 1).to(device)).shape)
print(summary_(model2, (3, 64, 64)))
print(model2(torch.rand(10, 3, 64, 64).to(device)).shape)
'''
