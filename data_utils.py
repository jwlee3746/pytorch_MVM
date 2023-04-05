from os import listdir
from os.path import join
from glob import glob
import numpy as np
import csv
from collections import OrderedDict
from PIL import Image
import torch
from torch.utils.data.dataset import Dataset
from torchvision.transforms import Compose, RandomCrop, ToTensor, ToPILImage, CenterCrop, Resize

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from torch.autograd import Variable


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])

def train_transform(size):
    transform = None
    transform = Compose([
        Resize((size, size), interpolation=Image.BICUBIC),
        ToTensor()
    ])
    return transform


class TrainDatasetFromFolder(Dataset):
    def __init__(self, dataset_dir, size, file_format="png", mode="RGB"):
        super(TrainDatasetFromFolder, self).__init__()
        self.image_filenames = glob(dataset_dir + '/**/*.{}'.format(file_format), recursive=True)
        self.resize_transform = train_transform(size)
        self.mode = mode

    def __getitem__(self, index):
        image = Image.open(self.image_filenames[index]).convert(self.mode) # convert RGB if is color image
        real_image = self.resize_transform(image)
        #real_image = real_image * 2 - 1 # if use tanh
        return real_image, -1

    def __len__(self):
        return len(self.image_filenames)

    
def generate_image(netG, dim, channel, batch_size, noise=None):
    if noise is None:
        noise = gen_rand_noise()

    with torch.no_grad():
    	noisev = noise 
    samples = netG(noisev)
    samples = samples.view(batch_size, channel, dim, dim) # 1 or 3
    #samples = (samples + 1) * 0.5 # if tanh
    return samples

def gen_rand_noise(batch_size, l_dim):
    #torch.manual_seed(42)
    noise = torch.randn(batch_size, l_dim, 1, 1)
    return noise

def gen_rand_noise_MNIST(batch_size, ):
    #torch.manual_seed(42)
    noise = torch.randn(batch_size, 128)
    return noise

def remove_module_str_in_state_dict(state_dict):
    state_dict_rename = OrderedDict()
    for k, v in state_dict.items():
        name = k.replace("module.", "") # remove `module.`
        state_dict_rename[name] = v
    return state_dict_rename

def requires_grad(model, tf):   
    for param in model.parameters():
        param.requires_grad = tf
    
# For parser boolean input
def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise Exception('Boolean value expected.')


def invert_grayscale(mnist_digits):
    """ Makes black white, and white black """
    return 1-mnist_digits
    

