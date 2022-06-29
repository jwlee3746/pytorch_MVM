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
    def __init__(self, dataset_dir, size):
        super(TrainDatasetFromFolder, self).__init__()
        self.image_filenames = glob(dataset_dir + '/**/*.jpg', recursive=True)
        self.resize_transform = train_transform(size)

    def __getitem__(self, index):
        image = Image.open(self.image_filenames[index]).convert('RGB') # convert RGB if is color image
        real_image = self.resize_transform(image)
        #real_image = real_image * 2 - 1 # if use tanh
        return real_image

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

def gen_rand_noise(batch_size, ):
    torch.manual_seed(42)
    noise = torch.randn(batch_size, 128,1,1)
    return noise

def gen_rand_noise_MNIST(batch_size, ):
    torch.manual_seed(42)
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

### Plot n-dim(n : 2-3) data scatter after train ###
def plot_in_train(epoch, real_embedding_np, fake_embedding_np, n_to_show, download_path = None):
       
    if len(real_embedding_np) != len(fake_embedding_np):
        raise Exception("len np_real_embdding != np_fake_embdding")
    if len(real_embedding_np) < n_to_show:
        raise Exception("Too large n_to_show ! input lower than {}".format(len(real_embedding_np)))
        
    embeddings = np.concatenate((real_embedding_np[:n_to_show], fake_embedding_np[:n_to_show]))
    print("Load total {} of embeddings".format(len(embeddings)))
    labels = np.concatenate((np.zeros(n_to_show), np.ones(n_to_show)))
    
    if n_to_show <= 3000: alpha2D, s2D, alpha3D, s3D =0.7, 5, 0.3, 4 
    else: alpha2D, s2D, alpha3D, s3D =0.1, 5, 0.1, 4 
    
    ### 2d PCA ###
    reducer = PCA(n_components=2)
    ml_out_reduced = reducer.fit_transform(embeddings)
    r = plt.scatter(
        ml_out_reduced[:n_to_show, 0], ml_out_reduced[:n_to_show, 1],
        c="dodgerblue", alpha=alpha2D, s=s2D)
    f = plt.scatter(
        ml_out_reduced[n_to_show:, 0], ml_out_reduced[n_to_show:, 1],
        c="red", alpha=alpha2D, s=s2D)
    plt.legend((r, f),
            ('Real', 'Fake'),
           scatterpoints=1,
           loc='lower left',
           ncol=3,
           fontsize=8)
    plt.gca().set_aspect('equal', 'datalim')
    plt.title('PCA 2D of a batch[epoch#{}]'.format(epoch), fontsize=18)
    if download_path:
        plt.savefig(str(download_path  +'/' + '2d_PCA_{}.pdf').format(epoch), dpi=200)
    plt.show()
    ### 2d PCA ###
    
    ### 3d PCA###   
    reducer = PCA(n_components=3)
    ml_out_reduced = reducer.fit_transform(embeddings)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    r = ax.scatter(
        ml_out_reduced[:n_to_show, 0], ml_out_reduced[:n_to_show, 1], ml_out_reduced[:n_to_show, 2],
        c='dodgerblue', alpha=alpha3D, s=s3D) 
    f = ax.scatter(
        ml_out_reduced[n_to_show:, 0], ml_out_reduced[n_to_show:, 1], ml_out_reduced[n_to_show:, 2],
        c='red', alpha=alpha3D, s=s3D) 
    plt.legend((r, f),
            ('Real', 'Fake'),
           numpoints=1,
           loc='upper left',
           ncol=3,
           fontsize=8,
           bbox_to_anchor=(0, 0))
    plt.gca().set_aspect('auto', 'datalim')
    plt.title('PCA 3D of a batch[epoch#{}]'.format(epoch), fontsize=18)
    if download_path:
        plt.savefig(str(download_path + '/' + '3d_PCA_{}.pdf').format(epoch), dpi=200)
    plt.show()
    ### 3d PCA###
    