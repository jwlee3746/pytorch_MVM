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
from sklearn.manifold import TSNE
import umap.umap_ as umap
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
def plot_after_train(netML, netG, trainset, n_to_show = 1500, download_path = None):
    # Dataset & Dataloader
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=n_to_show,
                                              shuffle=True, num_workers= 0)
    
    for data in train_loader:
        real_img = Variable(data).cuda()     #.cuda() 
            
        z = torch.randn(n_to_show, 128, 1, 1).cuda()    

        fake_img = netG(z)
        
        ml_real_out = netML(real_img).cpu().detach().numpy()
        ml_fake_out = netML(fake_img).cpu().detach().numpy()
        
        embeddings = np.concatenate((ml_real_out, ml_fake_out))
        labels = np.zeros(len(embeddings))
        for idx in range(len(embeddings)):
            if idx < len(ml_real_out): labels[idx] = 1
            else: break
            
        ### 2d U-map ###
        reducer = umap.UMAP()
        ml_out_reduced = reducer.fit_transform(embeddings)
        plt.scatter(
            ml_out_reduced[:, 0], ml_out_reduced[:, 1],
            cmap="rainbow", c=labels, alpha=0.7, s=7)
        plt.gca().set_aspect('equal', 'datalim')
        plt.title('UMAP 2D projection of {} data'.format(n_to_show), fontsize=18)
        if download_path:
            plt.savefig(str(download_path  +'/' + '2Dumap_{}.pdf').format(n_to_show), dpi=200)
        plt.show()
        ### 2d U-map ###
        ### 3d U-map ###
        reducer = umap.UMAP(n_components=3, random_state=42)       
        ml_out_reduced = reducer.fit_transform(embeddings)
        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(
            ml_out_reduced[:, 0], ml_out_reduced[:, 1], ml_out_reduced[:, 2],
            cmap='rainbow', c=labels, alpha=0.7, s=7)    
        plt.gca().set_aspect('auto', 'datalim')
        plt.title('UMAP 3D projection of {} data'.format(n_to_show), fontsize=18)
        if download_path:
            plt.savefig(str(download_path  +'/' + '3Dumap_{}.pdf').format(n_to_show), dpi=200)
        plt.show()
        ### 3d U-map ###
        """
        ### 2-dim ###
        figsize = 10
        tsne = TSNE(n_components=2, random_state=42)
        ml_out_reduced = tsne.fit_transform(embeddings)
    
        plt.figure(figsize=(figsize, figsize))
        plt.scatter(
            ml_out_reduced[:, 0], ml_out_reduced[:, 1],
            cmap='rainbow', c=labels, alpha=0.7, s=10)     
        plt.title('t-SNE 2D projection of {} data'.format(n_to_show), fontsize=18)
        if download_path:
            plt.savefig(str(download_path  +'/' + '2Dtsne_{}.pdf').format(n_to_show), dpi=200)       
        plt.show()       
        ### 2-dim ###
        
        ### 3-dim ###
        tsne = TSNE(n_components=3, random_state=42)
        ml_out_reduced = tsne.fit_transform(embeddings)
    
        fig = plt.figure(figsize=(figsize, figsize))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(
            ml_out_reduced[:, 0], ml_out_reduced[:, 1], ml_out_reduced[:, 2],
            cmap='rainbow', c=labels, alpha=0.7, s=10)    
        plt.title('t-SNE 3D projection of {} data'.format(n_to_show), fontsize=18)
        if download_path:
            plt.savefig(str(download_path  +'/' + '3Dtsne_{}.pdf').format(n_to_show), dpi=200)             
        plt.show()
        ### 3-dim ###
        """
        break