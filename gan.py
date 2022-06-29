# -*- coding: utf-8 -*-
"""
Created on Tue May 31 14:38:27 2022

@author: Jaewon
"""
import argparse
import os, time, sys
import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt
import itertools
import pickle
from math import log10,exp,sqrt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

import matplotlib
import matplotlib.pylab as plt

from torchvision import datasets, transforms
from torch.autograd import Variable
from torchvision.utils import make_grid, save_image
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.utils as utils

from data_utils import TrainDatasetFromFolder
from model_gan import generator, discriminator

from data_utils import is_image_file, train_transform, TrainDatasetFromFolder, \
                        generate_image, gen_rand_noise_MNIST, remove_module_str_in_state_dict, \
                            requires_grad, str2bool, plot_in_train

parser = argparse.ArgumentParser(description='Train Image Generation Models')
parser.add_argument('--data_path', default='/mnt/prj/Data/', type=str, help='dataset path')
parser.add_argument('--data_name', default='MNIST', type=str, help='dataset name')
parser.add_argument('--name', default='results', type=str, help='path to store results')
parser.add_argument('--size', default=28, type=int, help='training images size')
parser.add_argument('--nz', default=128, type=int, help='Latent Vector dim')
parser.add_argument('--num_epochs', default=80, type=int, help='train epoch number')
parser.add_argument('--num_samples', default=64, type=int, help='number of displayed samples')
parser.add_argument('--batch_size', default=128, type=int, help='train batch size')
parser.add_argument('--k', default=1, type=float, help='num training descriminator')
parser.add_argument('--lr', default=2e-4, type=float, help='train learning rate')
parser.add_argument('--load_model', default = 'no', type=str, choices=['yes', 'no'], help='if load previously trained model')
parser.add_argument('--g_model_name', default='GAN', type=str, help='generator model name')
parser.add_argument('--ml_model_name', default='', type=str, help='metric learning model name')
parser.add_argument('--margin', default=1, type=float, help='triplet loss margin')
parser.add_argument('--alpha', default=1e-2, type=float, help='triplet loss direction guidance weight parameter')
parser.add_argument('--n_threads', type=int, default=16)

parser.add_argument('--multi_gpu', default=False, type=str2bool, help='Use multi-gpu')
parser.add_argument('--device_ids', nargs='+', type=int, help='0 1 2 3: Use all 4 GPUs')
parser.add_argument('--write_logs', default=True, type=str2bool, help='Write logs or not')
parser.add_argument('--n_to_show', default=1000, type=int, help='Num of fixed latent vector to show')
parser.add_argument('--comment', default='', type=str, help='comment what is changed')

if __name__ == '__main__':
    opt = parser.parse_args()
    
    SIZE = opt.size
    learning_rate = opt.lr
    batch_size = opt.batch_size
    NUM_EPOCHS = opt.num_epochs
    num_samples = opt.num_samples
    LOAD_MODEL = opt.load_model
    G_MODEL_NAME = opt.g_model_name
    ML_MODEL_NAME = opt.ml_model_name
    margin = opt.margin
    alpha = opt.alpha
    
    multi_gpu = opt.multi_gpu
    device_ids = opt.device_ids
    write_logs = opt.write_logs
    n_to_show = opt.n_to_show
    COMMENT = opt.comment
    
    # Result path      
    now = datetime.datetime.now().strftime("%m%d_%H%M")
    output_path = str(opt.name + '/' + '{}_{}_{}_e{}_b{}_m{}').format(now, G_MODEL_NAME, opt.data_name, NUM_EPOCHS, batch_size, margin)
    if COMMENT:
        output_path += '_{}'.format(COMMENT)
    if write_logs and not os.path.exists(output_path):
        os.makedirs(output_path)
    log_path= str(output_path + '/logs')
    if write_logs and not os.path.exists(log_path):
        os.makedirs(log_path)
    pca_path = output_path + '/PCA' 
    if write_logs and not os.path.exists(pca_path):
        os.makedirs(pca_path)   
    embedding_path = output_path + '/embeddings' 
    if write_logs and not os.path.exists(embedding_path):
        os.makedirs(embedding_path)   
    sample_path = output_path + '/images' 
    if write_logs and not os.path.exists(sample_path):
        os.makedirs(sample_path)   
    A_sample_path = sample_path + '/a_image' 
    if write_logs and not os.path.exists(A_sample_path):
        os.makedirs(A_sample_path)      
    
    # jupyter-tensorboard writer
    writer = SummaryWriter(log_path)    
    
    # Dataset & Dataloader
    transform = transforms.Compose(
        [transforms.ToTensor(), 
         transforms.Normalize((0.5,), (0.5,)),
         ]
    )
    
    trainset = datasets.MNIST(root = opt.data_path,
                                   train = True,
                                   download = True,
                                   transform = transform
                                   )
    if multi_gpu:
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=False, num_workers= 4 * len(device_ids))
    else:
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=False, num_workers= 0)
    
    # Multi-GPU
    if multi_gpu and (len(device_ids) > 1):
        if (torch.cuda.is_available()) and (torch.cuda.device_count() > 1):
            print("Let's use", len(device_ids), "GPUs!")
        else : multi_gpu = False
    
    # Device
    if torch.cuda.is_available():
        if device_ids:
            DEVICE = torch.device('cuda:{}'.format(device_ids[0]))
        else:
            DEVICE = torch.device('cuda')
    else:
        DEVICE = torch.device('cpu')
    print('Using PyTorch version:', torch.__version__, ' Device:', DEVICE)
    
    # network
    netG = generator(128, 28*28)
    netD = discriminator(28*28, 1)
    #netG.weight_init(mean=0.0, std=0.02)
    #netD.weight_init(mean=0.0, std=0.02)
    netG.to(DEVICE) 
    netD.to(DEVICE) 
    
    # Binary Cross Entropy loss
    BCE_loss = nn.BCELoss().to(DEVICE) 
    
    # Adam optimizer
    G_optimizer = optim.Adam(netG.parameters(), lr=learning_rate, betas=(0.5, 0.999))
    D_optimizer = optim.Adam(netD.parameters(), lr=learning_rate, betas=(0.5, 0.999))
    
    # Train
    for epoch in range(1, NUM_EPOCHS + 1):
            
        train_bar = tqdm(train_loader) 
    
        netG.train()
        netD.train()    
        
        for images, labels in train_bar:            
            real_img = Variable(images).to(DEVICE)   
            #print("real_img")
            #print(real_img)    
            batch_size = real_img.size()[0]
            if batch_size != opt.batch_size:
                continue
            
            ############################ 
            # train discriminator D
            ############################         
            netD.zero_grad() 
            requires_grad(netG, False)
            requires_grad(netD, True)
    
            y_real_ = torch.ones(batch_size).to(DEVICE) 
            y_fake_ = torch.zeros(batch_size).to(DEVICE) 
                                        
            D_real_result = netD(real_img.view(-1, 28 * 28)).squeeze()  
            #print("D_real_result")
            #print(D_real_result)    
            D_real_loss = BCE_loss(D_real_result, y_real_)
            #print("D_real_loss")
            #print(D_real_loss)  
    
            z_ = torch.randn((batch_size, 128)).to(DEVICE)      
            z_.requires_grad_(True)
            G_result = netG(z_)
            #print("G_result")
            #print(G_result)  
            D_fake_result = netD(G_result).squeeze()
            #print("D_fake_result")
            #print(D_fake_result)  
            D_fake_loss = BCE_loss(D_fake_result, y_fake_)
            #print("D_fake_loss")
            #print(D_fake_loss)  
            D_fake_score = D_fake_result.data.mean()
    
            D_train_loss = D_real_loss + D_fake_loss
    
            D_train_loss.backward()
            D_optimizer.step()
    
    
            ############################ 
            # train generator G
            ############################         
            netG.zero_grad()  
            requires_grad(netG, True)
            requires_grad(netD, False)
    
            G_result = netG(z_)
            D_result = netD(G_result).squeeze()
            
            G_train_loss = BCE_loss(D_result, y_real_)
            
            G_train_loss.backward()
            G_optimizer.step()   
            
            train_bar.set_description(desc='[%d/%d]' % (epoch, NUM_EPOCHS))
        
        print(' D_train_loss:', D_train_loss.item(), ' G_train_loss:', G_train_loss.item())
        writer.add_scalars("loss/train", {"d_loss": D_train_loss, "g_loss": G_train_loss}, epoch)
        #writer.add_scalar("triplet_loss/train", triplet_loss, epoch)
        #writer.add_scalar("g_loss/train", g_loss, epoch)
        
        if not write_logs:
            continue

###------------------display generated samples--------------------------------------------------        
        fixed_noise = gen_rand_noise_MNIST(num_samples).to(DEVICE) 
        gen_images = generate_image(netG, dim=SIZE, channel=1, batch_size=num_samples, noise=fixed_noise)
        utils.save_image(gen_images, str(sample_path  +'/' + 'samples_{}.png').format(epoch), nrow=int(sqrt(num_samples)), padding=2)             
         
# 	#----------------------Save model----------------------
        torch.save(netG.state_dict(), str(output_path  +'/' + "generator_{}.pt").format(epoch))
        torch.save(netD.state_dict(), str(output_path  +'/' + "d_model_{}.pt").format(epoch))
        torch.save(netG.state_dict(), str(output_path  +'/' + "generator_latest.pt"))
        torch.save(netD.state_dict(), str(output_path  +'/' + "d_model_latest.pt"))
    
    writer.close()