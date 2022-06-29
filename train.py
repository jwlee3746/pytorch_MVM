import argparse
import os
import datetime
from math import log10,exp,sqrt
import numpy as np
import scipy.spatial

from torch import nn
import torch.optim as optim
import torch.utils.data

import torchvision
from torchvision import datasets, transforms
import torchvision.utils as utils

from torch.autograd import Variable, grad as torch_grad
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm
import pickle
import torch.nn.functional as F
from torch import autograd


from data_utils import is_image_file, train_transform, TrainDatasetFromFolder, \
                        generate_image, gen_rand_noise, remove_module_str_in_state_dict, \
                            requires_grad, str2bool, plot_in_train
from loss import TripletLoss, pairwise_distances
from model import Generator64, ML64
        
parser = argparse.ArgumentParser(description='Train Image Generation Models')
parser.add_argument('--data_path', default='/mnt/prj/Data/celebA/celeba', type=str, help='dataset path')
parser.add_argument('--data_name', default='celeba', type=str, help='dataset name')
parser.add_argument('--name', default='results', type=str, help='path to store results')
parser.add_argument('--size', default=64, type=int, help='training images size')
parser.add_argument('--out_dim', default=10, type=int, help='ML network output dim')
parser.add_argument('--num_epochs', default=80, type=int, help='train epoch number')
parser.add_argument('--num_samples', default=64, type=int, help='number of displayed samples')
parser.add_argument('--batch_size', default=64, type=int, help='train batch size')
parser.add_argument('--lr', default=1e-4, type=float, help='train learning rate')
parser.add_argument('--beta1', default=0, type=float, help='Adam optimizer beta1')
parser.add_argument('--beta2', default=0.9, type=float, help='Adam optimizer beta2')
parser.add_argument('--load_model', default = 'no', type=str, choices=['yes', 'no'], help='if load previously trained model')
parser.add_argument('--load_model_path', default = '', type=str, help='load model dir path')
parser.add_argument('--g_model_name', default='Generator64', type=str, help='generator model name')
parser.add_argument('--ml_model_name', default='ML64', type=str, help='metric learning model name')
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
    COMMENT = opt.comment
    out_dim = opt.out_dim
    learning_rate = opt.lr
    beta1 = opt.beta1
    beta2 = opt.beta2
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
    if write_logs:
        writer = SummaryWriter(log_path)
    
    # Dataset & Dataloader
    trainset = TrainDatasetFromFolder(opt.data_path, size=SIZE)
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
    
    # Generator
    netG = Generator64(channel=3)
    if multi_gpu and LOAD_MODEL == 'no':
        netG = nn.DataParallel(netG, device_ids)
    
    # ML
    netML = ML64(out_dim=out_dim, channel=3)
    if multi_gpu and LOAD_MODEL == 'no':
        netML = nn.DataParallel(netML, device_ids)
     
    
    if LOAD_MODEL == 'yes':
        netG.load_state_dict(remove_module_str_in_state_dict(torch.load(str(opt.load_model_path + "/generator_latest.pt"))))
        netG = nn.DataParallel(netG, device_ids)
        
        netML.load_state_dict(remove_module_str_in_state_dict(torch.load(str(opt.load_model_path + "/ml_model_latest.pt")))) 
        netML = nn.DataParallel(netML, device_ids)
        print("LOAD MODEL SUCCESSFULLY")
    
    optimizerG = optim.Adam(netG.parameters(), lr = learning_rate, betas=(beta1,beta2),eps= 1e-6) 
    optimizerML = optim.Adam(netML.parameters(), lr = learning_rate, betas=(0.5,0.999),eps= 1e-3)   
    
    netG.to(DEVICE)     
    netML.to(DEVICE)     
    
    # Losses    
    triplet_ = TripletLoss(margin, alpha).to(DEVICE)    
  

    #fixed_noise1 = gen_rand_noise(1).to(DEVICE) 

    for epoch in range(1, NUM_EPOCHS + 1):
            
        train_bar = tqdm(train_loader) 
    
        netG.train()
        netML.train()        
        
        ### For recording latent ###
        if write_logs:
            real_embedding = torch.zeros(1, out_dim).to(DEVICE)
            fake_embedding = torch.zeros(1, out_dim).to(DEVICE)
        ### For recording manifold ###
        for target in train_bar:
            real_img = Variable(target).to(DEVICE)     #.cuda() 
                
            batch_size = real_img.size()[0]
            if batch_size != opt.batch_size:
                continue
            
            ############################ 
            # Metric Learning
            ############################                                         
            requires_grad(netG, False)
            requires_grad(netML, True)                
                
            z = torch.randn(batch_size, 128, 1, 1).to(DEVICE)            
            z.requires_grad_(True)            

            fake_img = netG(z)
            ml_real_out = netML(real_img)
            ml_fake_out = netML(fake_img) 
            
            r1=torch.randperm(batch_size)
            r2=torch.randperm(batch_size)
            ml_real_out_shuffle = ml_real_out[r1[:, None]].view(ml_real_out.shape[0],ml_real_out.shape[-1])
            ml_fake_out_shuffle = ml_fake_out[r2[:, None]].view(ml_fake_out.shape[0],ml_fake_out.shape[-1])
            
            triplet_loss = triplet_(ml_real_out,ml_real_out_shuffle,ml_fake_out_shuffle)   
            netML.zero_grad()
            triplet_loss.backward()
            optimizerML.step() 
            
            ############################ 
            # Generative Model
            ############################           
            requires_grad(netG, True)
            requires_grad(netML, False)            
            
            fake_img = netG(z)            
            ml_real_out = netML(real_img)
            ml_fake_out = netML(fake_img)                  
            
            r1=torch.randperm(batch_size)
            r2=torch.randperm(batch_size)
            ml_real_out_shuffle = ml_real_out[r1[:, None]].view(ml_real_out.shape[0],ml_real_out.shape[-1])
            ml_fake_out_shuffle = ml_fake_out[r2[:, None]].view(ml_fake_out.shape[0],ml_fake_out.shape[-1])
                
            ############################
            pd_r = pairwise_distances(ml_real_out, ml_real_out) 
            pd_f = pairwise_distances(ml_fake_out, ml_fake_out)
        
            p_dist =  torch.dist(pd_r,pd_f,2)             
            c_dist = torch.dist(ml_real_out.mean(0),ml_fake_out.mean(0),2)         
        
            #############################
            g_loss = p_dist + c_dist   
            netG.zero_grad()            
            g_loss.backward()                        
            optimizerG.step()
            
            ### For recording manifold ###
            real_embedding = torch.cat([real_embedding, ml_real_out], dim=0)
            fake_embedding = torch.cat([fake_embedding, ml_fake_out], dim=0)
            ### For recording manifold ###
        
            train_bar.set_description(desc='[%d/%d]' % (epoch, NUM_EPOCHS))
###--------------------Save Embeddings----------------------------------------
        real_embedding_np = real_embedding.cpu().detach().numpy()
        real_embedding_np = np.delete(real_embedding_np, 0, 0)
        #print(real_embedding_np.shape)
        fake_embedding_np = fake_embedding.cpu().detach().numpy()
        fake_embedding_np = np.delete(fake_embedding_np, 0, 0)
        #print(fake_embedding_np.shape)
        
        if write_logs:            
            np.save(str(embedding_path  +'/' + 'real_embdding_{}').format(epoch), real_embedding_np)
            np.save(str(embedding_path  +'/' + 'fake_embdding_{}').format(epoch), fake_embedding_np)
        
###--------------------visualize by epoch----------------------------------------
        if write_logs:
            plot_in_train(epoch, real_embedding_np, fake_embedding_np, n_to_show, pca_path)
        else:
            plot_in_train(epoch, real_embedding_np, fake_embedding_np, n_to_show)
###--------------------write_logs----------------------------------------        
        print("triplet_loss: {}".format(triplet_loss))
        print("g_loss: {}".format(g_loss))
        if write_logs:
            writer.add_scalars("loss/train", {"triplet_loss": triplet_loss, "g_loss": g_loss}, epoch)
            #writer.add_scalar("triplet_loss/train", triplet_loss, epoch)
            #writer.add_scalar("g_loss/train", g_loss, epoch)
        
###--------------------Hausdorff distance----------------------------------------
        h_distance, idx1, idx2 = scipy.spatial.distance.directed_hausdorff(ml_real_out.clone().cpu().detach().numpy(), ml_fake_out.clone().cpu().detach().numpy(), seed=0)
        print('hausdorff distance: ', h_distance)        
        print(' c_dist:',c_dist.item(), ' p_dist:', p_dist.item(),' triplet_loss:',triplet_loss.item())

###------------------display generated samples--------------------------------------------------      
        if not write_logs:
            continue
        
        fixed_noise1 = gen_rand_noise(1).to(DEVICE)        
        gen_image1 = generate_image(netG, dim=SIZE, channel=3, batch_size=1, noise=fixed_noise1)
        
        fixed_noise = gen_rand_noise(num_samples).to(DEVICE)        
        gen_images = generate_image(netG, dim=SIZE, channel=3, batch_size=num_samples, noise=fixed_noise)
        if write_logs:
        #    utils.save_image(decode_images, str(decoded_path  +'/' + 'decoded_{}.png').format(epoch), nrow=int(sqrt(num_decoded)), padding=2)
            utils.save_image(gen_image1, str(A_sample_path  +'/' + 'sample1_{}.png').format(epoch), nrow=int(sqrt(1)), padding=2)
            utils.save_image(gen_images, str(sample_path  +'/' + 'samples_{}.png').format(epoch), nrow=int(sqrt(num_samples)), padding=2)             
        
# 	#----------------------Save model----------------------
        torch.save(netG.state_dict(), str(output_path  +'/' + "generator_{}.pt").format(epoch))
        torch.save(netML.state_dict(), str(output_path  +'/' + "ml_model_{}.pt").format(epoch))
        torch.save(netG.state_dict(), str(output_path  +'/' + "generator_latest.pt"))
        torch.save(netML.state_dict(), str(output_path  +'/' + "ml_model_latest.pt"))
    
    if write_logs:    
        writer.close()   

