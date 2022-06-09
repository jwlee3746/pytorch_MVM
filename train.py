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
                        generate_image, gen_rand_noise, remove_module_str_in_state_dict, requires_grad, str2bool
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
parser.add_argument('--batch_size', default=256, type=int, help='train batch size')
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
parser.add_argument('--device_ids', nargs='+', type=int, help='0: Use all 4 GPUs / 1: Use GPU0,1 / 2: Use GPU2,3')
parser.add_argument('--write_logs', default=True, type=str2bool, help='Write logs or not')

if __name__ == '__main__':
    opt = parser.parse_args()
    
    SIZE = opt.size
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
    
    # Result path    
    now = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    output_path = str(opt.name + '/' + '{}_{}_e{}_b{}_{}').format(G_MODEL_NAME, opt.data_name, NUM_EPOCHS, batch_size, now)
    if write_logs and not os.path.exists(output_path):
        os.makedirs(output_path)
    log_path= str(output_path + '/logs')
    if write_logs and not os.path.exists(log_path):
        os.makedirs(log_path)
    sample_path = output_path + '/images' 
    if write_logs and not os.path.exists(sample_path):
        os.makedirs(sample_path)          
    
    # jupyter-tensorboard writer
    if write_logs:
        writer = SummaryWriter(log_path)
    
    # Dataset & Dataloader
    trainset = TrainDatasetFromFolder(opt.data_path, size=SIZE)  
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=16)
    
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
    netG = Generator64()
    if multi_gpu:
        netG = nn.DataParallel(netG, device_ids)
    netG.to(DEVICE)  
    optimizerG = optim.Adam(netG.parameters(), lr = learning_rate, betas=(beta1,beta2),eps= 1e-6) 
    
    # ML
    netML = ML64(out_dim=out_dim)
    if multi_gpu:
        netML = nn.DataParallel(netML, device_ids)
    netML.to(DEVICE)      
    optimizerML = optim.Adam(netML.parameters(), lr = learning_rate, betas=(0.5,0.999),eps= 1e-3)   

    # Losses    
    triplet_ = TripletLoss(margin, alpha).to(DEVICE)    

    if LOAD_MODEL == 'yes':
        netG.load_state_dict(remove_module_str_in_state_dict(torch.load(str(opt.load_model_path + "/generator_latest.pt"))))
        netML.load_state_dict(remove_module_str_in_state_dict(torch.load(str(opt.load_model_path + "/ml_model_latest.pt")))) 
        print("LOAD MODEL SUCCESSFULLY")
        
    
    #fixed_noise1 = gen_rand_noise(1).to(DEVICE) 

    for epoch in range(1, NUM_EPOCHS + 1):
            
        train_bar = tqdm(train_loader) 
    
        netG.train()
        netML.train()        
        
        for target in train_bar:
            real_img = Variable(target).cuda().to(DEVICE)           
                
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
            """
            real_img = Variable(target).cuda()           
                
            batch_size = real_img.size()[0]
            if batch_size != opt.batch_size:
                continue
                
            z = torch.randn(batch_size, 128, 1, 1).to(DEVICE)
            z.requires_grad_(True)    
            fake_img = netG(z)
            
#             print(real_img.size())
#             print(fake_img.size())
            
            ############################ 
            # Metric Learning
            ############################
            
            netML.zero_grad()
            ml_real_out = netML(real_img)
            ml_fake_out = netML(fake_img)         
#             print(ml_real_out.size())
#             print(ml_fake_out.size())            
            
            if ml_real_out.size()[-1] == 1:
                ml_real_out = ml_real_out.squeeze(-1).squeeze(-1)
                ml_fake_out = ml_fake_out.squeeze(-1).squeeze(-1)
            
            r1=torch.randperm(batch_size)
            r2=torch.randperm(batch_size)
            ml_real_out_shuffle = ml_real_out[r1[:, None]].view(ml_real_out.shape[0],ml_real_out.shape[-1])
            ml_fake_out_shuffle = ml_fake_out[r2[:, None]].view(ml_fake_out.shape[0],ml_fake_out.shape[-1])
                
            ############################
            netG.zero_grad()

            pd_r = pairwise_distances(ml_real_out, ml_real_out) 
            pd_f = pairwise_distances(ml_fake_out, ml_fake_out)
        
            p_dist =  torch.dist(pd_r,pd_f,2)             
            c_dist = torch.dist(ml_real_out.mean(0),ml_fake_out.mean(0),2)         
        
            #############################
            g_loss = p_dist + c_dist   
            g_loss.backward(retain_graph=True)                        
            optimizerG.step()
            
            #############################
            fake_img = netG(z)
            
            triplet_loss = triplet_(ml_real_out, ml_real_out_shuffle, ml_fake_out_shuffle)
            triplet_loss.backward()
            optimizerML.step()    
            """            
        
            train_bar.set_description(desc='[%d/%d]' % (epoch, NUM_EPOCHS))
        
        if write_logs:
            writer.add_scalars("loss/train", {"triplet_loss": triplet_loss, "g_loss": g_loss}, epoch)
            #writer.add_scalar("triplet_loss/train", triplet_loss, epoch)
            #writer.add_scalar("g_loss/train", g_loss, epoch)
        
        if epoch % 2 != 0:
            continue    
        
###--------------------Hausdorff distance----------------------------------------
        h_distance, idx1, idx2 = scipy.spatial.distance.directed_hausdorff(ml_real_out.clone().cpu().detach().numpy(), ml_fake_out.clone().cpu().detach().numpy(), seed=0)
        print('hausdorff distance: ', h_distance)        
        print(' c_dist:',c_dist.item(), ' p_dist:', p_dist.item(),' triplet_loss:',triplet_loss.item())

###------------------display generated samples--------------------------------------------------
        fixed_noise = gen_rand_noise(num_samples).to(DEVICE)        
        gen_images = generate_image(netG, dim=SIZE, batch_size=num_samples, noise=fixed_noise)
        if write_logs:
            utils.save_image(gen_images, str(sample_path  +'/' + 'samples_{}.png').format(epoch), nrow=int(sqrt(num_samples)), padding=2)             
        
        if (not write_logs) or (epoch % 10 != 0):
            continue    
# 	#----------------------Save model----------------------
        torch.save(netG.state_dict(), str(output_path  +'/' + "generator_{}.pt").format(epoch))
        torch.save(netML.state_dict(), str(output_path  +'/' + "ml_model_{}.pt").format(epoch))
        torch.save(netG.state_dict(), str(output_path  +'/' + "generator_latest.pt"))
        torch.save(netML.state_dict(), str(output_path  +'/' + "ml_model_latest.pt"))
    
    if write_logs:    
        writer.close()   
################################################################## 
### The above code is for reproducing results in PyTorch 1.0. For PyTorch with higher versions, one can try replace the main body as follows:
# 1.  Like in implementation-stylegan2/train_MvM.py:

#             requires_grad(netG, False)
#             requires_grad(netML, True)                
                
#             z = torch.randn(batch_size, 128, 1, 1).to(DEVICE)          
#             z.requires_grad_(True)            

#             fake_img = netG(z)
#             ml_real_out = netML(real_img)
#             ml_fake_out = netML(fake_img)         
            
#             r1=torch.randperm(batch_size)
#             r2=torch.randperm(batch_size)
#             ml_real_out_shuffle = ml_real_out[r1[:, None]].view(ml_real_out.shape[0],ml_real_out.shape[-1])
#             ml_fake_out_shuffle = ml_fake_out[r2[:, None]].view(ml_fake_out.shape[0],ml_fake_out.shape[-1])
            
#             triplet_loss = triplet_(ml_real_out,ml_real_out_shuffle,ml_fake_out_shuffle)     
#             netML.zero_grad()
#             triplet_loss.backward()
#             optimizerML.step() 
            
#             ############################            
#             requires_grad(netG, True)
#             requires_grad(netML, False)            
            
#             fake_img = netG(z)            
#             ml_real_out = netML(real_img)
#             ml_fake_out = netML(fake_img)                  
            
#             r1=torch.randperm(batch_size)
#             r2=torch.randperm(batch_size)
#             ml_real_out_shuffle = ml_real_out[r1[:, None]].view(ml_real_out.shape[0],ml_real_out.shape[-1])
#             ml_fake_out_shuffle = ml_fake_out[r2[:, None]].view(ml_fake_out.shape[0],ml_fake_out.shape[-1])
                
#             ############################

#             pd_r = pairwise_distances(ml_real_out, ml_real_out) 
#             pd_f = pairwise_distances(ml_fake_out, ml_fake_out)
        
#             p_dist =  torch.dist(pd_r,pd_f,2)             
#             c_dist = torch.dist(ml_real_out.mean(0),ml_fake_out.mean(0),2)         
        
#             #############################
#             g_loss = p_dist + c_dist   
#             netG.zero_grad()            
#             g_loss.backward()                        
#             optimizerG.step()



# OR 
# 2. Tested with PytTorch 1.7.1:
#             netML.zero_grad()   
#             fake_img = netG(z) 
#             ml_real_out, _ = netML(real_img)
#             ml_fake_out, _ = netML(fake_img)
            
#             r1=torch.randperm(batch_size)
#             r2=torch.randperm(batch_size)
#             ml_real_out_shuffle = ml_real_out[r1[:, None]].view(ml_real_out.shape[0],ml_real_out.shape[-1])
#             ml_fake_out_shuffle = ml_fake_out[r2[:, None]].view(ml_fake_out.shape[0],ml_fake_out.shape[-1])                

#             ############################
            
#             pd_r = pairwise_distances(ml_real_out, ml_real_out) 
#             pd_f = pairwise_distances(ml_fake_out, ml_fake_out)
        
#             p_dist =  torch.dist(pd_r,pd_f,2)   
#             c_dist = torch.dist(ml_real_out.mean(0),ml_fake_out.mean(0),2)      

#             g_loss = p_dist + c_dist 
#             netG.zero_grad()
#             g_loss.backward()                        
#             optimizerG.step()
            
#             fake_img = netG(z) 
#             ml_real_out = netML(real_img)
#             ml_fake_out = netML(fake_img.detach())           
            
#             r1=torch.randperm(batch_size)
#             r2=torch.randperm(batch_size)
#             ml_real_out_shuffle = ml_real_out[r1[:, None]].view(ml_real_out.shape[0],ml_real_out.shape[-1])
#             ml_fake_out_shuffle = ml_fake_out[r2[:, None]].view(ml_fake_out.shape[0],ml_fake_out.shape[-1])

#             ml_loss = triplet_(ml_real_out,ml_real_out_shuffle,ml_fake_out_shuffle) 
#             ml_loss.backward()
#             optimizerML.step()

