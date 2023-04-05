import argparse
import os
import datetime
import logging

import numpy as np
import scipy.spatial
from math import log10,exp,sqrt

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


from loss import TripletLoss, pairwise_distances
from model import Generator28_linear, Generator64, Generator128, ML28, ML64, ML128
from utils import utils_dataset, data_utils
from models import hgan

parser = argparse.ArgumentParser(description='Train Image Generation Models')
# DATA
parser.add_argument('--data_path', default='/mnt/prj/Data/', type=str, help='dataset path')
parser.add_argument('--data_name', default='MNIST', type=str, help='dataset name')
parser.add_argument('--size', default=28, type=int, help='training images size')

# HYPERPARAMETER
parser.add_argument('--ep', default=200, type=int, help='train epoch number')
parser.add_argument('--batch_size', default=128, type=int, help='train batch size')
parser.add_argument('--lr', default=2e-4, type=float, help='train learning rate')

parser.add_argument('--out_dim', default=100, type=int, help='ML network output dim')
parser.add_argument('--beta1', default=0.5, type=float, help='Adam optimizer beta1')
parser.add_argument('--beta2', default=0.999, type=float, help='Adam optimizer beta2')
parser.add_argument('--margin', default=2, type=float, help='triplet loss margin')
parser.add_argument('--alpha', default=1e-2, type=float, help='triplet loss direction guidance weight parameter')

parser.add_argument('--c_d', type=float, default=0.000823, metavar='C', 
                    help='C of hiperbolic space of discriminator (default: 0.01)')
parser.add_argument('--c_g', type=float, default=0.00005, metavar='C',
                        help='C of hiperbolic space of generator (default: 0.01)')
parser.add_argument('--bias', help='bias is False', action='store_true')
parser.add_argument('--cfg_d', type=str, default='thhhh',
                    help='layes in the net')
parser.add_argument('--cfg_g', type=str, default='thhfee',
                        help='layes in the net')
parser.add_argument('--mode_train_c_g', type=int, default=1, help='0: do not train(defualt)\n 1:train c with abs(c*lr)+e\n 2: train c with exp(c*lr)+e in generator.')
parser.add_argument('--mode_train_c_d', type=int, default=1, help='0: do not train(defualt)\n 1:train c with abs(c*lr)+e\n 2: train c with exp(c*lr)+e in discriminator.')
parser.add_argument('--lrmul_c', type=float, default=0.01, metavar='L',
                    help='laerning rate multiplier for train c(defualt 0.01)')
# EMBEDDINGS
parser.add_argument('--num_samples', default=64, type=int, help='number of displayed samples')
parser.add_argument('--n_to_show', default=10000, type=int, help='Num of fixed latent vector to show embeddings')
parser.add_argument('--eval_ep', default='', type=str, help='epochs for evaluation, [] or range "r(start,end,step)", e.g. "r(10,70,20)+[200]" means 10, 30, 50, 200"""')

# PRETRAINED MODEL
parser.add_argument('--load_model', default = 'no', type=str, choices=['yes', 'no'], help='if load previously trained model')
parser.add_argument('--load_model_path', default = '', type=str, help='load model dir path')

# MODEL NAME
parser.add_argument('--g_model_name', default='Generator28_linear', type=str, help='generator model name')
parser.add_argument('--ml_model_name', default='ML28', type=str, help='metric learning model name')

# SETTING
parser.add_argument('--name', default='results', type=str, help='path to store results')
parser.add_argument('--comment', default='', type=str, help='comment what is changed')
parser.add_argument('--n_threads', type=int, default=16)



if __name__ == '__main__':
    opt = parser.parse_args()
    
    ### Result path ###   
    now = datetime.datetime.now().strftime("%m%d_%H%M")
    output_path = str(opt.name + '/' + '{}_{}_{}_e{}_b{}_d{}').format(now, opt.ml_model_name, opt.data_name, opt.ep, opt.batch_size, opt.out_dim)
    
    if opt.comment:
        output_path += '_{}'.format(opt.comment)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    log_path= output_path + '/logs'
    if not os.path.exists(log_path):
        os.makedirs(log_path)   
    embedding_path= output_path + '/embeddings'
    if not os.path.exists(embedding_path):
        os.makedirs(embedding_path)  
    sample_path = output_path + '/images' 
    if not os.path.exists(sample_path):
        os.makedirs(sample_path) 
    model_path = output_path + '/models' 
    if not os.path.exists(model_path):
        os.makedirs(model_path)     
    
    ### jupyter-tensorboard writer ###
    writer = SummaryWriter(log_path)
    writer2 = SummaryWriter(embedding_path)
    
    ### Dataset & Dataloader ###
    if opt.data_name == "MNIST":
        train_loader = utils_dataset.loader_MNIST(opt.batch_size)
        # nodes of network
        node_sizes_d = hgan.nodeSizesDiscriminatorMNIST
        node_sizes_d[-1] = opt.out_dim
        
        node_sizes_g = hgan.nodeSizesGeneratorMNIST
        dimImage = (1,28,28)
        sizeImageVect = 784
        
    elif opt.data_name == "FashionMNIST":
        train_loader = utils_dataset.loader_FashionMNIST(opt.batch_size)
        
    elif opt.data_name == "celebA":
        train_loader = utils_dataset.loader_celebA(opt.batch_size, opt.size)
        # nodes of network
        node_sizes_d = hgan.nodeSizesDiscriminatorMNIST
        node_sizes_d[-1] = opt.out_dim
        
        node_sizes_g = hgan.nodeSizesGeneratorMNIST
        dimImage = (3,opt.size,opt.size)
        sizeImageVect = 784
    
    else:
        train_loader = utils_dataset.loader_custom(opt.batch_size, opt.size, opt.data_path)
        
    
    ### Device ###
    if (torch.cuda.is_available()) and (torch.cuda.device_count() > 1):
        print("Let's use {}GPUs!".format(torch.cuda.device_count()))
    else : raise Exception("No multi-GPU")
    
    if torch.cuda.is_available():
        DEVICE = torch.device('cuda')
    else:
        DEVICE = torch.device('cpu')
    print('Using PyTorch version:', torch.__version__, ' Device:', DEVICE)
    
    # Load Discriminator network
    kwargs = { 'c' : opt.c_d }
    kwargs['bias'] = opt.bias
    kwargs['cfg'] = opt.cfg_d
    kwargs['nodeSizes'] = node_sizes_d
    kwargs['modeTrainC'] = opt.mode_train_c_d
    kwargs['lm_c'] = opt.lrmul_c
    netML = hgan.HypDiscriminator(**kwargs)
    logging.info('Load Discriminator network with cfg' + opt.cfg_d)
    logging.info(netML)
    print(netML)

    # Load Generator network
    kwargs = { 'c' : opt.c_g }
    kwargs['bias'] = opt.bias
    kwargs['cfg'] = opt.cfg_g
    kwargs['nodeSizes'] = node_sizes_g
    kwargs['modeTrainC'] = opt.mode_train_c_g
    kwargs['lm_c'] = opt.lrmul_c
    netG = hgan.HypGenerator(**kwargs)
    logging.info('Load Generator  network with cfg' + opt.cfg_g)
    logging.info(netG)
    print(netG)

    """
    ### Generator ###
    netG = eval(opt.g_model_name+'()')
    if opt.load_model == 'no':
        netG = nn.DataParallel(netG)
    print('G MODEL: '+opt.g_model_name)
    
    ### ML ###
    netML = eval(opt.ml_model_name+'(out_dim={})'.format(opt.out_dim))
    if opt.load_model == 'no':
        netML = nn.DataParallel(netML)
    print('ML MODEL: '+opt.ml_model_name)
     
    
    if opt.load_model == 'yes':
        netG.load_state_dict(data_utils.remove_module_str_in_state_dict(torch.load(str(opt.load_model_path + "/generator_latest.pt"))))
        netG = nn.DataParallel(netG)
        
        netML.load_state_dict(data_utils.remove_module_str_in_state_dict(torch.load(str(opt.load_model_path + "/ml_model_latest.pt")))) 
        netML = nn.DataParallel(netML)
        print("LOAD MODEL SUCCESSFULLY")
    """

    optimizerG = optim.Adam(netG.parameters(), lr = opt.lr, betas=(opt.beta1,opt.beta2),eps= 1e-6) 
    optimizerML = optim.Adam(netML.parameters(), lr = opt.lr, betas=(opt.beta1,opt.beta2),eps= 1e-6)   
    
    netG.to(DEVICE)     
    netML.to(DEVICE)     
    
    ### Losses ###   
    triplet_ = TripletLoss(opt.margin, opt.alpha).to(DEVICE)    
  
    ### eval_ep ###
    if opt.eval_ep == '':
        step = opt.ep // 10
        step = step if step != 0 else 2
        opt.eval_ep = '[1]+r({},{},{})+[{}]'.format(step, opt.ep, step, opt.ep)
    eval_ep = eval(opt.eval_ep.replace("r", "list(range").replace(")", "))"))
    print(eval_ep)

###---------------------------------------- Train ----------------------------------------
    for epoch in range(1, opt.ep + 1):
    
        netG.train()
        netML.train()        
        
        ### For recording latent ###
        fake_embedding = torch.zeros(1, opt.out_dim).to(DEVICE) 
        real_embedding = torch.zeros(1, opt.out_dim).to(DEVICE) 
        labels_embedding = []
        label_real_img = torch.zeros(1, 1, opt.size, opt.size).to(DEVICE) 
        label_fake_img = torch.zeros(1, 1, opt.size, opt.size).to(DEVICE) 
        ### For recording manifold ###
        for images, labels in tqdm(train_loader, leave=True, desc='[%d/%d]' % (epoch, opt.ep)) :
            real_img = Variable(images).to(DEVICE)     #.cuda() 
            
            batch_size = real_img.size()[0]
            if batch_size != opt.batch_size:
                continue
            
            label_real_img = torch.cat([label_real_img, data_utils.invert_grayscale(real_img)], dim=0)          
            
            ############################ 
            # Metric Learning
            ############################                                         
            data_utils.requires_grad(netG, False)
            data_utils.requires_grad(netML, True)                
                
            z = torch.randn(batch_size, node_sizes_g[0]).to(DEVICE)            
            z.requires_grad_(True)            

            fake_img = netG(z)
            ml_real_out = netML(data_utils.images_to_vectors(real_img, sizeImageVect))
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
            data_utils.requires_grad(netG, True)
            data_utils.requires_grad(netML, False)            
            
            fake_img = netG(z)            
            ml_real_out = netML(data_utils.images_to_vectors(real_img, sizeImageVect))
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
            labels_embedding += labels
            
            fake_img = data_utils.vectors_to_images(fake_img, dimImage)
            label_fake_img = torch.cat([label_fake_img, data_utils.invert_grayscale(fake_img)], dim=0)    
            ### For recording manifold ###
        
            #train_bar.set_description(desc='[%d/%d]' % (epoch, NUM_EPOCHS))
            
###--------------------logs----------------------------------------        
        print("triplet_loss: {}".format(triplet_loss))
        print("g_loss: {}".format(g_loss))
        writer.add_scalars("loss/train", {"triplet_loss": triplet_loss, "g_loss": g_loss}, epoch)
        #writer.add_scalar("triplet_loss/train", triplet_loss, epoch)
        #writer.add_scalar("g_loss/train", g_loss, epoch)
        
###--------------------Hausdorff distance----------------------------------------
        h_distance, idx1, idx2 = scipy.spatial.distance.directed_hausdorff(ml_real_out.clone().cpu().detach().numpy(), ml_fake_out.clone().cpu().detach().numpy(), seed=0)
        print('hausdorff distance: ', h_distance)        
        print(' c_dist:',c_dist.item(), ' p_dist:', p_dist.item(),' triplet_loss:',triplet_loss.item())

###------------------Save generated samples--------------------------------------------------      
        fixed_noise = data_utils.gen_rand_noise_MNIST(opt.num_samples).to(DEVICE)        
        gen_images = data_utils.generate_image(netG, dim=opt.size, channel=1, batch_size=opt.num_samples, noise=fixed_noise)
        utils.save_image(gen_images, str(sample_path  +'/' + 'samples_{}.png').format(epoch), nrow=int(sqrt(opt.num_samples)), padding=2)    
        
###--------------------Save Embeddings----------------------------------------        
        if epoch not in eval_ep:
            continue
        
        real_embedding = real_embedding.cpu().detach().numpy()
        real_embedding = np.delete(real_embedding, 0, 0)
        print("Real img embeddings size: {}".format(real_embedding.shape))
        fake_embedding = fake_embedding.cpu().detach().numpy()
        fake_embedding = np.delete(fake_embedding, 0, 0)
        print("Fake img embeddings size: {}".format(real_embedding.shape))
        #print(fake_embedding_np.shape)
        print("{} labels".format(len(labels_embedding)))
        label_real_img = label_real_img.cpu().detach()[1:]
        label_fake_img = label_fake_img.cpu().detach()[1:]
        
        writer2.add_embedding(real_embedding[:opt.n_to_show],
                    metadata=labels_embedding[:opt.n_to_show],
                    label_img=label_real_img[:opt.n_to_show],
                    global_step=epoch,
                    tag='supervised')
        
        n = opt.n_to_show//2
        concat_embedding = np.concatenate((real_embedding[:n], fake_embedding[:n]), axis=0)
        real_fake_labels = [labels_embedding[i] for i in range(n)] + ['fake' for i in range(n)]
        real_fake_label_img = torch.cat([label_real_img[:n], label_fake_img[:n]], dim=0)   
        
        writer2.add_embedding(concat_embedding,
                    metadata=real_fake_labels,
                    label_img=real_fake_label_img,
                    global_step=epoch,
                    tag='RealAndFake')
        
        del real_embedding
        del fake_embedding
        del label_real_img
        del label_fake_img
        torch.cuda.empty_cache()
        
        #    np.save(str(embedding_path  +'/' + 'real_embdding_{}').format(epoch), real_embedding_np)
        #    np.save(str(embedding_path  +'/' + 'fake_embdding_{}').format(epoch), fake_embedding_np)
                 
        
# 	#----------------------Save model----------------------
        torch.save(netG.state_dict(), str(model_path  +'/' + "generator_{}.pt").format(epoch))
        torch.save(netML.state_dict(), str(model_path  +'/' + "ml_model_{}.pt").format(epoch))
        torch.save(netG.state_dict(), str(model_path  +'/' + "generator_latest.pt"))
        torch.save(netML.state_dict(), str(model_path  +'/' + "ml_model_latest.pt"))
    
    writer.close()   

