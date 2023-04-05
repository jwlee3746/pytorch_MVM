import argparse
import os
import json
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
                        generate_image, gen_rand_noise, gen_rand_noise_MNIST, remove_module_str_in_state_dict, \
                            requires_grad, str2bool, invert_grayscale
from loss import TripletLoss, pairwise_distances
from model import Generator28_linear, Generator64, Generator128, ML28, ML64, ML128, MappingNetwork, gen64_DCGAN, ML64_DCGAN, discriminator

parser = argparse.ArgumentParser(description='Train Image Generation Models')
# DATA
parser.add_argument('--data_path', default='/mnt/prj/Data/', type=str, help='dataset path')
parser.add_argument('--data_name', default='MNIST', type=str, help='dataset name')
parser.add_argument('--name', default='results', type=str, help='path to store results')
parser.add_argument('--write_logs', default=True, type=str2bool, help='Write logs or not')
parser.add_argument('--comment', default='', type=str, help='comment what is changed')
parser.add_argument('--concat', default='', type=str, help='concat with ?')

# HYPERPARAMETER
parser.add_argument('--size', default=64, type=int, help='training images size')
#parser.add_argument('--out_dim', default=128, type=int, help='ML network output dim')
parser.add_argument('--out_dim', default=10, type=int, help='ML network output dim')
parser.add_argument('--n_layers', default=4, type=int, help='Discriminator n layers')
parser.add_argument('--ep', default=100, type=int, help='train epoch number')
parser.add_argument('--num_samples', default=256, type=int, help='number of displayed samples')
parser.add_argument('--batch_size', default=64, type=int, help='train batch size')
parser.add_argument('--g_lr', default=2e-4, type=float, help='train learning rate')
parser.add_argument('--ml_lr', default=2e-6, type=float, help='train learning rate')
parser.add_argument('--d_lr', default=2e-4, type=float, help='train learning rate')
parser.add_argument('--beta1', default=0.0, type=float, help='Adam optimizer beta1')
parser.add_argument('--beta2', default=0.9, type=float, help='Adam optimizer beta2')
parser.add_argument('--margin', default=10, type=float, help='triplet loss margin')
parser.add_argument('--alpha', default=1e-2, type=float, help='triplet loss direction guidance weight parameter')

# EMBEDDINGS
parser.add_argument('--n_to_show', default=10000, type=int, help='Num of fixed latent vector to show embeddings')
parser.add_argument('--eval_ep', default='', type=str, help='epochs for evaluation, [] or range "r(start,end,step)", e.g. "r(10,70,20)+[200]" means 10, 30, 50, 200"""')

# PRETRAINED MODEL
parser.add_argument('--load_model', default = 'no', type=str, choices=['yes', 'no'], help='if load previously trained model')
parser.add_argument('--load_model_path', default = '', type=str, help='load model dir path')

# MODEL NAME
parser.add_argument('--g_model_name', default='gen64_DCGAN', type=str, help='generator model name')
parser.add_argument('--ml_model_name', default='ML64_DCGAN', type=str, help='metric learning model name')

# SYSTEM
parser.add_argument('--n_threads', type=int, default=16)

class ConcatDataset(torch.utils.data.Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets

    def __getitem__(self, i):
        return tuple(d[i] for d in self.datasets)

    def __len__(self):
        return min(len(d) for d in self.datasets)
    
if __name__ == '__main__':
    opt = parser.parse_args()
    
    ### Result path ###   
    now = datetime.datetime.now().strftime("%m%d_%H%M")
    output_path = str(opt.name + '/' + '{}_{}_{}_e{}_b{}_d{}').format(now, opt.data_name, opt.ml_model_name, opt.ep, opt.batch_size, opt.out_dim)
    if opt.comment:
        output_path += '_{}'.format(opt.comment)
    if opt.write_logs and not os.path.exists(output_path):
        os.makedirs(output_path)
    log_path= str(output_path + '/logs')
    if opt.write_logs and not os.path.exists(log_path):
        os.makedirs(log_path)   
    embedding_path= str(output_path + '/embeddings')
    if opt.write_logs and not os.path.exists(embedding_path):
        os.makedirs(embedding_path)  
    sample_path = output_path + '/images' 
    if opt.write_logs and not os.path.exists(sample_path):
        os.makedirs(sample_path) 
    model_path = output_path + '/models' 
    if opt.write_logs and not os.path.exists(model_path):
        os.makedirs(model_path)        
    code_path = output_path + '/code' 
    if opt.write_logs and not os.path.exists(code_path):
        os.makedirs(code_path)      
    
    from shutil import copy
    copy('train_DCGAN.py', code_path+'/train_DCGAN.py')
    args_dict = vars(opt)
    with open(os.path.join(code_path, 'training_options.json'), 'wt+') as f:
        json.dump(args_dict, f, indent=2)
    
    ### jupyter-tensorboard writer ###
    if opt.write_logs:
        writer = SummaryWriter(log_path)
        writer2 = SummaryWriter(embedding_path)
    
    ### Dataset & Dataloader ###
    transform = transforms.Compose(
        [transforms.Scale(opt.size),
         transforms.ToTensor(), 
         transforms.Normalize((0.5,), (0.5,)),
         ]
    )
    
    if opt.data_name == 'MNIST':
        trainset = datasets.MNIST(root = opt.data_path,
                                       train = True,
                                       download = True,
                                       transform = transform
                                       )
    elif opt.data_name == 'FashionMNIST':
        trainset = datasets.FashionMNIST(root = opt.data_path,
                                       train = True,
                                       download = True,
                                       transform = transform
                                       )
    elif opt.data_name == 'star-MNIST':
        trainset = TrainDatasetFromFolder(opt.data_path + 'star-MNIST',
                                          size=opt.size,
                                          file_format='png',
                                          mode='L'
                                          )
    else:
        trainset = TrainDatasetFromFolder(opt.data_path,
                                          size=opt.size,
                                          file_format='jpg'
                                          )
        
    if opt.concat:
        trainset2 = getattr(datasets, opt.concat)(root = opt.data_path,
                                       train = True,
                                       download = True,
                                       transform = transform
                                       )
        
        concat_dataset = torch.utils.data.ConcatDataset([trainset, trainset2])
        
        train_loader = torch.utils.data.DataLoader(concat_dataset, batch_size=opt.batch_size,
                                              shuffle=True, num_workers= 16)
    else:
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=opt.batch_size,
                                              shuffle=True, num_workers= 16)
        
    ### Multi-GPU ###
    if (torch.cuda.is_available()) and (torch.cuda.device_count() > 1):
        print("Let's use {}GPUs!".format(torch.cuda.device_count()))
    else : raise Exception("no multi gpu")
    
    ### Device ###
    if torch.cuda.is_available():
        DEVICE = torch.device('cuda')
    else:
        DEVICE = torch.device('cpu')
    print('Using PyTorch version:', torch.__version__, ' Device:', DEVICE)
    
    ### Generator ###
    channel = 1
    netG = eval(opt.g_model_name+'(channel={})'.format(channel))
    if opt.load_model == 'no':
        netG = nn.DataParallel(netG)
    print('G MODEL: '+opt.g_model_name)
    
    ### ML ###
    netML = eval(opt.ml_model_name+'(channel={}, out_dim={})'.format(channel, opt.out_dim))
    if opt.load_model == 'no':
        netML = nn.DataParallel(netML)
    print('ML MODEL: '+opt.ml_model_name)
    
    ### Discriminator ###
    netD = discriminator(input_dim=opt.out_dim, n_layers=opt.n_layers)
    netD = nn.DataParallel(netD)
    print('D MODEL: '+ 'discriminator')
    
     
    if opt.load_model == 'yes':
        netG.load_state_dict(remove_module_str_in_state_dict(torch.load(str(opt.load_model_path + "/generator_latest.pt"))))
        netG = nn.DataParallel(netG)
        
        netML.load_state_dict(remove_module_str_in_state_dict(torch.load(str(opt.load_model_path + "/ml_model_latest.pt")))) 
        netML = nn.DataParallel(netML)
        print("LOAD MODEL SUCCESSFULLY")
    
    optimizerG = optim.Adam(netG.parameters(), lr = opt.g_lr, betas=(opt.beta1,opt.beta2)) 
    optimizerML = optim.Adam(netML.parameters(), lr = opt.ml_lr, betas=(0.5,0.999))   
    optimizerD = optim.Adam(netD.parameters(), lr = opt.d_lr, betas=(0.5,0.999))  
    #optimizerMap = optim.Adam(netMap.parameters(), lr = opt.lr, betas=(0.0,0.99))
    
    netG.to(DEVICE)     
    netML.to(DEVICE)       
    netD.to(DEVICE)   
    
    ### Losses ###   
    triplet_ = TripletLoss(opt.margin, opt.alpha).to(DEVICE)  
    BCE_loss = nn.BCELoss() 
  
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
        for_real_img = torch.tensor([]).to(DEVICE)
        fake_embedding = torch.tensor([]).to(DEVICE) 
        real_embedding = torch.tensor([]).to(DEVICE) 
        labels_embedding = []
        label_real_img = torch.tensor([]).to(DEVICE) 
        label_fake_img = torch.tensor([]).to(DEVICE) 
        ### For recording manifold ###
        for target in tqdm(train_loader, leave=True, desc='[%d/%d]' % (epoch, opt.ep)) :
            if type(target) == list:
                images, labels = target
            else:
                images = target
            real_img = Variable(images).to(DEVICE)
            ### For recording latent ###
            if for_real_img.size()[0] < opt.num_samples:
                for_real_img = torch.cat([for_real_img, real_img], dim=0)  
            ### For recording latent ###
            
            batch_size = real_img.size()[0]
            if batch_size != opt.batch_size:
                continue
            
            ### For recording latent ###
            if label_real_img.size()[0] < opt.n_to_show:
                label_real_img = torch.cat([label_real_img, invert_grayscale(real_img)], dim=0)          
            ### For recording latent ###
            
            ############################ 
            # Metric Learning
            ############################                                         
            requires_grad(netG, False)                                       
            requires_grad(netD, False)
            requires_grad(netML, True)                
                
            z = torch.randn(batch_size, 128, 1, 1).to(DEVICE)    
            #z = torch.randn(batch_size, opt.out_dim).to(DEVICE)    
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
            # Discriminator
            ############################                                         
            requires_grad(netG, False)
            requires_grad(netML, False)                                   
            requires_grad(netD, True)             
                
            y_real_ = torch.ones(batch_size).to(DEVICE)
            y_fake_ = torch.zeros(batch_size).to(DEVICE)
            
            ml_real_out = netML(real_img)
            D_result = netD(ml_real_out)
            D_real_loss = BCE_loss(D_result.squeeze(), y_real_)
            
            z = torch.randn(batch_size, 128, 1, 1).to(DEVICE)    
            #z = torch.randn(batch_size, opt.out_dim).to(DEVICE)    
            z.requires_grad_(True)            

            fake_img = netG(z)
            ml_fake_out = netML(fake_img)
            D_result = netD(ml_fake_out) 
            D_fake_loss = BCE_loss(D_result.squeeze(), y_fake_)
            D_fake_score = ml_fake_out.data.mean()
            
            D_train_loss = D_real_loss + D_fake_loss
            
            netD.zero_grad()
            D_train_loss.backward()
            optimizerD.step() 
            
            ############################ 
            # Generative Model
            ############################           
            requires_grad(netG, True)                                
            requires_grad(netD, False)
            requires_grad(netML, False)            
            
            fake_img = netG(z)            
            ml_real_out = netML(real_img)
            ml_fake_out = netML(fake_img) 
            
            D_result = netD(ml_fake_out)
            G_train_loss = BCE_loss(D_result.squeeze(), y_real_)              
            
            #r1=torch.randperm(batch_size)
            #r2=torch.randperm(batch_size)
            #ml_real_out_shuffle = ml_real_out[r1[:, None]].view(ml_real_out.shape[0],ml_real_out.shape[-1])
            #ml_fake_out_shuffle = ml_fake_out[r2[:, None]].view(ml_fake_out.shape[0],ml_fake_out.shape[-1])
                
            ############################
            pd_r = pairwise_distances(ml_real_out, ml_real_out) 
            pd_f = pairwise_distances(ml_fake_out, ml_fake_out)
        
            p_dist =  torch.dist(pd_r,pd_f,2)             
            c_dist = torch.dist(ml_real_out.mean(0),ml_fake_out.mean(0),2)         
        
            #############################
            g_loss = G_train_loss + p_dist + c_dist   
            netG.zero_grad()            
            g_loss.backward()                        
            optimizerG.step()
            
            ### For recording manifold ###
            real_embedding = torch.cat([real_embedding, ml_real_out], dim=0)
            fake_embedding = torch.cat([fake_embedding, ml_fake_out], dim=0)
            labels_embedding += labels
            if label_fake_img.size()[0] < opt.n_to_show:
                label_fake_img = torch.cat([label_fake_img, invert_grayscale(fake_img)], dim=0)    
            ### For recording manifold ###
        
            #train_bar.set_description(desc='[%d/%d]' % (epoch, NUM_EPOCHS))
            
###--------------------write_logs----------------------------------------        
        print("triplet_loss: {}".format(triplet_loss))
        print("D_train_loss: {}".format(D_train_loss))
        print("g_loss: {}".format(g_loss))
        
        h_distance, idx1, idx2 = scipy.spatial.distance.directed_hausdorff(ml_real_out.clone().cpu().detach().numpy(), ml_fake_out.clone().cpu().detach().numpy(), seed=0)
        print('hausdorff distance: ', h_distance)        
        print(' c_dist:',c_dist.item(), ' p_dist:', p_dist.item(), ' G_train_loss:', G_train_loss.item(), ' triplet_loss:',triplet_loss.item())

        if opt.write_logs:
            writer.add_scalars("loss/train", {"triplet_loss": triplet_loss, "D_train_loss": D_train_loss, \
                                              "G_train_loss": G_train_loss, "p_dist": p_dist, "c_dist": c_dist,\
                                                  "hausdorff_distance": h_distance}, epoch)
            #writer.add_scalar("triplet_loss/train", triplet_loss, epoch)
            #writer.add_scalar("g_loss/train", g_loss, epoch)
###------------------Save generated samples--------------------------------------------------      
        if not opt.write_logs:
            continue
        
        if epoch == 1:
            utils.save_image(for_real_img[:opt.num_samples], str(sample_path  +'/' + 'realimg.png'), \
                             nrow=int(sqrt(opt.num_samples)), padding=2)
        fixed_noise = gen_rand_noise(opt.num_samples).to(DEVICE)        
        gen_images = generate_image(netG, dim=opt.size, channel=1, batch_size=opt.num_samples, noise=fixed_noise)
        utils.save_image(gen_images, str(sample_path  +'/' + 'samples_{}.png').format(epoch), \
                                 nrow=int(sqrt(opt.num_samples)), padding=2)    
        
###--------------------Save Embeddings----------------------------------------        
        if epoch not in eval_ep:
            continue
        
        real_embedding = real_embedding.cpu().detach().numpy()
        print("Real img embeddings size: {}".format(real_embedding.shape))
        fake_embedding = fake_embedding.cpu().detach().numpy()
        print("Fake img embeddings size: {}".format(real_embedding.shape))
        #print(fake_embedding_np.shape)
        print("{} labels".format(len(labels_embedding)))
        label_real_img = label_real_img.cpu().detach()
        label_fake_img = label_fake_img.cpu().detach()
        
       
        writer2.add_embedding(real_embedding[:opt.n_to_show],
                    metadata=labels_embedding[:opt.n_to_show],
                    global_step=epoch,
                    tag='noSprite_Real')
        
        writer2.add_embedding(real_embedding[:opt.n_to_show],
                    metadata=labels_embedding[:opt.n_to_show],
                    label_img=label_real_img[:opt.n_to_show],
                    global_step=epoch,
                    tag='Real')
        
        n = opt.n_to_show//2
        concat_embedding = np.concatenate((real_embedding[:n], fake_embedding[:n]), axis=0)
        real_fake_labels = [labels_embedding[i] for i in range(n)] + ['fake' for i in range(n)]
        real_fake_label_img = torch.cat([label_real_img[:n], label_fake_img[:n]], dim=0)   
        
        writer2.add_embedding(concat_embedding,
                    metadata=real_fake_labels,
                    global_step=epoch,
                    tag='noSprite_RealFake')
        
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
        
        #if write_logs:
        #    np.save(str(embedding_path  +'/' + 'real_embdding_{}').format(epoch), real_embedding_np)
        #    np.save(str(embedding_path  +'/' + 'fake_embdding_{}').format(epoch), fake_embedding_np)
                 
        
# 	#----------------------Save model----------------------
        torch.save(netG.state_dict(), str(model_path  +'/' + "generator_{}.pt").format(epoch))
        torch.save(netML.state_dict(), str(model_path  +'/' + "ml_model_{}.pt").format(epoch))
        torch.save(netG.state_dict(), str(model_path  +'/' + "generator_latest.pt"))
        torch.save(netML.state_dict(), str(model_path  +'/' + "ml_model_latest.pt"))
    
    if opt.write_logs:    
        writer.close()   

