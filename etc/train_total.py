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

from tensorboard.plugins import projector

from tqdm import tqdm
import pickle
import torch.nn.functional as F
from torch import autograd


from data_utils import is_image_file, train_transform, TrainDatasetFromFolder, \
                        generate_image, gen_rand_noise_MNIST, remove_module_str_in_state_dict, \
                            requires_grad, str2bool, invert_grayscale
from loss import TripletLoss, pairwise_distances
from model import Generator28_linear, Generator64, Generator128, ML28, ML64, ML128

parser = argparse.ArgumentParser(description='Train Image Generation Models')
# DATA
parser.add_argument('--data_path', default='/mnt/prj/Data/', type=str, help='dataset path')
parser.add_argument('--data_name', default='MNIST', type=str, help='dataset name')
parser.add_argument('--name', default='results', type=str, help='path to store results')
parser.add_argument('--write_logs', default=True, type=str2bool, help='Write logs or not')
parser.add_argument('--comment', default='', type=str, help='comment what is changed')

# HYPERPARAMETER
parser.add_argument('--size', default=28, type=int, help='training images size')
parser.add_argument('--out_dim', default=100, type=int, help='ML network output dim')
parser.add_argument('--ep', default=200, type=int, help='train epoch number')
parser.add_argument('--num_samples', default=64, type=int, help='number of displayed samples')
parser.add_argument('--batch_size', default=128, type=int, help='train batch size')
parser.add_argument('--lr', default=2e-4, type=float, help='train learning rate')
parser.add_argument('--beta1', default=0.5, type=float, help='Adam optimizer beta1')
parser.add_argument('--beta2', default=0.999, type=float, help='Adam optimizer beta2')
parser.add_argument('--margin', default=2, type=float, help='triplet loss margin')
parser.add_argument('--alpha', default=1e-2, type=float, help='triplet loss direction guidance weight parameter')

# EMBEDDINGS
parser.add_argument('--n_to_show', default=10000, type=int, help='Num of fixed latent vector to show embeddings')
parser.add_argument('--eval_ep', default='', type=str, help='epochs for evaluation, [] or range "r(start,end,step)", e.g. "r(10,70,20)+[200]" means 10, 30, 50, 200"""')

# PRETRAINED MODEL
parser.add_argument('--load_model', default = 'no', type=str, choices=['yes', 'no'], help='if load previously trained model')
parser.add_argument('--load_model_path', default = '', type=str, help='load model dir path')

# MODEL NAME
parser.add_argument('--g_model_name', default='Generator28_linear', type=str, help='generator model name')
parser.add_argument('--ml_model_name', default='ML28', type=str, help='metric learning model name')

# SYSTEM
parser.add_argument('--n_threads', type=int, default=16)
parser.add_argument('--multi_gpu', default=False, type=str2bool, help='Use multi-gpu')



if __name__ == '__main__':
    opt = parser.parse_args()
    
    SIZE = opt.size
    out_dim = opt.out_dim
    learning_rate = opt.lr
    beta1 = opt.beta1
    beta2 = opt.beta2
    batch_size = opt.batch_size
    NUM_EPOCHS = opt.ep
    num_samples = opt.num_samples
    LOAD_MODEL = opt.load_model
    G_MODEL_NAME = opt.g_model_name
    ML_MODEL_NAME = opt.ml_model_name
    margin = opt.margin
    alpha = opt.alpha
    multi_gpu = opt.multi_gpu
    write_logs = opt.write_logs
    COMMENT = opt.comment
    N_TO_SHOW = opt.n_to_show
    
    ### Result path ###   
    now = datetime.datetime.now().strftime("%m%d_%H%M")
    output_path = str(opt.name + '/' + '{}_{}_{}_e{}_b{}_d{}').format(now, ML_MODEL_NAME, opt.data_name, NUM_EPOCHS, batch_size, out_dim)
    if COMMENT:
        output_path += '_{}'.format(COMMENT)
    if write_logs and not os.path.exists(output_path):
        os.makedirs(output_path)
    log_path= str(output_path + '/logs')
    if write_logs and not os.path.exists(log_path):
        os.makedirs(log_path)   
    embedding_path= str(output_path + '/embeddings')
    if write_logs and not os.path.exists(embedding_path):
        os.makedirs(embedding_path)  
    sample_path = output_path + '/images' 
    if write_logs and not os.path.exists(sample_path):
        os.makedirs(sample_path) 
    model_path = output_path + '/models' 
    if write_logs and not os.path.exists(model_path):
        os.makedirs(model_path)     
    
    ### jupyter-tensorboard writer ###
    if write_logs:
        writer = SummaryWriter(log_path)
        writer2 = SummaryWriter(embedding_path)
    
    ### Dataset & Dataloader ###
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
                                              shuffle=True, num_workers= 16)
    else:
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers= 0)
    
    ### Multi-GPU ###
    if multi_gpu:
        if (torch.cuda.is_available()) and (torch.cuda.device_count() > 1):
            print("Let's use {}GPUs!".format(torch.cuda.device_count()))
        else : multi_gpu = False
    
    ### Device ###
    if torch.cuda.is_available():
        DEVICE = torch.device('cuda')
    else:
        DEVICE = torch.device('cpu')
    print('Using PyTorch version:', torch.__version__, ' Device:', DEVICE)
    
    ### Generator ###
    netG = eval(G_MODEL_NAME+'()')
    if multi_gpu and LOAD_MODEL == 'no':
        netG = nn.DataParallel(netG)
    print('G MODEL: '+G_MODEL_NAME)
    
    ### ML ###
    netML = eval(ML_MODEL_NAME+'(out_dim={})'.format(out_dim))
    if multi_gpu and LOAD_MODEL == 'no':
        netML = nn.DataParallel(netML)
    print('ML MODEL: '+ML_MODEL_NAME)
     
    
    if LOAD_MODEL == 'yes':
        netG.load_state_dict(remove_module_str_in_state_dict(torch.load(str(opt.load_model_path + "/generator_latest.pt"))))
        netG = nn.DataParallel(netG)
        
        netML.load_state_dict(remove_module_str_in_state_dict(torch.load(str(opt.load_model_path + "/ml_model_latest.pt")))) 
        netML = nn.DataParallel(netML)
        print("LOAD MODEL SUCCESSFULLY")
    
    optimizerG = optim.Adam(netG.parameters(), lr = learning_rate, betas=(beta1,beta2),eps= 1e-6) 
    optimizerML = optim.Adam(netML.parameters(), lr = 2e-6, betas=(0.5,0.999),eps= 1e-3)   
    
    netG.to(DEVICE)     
    netML.to(DEVICE)     
    
    ### Losses ###   
    triplet_ = TripletLoss(margin, alpha).to(DEVICE)    
  
    ### eval_ep ###
    if opt.eval_ep == '':
        step = NUM_EPOCHS // 10
        step = step if step != 0 else 2
        opt.eval_ep = '[1]+r({},{},{})+[{}]'.format(step, opt.ep, step, NUM_EPOCHS)
    eval_ep = eval(opt.eval_ep.replace("r", "list(range").replace(")", "))"))
    print(eval_ep)

###---------------------------------------- Train ----------------------------------------
    for epoch in range(1, NUM_EPOCHS + 1):
    
        netG.train()
        netML.train()        
        
        ### For recording latent ###
        fake_embedding = torch.zeros(1, out_dim).to(DEVICE) 
        real_embedding = torch.zeros(1, out_dim).to(DEVICE) 
        labels_embedding = []
        label_real_img = torch.zeros(1, 1, SIZE, SIZE).to(DEVICE) 
        label_fake_img = torch.zeros(1, 1, SIZE, SIZE).to(DEVICE) 
        ### For recording manifold ###
        for images, labels in tqdm(train_loader, leave=True, desc='[%d/%d]' % (epoch, NUM_EPOCHS)) :
            real_img = Variable(images).to(DEVICE)     #.cuda() 
            
            batch_size = real_img.size()[0]
            if batch_size != opt.batch_size:
                continue
            
            label_real_img = torch.cat([label_real_img, invert_grayscale(real_img)], dim=0)          
            
            ############################ 
            # Metric Learning
            ############################                                         
            requires_grad(netG, False)
            requires_grad(netML, True)                
                
            z = torch.randn(batch_size, 128).to(DEVICE)            
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
            labels_embedding += labels
            label_fake_img = torch.cat([label_fake_img, invert_grayscale(fake_img)], dim=0)    
            ### For recording manifold ###
        
            #train_bar.set_description(desc='[%d/%d]' % (epoch, NUM_EPOCHS))
            
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

###------------------Save generated samples--------------------------------------------------      
        if not write_logs:
            continue
        
        fixed_noise = gen_rand_noise_MNIST(num_samples).to(DEVICE)        
        gen_images = generate_image(netG, dim=SIZE, channel=1, batch_size=num_samples, noise=fixed_noise)
        if write_logs:
            utils.save_image(gen_images, str(sample_path  +'/' + 'samples_{}.png').format(epoch), nrow=int(sqrt(num_samples)), padding=2)    
        
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
        
        writer2.add_embedding(real_embedding[:N_TO_SHOW],
                    metadata=labels_embedding[:N_TO_SHOW],
                    label_img=label_real_img[:N_TO_SHOW],
                    global_step=epoch,
                    tag='supervised')
        
        n = N_TO_SHOW//2
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
        
        #if write_logs:
        #    np.save(str(embedding_path  +'/' + 'real_embdding_{}').format(epoch), real_embedding_np)
        #    np.save(str(embedding_path  +'/' + 'fake_embdding_{}').format(epoch), fake_embedding_np)
                 
        
# 	#----------------------Save model----------------------
        torch.save(netG.state_dict(), str(model_path  +'/' + "generator_{}.pt").format(epoch))
        torch.save(netML.state_dict(), str(model_path  +'/' + "ml_model_{}.pt").format(epoch))
        torch.save(netG.state_dict(), str(model_path  +'/' + "generator_latest.pt"))
        torch.save(netML.state_dict(), str(model_path  +'/' + "ml_model_latest.pt"))
    
    if write_logs:    
        writer.close()   

