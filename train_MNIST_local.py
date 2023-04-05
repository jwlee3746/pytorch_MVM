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
from loss import TripletLoss, pairwise_distances, AveragedHausdorffLoss, cosLoss, pairwise_cosLoss
from model import Generator28_linear, Generator64, Generator128, ML28, ML64, ML128, MappingNetwork, gen64_DCGAN, ML64_DCGAN

import warnings
warnings.filterwarnings(action='ignore')

parser = argparse.ArgumentParser(description='Train Image Generation Models')
# DATA
parser.add_argument('--data_path', default='./Data', type=str, help='dataset path')
parser.add_argument('--data_name', default='MNIST', type=str, help='dataset name')
parser.add_argument('--name', default='results', type=str, help='path to store results')
parser.add_argument('--write_logs', default=True, type=str2bool, help='Write logs or not')
parser.add_argument('--concat', default='', type=str, help='concat with ?')
parser.add_argument('--comment', default='cos_', type=str, help='comment what is changed')

# HYPERPARAMETER
parser.add_argument('--size', default=64, type=int, help='training images size')
#parser.add_argument('--out_dim', default=128, type=int, help='ML network output dim')
parser.add_argument('--out_dim', default=3, type=int, help='ML network output dim')
parser.add_argument('--noise_dim', default=100, type=int, help='generatior noise vector dim')
parser.add_argument('--ep', default=400, type=int, help='train epoch number')
parser.add_argument('--num_samples', default=256, type=int, help='number of displayed samples')
parser.add_argument('--batch_size', default=128, type=int, help='train batch size')
parser.add_argument('--g_lr', default=2e-4, type=float, help='train learning rate')
parser.add_argument('--ml_lr', default=2e-5, type=float, help='train learning rate')
parser.add_argument('--beta1', default=0.0, type=float, help='Adam optimizer beta1')
parser.add_argument('--beta2', default=0.9, type=float, help='Adam optimizer beta2')
parser.add_argument('--margin', default=10, type=float, help='triplet loss margin')
#parser.add_argument('--alpha', default=1e-2, type=float, help='triplet loss direction guidance weight parameter')
#parser.add_argument('--delta', default=0.01, type=float, help='')

# EMBEDDINGS
parser.add_argument('--n_to_show', default=2000, type=int, help='Num of fixed latent vector to show embeddings')
parser.add_argument('--eval_ep', default='', type=str, help='epochs for evaluation, [] or range "r(start,end,step)", e.g. "r(10,70,20)+[200]" means 10, 30, 50, 200"""')

# PRETRAINED MODEL
parser.add_argument('--load_model', default = 'no', type=str, choices=['yes', 'no'], help='if load previously trained model')
parser.add_argument('--load_model_path', default = '', type=str, help='load model dir path')

# MODEL NAME
parser.add_argument('--g_model_name', default='gen64_DCGAN', type=str, help='generator model name')
parser.add_argument('--ml_model_name', default='ML64', type=str, help='metric learning model name')

# SYSTEM
parser.add_argument('--n_threads', type=int, default=4)

class ConcatDataset(torch.utils.data.Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets

    def __getitem__(self, i):
        return tuple(d[i] for d in self.datasets)

    def __len__(self):
        return min(len(d) for d in self.datasets)

import matplotlib.pyplot as plt
def plot(ml_loss, g_loss, NUM_EPOCHES, epoch, epoch_count, real_embedding, fake_embedding, labels, dim3, filename):
    '''
    Plots for the GAN gif
    '''
    f, ax = plt.subplots(2, 2, figsize=(8,8))
    f.suptitle(f'       {epoch:05d}', fontsize=10)

    
    # [0, 0]: plot loss and accuracy
    ax[0, 0].plot(epoch_count, 
                    g_loss,
                    label='g_loss',
                    c='darkred',
                    zorder=50,
                    alpha=0.8,)
    ax[0, 0].plot(epoch_count, 
                    ml_loss,
                    label='ml_loss',
                    c='darkblue',
                    zorder=55,
                    alpha=0.8,)
    ax[0, 0].set_xlim(-5, NUM_EPOCHES+5)
    ax[0, 0].set_ylim(-0.1, 2.1)
    ax[0, 0].legend(loc=1)
    
    # [0, 1]: Plot actual samples and fake samples embeddings
    if not dim3:
        ax[0, 1].scatter(real_embedding[:, 0], 
                         real_embedding[:, 1], 
                         cmap='rainbow', c=labels, s=5, alpha=1, 
                         linewidth=1, label='Real')
        ax[0, 1].legend(loc=1)
    
    else:
        ax[0, 1].remove()
        ax[0, 1] = f.add_subplot(222,projection='3d')
        ax[0, 1].scatter(real_embedding[:, 0], 
                         real_embedding[:, 1], 
                         real_embedding[:, 2],
                         cmap='rainbow', c=labels, s=5, alpha=1, 
                         linewidth=1, label='Real')
        ax[0, 1].dist = 12
        ax[0, 1].legend(loc=1)
    
    # [1, 0]: Plot actual samples and fake samples embeddings
    if not dim3:
        ax[1, 0].scatter(real_embedding[:, 0], 
                         real_embedding[:, 1], 
                         cmap='rainbow', c=labels, s=5, alpha=1, 
                         linewidth=1, label='Real')
        ax[1, 0].scatter(fake_embedding[:, 0], 
                         fake_embedding[:, 1], 
                         edgecolor='red', facecolor='None', s=5, alpha=1, 
                         linewidth=1, label='MVM+')
        ax[1, 0].legend(loc=1)
    else:
        ax[1, 0].remove()
        ax[1, 0] = f.add_subplot(223,projection='3d')
        ax[1, 0].scatter(real_embedding[:, 0],
                         real_embedding[:, 1],
                         real_embedding[:, 2],
                         cmap='rainbow', c=labels, s=5, alpha=0.2, 
                         linewidth=1, label='Real')
        ax[1, 0].scatter(fake_embedding[:, 0], 
                         fake_embedding[:, 1], 
                         fake_embedding[:, 2], 
                         edgecolor='red', facecolor='None', s=5, alpha=1, 
                         linewidth=1, label='MVM+')
        ax[1 ,0].dist = 12
        ax[1, 0].legend(loc=1)
        
    # [1, 1]: Plot actual samples and fake samples embeddings
    if not dim3:
        ax[1, 1].scatter(real_embedding[:, 0], 
                         real_embedding[:, 1], 
                         cmap='rainbow', c=labels, s=5, alpha=1, 
                         linewidth=1, label='Real')
        ax[1, 1].scatter(fake_embedding[:, 0], 
                         fake_embedding[:, 1], 
                         edgecolor='red', facecolor='None', s=5, alpha=1, 
                         linewidth=1, label='MVM+')
        ax[1, 1].set_xlim(-0.5, 0.5)
        ax[1, 1].set_ylim(-0.5, 0.5)
        ax[1, 1].legend(loc=1)
    else:
        ax[1, 1].remove()
        ax[1, 1] = f.add_subplot(224,projection='3d')
        ax[1, 1].scatter(real_embedding[:, 0],
                         real_embedding[:, 1],
                         real_embedding[:, 2],
                         cmap='rainbow', c=labels, s=5, alpha=0.2, 
                         linewidth=1, label='Real')
        ax[1, 1].scatter(fake_embedding[:, 0], 
                         fake_embedding[:, 1], 
                         fake_embedding[:, 2], 
                         edgecolor='red', facecolor='None', s=5, alpha=1, 
                         linewidth=1, label='MVM+')
        ax[1, 1].set_xlim(-0.5, 0.5)
        ax[1, 1].set_ylim(-0.5, 0.5)
        ax[1, 1].set_zlim(-0.5, 0.5)
        ax[1 ,1].dist = 12
        ax[1, 1].legend(loc=1)
        
    
    plt.tight_layout()
    plt.savefig(visualization_path+'/'+filename)
    plt.close()
    
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
    visualization_path = output_path + '/visualization' 
    if opt.write_logs and not os.path.exists(visualization_path):
        os.makedirs(visualization_path)    
    
    from shutil import copy
    copy('train_MNIST_local.py', code_path+'/train_MNIST_local.py')
    copy('model.py', code_path+'/model.py')
    copy('loss.py', code_path+'/loss.py')
    copy('data_utils.py', code_path+'/data_utils.py')
    args_dict = vars(opt)
    with open(os.path.join(code_path, 'training_options.json'), 'wt+') as f:
        json.dump(args_dict, f, indent=2)
    
    ### jupyter-tensorboard writer ###
    if opt.write_logs:
        writer = SummaryWriter(log_path)
        writer2 = SummaryWriter(embedding_path)
    
    ### Dataset & Dataloader ###
    transform = transforms.Compose(
        [transforms.Resize(opt.size),
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
                                              shuffle=True, num_workers= 0)
    else:
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=opt.batch_size,
                                              shuffle=True, num_workers= 0)
        
    ### Multi-GPU ###
    if (torch.cuda.is_available()) and (torch.cuda.device_count() >= 1):
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
    netG = eval(opt.g_model_name+'(nz={}, channel={})'.format(opt.noise_dim, channel))
    if opt.load_model == 'no':
        netG = nn.DataParallel(netG)
    print('G MODEL: '+opt.g_model_name)
    
    ### ML ###
    netML = eval(opt.ml_model_name+'(channel={}, out_dim={})'.format(channel, opt.out_dim))
    if opt.load_model == 'no':
        netML = nn.DataParallel(netML)
    print('ML MODEL: '+opt.ml_model_name)
    
    ### Mapping Network ###
    #netMap = MappingNetwork(opt.out_dim, 4).to(DEVICE)
     
    if opt.load_model == 'yes':
        netG.load_state_dict(remove_module_str_in_state_dict(torch.load(str(opt.load_model_path + "/generator_latest.pt"))))
        netG = nn.DataParallel(netG)
        
        netML.load_state_dict(remove_module_str_in_state_dict(torch.load(str(opt.load_model_path + "/ml_model_latest.pt")))) 
        netML = nn.DataParallel(netML)
        print("LOAD MODEL SUCCESSFULLY")
    
    optimizerG = optim.Adam(netG.parameters(), lr = opt.g_lr, betas=(opt.beta1,opt.beta2)) 
    optimizerML = optim.Adam(netML.parameters(), lr = opt.ml_lr, betas=(0.5,0.999))   
    #optimizerMap = optim.Adam(netMap.parameters(), lr = opt.lr, betas=(0.0,0.99))
    
    netG.to(DEVICE)     
    netML.to(DEVICE)     
    
    ### Losses ###   
    #triplet_ = TripletLoss(opt.margin, opt.alpha).to(DEVICE)    
    d_hausdorff_ = AveragedHausdorffLoss(directed = True).to(DEVICE)
    a_hausdorff_ = AveragedHausdorffLoss(directed = False).to(DEVICE)
    Cos_ = cosLoss().to(DEVICE)
    pCos_ = pairwise_cosLoss().to(DEVICE)
    
    ### eval_ep ###
    if opt.eval_ep == '':
        step = opt.ep // 20
        step = step if step != 0 else 2
        opt.eval_ep = '[1]+r({},{},{})+[{}]'.format(step, opt.ep, step, opt.ep)
    eval_ep = eval(opt.eval_ep.replace("r", "list(range").replace(")", "))"))
    print(eval_ep)
    
    epoch_count = []
    with torch.no_grad():
        ML_loss = torch.tensor([]).to(DEVICE)
        G_loss = torch.tensor([]).to(DEVICE)
###---------------------------------------- Train ----------------------------------------
    for epoch in range(1, opt.ep + 1):
    
        netG.train()
        netML.train()        
        
        ### For recording manifold ###
        with torch.no_grad():
            for_real_img = torch.tensor([]).to(DEVICE)
            fake_embedding = torch.tensor([]).to(DEVICE) 
            real_embedding = torch.tensor([]).to(DEVICE) 
            labels_embedding = []
            label_real_img = torch.tensor([]).to(DEVICE) 
            label_fake_img = torch.tensor([]).to(DEVICE)
        ### For recording manifold ###
        for idx, target in enumerate(tqdm(train_loader, leave=True, desc='[%d/%d]' % (epoch, opt.ep))) :
            '''
            if idx * opt.batch_size > 1000:
                break
            '''
            if type(target) == list:
                images, labels = target
            else:
                images = target
            real_img = Variable(images).to(DEVICE)
            
            batch_size = real_img.size()[0]
            if batch_size != opt.batch_size:
                continue
            ### For recording manifold ###
            with torch.no_grad():
                if for_real_img.size()[0] < opt.num_samples:
                    for_real_img = torch.cat([for_real_img, real_img], dim=0)  
               
                if label_real_img.size()[0] < opt.n_to_show:
                    label_real_img = torch.cat([label_real_img, invert_grayscale(real_img)], dim=0)          
            ### For recording manifold ###
            
            ############################ 
            # Metric Learning
            ############################                                         
            requires_grad(netG, False)
            requires_grad(netML, True)                
                
            z = torch.randn(batch_size, opt.noise_dim, 1, 1).to(DEVICE)  
            z.requires_grad_(True)            

            fake_img = netG(z)
            ml_real_out = netML(real_img)
            ml_fake_out = netML(fake_img) 
            '''
            if epoch in eval_ep and idx in [0,1]:
                print("fake_img_before_triplet: {}".format(fake_img[:3])) 
                print("ml_real_out_before_triplet: {}".format(ml_real_out[:3]))   
                print("ml_fake_out_before_triplet: {}".format(ml_fake_out[:3]))
            '''
            '''
            r1=torch.randperm(batch_size)
            r2=torch.randperm(batch_size)
            ml_real_out_shuffle = ml_real_out[r1[:, None]].view(ml_real_out.shape[0],ml_real_out.shape[-1])
            ml_fake_out_shuffle = ml_fake_out[r2[:, None]].view(ml_fake_out.shape[0],ml_fake_out.shape[-1])
            
            triplet_loss = triplet_(ml_real_out,ml_real_out_shuffle,ml_fake_out_shuffle)
            '''
            r1=torch.randperm(batch_size)
            ml_real_out_shuffle = ml_real_out[r1[:, None]].view(ml_real_out.shape[0],ml_real_out.shape[-1])
            
            ml_loss = Cos_(ml_real_out,ml_real_out_shuffle,ml_fake_out)
            #ml_loss = pCos_(ml_real_out, ml_fake_out)
                        
            #if d_hausdorff_(ml_real_out, ml_fake_out) < opt.delta:
            #    print("ml optimizing...")
            netML.zero_grad()
            ml_loss.backward()
            optimizerML.step() 
            
            ############################ 
            # Generative Model
            ############################           
            requires_grad(netG, True)
            requires_grad(netML, False)            
                  
            #z = torch.randn(1000, opt.noise_dim, 1, 1).to(DEVICE)  
            #z.requires_grad_(True)    
            
            fake_img = netG(z)            
            ml_real_out = netML(real_img)
            ml_fake_out = netML(fake_img)   
            '''
            if epoch in eval_ep and idx in [0,1]: 
                print("fake_img_after_triplet: {}".format(fake_img[:3])) 
                print("ml_real_out_after_triplet: {}".format(ml_real_out[:3]))   
                print("ml_fake_out_after_triplet: {}".format(ml_fake_out[:3]))
            '''
            #r1=torch.randperm(batch_size)
            #r2=torch.randperm(batch_size)
            #ml_real_out_shuffle = ml_real_out[r1[:, None]].view(ml_real_out.shape[0],ml_real_out.shape[-1])
            #ml_fake_out_shuffle = ml_fake_out[r2[:, None]].view(ml_fake_out.shape[0],ml_fake_out.shape[-1])
                
            ############################
            #pd_r = pairwise_distances(ml_real_out, ml_real_out) 
            #pd_f = pairwise_distances(ml_fake_out, ml_fake_out) 
            
            #p_dist =  torch.dist(pd_r,pd_f,2)         
            #c_dist = torch.dist(ml_real_out.mean(0),ml_fake_out.mean(0),2)         
            
            #############################
            #g_loss = p_dist + c_dist
            g_loss = d_hausdorff_(ml_real_out, ml_fake_out)
            #g_loss = a_hausdorff_(pd_r, pd_f) + c_dist
            
            netG.zero_grad()            
            g_loss.backward()                        
            optimizerG.step()
            '''
            if epoch in eval_ep and idx in [0,1]:
                fake_img = netG(z)            
                ml_real_out = netML(real_img)
                ml_fake_out = netML(fake_img) 
                print("fake_img_after_gloss: {}".format(fake_img[:3])) 
                print("ml_real_out_after_gloss: {}".format(ml_real_out[:3]))   
                print("ml_fake_out_after_gloss: {}".format(ml_fake_out[:3]))
            '''
            ### For recording manifold ###
            with torch.no_grad():
                real_embedding = torch.cat([real_embedding, ml_real_out], dim=0)
                fake_embedding = torch.cat([fake_embedding, ml_fake_out], dim=0)
                labels_embedding += labels
                if label_fake_img.size()[0] < opt.n_to_show:
                    label_fake_img = torch.cat([label_fake_img, invert_grayscale(fake_img)], dim=0) 
            ### For recording manifold ###
        
            #train_bar.set_description(desc='[%d/%d]' % (epoch, NUM_EPOCHS))
            
###--------------------write_logs----------------------------------------              
        #print("ml_real_out: {}".format(ml_real_out))   
        #print("ml_fake_out: {}".format(ml_fake_out))
        #print("pd_r: {}".format(pd_r))
        #print("pd_f: {}".format(pd_f))
        
        print("ml_loss: {}".format(ml_loss))
        print("g_loss: {}".format(g_loss))
        #print("cos_reg: {}".format(cos_reg))
        
        #h_distance, idx1, idx2 = scipy.spatial.distance.directed_hausdorff(ml_real_out.clone().cpu().detach().numpy(), ml_fake_out.clone().cpu().detach().numpy(), seed=0)
        #print('hausdorff distance: ', h_distance)        
        #print(' c_dist:',c_dist.item(), ' p_dist:', p_dist.item(),' triplet_loss:',triplet_loss.item())

        if opt.write_logs:
            #writer.add_scalars("loss/train", {"triplet_loss": triplet_loss, "p_dist": p_dist, "c_dist": c_dist,\
            #                                      "hausdorff_distance": h_distance}, epoch)
            writer.add_scalar("ml_loss/train", ml_loss, epoch)
            writer.add_scalar("g_loss/train", g_loss, epoch)
            #writer.add_scalar("cos_reg/train", cos_reg, epoch)
        
###--------------------visualize if dim 2-3 ----------------------------------------    
        real_embedding = real_embedding.cpu().detach().numpy()   
        fake_embedding = fake_embedding.cpu().detach().numpy()
        
        ML_loss = torch.cat([ML_loss, ml_loss.unsqueeze(0)], dim=0)
        G_loss = torch.cat([G_loss, g_loss.unsqueeze(0)], dim=0)
        if opt.out_dim <= 3:
            epoch_count.append(epoch)
            plot(epoch=epoch,
                 epoch_count = epoch_count,
                 #step_count=step_count,
                 ml_loss=ML_loss.detach().cpu().numpy(),
                 g_loss=G_loss.detach().cpu().numpy(),
                 NUM_EPOCHES= opt.ep,
                 real_embedding=real_embedding[:opt.n_to_show],
                 fake_embedding=fake_embedding[:opt.n_to_show],
                 labels=labels_embedding[:opt.n_to_show],
                 dim3= (opt.out_dim == 3),
                 filename=f'3.{epoch:03d}.png')
###------------------Save generated samples--------------------------------------------------      
        if not opt.write_logs:
            continue
        
        if epoch == 1:
            utils.save_image(for_real_img[:opt.num_samples], str(sample_path  +'/' + 'realimg.png'), \
                             nrow=int(sqrt(opt.num_samples)), padding=2)
        fixed_noise = gen_rand_noise(opt.num_samples, opt.noise_dim).to(DEVICE)        
        gen_images = generate_image(netG, dim=opt.size, channel=1, batch_size=opt.num_samples, noise=fixed_noise)
        utils.save_image(gen_images, str(sample_path  +'/' + 'samples_{}.png').format(epoch), \
                                 nrow=int(sqrt(opt.num_samples)), padding=2)    
        
###--------------------Save Embeddings----------------------------------------        
        if epoch not in eval_ep:
            del real_embedding
            del fake_embedding
            del label_real_img
            del label_fake_img
            torch.cuda.empty_cache()
            continue
        
        print("Real img embeddings size: {}".format(real_embedding.shape))
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
        
        
        n = opt.n_to_show//4
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

