import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
#import tensorflow.keras.layers as layers
#import tensorflow as tf
#from tensorflow.keras.models import Model
#from tensorflow.keras.optimizers import Adam

import os
import datetime
import warnings
warnings.filterwarnings(action='ignore')

from loss import TripletLoss, pairwise_distances, AveragedHausdorffLoss, cosLoss, pairwise_cosLoss


NUM_BATCHES = 30000
BATCH_SIZE = 256
PLOT_EVERY = 100
GRID_RESOLUTION = 400
DATA = 'sup'
MODEL = 'MVM'
COMMENT = 'original'

def requires_grad(model, tf):   
    for param in model.parameters():
        param.requires_grad = tf

def target_function4(Z):
    '''
    Map Z ([-1,1], [-1,1]) to a spiral
    '''
    r = 0.05 + 0.90*(Z[:, 0]+1)/2 + Z[:, 1]*0.05
    theta = 3 * Z[:, 0] * torch.pi
    results = torch.zeros((Z.shape[0], 2))
    results[:, 0] = r * torch.cos(theta)
    results[:, 1] = r * torch.sin(theta)
    return results

def target_function(Z):
    '''
    Map Z ([-1,1], [-1,1]) to a supervised distribution
    '''
    results = torch.zeros((Z.shape[0], 2))
    for idx in range(Z.shape[0]):
        if Z[idx, 0] < -0.5:
            x = -0.5
            y = 0.5
            r = 0.2 * (Z[idx, 1]+1.8)/2
            theta = 4*(Z[idx, 0]+0.75) * torch.pi
            
        elif -0.5 <= Z[idx, 0] < 0:
            x = 0.5
            y = 0.5
            r = 0.2 * (Z[idx, 1]+1.8)/2
            theta = 4*(Z[idx, 0]+0.25) * torch.pi
            
        elif 0 <= Z[idx, 0] < 0.5:
            x = 0.5
            y = -0.5
            r = 0.2 * (Z[idx, 1]+1.8)/2
            theta = 4*(Z[idx, 0]-0.25) * torch.pi
            
        else:
            x = -0.5
            y = -0.5
            r = 0.2 * (Z[idx, 1]+1.8)/2
            theta = 4*(Z[idx, 0]-0.75) * torch.pi
        results[idx, 0] = x + r * torch.cos(theta)
        results[idx, 1] = y + r * torch.sin(theta)
    return results

def target_function1(Z):
    '''
    Map Z ([-1,1], [-1,1]) to a sine
    '''
    r = 0.5
    theta = Z[:, 0] * torch.pi
    results = torch.zeros((Z.shape[0], 2))
    results[:, 0] = Z[:, 0]
    results[:, 1] = r * torch.sin(theta)
    return results

def target_function7(Z):
    '''
    Map Z ([-1,1], [-1,1]) to ([-0.9,0.9], [-0.9,0.9])
    '''
    r = 0.9
    results = torch.zeros((Z.shape[0], 2))
    results[:, 0] = r * Z[:, 0]
    results[:, 1] = r * Z[:, 1]
    return results

def target_function5(Z):
    '''
    Map Z ([-1,1], [-1,1]) to a circleplane
    '''
    r = 0.6 * (Z[:, 1]+1.8)/2
    theta = Z[:, 0] * torch.pi
    results = torch.zeros((Z.shape[0], 2))
    results[:, 0] = r * torch.cos(theta)
    results[:, 1] = r * torch.sin(theta)
    return results

def target_function3(Z):
    '''
    Map Z ([-1,1], [-1,1]) to a circle
    '''
    r = 0.75
    theta = Z[:, 0] * torch.pi
    results = torch.zeros((Z.shape[0], 2))
    results[:, 0] = r * torch.cos(theta)
    results[:, 1] = r * torch.sin(theta)
    return results

def generate_noise(samples):
    '''
    Generate `samples` samples of uniform noise in 
    ([-1,1], [-1,1])
    '''
    return (torch.rand(samples, 2)- 0.5) * 2

def sample_from_target_function(samples):
    '''
    sample from the target function
    '''
    Z = generate_noise(samples)
    return target_function(Z)

def plot(netG, netD, step, step_count, D_loss, G_loss, test_samples, test_noise, filename):
    '''
    Plots for the GAN gif
    '''
    f, ax = plt.subplots(2, 2, figsize=(8,8))
    f.suptitle(f'       {step:05d}', fontsize=10)


    # [0, 0]: plot loss and accuracy
    ax[0, 0].plot(step_count, 
                    G_loss,
                    label='G loss',
                    c='darkred',
                    zorder=50,
                    alpha=0.8,)
    ax[0, 0].plot(step_count, 
                    D_loss,
                    label='D loss',
                    c='darkblue',
                    zorder=55,
                    alpha=0.8,)
    ax[0, 0].set_xlim(-5, NUM_BATCHES+5)
    ax[0, 0].set_ylim(-0.1, 2.1)
    ax[0, 0].legend(loc=1)

    # [0, 1]: Plot actual samples and fake samples
    fake_samples = netG(test_noise)
    np_fake_samples = fake_samples.detach().cpu().numpy()
    ax[0, 1].scatter(test_samples[:, 0].detach().cpu().numpy(), 
                     test_samples[:, 1].detach().cpu().numpy(), 
                     edgecolor='blue', facecolor='None', s=5, alpha=1, 
                     linewidth=1, label='Real')
    ax[0, 1].scatter(np_fake_samples[:, 0], 
                     np_fake_samples[:, 1], 
                     edgecolor='red', facecolor='None', s=5, alpha=1, 
                     linewidth=1, label=MODEL)
    ax[0, 1].set_xlim(-1, 1)
    ax[0, 1].set_ylim(-1, 1)
    ax[0, 1].legend(loc=1)
    ax[0, 1].set_xticks([])
    ax[0, 1].set_yticks([])
    
    del np_fake_samples
    
    # [1, 0]: Plot actual samples and fake samples embeddings
    fake_samples = netG(test_noise)
    real_embedding = netD(test_samples).detach().cpu().numpy()
    fake_embedding = netD(fake_samples).detach().cpu().numpy()
    ax[1, 0].scatter(real_embedding[:, 0], 
                     real_embedding[:, 1], 
                     edgecolor='blue', facecolor='None', s=5, alpha=1, 
                     linewidth=1, label='Real')
    ax[1, 0].scatter(fake_embedding[:, 0], 
                     fake_embedding[:, 1], 
                     edgecolor='red', facecolor='None', s=5, alpha=1, 
                     linewidth=1, label=MODEL)
    ax[1, 0].legend(loc=1)
    
    # [1, 1]: Plot actual samples and fake samples embeddings
    fake_samples = netG(test_noise)
    real_embedding = netD(test_samples).detach().cpu().numpy()
    fake_embedding = netD(fake_samples).detach().cpu().numpy()
    ax[1, 1].scatter(real_embedding[:, 0], 
                     real_embedding[:, 1], 
                     edgecolor='blue', facecolor='None', s=5, alpha=1, 
                     linewidth=1, label='Real')
    ax[1, 1].scatter(fake_embedding[:, 0], 
                     fake_embedding[:, 1], 
                     edgecolor='red', facecolor='None', s=5, alpha=1, 
                     linewidth=1, label=MODEL)
    ax[1, 1].set_xlim(-0.5, 0.5)
    ax[1, 1].set_ylim(-0.5, 0.5)
    ax[1, 1].legend(loc=1)
    
    plt.tight_layout()
    plt.savefig(output_path+'/'+filename)
    plt.close()
    
class Generator(nn.Module):
    '''
    Build a generator mapping (R, R) to ([-1,1], [-1,1])
    '''
    # initializers
    def __init__(self, input_size = 2, out_dim = 2):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(input_size, 512)
        self.fc2 = nn.Linear(self.fc1.out_features, 512)
        self.fc3 = nn.Linear(self.fc2.out_features, 512)
        self.fc4 = nn.Linear(self.fc3.out_features, out_dim)

    # forward method
    def forward(self, input):
        x = F.leaky_relu(self.fc1(input), 0.1)
        x = F.leaky_relu(self.fc2(x), 0.1)
        x = F.leaky_relu(self.fc3(x), 0.1)
        x = torch.tanh(self.fc4(x))

        return x
    
class Discriminator(nn.Module):
    '''
    Build a discriminator mapping ([-1,1], [-1,1]) to ([-1,1], [-1,1])
    '''
    def __init__(self, input_size = 2, out_dim = 2):
        super(Discriminator, self).__init__()
        self.image_to_features = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.LeakyReLU(0.1),
            nn.Linear(512, 512),
            nn.LeakyReLU(0.1),
            nn.Linear(512, 512),
            nn.LeakyReLU(0.1)
        )

        self.features_to_prob = nn.Linear(512, out_dim)


    def forward(self, x):
        x = self.image_to_features(x)
        x = self.features_to_prob(x)
        #x = F.sigmoid(x)
        return x

if __name__ == '__main__':
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
    
    with torch.no_grad():
        grid = torch.zeros((GRID_RESOLUTION, GRID_RESOLUTION, 2)).to(DEVICE)
        grid[:, :, 0] = torch.linspace(-1, 1, GRID_RESOLUTION).reshape((1, -1))
        grid[:, :, 1] = torch.linspace(1, -1, GRID_RESOLUTION).reshape((-1, 1))
        flat_grid = grid.reshape((-1, 2))
    
    
    ## Set up directories
    paths = ['ims', 
                'ims/1', 
                    'ims/1/a', 'ims/1/b', 
                'ims/2', 
                    'ims/2/a', 'ims/2/b', 'ims/2/c',
                'ims/3',
                    'ims/3/a', 'ims/3/b'
            ]
    now = datetime.datetime.now().strftime("%m%d_%H%M")
    output_path = str('results' + '/' + '{}_{}_{}_{}_{}').format(now, MODEL, DATA, NUM_BATCHES, COMMENT)
    
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        
    for i in paths:
        if not os.path.exists(output_path+'/'+i):
            os.makedirs(output_path+'/'+i)
    
    
    f, ax = plt.subplots(1, 2, figsize=(12,6))
    with torch.no_grad():
        input_points = generate_noise(2000).to(DEVICE) 
        output_points = target_function(input_points).to(DEVICE) 
    
    np_input_points = generate_noise(2000).detach().cpu().numpy()
    np_output_points = output_points.detach().cpu().numpy()
    
    del input_points
    del output_points
    del np_input_points
    del np_output_points
    torch.cuda.empty_cache()
    
    
    
    ## Part 3: GAN
    with torch.no_grad():
        test_samples = sample_from_target_function(2000).to(DEVICE)
        test_noise = generate_noise(2000).to(DEVICE)
    
    netG = Generator().to(DEVICE)  
    optimizerG = optim.Adam(netG.parameters(), lr = 0.0002, betas=(0.5,0.999)) 
    netD = Discriminator().to(DEVICE) 
    optimizerD = optim.Adam(netD.parameters(), lr = 0.00002, betas=(0.5,0.999)) 
    
    triplet_ = TripletLoss(1, 1e-2).to(DEVICE)    
    Cos_ = cosLoss().to(DEVICE)
    d_hausdorff_ = AveragedHausdorffLoss(directed = True).to(DEVICE)
    a_hausdorff_ = AveragedHausdorffLoss(directed = False).to(DEVICE)
    pCos_ = pairwise_cosLoss().to(DEVICE)
    
    step_count = []
    with torch.no_grad():
        D_loss = torch.tensor([]).to(DEVICE)
        G_loss = torch.tensor([]).to(DEVICE)
    count = 1
    
    netD.train()
    netG.train()
    
    temp = 0
    for step in range(1,NUM_BATCHES+1):
        print(f'3a: {step:03d}/{NUM_BATCHES}', end='\r')
        
        ### Train discriminator ###
        requires_grad(netG, False)
        requires_grad(netD, True) 
        
        real_data = sample_from_target_function(BATCH_SIZE // 2).to(DEVICE)
        fake_data = netG(generate_noise(BATCH_SIZE // 2).to(DEVICE))
        #data = torch.cat((real_data, fake_data), axis=0)
        
        #real_labels = torch.ones((BATCH_SIZE // 2, 1)).to(DEVICE)
        #fake_labels = torch.zeros((BATCH_SIZE // 2, 1)).to(DEVICE)
        #labels = torch.cat((real_labels, fake_labels), axis=0)
        
        #prob = netD(data)
        #d_loss = bce_(prob, labels)
        ml_real_out = netD(real_data)
        ml_fake_out = netD(fake_data)
        
        r1=torch.randperm(BATCH_SIZE // 2)
        ml_real_out_shuffle = ml_real_out[r1[:, None]].view(ml_real_out.shape[0],ml_real_out.shape[-1])
        
        #d_loss = Cos_(ml_real_out,ml_real_out_shuffle,ml_fake_out)
        d_loss = triplet_(ml_real_out,ml_real_out_shuffle,ml_fake_out)
        #d_loss = pCos_(ml_real_out, ml_fake_out)
        
        #if step == 1: temp = d_loss
        #if temp + 0.1 >= d_loss:
        #    temp = d_loss   
        
        #if a_hausdorff_(ml_real_out, ml_fake_out) < 0.02:
        #    print("d optimizing...")
        netD.zero_grad()
        d_loss.backward()
        optimizerD.step()
        #else:
        #    print("collape!")
    
        ### Train generator ### 
        requires_grad(netG, True)
        requires_grad(netD, False) 
        
        real_data = sample_from_target_function(BATCH_SIZE // 2).to(DEVICE)
        fake_data = netG(generate_noise(BATCH_SIZE // 2).to(DEVICE))
        ml_real_out = netD(real_data)
        ml_fake_out = netD(fake_data)
        
        #labels = torch.ones((BATCH_SIZE // 2, 1)).to(DEVICE)
        
        #prob = netD(fake_data)
        #g_loss = bce_(prob, labels)
        
        pd_r = pairwise_distances(ml_real_out, ml_real_out) 
        pd_f = pairwise_distances(ml_fake_out, ml_fake_out) 
        
        p_dist =  torch.dist(pd_r,pd_f,2)         
        c_dist = torch.dist(ml_real_out.mean(0),ml_fake_out.mean(0),2) 
            
        g_loss = p_dist + c_dist
        
        #g_loss = a_hausdorff_(ml_real_out, ml_fake_out)
        
        netG.zero_grad()
        g_loss.backward()
        optimizerG.step() 
        
        if (step % PLOT_EVERY == 0) or (step <= 100):
            step_count.append(step)
            if step % (PLOT_EVERY * 5) == 0:
                print()
                print("d_loss: {}".format(d_loss))
                print("g_loss: {}".format(g_loss))
            D_loss = torch.cat([D_loss, d_loss.unsqueeze(0)], dim=0)
            G_loss = torch.cat([G_loss, g_loss.unsqueeze(0)], dim=0)
            plot(netG=netG, 
                 netD=netD,
                 step=step,
                 step_count=step_count,
                 D_loss=D_loss.detach().cpu().numpy(),
                 G_loss=G_loss.detach().cpu().numpy(),
                 test_samples=test_samples,
                 test_noise=test_noise,
                 filename=f'ims/3/a/3.{count:03d}.png')
            count += 1
    print("Saving models..")
    #os.system(f'ffmpeg -r 20 -i ims/3/3.%03d.png'
    #              f' -crf 15 ims/3/3.mp4')
    torch.save(netG.state_dict(), str(output_path+'/'+"ims/3/G_latest.pt"))
    torch.save(netD.state_dict(), str(output_path+'/'+"ims/3/D_latest.pt"))
    print()
    
    del real_data
    del fake_data
    #del data
    #del real_labels
    #del fake_labels
    #del labels
    del d_loss
    del g_loss
    del D_loss
    del G_loss
    torch.cuda.empty_cache()
    
    
    ### 3b
    
    steps = 100
    
    with torch.no_grad():
        input_points = generate_noise(2000).to(DEVICE) 
    
    np_input_points = generate_noise(2000).detach().cpu().numpy()
    
    generated_points = netG(input_points).detach().cpu().numpy()
    space = np.linspace(np_input_points, generated_points, steps)
    c_space = np.linspace(0, 255, steps)
    for step in range(steps):
        c = f'#0000{int(c_space[step]):02x}'
        print(f'3b: {step:03d}/{steps}', end='\r')
        plt.figure(figsize=(6,6))
        plt.scatter(space[step, :, 0], space[step, :, 1], edgecolor=c, facecolor='None', s=5, alpha=1, linewidth=1)
        plt.xlim(-1, 1)
        plt.ylim(-1, 1)
        plt.gca().set_aspect(1)
        plt.tight_layout()
        plt.savefig(output_path+'/'+f'ims/3/b/3b.{step:03d}.png')
        plt.close()
    print()
    
    del generated_points
    del input_points
    del np_input_points
    #os.system(f'ffmpeg -r 20 -i ims/1/b/1b.%03d.png'
    #              f' -crf 15 ims/1/b/1b.mp4')
    