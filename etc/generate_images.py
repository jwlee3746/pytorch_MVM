import argparse
import os

from torch import nn
import torch.utils.data

import torchvision.utils as utils

from data_utils import generate_image, gen_rand_noise, remove_module_str_in_state_dict
from model import Generator28_linear, Generator32, Generator64, Generator128, ML28, ML32, ML64, ML128, \
                    gen64_DCGAN, ML64_DCGAN, Generator512, ML512
        
parser = argparse.ArgumentParser(description='Train Image Generation Models')
# DATA
parser.add_argument('--path', type=str, help='load model dir path')
parser.add_argument('--output_path', type=str, help='output path')
parser.add_argument('--size', type=int, help='images size')
parser.add_argument('--name', type=str, help='name to store results')

parser.add_argument('--noise_dim', default=128, type=int, help='generatior noise vector dim')
parser.add_argument('--num_samples', default=50000, type=int, help='number of displayed samples')

# MODEL NAME
parser.add_argument('--g_model_name', default='Generator64', type=str, help='generator model name')
parser.add_argument('--ml_model_name', default='ML64', type=str, help='metric learning model name')
    
if __name__ == '__main__':
    opt = parser.parse_args()
    
    ### Result path ###   
    if not os.path.exists(opt.path):
        raise Exception("no model path")
    if not os.path.exists(opt.output_path):
        raise Exception("no output path")
    sample_path = opt.output_path + '/{}'.format(opt.name) 
    if not os.path.exists(sample_path):
        os.makedirs(sample_path) 
    
    ### Device ###
    if torch.cuda.is_available():
        DEVICE = torch.device('cuda')
    else:
        DEVICE = torch.device('cpu')
    print('Using PyTorch version:', torch.__version__, ' Device:', DEVICE)
    
    ### Generator ###
    netG = eval(opt.g_model_name+'(nz={}, channel={})'.format(opt.noise_dim, 3))
    print('G MODEL: '+opt.g_model_name)
         
    netG.load_state_dict(remove_module_str_in_state_dict(torch.load(str(opt.path))))
    netG = nn.DataParallel(netG)
    print("LOAD MODEL SUCCESSFULLY")
    
    netG.to(DEVICE)     
          
###--------------------Save images----------------------------------------
    for i in range(opt.num_samples):
        fixed_noise = gen_rand_noise(1, opt.noise_dim) #.to(DEVICE)      
        gen_images = generate_image(netG, dim=opt.size, channel=3, batch_size=1, noise=fixed_noise)
        utils.save_image(gen_images, str(sample_path  +'/' + 'samples_{}.png').format(i)) #, nrow=1, padding=2)  
        
        step = opt.num_samples // 10
        step = step if step != 0 else opt.num_samples
        if i % step == 0 :
            print("Iteration : {}/{}".format(i, opt.num_samples))
    print("SUCCES")            
    