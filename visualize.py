import os
import numpy as np
import argparse

import torch

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


parser = argparse.ArgumentParser(description='Visualize manifold space')
parser.add_argument('--output_path', type=str, help='result path')

parser.add_argument('--epoch', default=1, type=int, help='epoch num')
parser.add_argument('--real_or_fake', default = 'both', type=str, choices=['both', 'real', 'fake'])
parser.add_argument('--n_to_show', default=1000, type=int, help='ML network output dim')
parser.add_argument('--mode', default = 'PCA', type=str, choices=['PCA', 'real'])
parser.add_argument('--download', default = 'no', type=str, choices=['yes', 'no'])

if __name__ == '__main__':
    opt = parser.parse_args()
    
    output_path = opt.output_path
    epoch = opt.epoch
    real_or_fake = opt.real_or_fake
    n_to_show = opt.n_to_show
    DOWNLOAD = opt.download
    MODE = opt.mode

    base_dir = "/mnt/prj/jaewon/metric_learning_pytorch/pytorch-manifold-matching/results"
    output_path = base_dir + "/" + output_path
    pca_path = output_path + '/PCA'
    if not os.path.exists(pca_path):
        os.makedirs(pca_path)    
    
    if real_or_fake == "real":
        embeddings = np.load(str(output_path + '/embeddings/' + 'real_embdding_{}.npy').format(epoch))[:n_to_show]
        #real_embdding = torch.from_numpy(embeddings).to('cuda')
        print("Load {} of real embeddings".format(len(embeddings)))
        labels = np.zeros(n_to_show)
    elif real_or_fake == "fake":
        embeddings = np.load(str(output_path + '/embeddings/' + 'fake_embdding_{}.npy').format(epoch))[:n_to_show]
        #fake_embdding = torch.from_numpy(embeddings).to('cuda')
        print("Load {} of fake embeddings".format(len(embeddings)))
        labels = np.ones(n_to_show)
    else:
        np_real_embdding = np.load(str(output_path + '/embeddings/' + 'real_embdding_{}.npy').format(epoch))
        #real_embdding = torch.from_numpy(np_real_embdding).to('cuda')
        
        np_fake_embdding = np.load(str(output_path + '/embeddings/' + 'fake_embdding_{}.npy').format(epoch))
        #fake_embdding = torch.from_numpy(np_fake_embdding).to('cuda')
        
        if len(np_real_embdding) != len(np_fake_embdding):
            raise Exception("len np_real_embdding != np_fake_embdding")
        if len(np_real_embdding) < n_to_show:
            raise Exception("Too large n_to_show ! input lower than {}".format(len(np_real_embdding)))
        embeddings = np.concatenate((np_real_embdding[:n_to_show], np_fake_embdding[:n_to_show]))
        print("Load total {} of embeddings".format(len(embeddings)))
        labels = np.concatenate((np.zeros(n_to_show), np.ones(n_to_show)))
    
    if n_to_show <= 3000: alpha2D, s2D, alpha3D, s3D =0.7, 1, 0.3, 1 
    else: alpha2D, s2D, alpha3D, s3D =0.7, 1, 0.1, 1 
    
    if MODE == 'PCA':
        ### 2d PCA ###
        reducer = PCA(n_components=2)
        ml_out_reduced = reducer.fit_transform(embeddings)
        if real_or_fake == "real":
            r = plt.scatter(
                ml_out_reduced[:n_to_show, 0], ml_out_reduced[:n_to_show, 1],
                c="dodgerblue", alpha=alpha2D, s=s2D)
        elif real_or_fake == "fake":
            f = plt.scatter(
                ml_out_reduced[:n_to_show, 0], ml_out_reduced[:n_to_show, 1],
                c="red", alpha=alpha2D, s=s2D)
        else:
            r = plt.scatter(
                ml_out_reduced[:n_to_show, 0], ml_out_reduced[:n_to_show, 1],
                c="dodgerblue", alpha=alpha2D, s=s2D)
            f = plt.scatter(
                ml_out_reduced[n_to_show:, 0], ml_out_reduced[n_to_show:, 1],
                c="red", alpha=alpha2D, s=s2D)
            plt.legend((r, f),
                    ('Real', 'Fake'),
                   scatterpoints=1,
                   loc='lower left',
                   ncol=3,
                   fontsize=8)
        plt.gca().set_aspect('equal', 'datalim')
        plt.title('PCA 2D of {} data[epoch#{}]'.format(n_to_show, epoch), fontsize=18)
        if DOWNLOAD:
            plt.savefig(str(pca_path + '/' + '2DPCA_n{}_e{}_{}.pdf').format(n_to_show, epoch, real_or_fake), dpi=200)
        plt.show()
        ### 2d PCA ###
        
        ### 3d PCA###   
        reducer = PCA(n_components=3)
        ml_out_reduced = reducer.fit_transform(embeddings)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        
        if real_or_fake == "real":
            r = ax.scatter(
                ml_out_reduced[:n_to_show, 0], ml_out_reduced[:n_to_show, 1], ml_out_reduced[:n_to_show, 2],
                c='dodgerblue', alpha=alpha3D, s=s3D) 
        elif real_or_fake == "fake":
            f = ax.scatter(
                ml_out_reduced[:n_to_show, 0], ml_out_reduced[:n_to_show, 1], ml_out_reduced[:n_to_show, 2],
                c='red', alpha=alpha3D, s=s3D) 
        else:
            f = ax.scatter(
                ml_out_reduced[n_to_show:, 0], ml_out_reduced[n_to_show:, 1], ml_out_reduced[n_to_show:, 2],
                c='red', alpha=alpha3D, s=s3D) 
            r = ax.scatter(
                ml_out_reduced[:n_to_show, 0], ml_out_reduced[:n_to_show, 1], ml_out_reduced[:n_to_show, 2],
                c='dodgerblue', alpha=alpha3D, s=s3D) 
            plt.legend((r, f),
                    ('Real', 'Fake'),
                   numpoints=1,
                   loc='upper left',
                   ncol=3,
                   fontsize=8,
                   bbox_to_anchor=(0, 0))
    
        plt.gca().set_aspect('auto', 'datalim')
        plt.title('PCA 3D of {} data[epoch#{}]'.format(n_to_show, epoch), fontsize=18)
        if DOWNLOAD:
            plt.savefig(str(pca_path + '/' + '3D_n{}_e{}_{}.pdf').format(n_to_show, epoch, real_or_fake), dpi=200)
        plt.show()
        ### 3d PCA###
        
    if MODE == 'real':
        ### [x, y]components of embeddings ###   
        if real_or_fake == "real":
            r = plt.scatter(
                embeddings[:n_to_show, 0], embeddings[:n_to_show, 1],
                c="dodgerblue", alpha=alpha2D, s=s2D)
        elif real_or_fake == "fake":
            f = plt.scatter(
                embeddings[:n_to_show, 0], embeddings[:n_to_show, 1],
                c="red", alpha=alpha2D, s=s2D)
        else:
            r = plt.scatter(
                embeddings[:n_to_show, 0], embeddings[:n_to_show, 1],
                c="dodgerblue", alpha=alpha2D, s=s2D)
            f = plt.scatter(
                embeddings[n_to_show:, 0], embeddings[n_to_show:, 1],
                c="red", alpha=alpha2D, s=s2D)
            plt.legend((r, f),
                    ('Real', 'Fake'),
                   scatterpoints=1,
                   loc='lower left',
                   ncol=3,
                   fontsize=8)
        plt.gca().set_aspect('equal', 'datalim')
        plt.title('X,Y of {} data[epoch#{}]'.format(n_to_show, epoch), fontsize=18)
        if DOWNLOAD:
            plt.savefig(str(pca_path + '/' + 'XY_n{}_e{}_{}.pdf').format(n_to_show, epoch, real_or_fake), dpi=200)
        plt.show()
        ### [x, y]components of embeddings ###   

        ### [x, y, z]components of embeddings ###   
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        
        if real_or_fake == "real":
            r = ax.scatter(
                embeddings[:n_to_show, 0], embeddings[:n_to_show, 1], embeddings[:n_to_show, 2],
                c='dodgerblue', alpha=alpha3D, s=s3D) 
        elif real_or_fake == "fake":
            f = ax.scatter(
                embeddings[:n_to_show, 0], embeddings[:n_to_show, 1], embeddings[:n_to_show, 2],
                c='red', alpha=alpha3D, s=s3D) 
        else:
            r = ax.scatter(
                embeddings[:n_to_show, 0], embeddings[:n_to_show, 1], embeddings[:n_to_show, 2],
                c='dodgerblue', alpha=alpha3D, s=s3D) 
            f = ax.scatter(
                embeddings[n_to_show:, 0], embeddings[n_to_show:, 1], embeddings[n_to_show:, 2],
                c='red', alpha=alpha3D, s=s3D) 
            plt.legend((r, f),
                    ('Real', 'Fake'),
                   numpoints=1,
                   loc='upper left',
                   ncol=3,
                   fontsize=8,
                   bbox_to_anchor=(0, 0))
    
        plt.gca().set_aspect('auto', 'datalim')
        plt.title('X,Y,Z of {} data[epoch#{}]'.format(n_to_show, epoch), fontsize=18)
        if DOWNLOAD:
            plt.savefig(str(pca_path + '/' + 'XYZ_n{}_e{}_{}.pdf').format(n_to_show, epoch, real_or_fake), dpi=200)
        plt.show()
        ### [x, y, z]components of embeddings ###   
