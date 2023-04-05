import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

from torchmetrics.functional import pairwise_cosine_similarity

def pairwise_distances(x, y):
    """
    Input: x is a Nxd Tensor
           y is a Mxd Tensor
    Output: dist is a NxM matrix where dist[i,j] is the norm
           between x[i,:] and y[j,:]
    i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
    """
    differences = x.unsqueeze(1) - y.unsqueeze(0)
    distances = torch.sum(differences**2, -1)
    return distances

def cdist(x, y):
    '''
    Input: x is a Nxd Tensor
           y is a Mxd Tensor
    Output: dist is a NxM matrix where dist[i,j] is the norm
           between x[i,:] and y[j,:]
    i.e. dist[i,j] = ||x[i,:]-y[j,:]||
    '''
    differences = x.unsqueeze(1) - y.unsqueeze(0) + 1e-8
    distances = torch.sum(differences**2, -1).sqrt()
    return distances

class TripletLoss(nn.Module):
    """
    Triplet loss
    Takes embeddings of an anchor sample, a positive sample and a negative sample
    """

    def __init__(self, margin, alpha):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.alpha = alpha

    def forward(self, anchor, positive, negative, size_average=True):
        distance_positive = (anchor - positive).pow(2).sum(1)  # .pow(.5)
        distance_negative = (anchor - negative).pow(2).sum(1)  # .pow(.5)
        
        cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        cos_reg = cos(negative - anchor, positive - anchor).sum(0) 
        #print((distance_positive-distance_negative).mean())
        losses = F.relu(distance_positive - distance_negative + self.margin - self.alpha * cos_reg) #2e-2
        
        #d2_matrix = pairwise_distances(anchor, positive)
        #distance_positive = d2_matrix.diag().unsqueeze(1)
        
        #triplet = distance_positive - d2_matrix + 10
        #losses = F.relu(triplet - triplet.diag().diagflat())# - self.alpha * cos_reg) #2e-2
        
        return losses.mean()

    
class cosLoss(nn.Module):
    """
    Cos loss
    Takes embeddings of an anchor sample, a positive sample and a negative sample
    """

    def __init__(self):
        super(cosLoss, self).__init__()
        
    def forward(self, anchor, positive, negative, size_average=True):        
        cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        cos_reg = cos(negative - anchor, positive - anchor)
        
        cos_reg = cos_reg.mean()
        
        losses = F.relu(1 - cos_reg) #2e-2

        return losses.mean()

class pairwise_cosLoss(nn.Module):
    """
    Cos loss
    Takes embeddings of an anchor sample, a positive sample and a negative sample
    """

    def __init__(self):
        super(pairwise_cosLoss, self).__init__()
        
    def forward(self, anchor, negative, size_average=True):
        
        assert anchor.ndimension() == 2, 'got %s' % anchor.ndimension()
        assert negative.ndimension() == 2, 'got %s' % negative.ndimension()

        assert anchor.size()[1] == negative.size()[1], \
            'The points in both sets must have the same number of dimensions, got %s and %s.'\
            % (anchor.size()[1], negative.size()[1])

        #positive_matrix = anchor.unsqueeze(1) - anchor.unsqueeze(0)
        #negative_matrix = negative.unsqueeze(1) - anchor.unsqueeze(0)
        
        #cos = nn.CosineSimilarity(dim=2, eps=1e-6)
        #cos_reg = cos(negative_matrix, positive_matrix)
        
        positive_matrix = cdist(anchor, anchor)
        negative_matrix = cdist(anchor, negative)
        
        cos_reg = pairwise_cosine_similarity(negative_matrix, positive_matrix)
               
        cos_reg = cos_reg.mean()
        
        losses = F.relu(1 - cos_reg) #2e-2

        return losses.mean()

    
##################################### Hausdorff dist  ###################################### 
def averaged_hausdorff_distance(set1, set2, max_ahd=np.inf):
    """
    Compute the Averaged Hausdorff Distance function
     between two unordered sets of points (the function is symmetric).
     Batches are not supported, so squeeze your inputs first!
    :param set1: Array/list where each row/element is an N-dimensional point.
    :param set2: Array/list where each row/element is an N-dimensional point.
    :param max_ahd: Maximum AHD possible to return if any set is empty. Default: inf.
    :return: The Averaged Hausdorff Distance between set1 and set2.
    """

    if len(set1) == 0 or len(set2) == 0:
        return max_ahd

    assert set1.ndim == 2, 'got %s' % set1.ndim
    assert set2.ndim == 2, 'got %s' % set2.ndim

    assert set1.shape[1] == set2.shape[1], \
        'The points in both sets must have the same number of dimensions, got %s and %s.'\
        % (set2.shape[1], set2.shape[1])

    d2_matrix = cdist(set1, set2)

    term_1 = torch.mean(torch.min(d2_matrix, 1)[0])
    term_2 = torch.mean(torch.min(d2_matrix, 0)[0])
    
    res = term_1 + term_2

    return res


class AveragedHausdorffLoss(nn.Module):
    def __init__(self, directed = False):
        super(AveragedHausdorffLoss, self).__init__()
        self.directed = directed

    def forward(self, set1, set2):
        """
        Compute the Averaged Hausdorff Distance function
         between two unordered sets of points (the function is symmetric).
         Batches are not supported, so squeeze your inputs first!
        :param set1: Tensor where each row is an N-dimensional point.
        :param set2: Tensor where each row is an N-dimensional point.
        :return: The Averaged Hausdorff Distance between set1 and set2.
        """

        assert set1.ndimension() == 2, 'got %s' % set1.ndimension()
        assert set2.ndimension() == 2, 'got %s' % set2.ndimension()

        assert set1.size()[1] == set2.size()[1], \
            'The points in both sets must have the same number of dimensions, got %s and %s.'\
            % (set2.size()[1], set2.size()[1])

        d2_matrix = cdist(set1, set2)
        #d2_matrix = pairwise_distances(set1, set2)
        
        # Modified Chamfer Loss
        term_1 = torch.mean(torch.min(d2_matrix, 1)[0])
        term_2 = torch.mean(torch.min(d2_matrix, 0)[0])

        if self.directed:
            res = term_1            
        else:
            res = term_1 + term_2

        return res
    