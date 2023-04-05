from torchvision.transforms import transforms
#from data_aug.gaussian_blur import GaussianBlur
from torchvision import transforms, datasets
#from exceptions.exceptions import InvalidDatasetSelection

import numpy as np

np.random.seed(0)

class ContrastiveLearningViewGenerator(object):
    """Take two random crops of one image as the query and key."""

    def __init__(self, transform, n_views=2):
        self.transform = transform
        self.n_views = n_views

    def __call__(self, x):
        return [self.transform[0](x) if i == 0 else self.transform[1](x) for i in range(self.n_views)]
    
class ContrastiveLearningDataset:
    def __init__(self, root_folder):
        self.root_folder = root_folder

    @staticmethod
    def get_simclr_pipeline_transform(size, s=1):
        """Return a set of data augmentation transformations as described in the SimCLR paper."""
        #color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
        base_transforms = transforms.Compose([transforms.Resize(size=size),
                                              transforms.ToTensor(),
                                              transforms.Normalize((0.5,), (0.5,)),
                                              ])
        
        data_transforms = transforms.Compose([transforms.Resize(size=size),
                                              #transforms.RandomHorizontalFlip(),
                                              transforms.RandomRotation(10),
                                              #transforms.RandomApply([color_jitter], p=0.8),
                                              #transforms.RandomGrayscale(p=0.2),
                                              #GaussianBlur(kernel_size=int(0.1 * size)),
                                              transforms.ToTensor(),
                                              transforms.Normalize((0.5,), (0.5,)),
                                              ])
        return base_transforms, data_transforms

    def get_dataset(self, name, size, n_views):
        valid_datasets = {'cifar10': lambda: datasets.CIFAR10(self.root_folder, train=True,
                                                              transform=ContrastiveLearningViewGenerator(
                                                                  self.get_simclr_pipeline_transform(32),
                                                                  n_views),
                                                              download=True),

                          'MNIST': lambda: datasets.MNIST(self.root_folder, train=True,
                                                          transform=ContrastiveLearningViewGenerator(
                                                              self.get_simclr_pipeline_transform(size),
                                                              n_views),
                                                          download=True)}

        try:
            dataset_fn = valid_datasets[name]
        except KeyError:
            raise Exception('invalid data')
        else:
            return dataset_fn()