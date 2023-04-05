from glob import glob
from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision.transforms import Compose, RandomCrop, ToTensor, ToPILImage, CenterCrop, Resize

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])

def train_transform(size):
    transform = None
    transform = Compose([
        Resize((size, size), interpolation=Image.BICUBIC),
        ToTensor()
    ])
    return transform

class TrainDatasetFromFolder(Dataset):
    def __init__(self, dataset_dir, size, file_format="png", mode="RGB"):
        super(TrainDatasetFromFolder, self).__init__()
        self.image_filenames = glob(dataset_dir + '/**/*.{}'.format(file_format), recursive=True)
        self.resize_transform = train_transform(size)
        self.mode = mode

    def __getitem__(self, index):
        image = Image.open(self.image_filenames[index]).convert(self.mode) # convert RGB if is color image
        real_image = self.resize_transform(image)
        #real_image = real_image * 2 - 1 # if use tanh
        return real_image, -1

    def __len__(self):
        return len(self.image_filenames)


trainset = TrainDatasetFromFolder('/mnt/prj/Data/star-MNIST', # 데이터 저장 경로
                                  size=64, # 이미지 크기
                                  file_format='png',
                                  mode='L'
                                  )