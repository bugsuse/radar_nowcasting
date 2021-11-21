import os
from glob import glob
import numpy as np
from PIL import Image
import torch
from torch.utils.data.dataset import Dataset


class RadarSets(Dataset):
    def __init__(self, pics, img_size, mode='train'):
        super(RadarSets, self).__init__()
        self.pics = pics
        self.mode = mode
        self.height, self.width = img_size
        self.train_path = f'{os.sep}'.join([i for i in pics[0].split(os.sep)[:-2]])

    def __getitem__(self, index):
        if self.mode not in ['test']:
            mode = 'train'
        else:
            mode = 'test'

        inputs = []
        input_fn = os.path.join(self.train_path, 'examples',
                                f"{self.pics[index].split('/')[-1]}",
                                f"{self.pics[index].split('/')[-1]}-inputs-{mode}.txt")
        input_img = np.loadtxt(input_fn, dtype=str)
        inp_len = len(input_img)

        for i in range(0, inp_len):
            img = Image.open(os.path.join(self.train_path, 'data', input_img[i]))
            img = np.pad(img, ((10, 10), (10, 10)), 'constant', constant_values = (0, 0))
            img = np.array(Image.fromarray(img).resize((self.height, self.width)))
            img = torch.from_numpy(img.astype(np.float32))
            inputs.append(img)

        if self.mode in ['train', 'valid']:
            target_fn = os.path.join(self.train_path, 'examples',
                                            f"{self.pics[index].split('/')[-1]}",
                                            f"{self.pics[index].split('/')[-1]}-targets-train.txt")
            target_img = np.loadtxt(target_fn, dtype=str)
            tar_len = len(target_img)

            targets = []
            for i in range(0, tar_len):
                img = Image.open(os.path.join(self.train_path, 'data', target_img[i]))
                img = np.pad(img, ((10, 10), (10, 10)), 'constant', constant_values = (0, 0))
                img = np.array(Image.fromarray(img).resize((self.height, self.width)))
                img = torch.from_numpy(img.astype(np.float32))
                targets.append(img)

            return torch.stack(inputs, dim=0)/255, torch.stack(targets, dim=0)/255
        elif self.mode in ['test']:
            return torch.stack(inputs, dim=0)/255
        else:
            raise ValueError(f'{self.mode} is unknown and should be among train, valid and test!')

    def __len__(self):
        return len(self.pics)

