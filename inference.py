import os
import sys
import random
import numpy as np
from glob import glob
import torch
from torch.utils.data import DataLoader

from dataset import RadarSets
from models import SmaAt_UNet
from utils import check_dir, save_pred, array2img


if __name__ == '__main__':

    ckpts = 'ckpts/epoch_49_valid_4.356657_ckpt.pth'
    save_dir = './infer'
    device = 'cuda:0'
    test_path = '/data/ml/test/examples/'
    img_height = 256
    img_width = 256
    in_length = 10
    out_length = 10
    batch_size = 16
    num_workers = 12
    pin_memory = True
    matplot = True

    visual = {'norm_max': 80, 'cmap': 'NWSRef'}

    check_dir(save_dir)

    print('load dataset to inference...')
    # load dataset
    test_path = np.array(glob(os.path.join(test_path, '*')))
    test_sets = RadarSets(test_path, (img_height, img_width), mode='test')
    test_loader = DataLoader(test_sets,
                             batch_size=batch_size,
                             num_workers=num_workers,
                             pin_memory=pin_memory,
                             )

## set loss function
    loss_fx = None

    ckpts = torch.load(ckpts)['model_state_dict']
    model = SmaAt_UNet(in_length, out_length).to(device)
    model.load_state_dict(ckpts)
    model.eval().to(device)

    with torch.no_grad():
        num = 0
        for i, input_ in enumerate(test_loader):
            input_ = input_.to(device)
            input_ = input_.reshape(batch_size, -1, img_height, img_width)
            pred = model(input_)

            print(f'save prediction to {save_dir}...')
            for b in range(pred.shape[0]):
                for t in range(pred.shape[1]):
                    check_dir(save_dir + os.sep + f'{num+1:05d}')
                    save_path = save_dir + f'{os.sep}{num+1:05d}{os.sep}'

                    if matplot:
                        print(pred[b, :].detach().cpu().numpy().max())
                        save_pred(pred[b, :].detach().cpu().numpy() * visual['norm_max'], save_path)
                    else:
                        img = array2img(pred[b, t, 0].detach().cpu().numpy() * visual['norm_max'], visual['cmap'])
                        img.save(save_path + f'{t:02d}.png')
                num += 1

