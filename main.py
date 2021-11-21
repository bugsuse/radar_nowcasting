import os
import random
import logging
from tqdm import tqdm
from glob import glob
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from dataset import RadarSets
from models import SmaAt_UNet
from utils import check_dir, get_learning_rate, plot_compare


if __name__ == '__main__':

    LOG_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'
    DATE_FORMAT = '%m/%d/%Y %H:%M:%S'

    data_dir = '/data/ml/train/examples/'
    log_dir = './log'
    ckpt_dir = './ckpts'
    pred_dir = './pred'

    check_dir(log_dir)
    check_dir(ckpt_dir)
    check_dir(pred_dir)

    logging.basicConfig(filename=os.path.join(log_dir, 'smaat_unet.log'),
                        level=logging.INFO,
                        format=LOG_FORMAT,
                        datefmt=DATE_FORMAT
                        )

    logging.info('setting parameters and building model...')
    writer = SummaryWriter(logdir=log_dir)

    ## 常规参数设置
    device = 'cuda:0'
    height = 256
    width = 256
    in_length = 10
    out_length = 10
    bs = 16
    lr = 0.001
    num_epochs = 50
    train_ratio = 0.8
    path = np.array(glob(os.path.join(data_dir, '*')))
    nums = len(path)
    tnums = int(nums*train_ratio)
    values = np.array(range(nums))

    ## 恢复模型训练
    resume = None #'ckpts/epoch_49_valid_4.356657_ckpt.pth'

    ## 训练集和验证集分割
    random.shuffle(values)
    path = path[values]
    train_path = path[:tnums]
    valid_path = path[tnums:]

    train_sets = RadarSets(train_path, (height, width), mode='train')
    valid_sets = RadarSets(valid_path, (height, width), mode='valid')
    train_loader = DataLoader(train_sets, batch_size=bs, num_workers=12,
                              pin_memory=True, shuffle=True, drop_last=True
                              )
    valid_loader = DataLoader(valid_sets, batch_size=bs, num_workers=12,
                              pin_memory=True, shuffle=False, drop_last=True
                             )

    ## 模型，损失函数以及优化器设置
    model = SmaAt_UNet(in_length, out_length).to(device)
    loss_func = lambda pred, obs: (F.l1_loss(pred, obs, reduction='mean') + F.mse_loss(pred, obs, reduction='mean'))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                               factor=0.9, patience=3,
                                               min_lr=0.000001, eps=0.000001, verbose=True
                                               )
    if resume is not None:
        ckpts = torch.load(resume)

        model.load_state_dict(ckpts['model_state_dict'])
        start_epoch = ckpts['epoch']
        epoch_start = start_epoch + 1
        epoch_end = start_epoch + num_epochs
        optimizer.load_state_dict(ckpts['optimizer_state_dict'])
        loss_func.load_state_dict(ckpts['criterion_state_dict'])
    else:
        epoch_start = 0
        epoch_end = num_epochs

    for epoch in range(epoch_start, epoch_end):
        train_loss = 0
        for inputs, targets in tqdm(train_loader):
            torch.multiprocessing.freeze_support()
            optimizer.zero_grad()
            model.train()
            inputs, targets = inputs.cuda(), targets.cuda()
            enc_preds = model(inputs)
            loss = loss_func(enc_preds, targets)

            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        lr_rate = get_learning_rate(optimizer)

        model.eval()
        with torch.no_grad():
            valid_loss = 0
            for inputs, targets in tqdm(valid_loader):
                inputs, targets = inputs.cuda(), targets.cuda()
                enc_preds = model(inputs)
                valid_loss += loss_func(enc_preds, targets).item()

        writer.add_scalar('train_loss', train_loss, epoch)
        writer.add_scalar('valid_loss', valid_loss, epoch)
        logging.info(f'Epoch: {epoch+1}/{num_epochs}, lr: {lr_rate:.6f}, train loss: {train_loss:.6f}, valid loss: {valid_loss:.6f}')

        scheduler.step(valid_loss)
        lr_rate = optimizer.param_groups[0]['lr']

        checkpoint = {'model_state_dict': model.state_dict(),
                      'optimizer_state_dict': optimizer.state_dict(),
                      'criterion_state_dict': loss_func.state_dict(),
                      'epoch': epoch}
        torch.save(checkpoint, f'{ckpt_dir}/epoch_{epoch}_valid_{valid_loss:.6f}_ckpt.pth')
        torch.save(model, f'{ckpt_dir}/epoch_{epoch}_valid_{valid_loss:.6f}_model.pth')
        logging.info(f'save model to {ckpt_dir}/epoch_{epoch}_valid_{valid_loss:.6f}_ckpt.pth')

        idx = np.random.randint(bs-1)
        fig, ax = plot_compare(targets[idx, :].detach().cpu().numpy()*70,
                               np.clip(enc_preds[idx, :].detach().cpu().numpy(), 0, 1)*70,
                               epoch=epoch, save_path=pred_dir)

    logging.info('train successful.')

