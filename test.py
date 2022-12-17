"""
## Multi-Stage Progressive Image Restoration
## Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, Ming-Hsuan Yang, and Ling Shao
## https://arxiv.org/abs/2102.02808
"""

import numpy as np
import os
import argparse
from tqdm import tqdm

import torch.nn as nn
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import utils

from data_RGB import get_test_data
from TIDNet import TIDNet
from skimage import img_as_ubyte
from pdb import set_trace as stx

parser = argparse.ArgumentParser(description='Image Deraining using MPRNet')

parser.add_argument('--input_dir', default='./Datasets/test/', type=str, help='Directory of validation images')
parser.add_argument('--result_dir', default='./results/', type=str, help='Directory for results')
parser.add_argument('--weights', default='./pretrained_models/model_pretrained.pth', type=str, help='Path to weights')
parser.add_argument('--gpus', default='0, 1', type=str, help='CUDA_VISIBLE_DEVICES')

args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

model_restoration = TIDNet()

utils.load_checkpoint(model_restoration,args.weights)
print("===>Testing using weights: ",args.weights)
model_restoration.cuda()
model_restoration = nn.DataParallel(model_restoration)
model_restoration.eval()


if __name__ == '__main__':
    rgb_dir_test = os.path.join(args.input_dir, 'input')
    test_dataset = get_test_data(rgb_dir_test, img_options={})
    test_loader  = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=4, drop_last=False, pin_memory=True)

    result_dir  = args.result_dir
    utils.mkdir(result_dir)

    with torch.no_grad():
        for ii, data_test in enumerate(tqdm(test_loader), 0):
            torch.cuda.ipc_collect()
            torch.cuda.empty_cache()

            input_ = data_test[0].cuda()
            filenames = data_test[1]

            # input_ = F.interpolate(input_, scale_factor=2)

            H, W = input_.size(2), input_.size(3)
            size = 4
            new_H, new_W = (H + size - 1) // size * size, (W + size - 1) // size * size
            pad_H, pad_W = new_H - H, new_W - W
            input_ = F.pad(input_, (0,pad_W,0,pad_H), mode='reflect')

            restored = model_restoration(input_)[0][0]

            if pad_H != 0:
                restored = restored[:, :, :-pad_H, :]
            if pad_W != 0:
                restored = restored[:, :, :, :-pad_W]
            # restored = model_restoration(input_)
            # restored = F.interpolate(restored, scale_factor=0.5)
            restored = torch.clamp(restored,0,1)

            restored = restored.permute(0, 2, 3, 1).cpu().detach().numpy()

            for batch in range(len(restored)):
                restored_img = img_as_ubyte(restored[batch])
                utils.save_img((os.path.join(result_dir, filenames[batch]+'.png')), restored_img)
