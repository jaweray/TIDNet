import os
from config import Config 
opt = Config('training.yml')

gpus = ','.join([str(i) for i in opt.GPU])
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = gpus

import torch
torch.backends.cudnn.benchmark = True

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

import random
import time
import numpy as np

import utils
from data_RGB import get_training_data, get_validation_data
from TIDNet import TIDNet
import losses
from tqdm import tqdm
from skimage import img_as_ubyte
from pdb import set_trace as stx

import lib.models.crnn as crnn
import yaml
from easydict import EasyDict as edict
import lib.config.alphabets as alphabets
from lib.utils.utils import get_batch_label

if __name__ == '__main__':
    ######### Set Seeds ###########
    random.seed(1234)
    np.random.seed(1234)
    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)

    start_epoch = 1
    mode = opt.MODEL.MODE
    session = opt.MODEL.SESSION

    channel_dir = os.path.join(opt.TRAINING.SAVE_DIR, mode, 'channels', session)
    result_dir = os.path.join(opt.TRAINING.SAVE_DIR, mode, 'results', session)
    model_dir  = os.path.join(opt.TRAINING.SAVE_DIR, mode, 'models',  session)

    utils.mkdir(channel_dir)
    utils.mkdir(result_dir)
    utils.mkdir(model_dir)

    train_dir = opt.TRAINING.TRAIN_DIR
    val_dir   = opt.TRAINING.VAL_DIR

    ######### Model ###########
    model_restoration = TIDNet()
    model_restoration.cuda()

    device_ids = [i for i in range(torch.cuda.device_count())]
    if torch.cuda.device_count() > 1:
      print("\n\nLet's use", torch.cuda.device_count(), "GPUs!\n\n")


    new_lr = opt.OPTIM.LR_INITIAL

    optimizer = optim.Adam(model_restoration.parameters(), lr=new_lr, betas=(0.9, 0.999),eps=1e-8)


    ######### Scheduler ###########
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, opt.OPTIM.NUM_EPOCHS, eta_min=opt.OPTIM.LR_MIN)

    ######### Resume ###########
    if opt.TRAINING.RESUME:
        path_chk_rest    = utils.get_last_path(model_dir, '_latest.pth')
        utils.load_checkpoint(model_restoration,path_chk_rest)
        start_epoch = utils.load_start_epoch(path_chk_rest) + 1
        utils.load_optim(optimizer, path_chk_rest)

        for i in range(1, start_epoch):
            scheduler.step()
        new_lr = scheduler.get_lr()[0]
        print('------------------------------------------------------------------------------')
        print("==> Resuming Training with learning rate:", new_lr)
        print('------------------------------------------------------------------------------')

    if len(device_ids)>1:
        model_restoration = nn.DataParallel(model_restoration, device_ids = device_ids)

    #############################################
    ########### OCR MODEL INIT ##################
    #############################################
    with open('lib/config/360CC_config.yaml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        config = edict(config)

    config.DATASET.ALPHABETS = alphabets.alphabet
    config.MODEL.NUM_CLASSES = len(config.DATASET.ALPHABETS)

    ocr_model = crnn.get_crnn(config).cuda()
    checkpoint = torch.load('output/checkpoints/mixed_second_finetune_acc_97P7.pth')
    if 'state_dict' in checkpoint.keys():
        ocr_model.load_state_dict(checkpoint['state_dict'])
    else:
        ocr_model.load_state_dict(checkpoint)

    ocr_model = nn.DataParallel(ocr_model, device_ids=device_ids)
    for param in ocr_model.parameters():
        param.requires_grad = False
    ocr_model.eval()
    ###########################################################
    ###########################################################
    ###########################################################

    ######### Loss ###########
    criterion_char = losses.CharbonnierLoss()
    criterion_edge = losses.EdgeLoss()
    criterion_binary = losses.BinaryLoss()
    criterion_mask = losses.MaskLoss()
    criterion_ocr = losses.OcrLoss(ocr_model, config.DATASET.ALPHABETS)

    ######### DataLoaders ###########
    train_dataset = get_training_data(train_dir, {'patch_size':opt.TRAINING.TRAIN_PS})
    train_loader = DataLoader(dataset=train_dataset, batch_size=opt.OPTIM.BATCH_SIZE, shuffle=True, num_workers=8, drop_last=False, pin_memory=True)

    val_dataset = get_validation_data(val_dir, {'patch_size':opt.TRAINING.VAL_PS})
    val_loader = DataLoader(dataset=val_dataset, batch_size=4, shuffle=False, num_workers=8, drop_last=False, pin_memory=True)

    print('===> Start Epoch {} End Epoch {}'.format(start_epoch,opt.OPTIM.NUM_EPOCHS + 1))
    print('===> Loading datasets')

    best_psnr = 0
    best_epoch = 0


    for epoch in range(start_epoch, opt.OPTIM.NUM_EPOCHS + 1):
        epoch_start_time = time.time()
        epoch_loss = 0
        epoch_ocr_loss = 0
        epoch_binary_loss = 0
        train_id = 1

        ocr_char_sum = 0
        ocr_correct_sum = 0

        model_restoration.train()
        for i, data in enumerate(tqdm(train_loader), 0):

            # zero_grad
            for param in model_restoration.parameters():
                param.grad = None

            target = data[0].cuda()
            input_ = data[1].cuda()
            label_index = data[3]
            pos = data[4]
            labels = get_batch_label(train_dataset, label_index)

            restored, c_feat_list = model_restoration(input_)

            # Compute loss at each stage
            loss_char = sum([criterion_char(restored[j],target) for j in range(len(restored))])
            loss_mask = sum([criterion_mask(restored[j], target) for j in range(len(restored)-1)])
            loss_channel_feat = criterion_char(c_feat_list[0], c_feat_list[2]) + criterion_char(c_feat_list[1], c_feat_list[3])
            loss_ocr, correct, gt_len = 0, 0, 0
            for j in range(len(restored)-1):
                loss_ocr_temp, correct_temp, gt_len_temp = criterion_ocr(restored[j], labels, pos)
                loss_ocr += loss_ocr_temp
                correct += correct_temp
                gt_len += gt_len_temp
            loss = 0.5*loss_char + 0.2*loss_channel_feat + 0.001 * loss_ocr
            if epoch > 40:
                loss += 0.85*loss_mask

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            if loss_ocr != 0:
                epoch_ocr_loss += loss_ocr.item()
            ocr_correct_sum += correct
            ocr_char_sum += gt_len
            # epoch_binary_loss += loss_binary.item()

        #### Evaluation ####
        if epoch%opt.TRAINING.VAL_AFTER_EVERY == 0:
            model_restoration.eval()
            psnr_val_rgb = []
            for ii, data_val in enumerate((val_loader), 0):
                target = data_val[0].cuda()
                input_ = data_val[1].cuda()

                with torch.no_grad():
                    restored, c_feat_list = model_restoration(input_)
                restored = restored[0]

                for res,tar in zip(restored,target):
                    psnr_val_rgb.append(utils.torchPSNR(res, tar))

                # print img
                if ii == 0:
                    filenames = data_val[2]
                    restored = torch.clamp(restored, 0, 1)

                    restored = restored.permute(0, 2, 3, 1).cpu().detach().numpy()

                    d_b, d_g, d_r = c_feat_list[4]
                    for batch in range(len(restored)):
                        restored_img = img_as_ubyte(restored[batch])
                        utils.save_img((os.path.join(result_dir, 'epoch_{:0>3d}_{}.png'.format(epoch, filenames[batch]))), restored_img)
                        # utils.sv((os.path.join(channel_dir, 'epoch_{:0>3d}_{}_B.png'.format(epoch, filenames[batch]))), d_b[batch])
                        # utils.sv((os.path.join(channel_dir, 'epoch_{:0>3d}_{}_G.png'.format(epoch, filenames[batch]))), d_g[batch])
                        # utils.sv((os.path.join(channel_dir, 'epoch_{:0>3d}_{}_R.png'.format(epoch, filenames[batch]))), d_r[batch])
                        # utils.sv((os.path.join(channel_dir, 'epoch_{:0>3d}_{}_B-G.png'.format(epoch, filenames[batch]))), abs(d_b[batch]-d_g[batch]))
                        # utils.sv((os.path.join(channel_dir, 'epoch_{:0>3d}_{}_R-G.png'.format(epoch, filenames[batch]))), abs(d_r[batch]-d_g[batch]))


            psnr_val_rgb  = torch.stack(psnr_val_rgb).mean().item()

            if psnr_val_rgb > best_psnr:
                best_psnr = psnr_val_rgb
                best_epoch = epoch
                torch.save({'epoch': epoch,
                            'state_dict': model_restoration.state_dict(),
                            'optimizer' : optimizer.state_dict()
                            }, os.path.join(model_dir,"model_best.pth"))

            print("[epoch %d PSNR: %.4f --- best_epoch %d Best_PSNR %.4f]" % (epoch, psnr_val_rgb, best_epoch, best_psnr))

            torch.save({'epoch': epoch,
                        'state_dict': model_restoration.state_dict(),
                        'optimizer' : optimizer.state_dict()
                        }, os.path.join(model_dir,f"model_epoch_{epoch}.pth"))

        scheduler.step()

        print("------------------------------------------------------------------")
        print("Epoch: {}\tTime: {:.4f}\tLoss: {:.4f}\tLearningRate {:.8f}".format(epoch, time.time()-epoch_start_time, epoch_loss, scheduler.get_lr()[0]))
        print("OcrLoss: {:.4f}\tCorrect: {}\tSum_char: {}".format(epoch_ocr_loss, ocr_correct_sum, ocr_char_sum))
        print("------------------------------------------------------------------")

        torch.save({'epoch': epoch,
                    'state_dict': model_restoration.state_dict(),
                    'optimizer' : optimizer.state_dict()
                    }, os.path.join(model_dir,"model_latest.pth"))


