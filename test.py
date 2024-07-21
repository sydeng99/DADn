import torch
import torchvision
import torch.nn as nn
from torch.autograd import Variable

from tensorboardX import SummaryWriter
from collections import OrderedDict

from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

import logging
import argparse
import os
import re
import time
import numpy as np
import yaml
from attrdict import AttrDict
import pdb
from skimage import io

from utils.setup_speed import setup_seed
from utils.buffer import ReplayBuffer
from dataset.dataset_unpair import Provider
from networks import cbdnet
from networks.unsupervised import c2n, cycleGAN
from networks.rescaling import inv_arch


def load_dataset(cfg):
    print('Caching datasets ... ', end='', flush=True)
    t1 = time.time()
    train_provider = Provider('train', cfg)
    valid_provider = Provider('test', cfg)

    print('Done (time: %.2fs)' % (time.time() - t1))
    return train_provider, valid_provider

def build_model(cfg):
    print('Building model on ', end='', flush=True)
    t1 = time.time()
    device = torch.device('cuda:0')

    model_inn_s = inv_arch.InvRescaleNet(channel_in=cfg.DATA.input_channel, channel_out=cfg.DATA.input_channel,
                                         subnet_constructor=inv_arch.subnet(net_structure='DBNet'),
                                         block_num=[8], down_num=1, down_scale=2).to(device)
    model_inn_r = inv_arch.InvRescaleNet(channel_in=cfg.DATA.input_channel, channel_out=cfg.DATA.input_channel,
                                         subnet_constructor=inv_arch.subnet(net_structure='DBNet'),
                                         block_num=[8], down_num=1, down_scale=2).to(device)
    model_denoise = cbdnet.Network(channel=cfg.DATA.input_channel).to(device)

    model_discri_content = c2n.C2N_D(n_ch_in=cfg.DATA.input_channel).to(device)
    model_discri_simu = c2n.C2N_D(n_ch_in=cfg.DATA.input_channel).to(device)
    model_discri_real = c2n.C2N_D(n_ch_in=cfg.DATA.input_channel).to(device)
    model = {'INNS': model_inn_s,
             'INNR': model_inn_r,
             'denoise': model_denoise,
             'discri_content': model_discri_content,
             'discri_simu': model_discri_simu,
             'discri_real': model_discri_real,
             }
    print('Done (time: %.2fs)' % (time.time() - t1))
    return model



def load_pretrain_model_weights(cfg, model):
    logging.info('Load pre-trained model ...')
    ckpt_path = cfg.TEST.model_path
    model_names = cfg.TEST.model_names
    pretrained_dict = {}
    pretained_model_dict = {}
    checkpoint = torch.load(ckpt_path)

    for mn in model_names:
        pretrained_dict['model_weights_'+mn] = checkpoint['model_weights_'+mn]
        pretained_model_dict['model_weights_'+mn] = OrderedDict()

    if cfg.TEST.trained_gpus > 1:
        # pretained_model_dict = OrderedDict()
        for model_name in pretrained_dict.keys():
            for k, v in pretrained_dict[model_name].items():
                if k[:7]=='module.':
                    name = k[7:]  # remove module.
                    pretained_model_dict[model_name][name] = v
    else:
        pretained_model_dict = pretrained_dict

    for mn in cfg.TEST.model_names:
        model[mn].load_state_dict(pretained_model_dict['model_weights_'+mn])
    return model

def tensor2img(tensor):
    im = (255. * tensor).data.cpu().numpy()
    # clamp
    im[im > 255] = 255
    im[im < 0] = 0
    im = im.astype(np.uint8)
    return im


def test(cfg, valid_provider, model):
    if not os.path.exists(cfg.TEST.save_path+'denoised_real/'):
        os.makedirs(cfg.TEST.save_path+'denoised_real/')
    if not os.path.exists(cfg.TEST.save_path+'genNoisy/'):
        os.makedirs(cfg.TEST.save_path+'genNoisy/')

    text_record = os.path.join(cfg.TEST.save_path, 'test_psnr_ssim.txt')
    f = open(text_record, 'w')

    for k in model.keys():
        model[k] = model[k].eval()
    results_denoise_simu_psnrs = []
    results_denoise_swap_real_psnrs = []
    results_direct_denoise_real_psnrs = []
    results_direct_denoise_real_ssims = []
    device = torch.device('cuda:1')
    with torch.no_grad():
        for i in range(len(valid_provider)):
            batch = valid_provider.next()
            noisy_data_i = batch['noisy_data_i']
            simu_noisy_img_j = batch['simu_noisy_img_j']
            clean_data_j = batch['clean_data_j']
            noisy_data_j = batch['noisy_data_j']

            if cfg.TRAIN.if_cuda:
                noisy_data_i = noisy_data_i.to(device) # cuda()
                simu_noisy_img_j = simu_noisy_img_j.to(device)
                clean_im_j = clean_data_j.to(device)
                real_noisy_img_j = noisy_data_j.to(device)

            F_s = model['INNS'](x = simu_noisy_img_j)
            content_channel = int(cfg.DATA.input_channel) * int(cfg.TRAIN.inn_down_scale) * \
                              int(cfg.TRAIN.inn_down_scale) * float(cfg.TRAIN.content_channel_split)
            content_channel = int(content_channel)
            F_c_j, N_s = F_s[:,:content_channel], F_s[:,content_channel:]

            F_r = model['INNR'](x = noisy_data_i)
            F_c_i, N_r = F_r[:,:content_channel], F_r[:,content_channel:]

            F_rc_sn = torch.cat([F_c_i, N_s], dim=1)
            F_sc_rn = torch.cat([F_c_j, N_r], dim=1)

            valid_rec_simu_noisy_img_i = model['INNS'](x=F_s, rev=True)
            F_r_j = model['INNR'](x=real_noisy_img_j)
            valid_rec_real_noisy_img_j = model['INNR'](x=F_r_j, rev=True)
            valid_swap_realContent_simuNoise = model['INNS'](x=F_rc_sn, rev=True)
            valid_swap_simuContent_realNoise = model['INNR'](x=F_sc_rn, rev=True)

            noise_map_est_direct_real_j, valid_direct_denoised_real_j = model['denoise'](valid_rec_real_noisy_img_j)

            # metrics
            results_direct_denoise_real_psnr = psnr(tensor2img(clean_im_j[0,0]), tensor2img(valid_direct_denoised_real_j[0,0]))
            results_direct_denoise_real_ssim = ssim(tensor2img(clean_im_j[0,0]), tensor2img(valid_direct_denoised_real_j[0,0]))

            # write text_record
            f.writelines('image-'+str(i+1).zfill(4)+' PSNR=%.4f, SSIM=%.4f'%(results_direct_denoise_real_psnr,results_direct_denoise_real_ssim))
            f.writelines('\n')
            print('image-'+str(i+1).zfill(4)+' PSNR=%.4f, SSIM=%.4f'%(results_direct_denoise_real_psnr,results_direct_denoise_real_ssim))

            # logging.info('img-%d, psnr=%.4f, ssim=%.4f'%(i, results_psnr,results_ssim))
            results_direct_denoise_real_psnrs.append(results_direct_denoise_real_psnr)
            results_direct_denoise_real_ssims.append(results_direct_denoise_real_ssim)

            swapImg_simuContent_realNoise = tensor2img(valid_swap_simuContent_realNoise[0,0])
            denoisedImg_real = tensor2img(valid_direct_denoised_real_j[0,0])

            io.imsave(os.path.join(cfg.TEST.save_path+'genNoisy/', str(i+1).zfill(4)+'.png'), swapImg_simuContent_realNoise)
            io.imsave(os.path.join(cfg.TEST.save_path+'denoised_real/', str(i+1).zfill(4)+'.png'), denoisedImg_real)

    ave_psnr = sum(results_direct_denoise_real_psnrs) / len(results_direct_denoise_real_psnrs)
    ave_ssim = sum(results_direct_denoise_real_ssims) / len(results_direct_denoise_real_ssims)

    f.writelines('AVE PSNR=%.4f, AVE SSIM=%.4f' % (ave_psnr, ave_ssim))
    f.writelines('\n')
    f.close()
    return ave_psnr, ave_ssim


if __name__ =='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--cfg', type=str, default='./config/ours.yml', help='path to config file')
    args = parser.parse_args()

    with open(args.cfg, 'r') as f:
        cfg = AttrDict(yaml.safe_load(f))

    train_provider, valid_provider = load_dataset(cfg)
    model = build_model(cfg)
    model = load_pretrain_model_weights(cfg, model)

    p,s = test(cfg, valid_provider, model)
    print(p, s)