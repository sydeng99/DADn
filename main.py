import logging
import argparse
import os
import re
import time
import numpy as np
import yaml
from attrdict import AttrDict
import pdb

from utils.setup_speed import setup_seed
from utils.buffer import ReplayBuffer
from dataset.dataset_unpair import Provider
from networks import cbdnet
from networks.unsupervised import c2n
from networks.rescaling import inv_arch

import torch
import torchvision
import torch.nn as nn
from torch.autograd import Variable

from tensorboardX import SummaryWriter
from collections import OrderedDict

from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim


def init_project(cfg):
    def init_logging(path):
        logging.basicConfig(
            level=logging.INFO,
            format='%(message)s',
            datefmt='%m-%d %H:%M',
            filename=path,
            filemode='w')

        # define a Handler which writes INFO messages or higher to the sys.stderr
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)

        # set a format which is simpler for console use
        formatter = logging.Formatter('%(message)s')
        # tell the handler to use this format
        console.setFormatter(formatter)
        logging.getLogger('').addHandler(console)

    # seeds
    setup_seed(cfg.TRAIN.random_seed)
    if cfg.TRAIN.if_cuda:
        if torch.cuda.is_available() is False:
            raise AttributeError('No GPU available')

    prefix = cfg.time
    if cfg.TRAIN.if_resume:
        model_name = cfg.TRAIN.resume_model_name
    else:
        model_name = prefix + '_' + cfg.NAME

    cfg.save_path = os.path.join(cfg.TRAIN.save_path, model_name)
    cfg.record_path = os.path.join(cfg.TRAIN.save_path, model_name)
    cfg.valid_path = os.path.join(cfg.TRAIN.save_path, 'valid')
    if cfg.TRAIN.if_resume is False:
        if not os.path.exists(cfg.save_path):
            os.makedirs(cfg.save_path)
        if not os.path.exists(cfg.record_path):
            os.makedirs(cfg.record_path)
        if not os.path.exists(cfg.valid_path):
            os.makedirs(cfg.valid_path)
    init_logging(os.path.join(cfg.record_path, prefix + '.log'))
    logging.info(cfg)
    writer = SummaryWriter(cfg.record_path)  # tensorboard
    writer.add_text('cfg', str(cfg))
    return writer

def load_dataset(cfg):
    print('Caching datasets ... ', end='', flush=True)
    t1 = time.time()
    train_provider = Provider('train', cfg)
    if cfg.TRAIN.if_valid:
        valid_provider = Provider('valid', cfg)
    else:
        valid_provider = None
    print('Done (time: %.2fs)' % (time.time() - t1))
    return train_provider, valid_provider

def model_parallel(model):
    cuda_count = torch.cuda.device_count()
    if cuda_count > 1 and cfg.TRAIN.if_multiGPU:
        if cfg.TRAIN.batch_size % cuda_count == 0:
            print('%d GPUs ... ' % cuda_count, end='', flush=True)
            model = nn.DataParallel(model)
        else:
            raise AttributeError(
                'Batch size (%d) cannot be equally divided by GPU number (%d)' % (cfg.TRAIN.batch_size, cuda_count))
    else:
        print('a single GPU ... ', end='', flush=True)
    return model

def build_model(cfg):
    print('Building model on ', end='', flush=True)
    t1 = time.time()
    device = torch.device('cuda:0')

    if cfg.MODEL.network == 'DADn':
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
        for k in model.keys():
            model[k] = model_parallel(model[k])

    else:
        raise NotImplementedError("using invertible networks")

    print('Done (time: %.2fs)' % (time.time() - t1))
    return model


def load_pretrain_model_weights(cfg, model, model_path=None):
    logging.info('Load pre-trained model ...')
    ckpt_path = {}

    if not model_path:
        files = os.listdir(cfg.MODEL.trained_model_path)
        for ff in files:
            if '%06d'% cfg.MODEL.trained_model_id in ff:
                start = ff.find('PSNR')
                end = ff.find('.pth')
                psnr_value = np.float(ff[start+4:end])
        for k in cfg.TRAIN.model_name:
            ckpt_path[k] = os.path.join(cfg.MODEL.trained_model_path,
                                     'model-%s-%06d-PSNR%.4f.ckpt' % (cfg.TRAIN.model_name[k], cfg.MODEL.trained_model_id, psnr_value))
    else:
        ckpt_path = model_path

    checkpoint = {}
    pretrained_dict = {}
    for k in ckpt_path.keys():
        checkpoint[k] = torch.load(ckpt_path[k])
        pretrained_dict[k] = checkpoint[k]['model_weights']
    if cfg.MODEL.trained_gpus > 1:
        pretained_model_dict = OrderedDict()
        for model_name in pretrained_dict.keys():
            for k, v in pretrained_dict[model_name].items():
                if k.startwith('module.'):
                    name = k[7:]  # remove module.
                    pretained_model_dict[model_name][name] = v
    else:
        pretained_model_dict = pretrained_dict

    for k in cfg.TRAIN.model_name:
        model[k].load_state_dict(pretained_model_dict)
    return model

def resume_training(cfg, model, optimizer):
    if cfg.TRAIN.if_resume:
        t1 = time.time()
        model_path = {}
        if cfg.MODEL.trained_model_id:
            files = os.listdir(cfg.save_path)
            for ff in files:
                if '%06d' % cfg.MODEL.trained_model_id in ff:
                    start = ff.find('PSNR')
                    end = ff.find('.pth')
                    psnr_value = np.float(ff[start + 4:end])
            for k in cfg.TRAIN.model_name:
                model_path[k] = os.path.join(cfg.save_path,
                                  'model-%s-%06d-PSNR%.4f.ckpt' % (cfg.TRAIN.model_name[k], cfg.MODEL.trained_model_id, psnr_value))
        else:
            last_iter = 0
            for files in os.listdir(cfg.save_path):
                if 'model' in files:
                    it = int(re.sub('\D', '', files))
                    if it > last_iter:
                        last_iter = it
            files = os.listdir(cfg.save_path)
            for ff in files:
                if '%06d' % last_iter in ff:
                    start = ff.find('PSNR')
                    end = ff.find('.pth')
                    psnr_value = np.float(ff[start + 4:end])

            for k in cfg.TRAIN.model_name:
                model_path[k] = os.path.join(cfg.save_path, 'model-%s-%06d-PSNR%.4f.ckpt' % (cfg.TRAIN.model_name[k], last_iter, psnr_value))

        print('Resuming weights from ... ', end='', flush=True)
        for k in model_path.keys():
            if os.path.isfile(model_path[k]):
                model[k] = load_pretrain_model_weights(cfg, model[k], model_path=model_path[k])
                optimizer[k].load_state_dict(os.path.join(cfg.TRAIN.save_path, 'optim-%s-lastest.pth' % str(k)))
        else:
            raise AttributeError('No checkpoint found at model_path')

        t2 = time.time() - t1
        print('Done. Cost time: %.2fs' % t2)
        print('valid %d' % checkpoint[0]['current_iter'])
        return model, optimizer, checkpoint[0]['current_iter']
    else:
        return model, optimizer, 0


def calculate_lr(cfg, iters):
    opt = cfg.TRAIN
    if iters < opt.warmup_iters:
        current_lr = (opt.base_lr - opt.end_lr) * pow(float(iters) / opt.warmup_iters, opt.power) + opt.end_lr
    else:
        if iters < opt.decay_iters:
            current_lr = (opt.base_lr - opt.end_lr) * pow(1 - float(iters - opt.warmup_iters) / opt.decay_iters,
                                                          opt.power) + opt.end_lr
        else:
            current_lr = opt.end_lr
    return current_lr

def criterion_loss(cfg):
    if cfg.TRAIN.loss_func == 'L1Loss':
        criterion = nn.L1Loss()
    elif cfg.TRAIN.loss_func == 'L2Loss':
        criterion = nn.MSELoss()
    else:
        raise AttributeError("NO this criterion")
    return criterion

def adjust_lr(cfg, iters, optimizer):
    lr_strategies = ['steplr', 'multi_steplr', 'explr', 'lambdalr']

    if cfg.TRAIN.lr_mode == 'customized':
        if cfg.TRAIN.end_lr == cfg.TRAIN.base_lr:
            current_lr = cfg.TRAIN.base_lr
        else:
            current_lr = calculate_lr(cfg, iters)
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr
        return optimizer, current_lr

    elif cfg.TRAIN.lr_mode in lr_strategies:
        if cfg.TRAIN.lr_mode == 'steplr':
            print('Step LR')
            lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg.TRAIN.step_size,
                                                           gamma=cfg.TRAIN.gamma)
        elif cfg.TRAIN.lr_mode == 'multi_steplr':
            print('Multi step LR')
            lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100000, 150000],
                                                                gamma=cfg.TRAIN.gamma)
        elif cfg.TRAIN.lr_mode == 'explr':
            print('Exponential LR')
            lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.999)
        elif cfg.TRAIN.lr_mode == 'lambdalr':
            print('Lambda LR')
            lambda_func = lambda epoch: (1.0 - epoch / cfg.TRAIN.total_iters) ** 0.9
            lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_func)
        else:
            raise NotImplementedError

        return optimizer, lr_scheduler, lr_scheduler.get_lr()

def tensor2img(tensor):
    im = (255. * tensor).data.cpu().numpy()
    # clamp
    im[im > 255] = 255
    im[im < 0] = 0
    im = im.astype(np.uint8)
    return im

class ReconstructionLoss(nn.Module):
    def __init__(self, losstype='l2', eps=1e-6):
        super(ReconstructionLoss, self).__init__()
        self.losstype = losstype
        self.eps = eps

    def forward(self, x, target):
        if self.losstype == 'l2':
            return torch.mean(torch.sum((x - target)**2, (1, 2, 3)))
        elif self.losstype == 'l1':
            diff = x - target
            return torch.mean(torch.sum(torch.sqrt(diff * diff + self.eps), (1, 2, 3)))
        else:
            print("reconstruction loss type error!")
            return 0

def validation(model, valid_provider):
    for k in model.keys():
        model[k] = model[k].eval()
    results_LR_simu_psnrs = []
    results_LR_real_psnrs = []
    results_recon_simu_psnrs = []
    results_recon_real_psnrs = []
    results_denoise_simu_psnrs = []
    results_denoise_swap_real_psnrs = []
    results_denoise_real_psnrs = []
    results_denoise_real_ssims = []
    results_direct_denoise_real_psnrs = []
    results_direct_denoise_real_ssims = []

    t1 = time.time()
    with torch.no_grad():
        for i in range(len(valid_provider)):
            batch = valid_provider.next()
            clean_im_i = batch['clean_data_i']
            simu_noisy_img_i = batch['simu_noisy_img_i']
            clean_im_j = batch['clean_data_j']
            real_noisy_img_j = batch['noisy_data_j']
            simu_noisy_lr = batch['simu_noisy_lr']
            real_noisy_lr = batch['real_noisy_lr']

            if cfg.TRAIN.if_cuda:
                clean_im_i = clean_im_i.cuda()
                simu_noisy_img_i = simu_noisy_img_i.cuda()
                clean_im_j = clean_im_j.cuda()
                real_noisy_img_j = real_noisy_img_j.cuda()
                simu_noisy_lr = simu_noisy_lr.cuda()
                real_noisy_lr = real_noisy_lr.cuda()


            F_s = model['INNS'](x = simu_noisy_img_i)
            content_channel = int(cfg.DATA.input_channel) * int(cfg.TRAIN.inn_down_scale) * \
                              int(cfg.TRAIN.inn_down_scale) * float(cfg.TRAIN.content_channel_split)
            content_channel = int(content_channel)
            F_c_i, N_s = F_s[:,:content_channel], F_s[:,content_channel:]

            F_r = model['INNR'](x = real_noisy_img_j)
            F_c_j, N_r = F_r[:,:content_channel], F_r[:,content_channel:]

            F_rc_sn = torch.cat([F_c_j, N_s], dim=1)
            F_sc_rn = torch.cat([F_c_i, N_r], dim=1)

            valid_rec_simu_noisy_img_i = model['INNS'](x=F_s, rev=True)
            valid_rec_real_noisy_img_j = model['INNR'](x=F_r, rev=True)
            valid_swap_realContent_simuNoise = model['INNS'](x=F_rc_sn, rev=True)
            valid_swap_simuContent_realNoise = model['INNR'](x=F_sc_rn, rev=True)

            noise_map_est_real, valid_denoised_real = model['denoise'](valid_swap_simuContent_realNoise)
            noise_map_est_simu, valid_denoised_simu = model['denoise'](valid_rec_simu_noisy_img_i)
            noise_map_est_real_j, valid_denoised_real_j = model['denoise'](valid_rec_real_noisy_img_j)
            noise_map_est_direct_real_j, valid_direct_denoised_real_j = model['denoise'](real_noisy_img_j)


            # metrics
            results_LR_simu_psnr = psnr(tensor2img(simu_noisy_lr[0,0]), tensor2img(F_c_i[0,0]))
            results_LR_real_psnr = psnr(tensor2img(real_noisy_lr[0,0]), tensor2img(F_c_j[0,0]))
            results_recon_simu_psnr = psnr(tensor2img(simu_noisy_img_i[0,0]), tensor2img(valid_rec_simu_noisy_img_i[0,0]))
            results_recon_real_psnr = psnr(tensor2img(real_noisy_img_j[0,0]), tensor2img(valid_rec_real_noisy_img_j[0,0]))
            results_denoise_simu_psnr = psnr(tensor2img(clean_im_i[0,0]), tensor2img(valid_denoised_simu[0,0]))
            results_denoise_swap_real_psnr = psnr(tensor2img(clean_im_i[0,0]), tensor2img(valid_denoised_real[0,0]))
            results_denoise_real_psnr = psnr(tensor2img(clean_im_j[0,0]), tensor2img(valid_denoised_real_j[0,0]))
            results_denoise_real_ssim = ssim(tensor2img(clean_im_j[0,0]), tensor2img(valid_denoised_real_j[0,0]))
            results_direct_denoise_real_psnr = psnr(tensor2img(clean_im_j[0,0]), tensor2img(valid_direct_denoised_real_j[0,0]))
            results_direct_denoise_real_ssim = ssim(tensor2img(clean_im_j[0,0]), tensor2img(valid_direct_denoised_real_j[0,0]))

            # logging.info('img-%d, psnr=%.4f, ssim=%.4f'%(i, results_psnr,results_ssim))
            results_LR_simu_psnrs.append(results_LR_simu_psnr)
            results_LR_real_psnrs.append(results_LR_real_psnr)
            results_recon_simu_psnrs.append(results_recon_simu_psnr)
            results_recon_real_psnrs.append(results_recon_real_psnr)
            results_denoise_simu_psnrs.append(results_denoise_simu_psnr)
            results_denoise_swap_real_psnrs.append(results_denoise_swap_real_psnr)
            results_denoise_real_psnrs.append(results_denoise_real_psnr)
            results_denoise_real_ssims.append(results_denoise_real_ssim)
            results_direct_denoise_real_psnrs.append(results_direct_denoise_real_psnr)
            results_direct_denoise_real_ssims.append(results_direct_denoise_real_ssim)

    t2 = time.time()

    ave_LR_simu_psnr = sum(results_LR_simu_psnrs) / len(results_LR_simu_psnrs)
    ave_LR_real_psnr = sum(results_LR_real_psnrs) / len(results_LR_real_psnrs)
    ave_recon_simu_psnr = sum(results_recon_simu_psnrs) / len(results_recon_simu_psnrs)
    ave_recon_real_psnr = sum(results_recon_real_psnrs) / len(results_recon_real_psnrs)
    ave_denoise_simu_psnr = sum(results_denoise_simu_psnrs) / len(results_denoise_simu_psnrs)
    ave_denoise_swap_real_psnr = sum(results_denoise_swap_real_psnrs) / len(results_denoise_swap_real_psnrs)
    ave_denoise_real_psnr = sum(results_denoise_real_psnrs) / len(results_denoise_real_psnrs)
    ave_denoise_real_ssim = sum(results_denoise_real_ssims) / len(results_denoise_real_ssims)
    ave_direct_denoise_real_psnr = sum(results_direct_denoise_real_psnrs) / len(results_direct_denoise_real_psnrs)
    ave_direct_denoise_real_ssim = sum(results_direct_denoise_real_ssims) / len(results_direct_denoise_real_ssims)

    return ave_LR_simu_psnr, ave_LR_real_psnr, ave_recon_simu_psnr, ave_recon_real_psnr, \
           ave_denoise_simu_psnr, ave_denoise_swap_real_psnr, ave_denoise_real_psnr, ave_denoise_real_ssim, \
           ave_direct_denoise_real_psnr, ave_direct_denoise_real_ssim, \
           t2-t1



def train_loop(cfg, train_provider, valid_provider, model, optimizer, iters, writer):
    rcd_time = []
    sum_time = 0
    sum_loss = 0

    current_lr = {}
    # Buffers of previously generated samples
    fake_real_content_buffer = ReplayBuffer()
    fake_simu_noisy_img_buffer = ReplayBuffer()
    fake_real_noisy_img_buffer = ReplayBuffer()


    FloatTensor = torch.cuda.FloatTensor if cfg.TRAIN.if_cuda else torch.FloatTensor
    real_label = Variable(FloatTensor(np.ones((cfg.TRAIN.batch_size, cfg.DATA.input_channel,
                                               cfg.DATA.train_patch_size[0], cfg.DATA.train_patch_size[1]))), requires_grad = False)
    fake_label =Variable(FloatTensor(np.zeros((cfg.TRAIN.batch_size, cfg.DATA.input_channel,
                                               cfg.DATA.train_patch_size[0], cfg.DATA.train_patch_size[1]))), requires_grad = False)

    real_label_lr = Variable(FloatTensor(np.ones((cfg.TRAIN.batch_size, cfg.DATA.input_channel,
                                               cfg.DATA.train_patch_size[0]//2, cfg.DATA.train_patch_size[1]//2))), requires_grad = False)
    fake_label_lr =Variable(FloatTensor(np.zeros((cfg.TRAIN.batch_size, cfg.DATA.input_channel,
                                               cfg.DATA.train_patch_size[0]//2, cfg.DATA.train_patch_size[1]//2))), requires_grad = False)

    torch.backends.cudnn.benchmark = True

    model_names = cfg.TRAIN.model_name
    while iters <= cfg.TRAIN.total_iters:
        # train
        iters += 1
        t1 = time.time()
        batch_data = train_provider.next()

        real_noisy_input = batch_data['noisy_img']      # j
        simu_noisy_input = batch_data['simu_noisy_img'] # i
        clean_input = batch_data['clean_img']
        simu_noisy_lr = batch_data['simu_noisy_lr']
        real_noisy_lr = batch_data['real_noisy_lr']
        simu_noisy_lr = simu_noisy_lr.detach()
        real_noisy_lr = real_noisy_lr.detach()

        if cfg.TRAIN.if_cuda:
            real_noisy_input = real_noisy_input.cuda()
            simu_noisy_input = simu_noisy_input.cuda()
            clean_input = clean_input.cuda()
            simu_noisy_lr = simu_noisy_lr.cuda()
            real_noisy_lr = real_noisy_lr.cuda()

        # decay learning rate
        if cfg.TRAIN.lr_mode == 'customized':
            for mn in model_names:
                optimizer[mn], current_lr[mn] = adjust_lr(cfg, iters, optimizer[mn])
        else:
            for mn in model_names:
                optimizer[mn], lr_scheduler, current_lr[mn] = adjust_lr(cfg, iters, optimizer[mn])

        for mn in model_names:
            model[mn] = model[mn].train()

        # generator, ['INNS', 'INNR', 'denoise', 'discri_content', 'discri_simu', 'discri_real']
        for mi in range(3):
            optimizer[model_names[mi]].zero_grad()


        F_simu = model['INNS'](simu_noisy_input)
        content_channel = int(cfg.DATA.input_channel) * int(cfg.TRAIN.inn_down_scale) * \
                          int(cfg.TRAIN.inn_down_scale) * float(cfg.TRAIN.content_channel_split)
        content_channel = int(content_channel)
        F_simu_content, N_s = F_simu[:, :content_channel], F_simu[:, content_channel:]

        F_real = model['INNR'](real_noisy_input)
        F_real_content, N_r = F_real[:, :content_channel], F_real[:, content_channel:]

        rec_simu_img = model['INNS'](x=torch.cat([F_simu_content, N_s], dim = 1), rev=True)
        rec_real_img = model['INNR'](x=torch.cat([F_real_content, N_r], dim = 1), rev=True)
        swap_img_rc_sn = model['INNS'](x=torch.cat([F_real_content, N_s], dim = 1), rev=True)
        swap_img_sc_rn = model['INNR'](x=torch.cat([F_simu_content, N_r], dim = 1), rev=True)

        noise_map_est_real, denoised_real = model['denoise'](swap_img_sc_rn)
        noise_map_est_simu, denoised_simu = model['denoise'](rec_simu_img)

        loss_func = criterion_loss(cfg)
        loss_func2 = nn.MSELoss()

        loss_recon_simu = loss_func(rec_simu_img, simu_noisy_input)
        loss_recon_real = loss_func(rec_real_img, real_noisy_input)
        loss_simu_lr_bicubic = loss_func(F_simu_content, simu_noisy_lr)
        loss_real_lr_bicubic = loss_func(F_real_content, real_noisy_lr)
        loss_denoised_simu = loss_func(denoised_simu, clean_input)
        loss_denoised_real = loss_func(denoised_real, clean_input)

        # semantic consistency
        semantic_model_path = cfg.TRAIN.semantic_model_path
        resnet50 = torchvision.models.resnet50()
        resnet50.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False)
        resnet50.fc = nn.Identity()
        checkpoint = torch.load(semantic_model_path)
        pretrained_dict = checkpoint['model']
        pretained_model_dict = OrderedDict()
        for k in pretrained_dict.keys():
            if k[:16] == 'module.backbone.':
                name = k[16:]
                pretained_model_dict[name] = pretrained_dict[k]
        resnet50.load_state_dict(pretained_model_dict)
        resnet50 = resnet50.eval()
        for param in resnet50.parameters():
            param.requires_grad = False
        new_m = torchvision.models._utils.IntermediateLayerGetter(resnet50, {'layer1': 'feat1', 'layer2': 'feat2',
                                                                             'layer3': 'feat3', 'layer4': 'feat4'})
        if cfg.TRAIN.if_cuda:
            new_m = new_m.cuda()

        feat_simu_input = new_m(simu_noisy_input)['feat2']
        feat_swap_sc_rn = new_m(swap_img_sc_rn)['feat2']
        feat_real_input = new_m(real_noisy_input)['feat2']
        feat_swap_rc_sn = new_m(swap_img_rc_sn)['feat2']

        loss_percep_sc = loss_func(feat_simu_input, feat_swap_sc_rn)
        loss_percep_rc = loss_func(feat_real_input, feat_swap_rc_sn)



        # Discriminator-content, real content=real label
        # -- Generator
        loss_GAN_content = loss_func2(model['discri_content'](F_simu_content), real_label_lr)

        # Discriminator - discri_simu noise, simu_input = real label
        # -- Generator
        loss_GAN_simu_noise = loss_func2(model['discri_simu'](swap_img_rc_sn), real_label)

        # Discriminator - discri_real noise, real noisy input = real label
        # -- Generator
        loss_GAN_real_noise = loss_func2(model['discri_real'](swap_img_sc_rn), real_label)

        # loss = loss_recon * cfg.TRAIN.lr_lambda
        loss_recon_simu = loss_recon_simu * cfg.LOSS.lambda_loss_recon_simu
        loss_recon_real = loss_recon_real * cfg.LOSS.lambda_loss_recon_real
        loss_simu_lr_bicubic = loss_simu_lr_bicubic * cfg.LOSS.lambda_loss_simu_lr_bicubic
        loss_real_lr_bicubic = loss_real_lr_bicubic * cfg.LOSS.lambda_loss_real_lr_bicubic
        loss_denoised_simu = loss_denoised_simu * cfg.LOSS.lambda_loss_denoised_simu
        loss_denoised_real = loss_denoised_real * cfg.LOSS.lambda_loss_denoised_real
        loss_percep_sc = loss_percep_sc * cfg.LOSS.lambda_loss_percep_sc
        loss_percep_rc = loss_percep_rc * cfg.LOSS.lambda_loss_percep_rc
        loss_GAN_content = loss_GAN_content * cfg.LOSS.lambda_loss_GAN_content
        loss_GAN_simu_noise = loss_GAN_simu_noise * cfg.LOSS.lambda_loss_GAN_simu_noise
        loss_GAN_real_noise = loss_GAN_real_noise * cfg.LOSS.lambda_loss_GAN_real_noise

        loss =  loss_recon_simu + \
                loss_recon_real + \
                loss_simu_lr_bicubic + \
                loss_real_lr_bicubic + \
                loss_denoised_simu + \
                loss_denoised_real + \
                loss_percep_sc + \
                loss_percep_rc + \
                loss_GAN_content + \
                loss_GAN_simu_noise + \
                loss_GAN_real_noise


        loss.backward(retain_graph=True)


        # =============================
        #    Train Discriminators
        # =============================
        for mi in range(3, len(model_names)):
            optimizer[model_names[mi]].zero_grad()

        # Discriminator-content, real content=real label
        # -- Discriminator
        fake_real_content_ = fake_real_content_buffer.push_and_pop(F_simu_content)
        loss_real = loss_func2(model['discri_content'](F_real_content), real_label_lr)
        loss_fake = loss_func2(model['discri_content'](fake_real_content_.detach()), fake_label_lr)
        loss_D_content = (loss_real + loss_fake) / 2
        loss_D_content = loss_D_content * cfg.LOSS.lambda_loss_D_content
        loss_D_content.backward()
        torch.autograd.set_detect_anomaly(True)
        optimizer['discri_content'].step()


        for mi in range(3):
            optimizer[model_names[mi]].step()


        # Discriminator - discri_simu noise, simu_input = real label
        # -- Discriminator
        fake_simu_noisy_img_ = fake_simu_noisy_img_buffer.push_and_pop(swap_img_rc_sn)
        loss_real = loss_func2(model['discri_simu'](simu_noisy_input), real_label)
        loss_fake = loss_func2(model['discri_simu'](fake_simu_noisy_img_.detach()), fake_label)
        loss_D_simu_noise = (loss_real + loss_fake) / 2
        loss_D_simu_noise = loss_D_simu_noise * cfg.LOSS.lambda_loss_D_simu_noise
        loss_D_simu_noise.backward()
        optimizer['discri_simu'].step()

        # Discriminator - discri_real noise, real noisy input = real label
        # -- Discriminator
        fake_real_noisy_img_ = fake_real_noisy_img_buffer.push_and_pop(swap_img_sc_rn)
        loss_real = loss_func2(model['discri_real'](real_noisy_input), real_label)
        loss_fake = loss_func2(model['discri_real'](fake_real_noisy_img_.detach()), fake_label)
        loss_D_real_noise = (loss_real + loss_fake) / 2
        loss_D_real_noise = loss_D_real_noise * cfg.LOSS.lambda_loss_D_real_noise
        loss_D_real_noise.backward()
        optimizer['discri_real'].step()

        loss_D = loss_D_content + \
                 loss_D_simu_noise + \
                 loss_D_real_noise

        torch.autograd.set_detect_anomaly(True)


        if cfg.TRAIN.lr_mode == 'customized':
            if cfg.TRAIN.weight_decay is not None:
                for mn in model_names:
                    for group in optimizer[mn].param_groups:
                        for param in group['params']:
                            param.data = param.data.add(-cfg.TRAIN.weight_decay * group['lr'], param.data)
        else:
            lr_scheduler.step()

        # for mn in model_names:
        #     optimizer[mn].step()

        sum_loss += loss.item()
        sum_time += time.time() - t1

        # log train
        df = cfg.TRAIN.display_freq
        if iters % df == 0 and iters>0:
            rcd_time.append(sum_time)
            logging.info('step %d, loss = %.4f, '
                         'loss_denoised_simu = %.4f, loss_denoised_real = %.4f, '
                         'loss_percep_sc = %.4f, loss_percep_rc = %.4f, '
                         'loss_GAN_content = %.4f, loss_GAN_simu_noise = %.4f, loss_GAN_real_noise = %.4f '
                         'loss_simu_lr_bicubic = %.4f, loss_real_lr_bicubic = %.4f, '
                         'loss_recon_simu = %.4f, loss_recon_real = %.4f, '
                         'loss_D_content=%.4f, loss_D_simu_noise=%.4f, loss_D_real_noise=%.4f'
                         '(lr-denoise:%g, et:%.2f sec, rd:%.2f h)'
                         % (iters, sum_loss/cfg.TRAIN.display_freq,
                            loss_denoised_simu.item(), loss_denoised_real.item(), loss_percep_sc.item(), loss_percep_rc.item(),
                            loss_GAN_content.item(), loss_GAN_simu_noise.item(), loss_GAN_real_noise.item(),
                            loss_simu_lr_bicubic.item(), loss_real_lr_bicubic.item(), loss_recon_simu.item(), loss_recon_real.item(),
                            loss_D_content.item(), loss_D_simu_noise.item(), loss_D_real_noise.item(),
                            current_lr['INNS'], sum_time,
                            (cfg.TRAIN.total_iters - iters) / df * np.mean(np.asarray(rcd_time)) / 3600))

            writer.add_scalar('loss', sum_loss / cfg.TRAIN.display_freq , iters)
            writer.add_scalar('loss_recon/loss_recon_simu', loss_recon_simu.item(), iters)
            writer.add_scalar('loss_recon/loss_recon_real', loss_recon_real.item() , iters)
            writer.add_scalar('loss_recon/loss_simu_lr_bicubic', loss_simu_lr_bicubic.item() , iters)
            writer.add_scalar('loss_recon/loss_real_lr_bicubic', loss_real_lr_bicubic.item() , iters)
            writer.add_scalar('loss_denoise/loss_denoised_simu', loss_denoised_simu.item() , iters)
            writer.add_scalar('loss_denoise/loss_denoised_real', loss_denoised_real.item() , iters)
            writer.add_scalar('loss_percep/loss_percep_sc', loss_percep_sc.item() , iters)
            writer.add_scalar('loss_percep/loss_percep_rc', loss_percep_rc.item() , iters)
            writer.add_scalar('loss_GAN/loss_GAN_content', loss_GAN_content.item() , iters)
            writer.add_scalar('loss_GAN/loss_GAN_simu_noise', loss_GAN_simu_noise.item() , iters)
            writer.add_scalar('loss_GAN/loss_GAN_real_noise', loss_GAN_real_noise.item() , iters)
            writer.add_scalar('loss_GAN/loss_D_content', loss_D_content.item() , iters)
            writer.add_scalar('loss_GAN/loss_D_simu_noise', loss_D_simu_noise.item() , iters)
            writer.add_scalar('loss_GAN/loss_D_real_noise', loss_D_real_noise.item() , iters)


            sum_time = 0
            sum_loss = 0

        # valid
        if cfg.TRAIN.if_valid:
            if (iters % cfg.TRAIN.valid_freq == 0) & (iters>0):
                ave_LR_simu_psnr, ave_LR_real_psnr, ave_recon_simu_psnr, ave_recon_real_psnr, \
                ave_denoise_simu_psnr, ave_denoise_swap_real_psnr, ave_denoise_real_psnr, ave_denoise_real_ssim, \
                ave_direct_denoise_real_psnr, ave_direct_denoise_real_ssim, \
                valid_cost_time = validation(model, valid_provider)

                logging.info('valid-iter%d, '
                             'ave_denoise_real_psnr=%.4f, ave_denoise_real_ssim=%.4f, '
                             'ave_denoise_simu_psnr=%.4f, ave_denoise_swap_real_psnr=%.4f, '
                             'ave_LR_simu_psnr=%.4f, ave_LR_real_psnr=%.4f, '
                             'ave_recon_simu_psnr=%.4f, ave_recon_real_psnr=%.4f, '
                             'valid_cost_time:%.2f sec' %
                             (iters,
                              ave_denoise_real_psnr, ave_denoise_real_ssim,
                              ave_denoise_simu_psnr, ave_denoise_swap_real_psnr,
                              ave_LR_simu_psnr, ave_LR_real_psnr,
                              ave_recon_simu_psnr,ave_recon_real_psnr,
                              valid_cost_time))

                writer.add_scalar('valid/ave_LR_simu_psnr', ave_LR_simu_psnr, iters)
                writer.add_scalar('valid/ave_LR_real_psnr', ave_LR_real_psnr, iters)
                writer.add_scalar('valid/ave_recon_simu_psnr', ave_recon_simu_psnr, iters)
                writer.add_scalar('valid/ave_recon_real_psnr', ave_recon_real_psnr, iters)
                writer.add_scalar('valid/ave_denoise_simu_psnr', ave_denoise_simu_psnr, iters)
                writer.add_scalar('valid/ave_denoise_swap_real_psnr', ave_denoise_swap_real_psnr, iters)
                writer.add_scalar('valid/ave_denoise_real_psnr', ave_denoise_real_psnr, iters)

                writer.add_image('valid/content_i/simu_noisy_input', tensor2img(simu_noisy_input[0,0]), iters, dataformats='HW')
                writer.add_image('valid/content_i/clean_input', tensor2img(clean_input[0,0]), iters, dataformats='HW')
                writer.add_image('valid/content_i/F_simu_content', tensor2img(F_simu_content[0,0]), iters, dataformats='HW')
                writer.add_image('valid/content_i/rec_simu_img', tensor2img(rec_simu_img[0,0]), iters, dataformats='HW')
                writer.add_image('valid/content_i/swap_img_sc_rn', tensor2img(swap_img_sc_rn[0,0]), iters, dataformats='HW')
                writer.add_image('valid/content_i/denoised_real', tensor2img(denoised_real[0,0]), iters, dataformats='HW')
                writer.add_image('valid/content_i/denoised_simu', tensor2img(denoised_simu[0,0]), iters, dataformats='HW')

                writer.add_image('valid/content_j/real_noisy_input', tensor2img(real_noisy_input[0,0]), iters, dataformats='HW')
                writer.add_image('valid/content_j/F_real_content', tensor2img(F_real_content[0,0]), iters, dataformats='HW')
                writer.add_image('valid/content_j/rec_real_img', tensor2img(rec_real_img[0,0]), iters, dataformats='HW')
                writer.add_image('valid/content_j/swap_img_rc_sn', tensor2img(swap_img_rc_sn[0,0]), iters, dataformats='HW')
                writer.add_image('valid/noise_map/noise_map_est_simu', tensor2img(noise_map_est_simu[0,0]), iters, dataformats='HW')
                writer.add_image('valid/noise_map/noise_map_est_real', tensor2img(noise_map_est_real[0,0]), iters, dataformats='HW')

        # save
        if (iters % cfg.TRAIN.save_freq == 0) & (iters>0):
            states_denoise = {'current_iter': iters,
                              'valid_result': (ave_LR_simu_psnr, ave_LR_real_psnr, ave_recon_simu_psnr, ave_recon_real_psnr,
                                               ave_denoise_simu_psnr,ave_denoise_swap_real_psnr,ave_denoise_real_psnr),
                              'model_weights_INNS': model['INNS'].state_dict(),
                              'model_weights_INNR': model['INNR'].state_dict(),
                              'model_weights_denoise': model['denoise'].state_dict(),
                              'model_weights_discri_content': model['discri_content'].state_dict(),
                              'model_weights_discri_simu': model['discri_simu'].state_dict(),
                              'model_weights_discri_real': model['discri_real'].state_dict()
                              }
            torch.save(states_denoise, os.path.join(cfg.save_path, 'models-%06d-PSNR%.4f.pth' % (iters, ave_denoise_real_psnr)))

            torch.save(optimizer['INNS'].state_dict(), os.path.join(cfg.save_path, 'optim-%s-lastest.pth'%'INNS'))
            torch.save(optimizer['INNR'].state_dict(), os.path.join(cfg.save_path, 'optim-%s-lastest.pth'%'INNR'))
            torch.save(optimizer['denoise'].state_dict(), os.path.join(cfg.save_path, 'optim-%s-lastest.pth'%'denoise'))
            torch.save(optimizer['discri_content'].state_dict(), os.path.join(cfg.save_path, 'optim-%s-lastest.pth'%'discri_content'))
            torch.save(optimizer['discri_simu'].state_dict(), os.path.join(cfg.save_path, 'optim-%s-lastest.pth'%'discri_simu'))
            torch.save(optimizer['discri_real'].state_dict(), os.path.join(cfg.save_path, 'optim-%s-lastest.pth'%'discri_real'))

            logging.info('***************save modol, iters = %d.***************' % (iters))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--cfg', type=str, default='./config/ours.yml', help='path to config file')
    args = parser.parse_args()

    with open(args.cfg, 'r') as f:
        cfg = AttrDict(yaml.load(f))

    print('*' * 20 + 'import data_provider_ours' + '*' * 20)
    timeArray = time.localtime()
    # time correction
    hour = int(timeArray[3]) + 8
    if hour>=24:
        hour-=24
        day = timeArray[2]+1
    else:
        day = timeArray[2]
    time_stamp = '%04d-%02d-%02d-%02d-%02d-%02d'%(timeArray[0],timeArray[1],day, hour,timeArray[4],timeArray[5])
    print('time stamp:', time_stamp)
    cfg.time = time_stamp

    writer = init_project(cfg = cfg)
    train_provider, valid_provider = load_dataset(cfg)
    model = build_model(cfg)
    optimizer = {}
    optimizer['INNS'] = torch.optim.Adam(model['INNS'].parameters(), lr=cfg.TRAIN.base_lr, betas=(0.9, 0.999), eps=1e-8, amsgrad=False)
    optimizer['INNR'] = torch.optim.Adam(model['INNR'].parameters(), lr=cfg.TRAIN.base_lr, betas=(0.9, 0.999), eps=1e-8, amsgrad=False)
    optimizer['denoise'] = torch.optim.Adam(model['denoise'].parameters(), lr=cfg.TRAIN.base_lr, betas=(0.9, 0.999), eps=1e-8, amsgrad=False)
    optimizer['discri_content'] = torch.optim.Adam(model['discri_content'].parameters(), lr=cfg.TRAIN.base_lr, betas=(0.9, 0.999), eps=1e-8, amsgrad=False)
    optimizer['discri_simu'] = torch.optim.Adam(model['discri_simu'].parameters(), lr=cfg.TRAIN.base_lr, betas=(0.9, 0.999), eps=1e-8, amsgrad=False)
    optimizer['discri_real'] = torch.optim.Adam(model['discri_real'].parameters(), lr=cfg.TRAIN.base_lr, betas=(0.9, 0.999), eps=1e-8, amsgrad=False)

    model, optimizer, init_iters = resume_training(cfg, model, optimizer)

    train_loop(cfg, train_provider, valid_provider, model, optimizer, init_iters, writer)

    writer.close()