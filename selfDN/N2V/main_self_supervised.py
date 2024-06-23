import logging
import argparse
import os
import re
import time
import numpy as np
import yaml
from attrdict import AttrDict

from setup_speed import setup_seed
from dataset_self_Noisy import Provider
import n2v

import torch
import torch.nn as nn

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
        model_name = cfg.TRAIN.model_name
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
        valid_provider = Provider('test', cfg)
    else:
        valid_provider = None
    print('Done (time: %.2fs)' % (time.time() - t1))
    print('Train data num: %d' % len(train_provider))
    return train_provider, valid_provider

def build_model(cfg):
    print('Building model on ', end='', flush=True)
    t1 = time.time()
    device = torch.device('cuda:0')

    if cfg.MODEL.model_type =='self-supervised':
        elif cfg.MODEL.network =='n2v':
            model = n2v.ResNet(nch_in=cfg.DATA.input_channel,
                           nch_out=cfg.DATA.input_channel).to(device)
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError('This main.py is for supervised learning: paired clean/noisy training')

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
    print('Done (time: %.2fs)' % (time.time() - t1))
    return model


def load_pretrain_model_weights(cfg, model, model_path=None):
    logging.info('Load pre-trained model ...')
    if not model_path:
        ckpt_path = os.path.join(cfg.MODEL.trained_model_path, 'model-%06d.ckpt' % cfg.MODEL.trained_model_id)
    else:
        ckpt_path = model_path
    checkpoint = torch.load(ckpt_path)
    pretrained_dict = checkpoint['model_weights']
    if cfg.MODEL.trained_gpus > 1:
        pretained_model_dict = OrderedDict()
        for k, v in pretrained_dict.items():
            if k.startwith('module.'):
                name = k[7:]  # remove module.
                pretained_model_dict[name] = v
    else:
        pretained_model_dict = pretrained_dict
    model.load_state_dict(pretained_model_dict)
    return model

def resume_training(cfg, model, optimizer):
    # state = {'epoch': epoch, 'iter': current_iter, 'optimizers': [], 'schedulers': []}
    if cfg.TRAIN.if_resume:
        t1 = time.time()
        if cfg.MODEL.trained_model_id:
            model_path = os.path.join(cfg.save_path,
                                  'model-%06d.ckpt' % cfg.MODEL.trained_model_id)
        else:
            last_iter = 0
            for files in os.listdir(cfg.save_path):
                if 'model' in files:
                    it = int(re.sub('\D', '', files))
                    if it > last_iter:
                        last_iter = it
            model_path = os.path.join(cfg.save_path, 'model-%d.ckpt' % last_iter)

        print('Resuming weights from %s ... ' % model_path, end='', flush=True)
        if os.path.isfile(model_path):
            checkpoint = torch.load(model_path)
            model = load_pretrain_model_weights(cfg, model, model_path=model_path)
            # model.load_state_dict(checkpoint['model_weights'])
            optimizer.load_state_dict(os.path.join(cfg.TRAIN.save_path, 'optim-lastest.pth'))
        else:
            raise AttributeError('No checkpoint found at %s' % model_path)
        print('Done (time: %.2fs)' % (time.time() - t1))
        print('valid %d' % checkpoint['current_iter'])
        return model, optimizer, checkpoint['current_iter']
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

def denormalize(input):
    mean = 0.5
    std = 0.5
    output = input * std +mean
    return output

def validation(model, valid_provider):
    model = model.eval()
    psnrs = []
    ssims = []

    t1 = time.time()
    with torch.no_grad():
        for i in range(len(valid_provider)):
            batch = valid_provider.next()
            noisy_im = batch['label']
            clean_im = batch['unseen_gt']
            input_im = batch['input']
            mask = batch['mask']

            if cfg.TRAIN.if_cuda:
                noisy_im = noisy_im.cuda()
                clean_im = clean_im.cuda()
                input_im = input_im.cuda()
                mask = mask.cuda()

            pred = model(input_im)

            # if data normalization
            if cfg.DATA.if_normalization:
                pred = denormalize(pred)

            # metrics
            results_psnr = psnr(tensor2img(clean_im[0,0]), tensor2img(pred[0,0]))
            results_ssim = ssim(tensor2img(clean_im[0,0]), tensor2img(pred[0,0]))
            psnrs.append(results_psnr)
            ssims.append(results_ssim)
    t2 = time.time()

    ave_psnr = sum(psnrs) / len(psnrs)
    ave_ssim = sum(ssims) / len(ssims)

    return ave_psnr, ave_ssim, t2-t1



def train_loop(cfg, train_provider, valid_provider, model, optimizer, iters, writer):
    rcd_time = []
    sum_time = 0
    sum_loss = 0

    while iters <= cfg.TRAIN.total_iters:
        # train
        model.train()
        iters += 1
        t1 = time.time()
        batch_data = train_provider.next()

        noisy_label = batch_data['label']
        input_noisy = batch_data['input']
        mask = batch_data['mask']
        if cfg.TRAIN.if_cuda:
            noisy_label = noisy_label.cuda()
            input_noisy = input_noisy.cuda()
            mask = mask.cuda()

        # decay learning rate
        if cfg.TRAIN.lr_mode == 'customized':
            optimizer, current_lr = adjust_lr(cfg, iters, optimizer)
        else:
            optimizer, lr_scheduler, current_lr = adjust_lr(cfg, iters, optimizer)

        optimizer.zero_grad()
        model = model.train()

        out = model(input_noisy)

        loss_fuc = criterion_loss(cfg)
        loss = loss_fuc(out * (1 - mask), noisy_label * (1 - mask))

        loss.backward()
        if cfg.TRAIN.lr_mode == 'customized':
            if cfg.TRAIN.weight_decay is not None:
                for group in optimizer.param_groups:
                    for param in group['params']:
                        param.data = param.data.add(-cfg.TRAIN.weight_decay * group['lr'], param.data)
        else:
            lr_scheduler.step()

        optimizer.step()

        sum_loss += loss.item()
        sum_time += time.time() - t1

        # log train
        df = cfg.TRAIN.display_freq
        if iters % df == 0 and iters>0:
            rcd_time.append(sum_time)
            logging.info('step %d, loss = %.4f (lr:%.8f, et:%.2f sec, rd:%.2f h)'
                         % (iters, sum_loss/cfg.TRAIN.display_freq, current_lr, sum_time,
                            (cfg.TRAIN.total_iters - iters) / df * np.mean(np.asarray(rcd_time)) / 3600))
            writer.add_scalar('loss', sum_loss / cfg.TRAIN.display_freq , iters)
            sum_time = 0
            sum_loss = 0

        # valid
        if cfg.TRAIN.if_valid:
            if (iters % cfg.TRAIN.valid_freq == 0) & (iters>0):
                ave_psnr, ave_ssim, valid_cost_time = validation(model, valid_provider)
                logging.info('valid-iter%d, ave_psnr=%.4f, ave_ssim=%.4f, valid_cost_time:%.2f sec' %
                             (iters, ave_psnr, ave_ssim, valid_cost_time))
                writer.add_scalar('valid/psnr', ave_psnr, iters)
                writer.add_scalar('valid/ssim', ave_ssim, iters)

        # save
        if (iters % cfg.TRAIN.save_freq == 0) & (iters>0):
            states = {'current_iter': iters, 'valid_result': (ave_psnr, ave_ssim),
                      'model_weights': model.state_dict()}
            torch.save(states, os.path.join(cfg.save_path, 'model-%06d-PSNR%.4f.pth' % (iters, ave_psnr)))
            torch.save(optimizer.state_dict(), os.path.join(cfg.save_path, 'optim-lastest.pth'))
            logging.info('***************save modol, iters = %d.***************' % (iters))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--cfg', type=str, default='n2v.yml', help='path to config file')
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
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.TRAIN.base_lr, betas=(0.9, 0.999), eps=1e-8, amsgrad=False)
    model, optimizer, init_iters = resume_training(cfg, model, optimizer)

    train_loop(cfg, train_provider, valid_provider, model, optimizer, init_iters, writer)

    writer.close()