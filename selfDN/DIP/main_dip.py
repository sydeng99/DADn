from __future__ import print_function
import matplotlib.pyplot as plt
from dip_model import get_net
from utils import crop_image, get_image, pil_to_np, np_to_torch, get_noise, torch_to_np, get_params, optimize
import torch
import torch.optim
import numpy as np
import os
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
from skimage import io
import argparse
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark =True
dtype = torch.cuda.FloatTensor
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
imsize =-1
PLOT = True
INPUT = 'noise'  # 'meshgrid'
pad = 'reflection'
OPT_OVER = 'net'  # 'net,input'

reg_noise_std = 1. / 20.  # set to 1./20. for sigma=50
LR = 0.01

OPTIMIZER = 'adam'  # 'LBFGS'
show_every = 100
exp_weight = 0.99

num_iter = 2000
input_depth = 32
figsize = 4

parser = argparse.ArgumentParser()
parser.add_argument('--savepath', type=str, default='experiments/dip/')
parser.add_argument('--datapath', type=str, default='dataset/validset/')
opt, _ = parser.parse_known_args()
savepath = opt.savepath

if not os.path.exists(savepath):
    os.makedirs(savepath)

files = os.listdir(opt.datapath+'noisy/')
for name in files:
    frame_clean = opt.datapath + 'clean/' +name
    img_pil = crop_image(get_image(frame_clean, imsize)[0], d=32)
    img_np = pil_to_np(img_pil)

    frame_noisy = opt.datapath + 'noisy/' +name
    img_noisy_pil = crop_image(get_image(frame_noisy, imsize)[0], d=32)
    img_noisy_np = pil_to_np(img_noisy_pil)

    '''
    setup
    '''
    net = get_net(input_depth, 'skip', pad,
                  skip_n33d=64,
                  skip_n33u=64,
                  skip_n11=4,
                  num_scales=5,
                  upsample_mode='bilinear').type(dtype)

    net_input = get_noise(input_depth, INPUT, (img_pil.size[1], img_pil.size[0])).type(dtype).detach()

    # Compute number of parameters
    s = sum([np.prod(list(p.size())) for p in net.parameters()])
    print('Number of params: %d' % s)

    # Loss
    mse = torch.nn.MSELoss().type(dtype)
    img_noisy_torch = np_to_torch(img_noisy_np).type(dtype)

    '''
    optimize
    '''
    net_input_saved = net_input.detach().clone()
    noise = net_input.detach().clone()
    out_avg = None
    last_net = None
    psrn_noisy_last = 0

    i = 0


    def closure():
        global i, out_avg, psrn_noisy_last, last_net, net_input

        if reg_noise_std > 0:
            net_input = net_input_saved + (noise.normal_() * reg_noise_std)

        out = net(net_input)

        # Smoothing
        if out_avg is None:
            out_avg = out.detach()
        else:
            out_avg = out_avg * exp_weight + out.detach() * (1 - exp_weight)

        total_loss = mse(out, img_noisy_torch)
        total_loss.backward()

        psrn_noisy = compare_psnr(img_noisy_np, out.detach().cpu().numpy()[0])
        psrn_gt = compare_psnr(img_np, out.detach().cpu().numpy()[0])
        psrn_gt_sm = compare_psnr(img_np, out_avg.detach().cpu().numpy()[0])

        # Note that we do not have GT for the "snail" example
        # So 'PSRN_gt', 'PSNR_gt_sm' make no sense
        print('Iteration %05d    Loss %f   PSNR_noisy: %f   PSRN_gt: %f PSNR_gt_sm: %f' % (
        i, total_loss.item(), psrn_noisy, psrn_gt, psrn_gt_sm), '\r', end='')

        # Backtracking
        if i % show_every:
            if psrn_noisy - psrn_noisy_last < -5:
                print('Falling back to previous checkpoint.')

                for new_param, net_param in zip(last_net, net.parameters()):
                    net_param.data.copy_(new_param.cuda())

                return total_loss * 0
            else:
                last_net = [x.detach().cpu() for x in net.parameters()]
                psrn_noisy_last = psrn_noisy

        i += 1

        return total_loss


    p = get_params(OPT_OVER, net, net_input)
    optimize(OPTIMIZER, p, closure, LR, num_iter)

    out_np = torch_to_np(net(net_input))
    ssim_v = compare_ssim(out_np[0], img_np[0])
    psnr_v = compare_psnr(out_np[0], img_np[0])
    print('\n')
    print(name, '%.4f'%psnr_v, '%.4f'%ssim_v)

    out_np = out_np[0]*255.
    out = out_np.astype('uint8')
    io.imsave(savepath + name, out)


