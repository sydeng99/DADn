import torch
import numpy as np

def tensor2img(tensor):
    im = (255. * tensor).data.cpu().numpy()
    # clamp
    im[im > 255] = 255
    im[im < 0] = 0
    im = im.astype(np.uint8)
    if im.shape[1] == 3:
        im = im[0]
        im = np.transpose(im, (1, 2, 0))  # c,h,w => h,w,c
    else:
        im = im[0,0]
    return im