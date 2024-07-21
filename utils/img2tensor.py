import numpy as np
from skimage import io
import torch

def img2tensor(img):
    img = (img / 255.).astype('float32')
    if img.ndim ==2:
        img = np.expand_dims(np.expand_dims(img, axis = 0),axis=0)
    else:
        img = np.transpose(img, (2, 0, 1))  # C, H, W
        img = np.expand_dims(img, axis=0)
    img = np.ascontiguousarray(img, dtype=np.float32)
    tensor = torch.from_numpy(img)
    return tensor