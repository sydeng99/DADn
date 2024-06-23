import numpy as np
import torch
import multiprocessing
from joblib import Parallel
from skimage import io
import os
import copy
import random
from torch.utils.data import DataLoader


class Dataset(torch.utils.data.Dataset):
    def __init__(self, cfg, mode='train'):
        self.cfg= cfg
        self.mode = mode
        num_cores=multiprocessing.cpu_count()
        self.parallel = Parallel(n_jobs=num_cores, backend='threading')

        self.data_dir = cfg.DATA.data_dir
        lst_noisy_data = cfg.DATA.dataset_noisy_name

        self.lst_noisy_data = lst_noisy_data

        self.patch_size = cfg.DATA.train_patch_size

        # augmentation
        self.if_RotateFlip_aug = cfg.DATA.if_RotateFlip_aug

        # if use spot-blind network
        self.if_sbn = cfg.DATA.if_spot_blind_network
        self.ratio = cfg.DATA.ratio
        self.size_window = cfg.DATA.size_window

        self.noisy_dir = os.path.join(self.data_dir, lst_noisy_data)
        self.noisy_files = os.listdir(self.noisy_dir)

        # if data normalization
        self.if_normalization = cfg.DATA.if_normalization


    def __getitem__(self, index):
        noisy_data = self.noisy_files[index % len(self.noisy_files)]
        noisy_data = io.imread(self.noisy_dir + noisy_data) /255.

        self.data_shape = noisy_data.shape

        if self.data_shape[0] > self.patch_size[0]:
            random_y = random.randint(0, self.data_shape[0] - self.patch_size[0])
            random_x = random.randint(0, self.data_shape[1] - self.patch_size[1])

            noisy_img = noisy_data[random_y:random_y+self.patch_size[0], \
                       random_x:random_x+self.patch_size[1]].copy()
        else:
            noisy_img = noisy_data


        if self.if_RotateFlip_aug:
            rand_num = np.random.randint(65535)
            noisy_img = self.FlipAug(noisy_img, rand_num)

        if self.if_sbn:
            input, mask = self.generate_mask(copy.deepcopy(noisy_img))

            noisy_img = np.expand_dims(np.ascontiguousarray(noisy_img, dtype=np.float32), axis=0)
            input = np.expand_dims(np.ascontiguousarray(input, dtype=np.float32), axis=0)
            mask = np.expand_dims(np.ascontiguousarray(mask, dtype=np.float32), axis=0)

            ones = np.ones_like(mask)
            ones = np.expand_dims(np.ascontiguousarray(ones, dtype=np.float32), axis=0)

            if self.if_normalization:
                noisy_img = self.normalize(noisy_img)
                input = self.normalize(input)
                mask = self.normalize(mask)
                ones = self.normalize(ones)

            data = {'label': noisy_img, 'input': input, 'mask': mask, 'unseen_gt': ones} 
        return data

    def __len__(self):
        return len(self.noisy_files)


    def FlipAug(self, img, random):
        if img.ndim==3:  # [z,x,y]
            if random % 8 ==0:
                return img
            elif random % 8==1:
                return img[:, ::-1, ...]
            elif random % 8 ==2:
                return img[:, :, ::-1]
            elif random % 8==3:
                return img[:, ::-1, ::-1]
            elif random % 8==4:
                return np.transpose(img, axes=(0, 2, 1))
            elif random % 8==5:
                return np.transpose(img[:, ::-1, ...], axes=(0, 2, 1))
            elif random % 8==6:
                return np.transpose(img[:, :, ::-1], axes=(0, 2, 1))
            elif random % 8==7:
                return np.transpose(img[:, ::-1, ::-1], axes=(0, 2, 1))
        elif img.ndim==2:
            if random % 8 ==0:
                return img
            elif random % 8==1:
                return img[::-1, ...]
            elif random % 8 ==2:
                return img[:, ::-1]
            elif random % 8==3:
                return img[::-1, ::-1]
            elif random % 8==4:
                return np.transpose(img, axes=(1,0))
            elif random % 8==5:
                return np.transpose(img[::-1, ...], axes=(1,0))
            elif random % 8==6:
                return np.transpose(img[:, ::-1], axes=(1,0))
            elif random % 8==7:
                return np.transpose(img[::-1, ::-1], axes=(1,0))

    def generate_mask(self, input):
        ratio = self.ratio
        size_window = self.size_window
        size_data = self.patch_size
        num_sample = int(size_data[0] * size_data[1] * ratio)

        mask = np.ones(size_data)
        output = input

        idy_msk = np.random.randint(0, size_data[0], num_sample)
        idx_msk = np.random.randint(0, size_data[1], num_sample)

        idy_neigh = np.random.randint(-size_window[0] // 2 + size_window[0] % 2, size_window[0] // 2 + size_window[0] % 2, num_sample)
        idx_neigh = np.random.randint(-size_window[1] // 2 + size_window[1] % 2, size_window[1] // 2 + size_window[1] % 2, num_sample)

        idy_msk_neigh = idy_msk + idy_neigh
        idx_msk_neigh = idx_msk + idx_neigh

        idy_msk_neigh = idy_msk_neigh + (idy_msk_neigh < 0) * size_data[0] - (idy_msk_neigh >= size_data[0]) * size_data[0]
        idx_msk_neigh = idx_msk_neigh + (idx_msk_neigh < 0) * size_data[1] - (idx_msk_neigh >= size_data[1]) * size_data[1]

        id_msk = (idy_msk, idx_msk)
        id_msk_neigh = (idy_msk_neigh, idx_msk_neigh)

        output[id_msk] = input[id_msk_neigh]
        mask[id_msk] = 0.0

        return output, mask

    def normalize(self, input):
        mean = 0.5
        std = 0.5
        output = (input- mean) / std
        return output

    def denormalize(self, input):
        mean = 0.5
        std = 0.5
        output = input * std +mean
        return output


class Provider(object):
    def __init__(self, stage, cfg):
        self.stage = stage
        if self.stage == 'train':
            self.data = Dataset(cfg)
            self.batch_size = cfg.TRAIN.batch_size
            self.num_workers = cfg.TRAIN.num_workers
        else:
            self.data=ValidData(cfg)
        self.is_cuda = cfg.TRAIN.if_cuda
        self.data_iter = None
        self.iteration = 0
        self.epoch = 1

    def __len__(self):
        return self.data.__len__()

    def build(self):
        if self.stage == 'train':
            self.data_iter = iter(
                DataLoader(dataset=self.data, batch_size=self.batch_size, num_workers=self.num_workers,
                           shuffle=True, drop_last=False, pin_memory=True))
        else:
            self.data_iter = iter(DataLoader(dataset=self.data, batch_size=1, num_workers=0,
                                             shuffle=False, drop_last=False, pin_memory=True))

    def next(self):
        if self.data_iter is None:
            self.build()
        try:
            batch = self.data_iter.next()
            self.iteration += 1
            if self.is_cuda:
                if 'mask' in batch:
                    batch[1] = batch['label'].cuda()
                    batch[2] = batch['input'].cuda()
                    batch[3] = batch['mask'].cuda()
                    batch[4] = batch['unseen_gt'].cuda()

            return batch
        except StopIteration:
            self.epoch += 1
            self.build()
            self.iteration += 1
            batch = self.data_iter.next()
            if self.is_cuda:
                if 'mask' in batch:
                    batch[1] = batch['label'].cuda()
                    batch[2] = batch['input'].cuda()
                    batch[3] = batch['mask'].cuda()
                    batch[4] = batch['unseen_gt'].cuda()
            return batch


class ValidData(Dataset):
    def __init__(self, cfg):
        super(ValidData, self).__init__(cfg)
        num_cores=multiprocessing.cpu_count()
        self.parallel = Parallel(n_jobs=num_cores, backend='threading')
        self.data_dir = cfg.DATA.data_dir

        # if use spot-blind network
        self.if_sbn = cfg.DATA.if_spot_blind_network
        self.ratio = cfg.DATA.ratio
        # self.ratio = 0
        self.size_window = cfg.DATA.size_window

        self.clean_dir = os.path.join(self.data_dir, cfg.DATA.testset_clean_name)
        self.noisy_dir = os.path.join(self.data_dir, cfg.DATA.testset_noisy_name)
        self.clean_files = os.listdir(self.clean_dir)
        self.noisy_files = os.listdir(self.noisy_dir)

        # if data normalization
        self.if_normalization = cfg.DATA.if_normalization

    def __getitem__(self, index):
        clean_data = self.clean_files[index % len(self.clean_files)]
        clean_data = io.imread(self.clean_dir + clean_data) /255.
        noisy_data = self.noisy_files[index % len(self.noisy_files)]
        noisy_data = io.imread(self.noisy_dir + noisy_data) /255.
        self.patch_size = noisy_data.shape

        if self.if_sbn:
            input, mask = self.generate_mask(copy.deepcopy(noisy_data))

            input = np.expand_dims(np.ascontiguousarray(input, dtype=np.float32), axis=0)
            mask = np.expand_dims(np.ascontiguousarray(mask, dtype=np.float32), axis=0)
            noisy_img = np.expand_dims(np.ascontiguousarray(noisy_data, dtype=np.float32), axis=0)
            clean_img = np.expand_dims(np.ascontiguousarray(clean_data, dtype=np.float32), axis=0)


            if self.if_normalization:
                noisy_img = self.normalize(noisy_img)
                input = self.normalize(input)
                mask = self.normalize(mask)


            data = {'label': noisy_img, 'input': input, 'mask': mask, 'unseen_gt': clean_img}
        return data

    def __len__(self):
        return len(self.clean_files)


    def normalize(self, input):
        mean = 0.5
        std = 0.5
        output = (input- mean) / std
        return output

    def denormalize(self, input):
        mean = 0.5
        std = 0.5
        output = input * std +mean
        return output

    def generate_mask(self, input):
        ratio = self.ratio
        size_window = self.size_window
        size_data = self.patch_size
        num_sample = int(size_data[0] * size_data[1] * ratio)  # ration: mask ratio

        mask = np.ones(size_data)
        output = input

        idy_msk = np.random.randint(0, size_data[0], num_sample)
        idx_msk = np.random.randint(0, size_data[1], num_sample)

        idy_neigh = np.random.randint(-size_window[0] // 2 + size_window[0] % 2, size_window[0] // 2 + size_window[0] % 2, num_sample) 
        idx_neigh = np.random.randint(-size_window[1] // 2 + size_window[1] % 2, size_window[1] // 2 + size_window[1] % 2, num_sample)

        idy_msk_neigh = idy_msk + idy_neigh
        idx_msk_neigh = idx_msk + idx_neigh

        idy_msk_neigh = idy_msk_neigh + (idy_msk_neigh < 0) * size_data[0] - (idy_msk_neigh >= size_data[0]) * size_data[0] 
        idx_msk_neigh = idx_msk_neigh + (idx_msk_neigh < 0) * size_data[1] - (idx_msk_neigh >= size_data[1]) * size_data[1]

        id_msk = (idy_msk, idx_msk)
        id_msk_neigh = (idy_msk_neigh, idx_msk_neigh)

        output[id_msk] = input[id_msk_neigh]

        return output, mask


if __name__ =='__main__':
    import yaml
    import time
    from attrdict import AttrDict
    from skimage.metrics import peak_signal_noise_ratio as psnr
    from skimage.metrics import structural_similarity as ssim

    seed = 555
    np.random.seed(seed)
    random.seed(seed)
    cfg_file = 'n2v.yml'
    with open('../config/' + cfg_file, 'r') as f:
        cfg = AttrDict(yaml.load(f, Loader=yaml.FullLoader))

    out_path = os.path.join('./', 'data_temp')
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    data = Dataset(cfg)
    data = ValidData(cfg)
    t = time.time()

    for i in range(0, 10):
        t1 = time.time()

        # print(data[10])

        dict = iter(data).__next__()

        if 'mask' in dict:
            img_noise = dict['label']
            img_input = dict['input']
            mask = dict['mask']
            mask = (np.squeeze(mask) * 255.).astype('uint8')
            unseen_gt = dict['unseen_gt']
            unseen_gt = (np.squeeze(unseen_gt) * 255.).astype('uint8')
            img_noise = (np.squeeze(img_noise) * 255.).astype('uint8')
            img_input = (np.squeeze(img_input) * 255.).astype('uint8')
        else:
            img_noise = dict['noisy_img']
            img_clean = dict['clean_img']
            img_noise = (np.squeeze(img_noise) * 255.).astype('uint8')
            img_clean = (np.squeeze(img_clean) * 255.).astype('uint8')

        print('single cost time: ', time.time() - t1)

        if 'mask' in dict:
            print('PSBR=', psnr(img_noise, unseen_gt))
            print('SSIM=', ssim(img_noise, unseen_gt))
        else:
            print('PSNR=',psnr(img_noise, img_clean))
            print('SSIM=',ssim(img_noise, img_clean))

        ones = (np.ones_like(img_noise)*255.).astype('uint8')
        if 'mask' in dict:
            im_cat0 = np.concatenate([unseen_gt, mask], axis=0)
            im_cat1 = np.concatenate([img_noise, img_input], axis=0)
            im_cat = np.concatenate([im_cat0, im_cat1], axis=1)
        else:
            im_cat = np.concatenate([img_noise, img_clean], axis=0)

        io.imsave(os.path.join(out_path, str(i).zfill(4) + '.png'), im_cat.astype('uint8'))

    print(time.time() - t, 's')

    from torch.utils.data import DataLoader

    dd = iter(DataLoader(dataset=data, batch_size=4, shuffle=False, num_workers=cfg.TRAIN.num_workers)) # cfg.TRAIN.num_workers
    # for i, val_data in enumerate(dd):
    #     print(i, val_data['clean_img'].shape)   # [4, 256, 256]

    # train_provider = Provider('train', cfg)
    # for i in range(3):
    #     batch = train_provider.next()
    #     print(batch['clean_img'].shape)

    def tensor2img(tensor):
        im = (255. * tensor).data.cpu().numpy()
        # clamp
        im[im > 255] = 255
        im[im < 0] = 0
        im = im.astype(np.uint8)
        return im

    valid_provider = Provider('test', cfg)
    # ValidData =
    t1 = time.time()
    for i in range(len(valid_provider)):
        batch = valid_provider.next()
        # print(batch['clean_img'].shape)
        # print(batch['clean_img'].cuda().dtype)
        print(psnr(tensor2img(batch['input'][0,0]), tensor2img(batch['unseen_gt'][0,0])))
        print(ssim(tensor2img(batch['input'][0,0]), tensor2img(batch['unseen_gt'][0,0])))
        print(i)
    t2 = time.time()
    print(t2-t1)   # 162s