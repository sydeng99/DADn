import numpy as np
import torch
import multiprocessing
from joblib import Parallel
from skimage import io
import os
import copy
import random
from torch.utils.data import DataLoader
from .bicubic_imresize import imresize

class Dataset(torch.utils.data.Dataset):
    def __init__(self, cfg, mode='train'):
        self.cfg= cfg
        self.mode = mode
        num_cores=multiprocessing.cpu_count()
        self.parallel = Parallel(n_jobs=num_cores, backend='threading')

        self.data_dir = cfg.DATA.data_dir
        lst_clean_data = cfg.DATA.dataset_clean_name
        lst_noisy_data = cfg.DATA.dataset_noisy_name

        self.lst_clean_data = lst_clean_data
        self.lst_noisy_data = lst_noisy_data

        self.patch_size = cfg.DATA.train_patch_size

        # augmentation
        self.if_RotateFlip_aug = cfg.DATA.if_RotateFlip_aug

        # simulate noise
        self.noise_mode = cfg.DATA.noise_mode
        self.noise_level = cfg.DATA.noise_level
        self.down_scale = cfg.TRAIN.inn_down_scale

        self.clean_dir = os.path.join(self.data_dir, lst_clean_data)
        self.noisy_dir = os.path.join(self.data_dir, lst_noisy_data)
        self.clean_files = os.listdir(self.clean_dir)
        self.noisy_files = os.listdir(self.noisy_dir)
        self.paired = cfg.DATA.paired


    def __getitem__(self, index):
        clean_data = self.clean_files[index % len(self.clean_files)]
        clean_data = io.imread(self.clean_dir + clean_data) /255.
        if self.paired:
            noisy_data = self.noisy_files[(index) % len(self.noisy_files)]
            noisy_data = io.imread(self.noisy_dir + noisy_data) /255.

        else: # default
            random_num = np.random.randint(1, self.__len__(), 1)
            noisy_data = self.noisy_files[(index+random_num[0]) % len(self.noisy_files)]
            noisy_data = io.imread(self.noisy_dir + noisy_data) /255.

        # assert noisy_data.shape == clean_data.shape
        self.data_shape = clean_data.shape

        if self.data_shape[0] > self.patch_size[0]:
            random_y = random.randint(0, self.data_shape[0] - self.patch_size[0])
            random_x = random.randint(0, self.data_shape[1] - self.patch_size[1])

            noisy_img = noisy_data[random_y:random_y+self.patch_size[0], \
                       random_x:random_x+self.patch_size[1]].copy()
            clean_img = clean_data[random_y:random_y+self.patch_size[0], \
                       random_x:random_x+self.patch_size[1]].copy()
        else:
            noisy_img = noisy_data
            clean_img = clean_data

        # add noise
        if self.noise_mode == 'S':
            noise = np.float32(np.random.randn(*(clean_img.shape))) * self.noise_level / 255.
        elif self.noise_mode == 'B':
            stdN = np.random.uniform(self.noise_level[0], self.noise_level[1])
            noise = np.float32(np.random.randn(*(clean_img.shape))) * stdN / 255.
        else:
            raise NotImplementedError('noise_mode must be S or B')

        simu_noisy_img = clean_img + noise
        simu_noisy_img = np.clip(simu_noisy_img, 0, 1)
        if simu_noisy_img.ndim==3:
            [hh, ww, c] = simu_noisy_img.shape
        else:
            [hh,ww] = simu_noisy_img.shape

        lr_img = imresize(clean_img, output_shape=[int(hh//self.down_scale), int(ww//self.down_scale)])
        real_noisy_lr = imresize(noisy_img, output_shape=[int(hh//self.down_scale), int(ww//self.down_scale)])
        simu_noisy_lr = imresize(simu_noisy_img, output_shape=[int(hh//self.down_scale), int(ww//self.down_scale)])
        if self.if_RotateFlip_aug:
            rand_num = np.random.randint(65535)
            noisy_img = self.FlipAug(noisy_img, rand_num)
            simu_noisy_img = self.FlipAug(simu_noisy_img, rand_num)
            clean_img = self.FlipAug(clean_img, rand_num)
            lr_img = self.FlipAug(lr_img, rand_num)
            simu_noisy_lr = self.FlipAug(simu_noisy_lr, rand_num)
            real_noisy_lr = self.FlipAug(real_noisy_lr, rand_num)

        if noisy_img.ndim==3:
            noisy_img = np.transpose(np.ascontiguousarray(noisy_img, dtype=np.float32),axes=(2,0,1))
            simu_noisy_img = np.transpose(np.ascontiguousarray(simu_noisy_img, dtype=np.float32),axes=(2,0,1))
            clean_img = np.transpose(np.ascontiguousarray(clean_img, dtype=np.float32),axes=(2,0,1))
            lr_img = np.transpose(np.ascontiguousarray(lr_img, dtype=np.float32),axes=(2,0,1))
            simu_noisy_lr = np.transpose(np.ascontiguousarray(simu_noisy_lr, dtype=np.float32),axes=(2,0,1))
            real_noisy_lr = np.transpose(np.ascontiguousarray(real_noisy_lr, dtype=np.float32),axes=(2,0,1))
        else:
            noisy_img = np.expand_dims(np.ascontiguousarray(noisy_img, dtype=np.float32), axis=0)
            simu_noisy_img = np.expand_dims(np.ascontiguousarray(simu_noisy_img, dtype=np.float32), axis=0)
            clean_img = np.expand_dims(np.ascontiguousarray(clean_img, dtype=np.float32), axis=0)
            lr_img = np.expand_dims(np.ascontiguousarray(lr_img, dtype=np.float32), axis=0)
            simu_noisy_lr = np.expand_dims(np.ascontiguousarray(simu_noisy_lr, dtype=np.float32), axis=0)
            real_noisy_lr = np.expand_dims(np.ascontiguousarray(real_noisy_lr, dtype=np.float32), axis=0)

        data = {'noisy_img':noisy_img, 'clean_img': clean_img, 'simu_noisy_img': simu_noisy_img, 'clean_lr':lr_img,
                    'simu_noisy_lr':simu_noisy_lr, 'real_noisy_lr': real_noisy_lr}

        return data

    def __len__(self):
        return len(self.clean_files)


    def FlipAug(self, img, random):
        if img.ndim==3 and img.shape[0]!=3 and img.shape[2]!=3:  # [z,x,y]
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
        elif img.ndim==3:
            if random % 8 ==0:
                return img
            elif random % 8==1:
                return img[::-1, ..., :]
            elif random % 8 ==2:
                return img[:, ::-1, :]
            elif random % 8==3:
                return img[::-1, ::-1, :]
            elif random % 8==4:
                return np.transpose(img, axes=(1,0,2))
            elif random % 8==5:
                return np.transpose(img[::-1, ..., :], axes=(1,0,2))
            elif random % 8==6:
                return np.transpose(img[:, ::-1, :], axes=(1,0,2))
            elif random % 8==7:
                return np.transpose(img[::-1, ::-1,: ], axes=(1,0,2))
        else:
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

class Provider(object):
    def __init__(self, stage, cfg):
        self.stage = stage
        if self.stage == 'train':
            self.data = Dataset(cfg)
            self.batch_size = cfg.TRAIN.batch_size
            self.num_workers = cfg.TRAIN.num_workers
        elif self.stage =='test':
            self.data = TestData(cfg)
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
                           shuffle=True, drop_last=True, pin_memory=True))
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
                else:
                    if 'noisy_img' in batch:
                        batch[1] = batch['noisy_img'].cuda()
                        batch[2] = batch['clean_img'].cuda()
                        batch[3] = batch['simu_noisy_img'].cuda()

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
                else:
                    if 'noisy_img' in batch:
                        batch[1] = batch['noisy_img'].cuda()
                        batch[2] = batch['clean_img'].cuda()
                        batch[3] = batch['simu_noisy_img'].cuda()
            return batch


class ValidData(Dataset):
    def __init__(self, cfg):
        super(ValidData, self).__init__(cfg)
        num_cores=multiprocessing.cpu_count()
        self.parallel = Parallel(n_jobs=num_cores, backend='threading')
        self.data_dir = cfg.DATA.data_dir

        self.clean_dir = os.path.join(self.data_dir, cfg.DATA.testset_clean_name)
        self.noisy_dir = os.path.join(self.data_dir, cfg.DATA.testset_noisy_name)
        self.clean_files = os.listdir(self.clean_dir)
        self.noisy_files = os.listdir(self.noisy_dir)
        self.random_num = np.random.randint(1, self.__len__(), 1)

    def __getitem__(self, index):
        # source
        clean_data_i = self.clean_files[index % len(self.clean_files)]
        clean_data_i = io.imread(self.clean_dir + clean_data_i) /255.
        # target
        random_num = np.random.randint(1, self.__len__(), 1)
        noisy_data_j = self.noisy_files[(index+random_num[0]) % len(self.noisy_files)]
        noisy_data_j = io.imread(self.noisy_dir + noisy_data_j) /255.
        clean_data_j = self.clean_files[(index+random_num[0]) % len(self.clean_files)]
        clean_data_j = io.imread(self.clean_dir + clean_data_j) /255.

        # add noise
        if self.noise_mode == 'S':
            noise = np.float32(np.random.randn(*(clean_data_i.shape))) * self.noise_level / 255.
        elif self.noise_mode == 'B':
            stdN = np.random.uniform(self.noise_level[0], self.noise_level[1])
            noise = np.float32(np.random.randn(*(clean_data_i.shape))) * stdN / 255.
        else:
            raise NotImplementedError('noise_mode must be S or B')

        simu_noisy_img_i = clean_data_i + noise
        simu_noisy_img_i = np.clip(simu_noisy_img_i, 0, 1)
        if simu_noisy_img_i.ndim==3:
            [hh, ww, c] = simu_noisy_img_i.shape
        else:
            [hh, ww] = simu_noisy_img_i.shape

        simu_noisy_lr = imresize(simu_noisy_img_i, output_shape=[int(hh // self.down_scale), int(ww // self.down_scale)])
        real_noisy_lr = imresize(noisy_data_j, output_shape=[int(hh // self.down_scale), int(ww // self.down_scale)])

        if clean_data_i.ndim==3:
            clean_data_i = np.transpose(np.ascontiguousarray(clean_data_i, dtype=np.float32),axes=(2,0,1))
            simu_noisy_img_i = np.transpose(np.ascontiguousarray(simu_noisy_img_i, dtype=np.float32),axes=(2,0,1))
            clean_data_j = np.transpose(np.ascontiguousarray(clean_data_j, dtype=np.float32),axes=(2,0,1))
            noisy_data_j = np.transpose(np.ascontiguousarray(noisy_data_j, dtype=np.float32),axes=(2,0,1))
            simu_noisy_lr = np.transpose(np.ascontiguousarray(simu_noisy_lr, dtype=np.float32),axes=(2,0,1))
            real_noisy_lr = np.transpose(np.ascontiguousarray(real_noisy_lr, dtype=np.float32),axes=(2,0,1))
        else:
            clean_data_i = np.expand_dims(np.ascontiguousarray(clean_data_i, dtype=np.float32), axis=0)
            simu_noisy_img_i = np.expand_dims(np.ascontiguousarray(simu_noisy_img_i, dtype=np.float32), axis=0)
            clean_data_j = np.expand_dims(np.ascontiguousarray(clean_data_j, dtype=np.float32), axis=0)
            noisy_data_j = np.expand_dims(np.ascontiguousarray(noisy_data_j, dtype=np.float32), axis=0)
            simu_noisy_lr = np.expand_dims(np.ascontiguousarray(simu_noisy_lr, dtype=np.float32), axis=0)
            real_noisy_lr = np.expand_dims(np.ascontiguousarray(real_noisy_lr, dtype=np.float32), axis=0)

        data = {'clean_data_i':clean_data_i, 'simu_noisy_img_i': simu_noisy_img_i,
                'clean_data_j': clean_data_j, 'noisy_data_j': noisy_data_j,
                'simu_noisy_lr': simu_noisy_lr, 'real_noisy_lr': real_noisy_lr, 'random_num': self.random_num}
        return data

    def __len__(self):
        return len(self.clean_files)



class TestData(Dataset):
    def __init__(self, cfg):
        super(TestData, self).__init__(cfg)
        num_cores=multiprocessing.cpu_count()
        self.parallel = Parallel(n_jobs=num_cores, backend='threading')
        self.data_dir = cfg.DATA.data_dir

        self.clean_dir = os.path.join(self.data_dir, cfg.TEST.testset_clean_name)
        self.noisy_dir = os.path.join(self.data_dir, cfg.TEST.testset_noisy_name)
        self.clean_files = os.listdir(self.clean_dir)
        self.noisy_files = os.listdir(self.noisy_dir)
        self.random_num = np.random.randint(1, self.__len__(), 1)

    def __getitem__(self, index):
        # source
        noisy_data_i = self.clean_files[(index+self.random_num[0]) % len(self.noisy_files)]
        noisy_data_i = io.imread(self.noisy_dir + noisy_data_i) /255.
        # target
        noisy_data_j = str(index+1).zfill(4)+'.png'
        noisy_data_j = io.imread(self.noisy_dir + noisy_data_j) /255.
        clean_data_j = str(index+1).zfill(4)+'.png'
        clean_data_j = io.imread(self.clean_dir + clean_data_j) /255.

        # add noise
        if self.noise_mode == 'S':
            noise = np.float32(np.random.randn(*(clean_data_j.shape))) * self.noise_level / 255.
        elif self.noise_mode == 'B':
            stdN = np.random.uniform(self.noise_level[0], self.noise_level[1])
            noise = np.float32(np.random.randn(*(clean_data_j.shape))) * stdN / 255.
        else:
            raise NotImplementedError('noise_mode must be S or B')

        simu_noisy_img_j = clean_data_j + noise
        simu_noisy_img_j = np.clip(simu_noisy_img_j, 0, 1)
        if simu_noisy_img_j.ndim==3:
            [hh, ww, c] = simu_noisy_img_j.shape
        else:
            [hh, ww] = simu_noisy_img_j.shape

        if clean_data_j.ndim==3:
            noisy_data_i = np.transpose(np.ascontiguousarray(noisy_data_i, dtype=np.float32),axes=(2,0,1))
            simu_noisy_img_j = np.transpose(np.ascontiguousarray(simu_noisy_img_j, dtype=np.float32),axes=(2,0,1))
            clean_data_j = np.transpose(np.ascontiguousarray(clean_data_j, dtype=np.float32),axes=(2,0,1))
            noisy_data_j = np.transpose(np.ascontiguousarray(noisy_data_j, dtype=np.float32),axes=(2,0,1))
        else:
            noisy_data_i = np.expand_dims(np.ascontiguousarray(noisy_data_i, dtype=np.float32), axis=0)
            simu_noisy_img_j = np.expand_dims(np.ascontiguousarray(simu_noisy_img_j, dtype=np.float32), axis=0)
            clean_data_j = np.expand_dims(np.ascontiguousarray(clean_data_j, dtype=np.float32), axis=0)
            noisy_data_j = np.expand_dims(np.ascontiguousarray(noisy_data_j, dtype=np.float32), axis=0)

        data = {'noisy_data_i':noisy_data_i, 'simu_noisy_img_j': simu_noisy_img_j,
                'clean_data_j': clean_data_j, 'noisy_data_j': noisy_data_j}
        return data

    def __len__(self):
        return len(self.clean_files)

