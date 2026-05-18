import os
import random
import numpy as np
import cv2
from torch.utils.data import Dataset
from utils.common import hwc_to_chw, read_img



def augment(imgs=[], size=256, edge_decay=0., data_augment=True):
    H, W, _ = imgs[0].shape
    Hc, Wc = [size, size]

    # simple re-weight for the edge
    if random.random() < Hc / H * edge_decay:
        Hs = 0 if random.randint(0, 1) == 0 else H - Hc
    else:
        Hs = random.randint(0, H - Hc)

    if random.random() < Wc / W * edge_decay:
        Ws = 0 if random.randint(0, 1) == 0 else W - Wc
    else:
        Ws = random.randint(0, W - Wc)

    for i in range(len(imgs)):
        imgs[i] = imgs[i][Hs:(Hs + Hc), Ws:(Ws + Wc), :]

    if data_augment:
        # horizontal flip
        if random.randint(0, 1) == 1:
            for i in range(len(imgs)):
                imgs[i] = np.flip(imgs[i], axis=1)

        # bad data augmentations for outdoor dehazing
        rot_deg = random.randint(0, 3)
        for i in range(len(imgs)):
            imgs[i] = np.rot90(imgs[i], rot_deg, (0, 1))

    return imgs


def align(imgs=[], size=256):
    H, W, _ = imgs[0].shape
    Hc, Wc = [size, size]

    Hs = (H - Hc) // 2
    Ws = (W - Wc) // 2
    for i in range(len(imgs)):
        imgs[i] = imgs[i][Hs:(Hs + Hc), Ws:(Ws + Wc), :]

    return imgs

def align_crop(imgs=[]):
    H, W, _ = imgs[0].shape
    W = W - W % 32
    H = H - H % 32
    for i in range(len(imgs)):
        imgs[i] = imgs[i][0:H, 0:W, :]
    return imgs


class Test_Syn_Loader(Dataset):
    def __init__(self, root_dir, size=256, edge_decay=0, data_augment=True, cache_memory=False):
        self.size = size
        self.edge_decay = edge_decay
        self.data_augment = data_augment

        self.root_dir = root_dir
        self.real_hazy_imgs = sorted(os.listdir(os.path.join(self.root_dir, 'hazy')))
        self.img_num = len(self.real_hazy_imgs)

        self.clear_imgs = sorted(os.listdir(os.path.join(self.root_dir, "clear")))
        self.clear_num = len(self.clear_imgs)

        self.cache_memory = cache_memory
        self.real_hazy_files = {}
        self.clear_img_files = {}

    def __len__(self):
        return self.img_num

    def __getitem__(self, idx):
        cv2.setNumThreads(0)
        cv2.ocl.setUseOpenCL(False)

        # select a image pair
        real_hazy_name = self.real_hazy_imgs[idx]

        a = real_hazy_name.split("_")
        clear_name = real_hazy_name


        # read images
        if real_hazy_name not in self.real_hazy_files and clear_name not in self.clear_img_files:
            real_hazy_img = read_img(os.path.join(self.root_dir, 'hazy', real_hazy_name), to_float=False)
            clear_img = read_img(os.path.join(self.root_dir, 'clear', clear_name), to_float=False)


            # cache in memory if specific (uint8 to save memory), need num_workers=0
            if self.cache_memory:
                self.real_hazy_files[real_hazy_img] = real_hazy_img
                self.clear_img_files[clear_img] = clear_img

        else:
            # load cached images
            real_hazy_img = self.real_hazy_files[real_hazy_name]
            clear_img = self.clear_img_files[clear_name]


        # [0, 1] to [-1, 1]
        real_hazy_img = real_hazy_img.astype('float32') / 255.0
        clear_img = clear_img.astype('float32') / 255.0


        # data augmentation
        [real_hazy_img, clear_img] = align_crop([real_hazy_img, clear_img])

        return {'hazy': hwc_to_chw(real_hazy_img), 'clear_img': hwc_to_chw(clear_img),
				'image_name': real_hazy_name, 'clear_name': clear_name}


class UnpairedLoader(Dataset):
    def __init__(self, root_dir, size=256, edge_decay=0, data_augment=True, cache_memory=False):
        self.size = size
        self.edge_decay = edge_decay
        self.data_augment = data_augment

        self.root_dir = root_dir
        self.hazy_img_names = sorted(os.listdir(os.path.join(self.root_dir, 'trainA')))
        self.clear_img_names = sorted(os.listdir(os.path.join(self.root_dir, 'trainB')))
        self.hazy_size = len(self.hazy_img_names)  # get the size of dataset A
        self.clear_size = len(self.clear_img_names)

        self.cache_memory = cache_memory
        self.real_hazy_files = {}
        self.clear_img_files = {}
        self.depth_img_files = {}

    def __len__(self):
        return max(self.hazy_size, self.clear_size)

    def __getitem__(self, idx):
        cv2.setNumThreads(0)
        cv2.ocl.setUseOpenCL(False)

        clear_img_name = self.clear_img_names[idx % self.clear_size]
        hazy_img_name = self.hazy_img_names[idx % self.hazy_size]
        depth_name = clear_img_name

        # read images
        if hazy_img_name not in self.real_hazy_files:
            real_hazy_img = read_img(os.path.join(self.root_dir, 'trainA', hazy_img_name), to_float=False)
            clear_img = read_img(os.path.join(self.root_dir, 'trainB', clear_img_name), to_float=False)
            depth_img = read_img(os.path.join(self.root_dir, 'trainB_depth', depth_name), to_float=False)
            # cache in memory if specific (uint8 to save memory), need num_workers=0
            if self.cache_memory:
                self.real_hazy_files[hazy_img_name] = real_hazy_img
                self.clear_img_files[clear_img] = clear_img
                self.depth_img_files[depth_img] = depth_img
        else:
            # load cached images
            real_hazy_img = self.real_hazy_files[hazy_img_name]
            clear_img = self.clear_img_files[clear_img_name]
            depth_img = self.depth_img_files[clear_img_name]

        # [0, 1]
        real_hazy_img = real_hazy_img.astype('float32') / 255.0
        clear_img = clear_img.astype('float32') / 255.0
        depth_img = depth_img.astype('float32') / 255.0

        # data augmentation
        [real_hazy_img] = augment(imgs=[real_hazy_img], size=self.size, edge_decay=self.edge_decay, data_augment=True)
        [clear_img, depth_img] = augment(imgs=[clear_img, depth_img], size=self.size, edge_decay=self.edge_decay, data_augment=True)

        return {'hazy': hwc_to_chw(real_hazy_img), 'clear': hwc_to_chw(clear_img),'depth': hwc_to_chw(depth_img),'hazyname': hazy_img_name, 'clearname': clear_img_name}



class SingleLoader(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.img_names = sorted(os.listdir(self.root_dir))
        self.img_num = len(self.img_names)

    def __len__(self):
        return self.img_num

    def __getitem__(self, idx):
        cv2.setNumThreads(0)
        cv2.ocl.setUseOpenCL(False)

        img_name = self.img_names[idx]
        img = read_img(os.path.join(self.root_dir, img_name))
        [img] = align_crop([img])

        return {'img': hwc_to_chw(img), 'filename': img_name}
