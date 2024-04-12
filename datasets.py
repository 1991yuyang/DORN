from torch.utils import data
import os
import numpy as np
from numpy import random as rd
from torchvision import transforms as T
from utils import SID
import torch as t
import cv2
from torch.nn import functional as F
from torchvision.transforms import functional as TF
from PIL import Image


"""
data_root_dir
    train
        image
            1.png
            2.png
        depth
            1.npy
            2.npy
    val
        image
            1.png
            2.png
        depth
            1.npy
            2.npy
"""


class MySet(data.Dataset):

    def __init__(self, data_root_dir, is_train, image_crop_size, alpha, beta, K, scale_max_factor, input_image_size):
        """

        :param data_root_dir:
        :param is_train:
        :param image_crop_size: (w, h)
        :param alpha: min depth of depth interval [alpha, beta]
        :param beta: max depth of depth interval [alpha, beta]
        :param K: depth interval[alpha, beta] needs to be discretized into K sub-intervals
        :param scale_max_factor: max ratio of scale data augmentation
        :param input_image_size: (w, h), first resize original image to this size, then cropping and training
        """
        if is_train:
            self.img_dir = os.path.join(data_root_dir, "train", "image")
            self.depth_dir = os.path.join(data_root_dir, "train", "depth")
        else:
            self.img_dir = os.path.join(data_root_dir, "val", "image")
            self.depth_dir = os.path.join(data_root_dir, "val", "depth")
        self.is_train = is_train
        self.image_crop_size = image_crop_size
        self.input_image_size = input_image_size
        self.img_names = os.listdir(self.img_dir)
        self.depth_names = [name[:name.rfind(".")] + ".npy" for name in self.img_names]
        self.epsilon = 1 - alpha  # shift value
        self.alpha_star = 1
        self.beta_star = beta + self.epsilon
        self.ti = SID(self.alpha_star, self.beta_star, K)  # [1, 1, K + 1]
        self.cj = T.ColorJitter(0.4, 0.4, 0.4)
        self.gb = T.GaussianBlur(5)
        self.gray = T.Grayscale(num_output_channels=3)
        self.scale_max_factor = scale_max_factor

    def __getitem__(self, index):
        img_pth = os.path.join(self.img_dir, self.img_names[index])
        img = cv2.imread(img_pth)
        img = cv2.resize(img, self.input_image_size)
        depth_pth = os.path.join(self.depth_dir, self.depth_names[index])
        depth = np.load(depth_pth).astype(np.uint8)
        depth = cv2.resize(depth, self.input_image_size)
        if self.is_train:
            img, depth = self.random_scale(img, depth)
            img, depth = self.random_h_flip(img, depth)
            img = self.random_color_jitter(img)
            img = self.random_gauss_blur(img)
            img = self.random_gray(img)
        img, depth = self.random_crop(img, depth)
        depth = t.from_numpy(depth)
        img = t.from_numpy(np.transpose(img, axes=[2, 0, 1]) / 255).type(t.FloatTensor)  # [3, h, w]
        depth = depth + self.epsilon  # [h, w]
        depth = depth.view((depth.size()[0], depth.size()[1], 1))  # [h, w, 1]
        depth = t.cat([depth] * (self.ti.size()[-1] - 1), dim=2)  # [h, w, K]
        l = t.sum(depth >= self.ti[:, :, :-1], dim=2) - 1  # [h, w], l(y,x) ∈ {0, 1, ..., K − 1} is the discrete label produced by SID at spatial location (x, y)
        l[l < 0] = 0
        return img, l

    def __len__(self):
        return len(self.img_names)

    def random_crop(self, img, depth):
        h, w = img.shape[:2]
        w_begin = rd.randint(0, w - self.image_crop_size[0])
        w_end = w_begin + self.image_crop_size[0]
        h_begin = rd.randint(0, h - self.image_crop_size[1])
        h_end = h_begin + self.image_crop_size[1]
        img = img[h_begin:h_end, w_begin:w_end, :]
        depth = depth[h_begin:h_end, w_begin:w_end]
        return img, depth

    def random_scale(self, img, depth):
        h, w = depth.shape
        img = Image.fromarray(img)
        depth = Image.fromarray(depth)
        s = rd.uniform(1.0, self.scale_max_factor)
        img = TF.resize(img, size=[round(h * s), round(w * s)])
        depth = TF.resize(depth, size=[round(h * s), round(w * s)])
        img = np.array(img)
        depth = np.array(depth) / s
        return img, depth

    def random_h_flip(self, img, depth):
        img = Image.fromarray(img)
        depth = Image.fromarray(depth)
        if rd.random() < 0.5:
            img = TF.hflip(img)
            depth = TF.hflip(depth)
        img = np.array(img)
        depth = np.array(depth)
        return img, depth

    def random_color_jitter(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        if rd.random() < 0.5:
            img = self.cj(img)
        img = np.array(img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return img

    def random_gauss_blur(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        if rd.random() < 0.5:
            img = self.gb(img)
        img = np.array(img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return img

    def random_gray(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        if rd.random() < 0.5:
            img = self.gray(img)
        img = np.array(img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return img


def make_loader(data_root_dir, is_train, image_size, alpha, beta, K, batch_size, num_workers, scale_max_factor, input_image_size):
    loader = iter(data.DataLoader(MySet(data_root_dir, is_train, image_size, alpha, beta, K, scale_max_factor, input_image_size), batch_size=batch_size, shuffle=True, drop_last=True, num_workers=num_workers))
    return loader


if __name__ == "__main__":
    s = MySet(r"/home/yuyang/data/make3d", True, (512, 256), 2, 80, 100, 1.5, (512, 256))
    print(s[0][0].size())
    print(s[0][1].size())