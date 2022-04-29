import torch
import torchvision.transforms.functional as F
from PIL import Image
import os
import glob
import numpy as np
import cv2
import random


def random_horizontal_flip(imgs):
    if random.random() < 0.3:
        for i in range(len(imgs)):
            imgs[i] = imgs[i].transpose(Image.FLIP_LEFT_RIGHT)
    return imgs

def random_rotate(imgs):
    if random.random() < 0.3:
        max_angle = 10
        angle = random.random() * 2 * max_angle - max_angle
        # print(angle)
        for i in range(len(imgs)):
            img = np.array(imgs[i])
            w, h = img.shape[:2]
            rotation_matrix = cv2.getRotationMatrix2D((h / 2, w / 2), angle, 1)
            img_rotation = cv2.warpAffine(img, rotation_matrix, (h, w))
            imgs[i] =Image.fromarray(img_rotation)
    return imgs

class Dataset(torch.utils.data.Dataset):
    def __init__(self, text_path, mask_path=None,gt_path=None, training=True, mask_reverse=False):
        super(Dataset, self).__init__()
        self.training = training
        self.has_gt = gt_path
        if self.has_gt:
            self.gt = self.load_list(gt_path)
        self.has_mask = mask_path
        if self.has_mask:
            self.mask = self.load_list(mask_path)
        self.text = self.load_list(text_path)
        self.mask_reverse = mask_reverse

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        try:
            item = self.load_item(index)
        except:
            print('loading error: ' + self.text[index])
            item = self.load_item(0)
        return item

    def load_item(self, index):

        if self.training:
            gt = Image.open(self.gt[index]).convert('RGB')
            mask = Image.open(self.mask[index]).convert('L')
            text = Image.open(self.text[index]).convert('RGB')
            all_input = [text, mask, gt]
            all_input = random_horizontal_flip(all_input)
            all_input = random_rotate(all_input)
            text = all_input[0]
            mask = all_input[1]
            gt = all_input[2]
            if self.mask_reverse:
                return self.to_tensor(gt),1-self.to_tensor(mask) ,self.to_tensor(text)
            else:
                return self.to_tensor(gt), self.to_tensor(mask), self.to_tensor(text)
        else:
            gt = Image.open(self.gt[index]).convert('RGB')
            if self.has_mask:
                mask = Image.open(self.mask[index]).convert('L')
            text = Image.open(self.text[index]).convert('RGB')
            name = self.text[index].split('/')[-1][:-4]
            if self.has_mask:
                if self.mask_reverse:
                    return self.to_tensor(gt), 1 - self.to_tensor(mask), self.to_tensor(text),name
                else:
                    return self.to_tensor(gt), self.to_tensor(mask), self.to_tensor(text), name
            else:
                return self.to_tensor(gt), self.to_tensor(text),name

    def to_tensor(self, img):
        img_t = F.to_tensor(img).float()
        return img_t

    def load_list(self, path):
        if isinstance(path, str):
            if os.path.isdir(path):
                path = list(glob.glob(path + '/*.jpg')) + list(glob.glob(path + '/*.png'))
                path.sort()
                return path
        return []