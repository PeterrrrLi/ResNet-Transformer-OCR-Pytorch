from data import make_label
from torch.utils.data import Dataset
from fake_chs_lp.random_plate import Draw
import os
from einops import rearrange
import random
import cv2
from utils import data_augmentation
import numpy
import torch
import ocr_config
import config
import re


class OcrDataSet(Dataset):
    
    '''
    Input: 3, 48, 144
    
    Output: 27, Batch, Num of Classes
    '''

    def __init__(self):
        super(OcrDataSet, self).__init__()
        self.dataset = []
        self.draw = Draw()
        for i in range(10000):
            self.dataset.append(1)
        self.smudge = data_augmentation.Smudge()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        plate, label = self.draw()
        target = []
        for i in label:
            target.append(ocr_config.class_name.index(i))
        plate = cv2.cvtColor(plate, cv2.COLOR_RGB2BGR)

        '''
        Data Augmentation
        '''
        
        plate = self.data_to_enhance(plate)

        image = torch.from_numpy(plate).permute(2, 0, 1) / 255
        target_length = torch.tensor(len(target)).long()
        target = torch.tensor(target).reshape(-1).long()
        _target = torch.full(size=(15,), fill_value=0, dtype=torch.long)
        _target[:len(target)] = target

        return image, _target, target_length

    def data_to_enhance(self, plate):
        # Smudge
        plate = self.smudge(plate)
        # Gaussian Blur
        plate = data_augmentation.gauss_blur(plate)
        # Gaussian Noise
        plate = data_augmentation.gauss_noise(plate)
        plate, pts = data_augmentation.augment_sample(plate)
        plate = data_augmentation.reconstruct_plates(plate, [numpy.array(pts).reshape((2, 4))])[0]
        return plate


class DetectDataset(Dataset):

    def __init__(self):
        super(DetectDataset, self).__init__()
        self.dataset = []
        self.draw = Draw()
        self.smudge = data_augmentation.Smudge()
        root = config.image_root
        for image_name in os.listdir(root):
            box = self.get_box(image_name)
            x3, y3, x4, y4, x1, y1, x2, y2 = box
            box = [x1, y1, x2, y2, x4, y4, x3, y3]
            self.dataset.append((f'{root}/{image_name}', box))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        """return (img,label) -- img shape is [3, 208, 208], label shape is [13, 13, 9]"""
        """[364, 517, 440, 515, 368, 549, 444, 547] tl tr bl br"""
        image_path, points = self.dataset[item]
        image = cv2.imread(image_path)

        if random.random() < 0.5:
            plate, _ = self.draw()
            plate = cv2.cvtColor(plate, cv2.COLOR_RGB2BGR)
            plate = self.smudge(plate)
            image = data_augmentation.apply_plate(image, points, plate)
        [x1, y1, x2, y2, x4, y4, x3, y3] = points
        points = [x1, x2, x3, x4, y1, y2, y3, y4]
        image, pts = data_augmentation.augment_detect(image, points, 208)
        image_tensor = torch.from_numpy(image)/255
        image_tensor = rearrange(image_tensor, 'h w c -> c h w')
        label = make_label.object_label(pts,208,16)
        label = torch.from_numpy(label).float()
        return image_tensor,label

    @staticmethod
    def up_background(image):
        image = data_augmentation.gauss_blur(image)
        image = data_augmentation.gauss_noise(image)
        image = data_augmentation.random_cut(image, (208, 208))
        return image

    def data_to_enhance(self, plate):
        plate = self.smudge(plate)
        plate = data_augmentation.gauss_blur(plate)
        plate = data_augmentation.gauss_noise(plate)
        plate, pts = data_augmentation.augment_sample(plate)
        plate = data_augmentation.reconstruct_plates(plate, [numpy.array(pts).reshape((2, 4))])[0]
        return plate

    @staticmethod
    def get_box(name):
        name = re.split('[.&_-]', name)[7:15]
        name = [int(i) for i in name]
        return name


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import numpy as np
    data_set = DetectDataset()
    data_ocr = OcrDataSet()
    img, target, tl = data_ocr[0]
    img = torch.permute(img, [1,2,0])
    print(img.shape, target.shape,tl)
    plt.imshow(img)
    plt.show()