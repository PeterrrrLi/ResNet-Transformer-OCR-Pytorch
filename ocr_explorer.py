from models.ocr_nn import OcrNet
import ocr_config as config
import torch
import cv2
import numpy as np
import os


class Explorer:

    def __init__(self, is_cuda=False):
        self.device = config.device
        self.net = OcrNet(config.num_class)
        if os.path.exists(config.weight):
            self.net.load_state_dict(torch.load(config.weight, map_location='cpu'))
            print(f'Load success:{config.weight.split("/")[-1]}')
        else:
            raise RuntimeError('Model parameters are not loaded')
        self.net = self.net.to(self.device).eval()

    def __call__(self, image):
        with torch.no_grad():
            image = torch.from_numpy(image).permute(2, 0, 1) / 255
            image = image.unsqueeze(0).to(self.device)
            # print(image.shape)
            out = self.net(image).reshape(-1, 70)
            out = torch.argmax(out, dim=1).cpu().numpy().tolist()
            c = ''
            for i in out:
                c += config.class_name[i]
            return self.deduplication(c)

    def deduplication(self, c):
        temp = ''
        new = ''
        for i in c:
            if i == temp:
                continue
            else:
                if i == '*':
                    temp = i
                    continue
                new += i
                temp = i
        return new


if __name__ == '__main__':
    import os
    import matplotlib.pyplot as plt
    e = Explorer()
    co = 0
    i = 0
    from fake_chs_lp.random_plate import Draw

    draw = Draw()
    for i in range(10):
        plate, label = draw()
        plate = cv2.resize(plate,(144,48))
        c = e(plate)
        print(i, c, label)
        if c == label:
            co += 1
        # cv2.imshow('a', plate)
        # cv2.waitKey(0)
    print(co, i, co / i)
