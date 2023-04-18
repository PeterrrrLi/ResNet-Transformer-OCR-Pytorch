# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    detect_explorer.py                                 :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: peterli <j2457li@uwaterloo.ca>             +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2023/04/18 17:29:13 by peterli           #+#    #+#              #
#    Updated: 2023/04/18 17:29:14 by peterli          ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import config as config
import cv2
import torch
from einops import rearrange
import os
import numpy


class DExplorer:

    def __init__(self):
        self.net = config.net()
        if os.path.exists(config.weight):
            self.net.load_state_dict(torch.load(config.weight, map_location='cpu'))
        else:
            raise RuntimeError('Model parameters are not loaded')
        # self.net.to(config.device)
        self.net.eval()

    def __call__(self, image_o):
        image = image_o.copy()
        h, w, c = image.shape
        f = min(288 * max(h, w) / min(h, w), 608) / min(h, w)
        _w = int(w * f) + (0 if w % 16 == 0 else 16 - w % 16)
        _h = int(h * f) + (0 if h % 16 == 0 else 16 - h % 16)
        image = cv2.resize(image, (_w, _h), interpolation=cv2.INTER_AREA)
        image_tensor = torch.from_numpy(image) / 255
        image_tensor = rearrange(image_tensor, 'h w c ->() c h w')
        with torch.no_grad():
            y = self.net(image_tensor).cpu()
            points = self.select_box(y, (_w, _h))
            return points

    def select_box(self, predict, size, dims=208, stride=16):
        wh = numpy.array([[size[0]], [size[1]]])
        probs = predict[0, :, :, 0:2]
        probs = torch.softmax(probs, dim=-1).numpy()
        affines = torch.cat(
            (
                predict[0, :, :, 2:3],
                predict[0, :, :, 3:4],
                predict[0, :, :, 4:5],
                predict[0, :, :, 5:6],
                predict[0, :, :, 6:7],
                predict[0, :, :, 7:8]
            ),
            dim=2
        )
        h, w, c = affines.shape
        affines = affines.reshape(h, w, 2, 3).numpy()
        scale = ((dims + 40.0) / 2.0) / stride
        unit = numpy.array([[-0.5, -0.5, 1], [0.5, -0.5, 1], [0.5, 0.5, 1], [-0.5, 0.5, 1]]).transpose((1, 0))
        h, w, _ = probs.shape
        candidates = []
        for i in range(h):
            for j in range(w):
                if probs[i, j, 1] > config.confidence_threshold:
                    affine = affines[i, j]
                    pts = affine @ unit
                    # print(affine)
                    # print(affine)
                    pts *= scale
                    pts += numpy.array([[j + 0.5], [i + 0.5]])
                    pts *= stride
                    # print(pts)
                    pts /= wh
                    # exit()
                    candidates.append((pts, probs[i, j, 1]))
                    # break

        candidates.sort(key=lambda x: x[1], reverse=True)
        labels = []
        for pts_c, prob_c in candidates:
            tl_c = pts_c.min(axis=1)
            br_c = pts_c.max(axis=1)
            overlap = False
            for pts_l, _ in labels:
                tl_l = pts_l.min(axis=1)
                br_l = pts_l.max(axis=1)
                if self.iou(tl_c, br_c, tl_l, br_l) > 0.1:
                    overlap = True
                    break
            if not overlap:
                labels.append((pts_c, prob_c))
        return labels

    @staticmethod
    def iou(tl1, br1, tl2, br2):
        x1, y1 = tl1
        x2, y2 = br1
        x3, y3 = tl2
        x4, y4 = br2
        wh1 = br1 - tl1
        wh2 = br2 - tl2
        assert ((wh1 >= 0).sum() > 0 and (wh2 >= 0).sum() > 0)
        s1 = (y2 - y1) * (x2 - x1)
        s2 = (y4 - y3) * (x4 - x3)
        _x1 = max(x1, x3)
        _y1 = max(y1, y3)
        _x2 = min(x2, x4)
        _y2 = max(y2, y4)
        w = max(0, _x2 - _x1)
        h = max(0, _y2 - _y1)
        i = w * h
        return i / (s1 + s2 - i)


if __name__ == '__main__':
    e = DExplorer()
    image = cv2.imread('test/test_image.jpg')
    label = e(image)
    print(label)
