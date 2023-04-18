import os
import cv2
import numpy as np
from PIL import Image, ImageFont, ImageDraw
import matplotlib.pyplot as plt


def load_font():
    return {
        "京": cv2.imread("res/ne000.png"), "津": cv2.imread("res/ne001.png"), "冀": cv2.imread("res/ne002.png"),
        "晋": cv2.imread("res/ne003.png"),"蒙": cv2.imread("res/ne004.png"),"辽": cv2.imread("res/ne005.png"),
        "吉": cv2.imread("res/ne006.png"),"黑": cv2.imread("res/ne007.png"),"沪": cv2.imread("res/ne008.png"),
        "苏": cv2.imread("res/ne009.png"),"浙": cv2.imread("res/ne010.png"),"皖": cv2.imread("res/ne011.png"),
        "闽": cv2.imread("res/ne012.png"),"赣": cv2.imread("res/ne013.png"),"鲁": cv2.imread("res/ne014.png"),
        "豫": cv2.imread("res/ne015.png"),"鄂": cv2.imread("res/ne016.png"),"湘": cv2.imread("res/ne017.png"),
        "粤": cv2.imread("res/ne018.png"),"桂": cv2.imread("res/ne019.png"),"琼": cv2.imread("res/ne020.png"),
        "渝": cv2.imread("res/ne021.png"),"川": cv2.imread("res/ne022.png"),"贵": cv2.imread("res/ne023.png"),
        "云": cv2.imread("res/ne024.png"),"藏": cv2.imread("res/ne025.png"),"陕": cv2.imread("res/ne026.png"),
        "甘": cv2.imread("res/ne027.png"),"青": cv2.imread("res/ne028.png"),"宁": cv2.imread("res/ne029.png"),
        "新": cv2.imread("res/ne030.png"),"A": cv2.imread("res/ne100.png"),"B": cv2.imread("res/ne101.png"),
        "C": cv2.imread("res/ne102.png"),"D": cv2.imread("res/ne103.png"),"E": cv2.imread("res/ne104.png"),
        "F": cv2.imread("res/ne105.png"),"G": cv2.imread("res/ne106.png"),"H": cv2.imread("res/ne107.png"),
        "J": cv2.imread("res/ne108.png"),"K": cv2.imread("res/ne109.png"),"L": cv2.imread("res/ne110.png"),
        "M": cv2.imread("res/ne111.png"),"N": cv2.imread("res/ne112.png"),"P": cv2.imread("res/ne113.png"),
        "Q": cv2.imread("res/ne114.png"),"R": cv2.imread("res/ne115.png"),"S": cv2.imread("res/ne116.png"),
        "T": cv2.imread("res/ne117.png"),"U": cv2.imread("res/ne118.png"),"V": cv2.imread("res/ne119.png"),
        "W": cv2.imread("res/ne120.png"),"X": cv2.imread("res/ne121.png"),"Y": cv2.imread("res/ne122.png"),
        "Z": cv2.imread("res/ne123.png"),"0": cv2.imread("res/ne124.png"),"1": cv2.imread("res/ne125.png"),
        "2": cv2.imread("res/ne126.png"),"3": cv2.imread("res/ne127.png"),"4": cv2.imread("res/ne128.png"),
        "5": cv2.imread("res/ne129.png"),"6": cv2.imread("res/ne130.png"),"7": cv2.imread("res/ne131.png"),
        "8": cv2.imread("res/ne132.png"),"9": cv2.imread("res/ne133.png")
    }


class Draw:
    def __init__(self, bg):
        self._font = [
            ImageFont.truetype(os.path.join(os.path.dirname(__file__), "res/eng_92.ttf"), 126), # 文件路径
            ImageFont.truetype(os.path.join(os.path.dirname(__file__), "res/zh_cn_92.ttf"), 95)
        ] if bg not in ["green_0", "green_1"] else load_font()
        self.bg = bg
        self.size = (440, 140)
        self.color = []
        self._bg = None
        self.plane = (0, 140, 45, 9, 34)
        if bg == "black":
            self.color = [(0, 0, 0), (255, 255, 255)]
            self._bg = cv2.resize(cv2.imread(r"\res\black_bg.png"), self.size)
        elif bg == "blue":
            self.color = [(0, 0, 0), (255, 255, 255)]
            self._bg = cv2.resize(cv2.imread(r"\res\blue_bg.png"), self.size)
        elif bg == "green_0":
            self.city = 43
            self.size = (480, 140)
            self.plane = (25, 115, 43, 9, 49)
            self.color = [(255, 255, 255), (0, 0, 0)]
            self._bg = cv2.resize(cv2.imread(r"\res\green_bg_0.png"), self.size)
        elif bg == "green_1":
            self.city = 43
            self.size = (480, 140)
            self.plane = (25, 115, 43, 9, 49)
            self.color = [(255, 255, 255), (0, 0, 0)]
            self._bg = cv2.resize(cv2.imread(r"\res\green_bg_1.png"), self.size)
        elif bg == "yellow":
            self.color = [(255, 255, 255), (0, 0, 0)]
            self._bg = cv2.resize(cv2.imread(r"\res\yellow_bg.png"), self.size)

    def __call__(self, car_num):
        assert len(car_num) in (7, 8), print("Error, car number length must be 7 or 8! but got {}->{}".format(car_num,len(car_num)))
        fg = self._draw_fg(car_num)
        if self.bg in ["black", "blue"]:
            fg = cv2.bitwise_or(fg, self._bg)
        elif self.bg in ["yellow", "green_0", "green_1"]:
            fg = cv2.bitwise_and(fg, self._bg)
        return cv2.cvtColor(fg, cv2.COLOR_BGR2RGB)

    def _draw_ch(self, ch):
        if self.bg in ["green_0", "green_1"]:
            return cv2.resize(self._font[ch], (43 if ch.isupper() or ch.isdigit() else 45, 90))
        img = Image.new("RGB", (45 if ch.isupper() or ch.isdigit() else 90, 140), self.color[0])
        draw = ImageDraw.Draw(img)
        draw.text(
            (0, -11 if ch.isupper() or ch.isdigit() else 3), ch,
            fill=self.color[1],
            font=self._font[0] if ch.isupper() or ch.isdigit() else self._font[1]
        )
        if img.width > 45:
            img = img.resize((45, 140))
        return np.array(img)

    def _draw_fg(self, car_num):
        img = np.array(Image.new("RGB", self.size, self.color[0]))
        offset = 15
        img[self.plane[0]:self.plane[1], offset:offset + 45] = self._draw_ch(car_num[0])
        offset = offset + 45 + self.plane[3]
        img[self.plane[0]:self.plane[1], offset:offset + self.plane[2]] = self._draw_ch(car_num[1])
        offset = offset + self.plane[2] + self.plane[4]
        for i in range(2, len(car_num)):
            img[self.plane[0]:self.plane[1], offset:offset + self.plane[2]] = self._draw_ch(car_num[i])
            offset = offset + self.plane[2] + self.plane[3]
        return img


if __name__ == '__main__':
    draw = Draw("yellow")
    plate = draw("京A12345")
    plt.imshow(plate)
    plt.show()
