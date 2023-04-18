from fake_chs_lp.draw import Draw
import random
import math
import matplotlib.pyplot as plt
import argparse


class RandomDraw:
    def __init__(self):
        self._draw = ["black", "blue", "green", "yellow"]
        self._province = ["皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "京", "闽", "赣", "鲁", "豫", "鄂", "湘",
                          "粤", "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新"]
        self._alpha = ["A", "B", "C", "D", "E", "F", "G", "H", "J", "K", "L", "M", "N", "P", "Q", "R", "S",
                       "T", "U", "V", "W", "X", "Y", "Z"]
        self._ads = ["A", "B", "C", "D", "E", "F", "G", "H", "J", "K", "L", "M", "N", "P", "Q", "R", "S",
                     "T", "U", "V", "W", "X", "Y", "Z", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

    def __call__(self):
        draw = random.choice(self._draw)
        candidates = [self._province, self._alpha]
        if draw == "green":
            candidates += [self._ads] * 6
            label = "".join([random.choice(c) for c in candidates])
            draw = draw + "_" + str(random.choice([0,1]))
            d = Draw(draw)
            return d(label), label
        elif draw == "black":
            if random.random() < 0.3:
                candidates += [self._ads] * 4
                candidates += [["港", "澳"]]
            else:
                candidates += [self._ads] * 5
            label = "".join([random.choice(c) for c in candidates])
            d = Draw(draw)
            return d(label), label
        elif draw == "yellow":
            if random.random() < 0.3:
                candidates += [self._ads] * 4
                candidates += [["学"]]
            else:
                candidates += [self._ads] * 5
            label = "".join([random.choice(c) for c in candidates])
            d = Draw(draw)
            return d(label), label
        else:
            candidates += [self._ads] * 5
            label = "".join([random.choice(c) for c in candidates])
            d = Draw(draw)
            return d(label), label

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="The input of car numbers")
    parser.add_argument("--num", help="numbers to random generate plates", default=9, type=int)
    args = parser.parse_args()

    random_draw = RandomDraw()
    rows = math.ceil(args.num / 3)
    cols = min(args.num, 3)
    for i in range(args.num):
        plate, label = random_draw()
        print(label)
        plt.subplot(rows, cols, i+1)
        plt.imshow(plate)
        plt.axis("off")
    plt.show()
