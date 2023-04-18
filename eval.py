from detect_explorer import DExplorer
from ocr_explorer import Explorer
import cv2
import numpy
import os

class ReadPlate:

    def __init__(self):
        self.detect_exp = DExplorer()
        self.ocr_exp = Explorer()

    def __call__(self, image):
        points = self.detect_exp(image)
        h, w, _ = image.shape
        result = []
        # print(points)
        for point, _ in points:
            plate, box = self.cutout_plate(image, point)
            # print(box)
            lp = self.ocr_exp(plate)
            result.append([lp, box])
            # cv2.imshow('a', plate)
            # cv2.waitKey()
        return result

    def cutout_plate(self, image, point):
        h, w, _ = image.shape
        x1, x2, x3, x4, y1, y2, y3, y4 = point.reshape(-1)
        x1, x2, x3, x4 = x1 * w, x2 * w, x3 * w, x4 * w
        y1, y2, y3, y4 = y1 * h, y2 * h, y3 * h, y4 * h
        src = numpy.array([[x1, y1], [x2, y2], [x4, y4], [x3, y3]], dtype="float32")
        dst = numpy.array([[0, 0], [144, 0], [0, 48], [144, 48]], dtype="float32")
        box = [min(x1, x2, x3, x4), min(y1, y2, y3, y4), max(x1, x2, x3, x4), max(y1, y2, y3, y4)]
        M = cv2.getPerspectiveTransform(src, dst)
        out_img = cv2.warpPerspective(image, M, (144, 48))
        return out_img, box


def val():
    provinces = ["皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "京", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤", "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "警", "学", "O"]
    alphabets = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W',
                 'X', 'Y', 'Z', 'O']
    ads = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X',
           'Y', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'O']
    count = 0
    correct = 0
    read_plate = ReadPlate()
    for pic in os.listdir("test"):
        image = cv2.imread('test/' + pic)
        predict = read_plate(image)[0][0]
        num = pic.split("-")[-3]
        nums = num.split("_")
        pro = provinces[int(nums[0])]
        city = alphabets[int(nums[1])]

        plates = [ads[int(n)] for n in nums[2:]]
        true_label = pro + city + "".join(plates)
        if true_label == predict:
            result = "√"
            correct += 1
        else:
            result = "×"
        count += 1
        print(f"{true_label}--{predict}--{result}")
    print(f"accuracy={correct / count}")
    return

if __name__ == '__main__':
    val()