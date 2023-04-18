from torch.utils.data import DataLoader
from torch import nn
from utils.loss import FocalLossManyClassification
from utils.dataset import DetectDataset
from einops import rearrange
from tqdm import tqdm
import config as config
import torch
import os


class Trainer:

    def __init__(self):
        self.net = config.net()
        if os.path.exists(config.weight):
            self.net.load_state_dict(torch.load(config.weight, map_location='cpu'))
            print('Loaded Weights')
        else:
            print('Do not load weights')

        self.l1_loss = nn.L1Loss()
        self.c_loss = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.net.parameters(),lr=0.00001)
        self.dataset = DetectDataset()
        self.data_loader = DataLoader(self.dataset, config.batch_size, drop_last=True)
        self.net.to(config.device)

    def train(self):

        for epoch in range(config.epoch):
            self.net.train()
            loss_sum = 0
            for i, (images, labels) in enumerate(tqdm(self.data_loader)):
                images = images.to(config.device)
                labels = labels.to(config.device)

                predict = self.net(images)# [b,13,13,8]  [b, 13, 13, 9]
                loss_c, loss_p = self.count_loss(predict, labels)
                loss = loss_c + loss_p
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                if (i+1) % 10==0:
                    print(f"epoch={epoch+1},step={i},loss={loss.item()},loss_c={loss_c.item()},loss_p={loss_p.item()}")
                if (i+1) % 100 == 0:
                    torch.save(self.net.state_dict(), config.weight)
                loss_sum += loss.item()
            logs = f'epoch:{epoch+1},loss:{loss_sum / len(self.data_loader)}'
            print(logs)
            torch.save(self.net.state_dict(), config.weight)

    def count_loss(self, predict, target):
        condition_positive = target[:, :, :, 0] == 1# [b, 13, 13]
        condition_negative = target[:, :, :, 0] == 0

        predict_positive = predict[condition_positive]
        predict_negative = predict[condition_negative]

        target_positive = target[condition_positive]
        target_negative = target[condition_negative]
        # print(target_positive.shape)
        n, v = predict_positive.shape
        if n > 0:
            loss_c_positive = self.c_loss(predict_positive[:, 0:2], target_positive[:, 0].long())
        else:
            loss_c_positive = 0
        loss_c_nagative = self.c_loss(predict_negative[:, 0:2], target_negative[:, 0].long())
        loss_c = loss_c_nagative + loss_c_positive

        if n > 0:
            affine = torch.cat(
                (
                    predict_positive[:, 2:3],
                    predict_positive[:,3:4],
                    predict_positive[:,4:5],
                    predict_positive[:,5:6],
                    predict_positive[:,6:7],
                    predict_positive[:,7:8]
                ),
                dim=1
            )
            # print(affine.shape)
            # exit()
            trans_m = affine.reshape(-1, 2, 3)
            unit = torch.tensor([[-0.5, -0.5, 1], [0.5, -0.5, 1], [0.5, 0.5, 1], [-0.5, 0.5, 1]]).transpose(0, 1).to(
                trans_m.device).float()
            # print(unit)
            point_pred = torch.einsum('n j k, k d -> n j d', trans_m, unit) # 等价于求和
            point_pred = rearrange(point_pred, 'n j k -> n (j k)')
            loss_p = self.l1_loss(point_pred, target_positive[:, 1:])
        else:
            loss_p = 0
        # exit()
        return loss_c, loss_p

        # return loss


if __name__ == '__main__':
    trainer = Trainer()
    trainer.train()
