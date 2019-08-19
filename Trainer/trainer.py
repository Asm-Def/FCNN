import torch
import torch.nn as nn
import math
from torch.utils.data import DataLoader
import numpy as np
import pickle
import os
import random
from tensorboardX import SummaryWriter
import cv2 as cv
import threading

class MyThread(threading.Thread):
    def __init__(self, func, cur_batch, cur_img):
        threading.Thread.__init__(self)
        self.func = func
        self.cur_batch = cur_batch
        self.cur_img = cur_img

    def run(self):
        return self.func(self.cur_batch, self.cur_img)

class Trainer(object):
    def __init__(self, model, train_loader: DataLoader, val_loader: DataLoader, optimizer, max_iter, model_name, log_dir):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.max_iter = max_iter
        self.model_name = model_name
        self.log_dir = log_dir
        self.model.cuda()
        self.best_dice = .0
        self.i_acc = 0
        self.threshold = .5
        self.gauss = torch.empty((5, 5), dtype=torch.float32)
        for i in range(5):
            for j in range(5):
                t = (i-2)**2 + (j-2)**2
                self.gauss[i,j] = math.exp(- t / 8) / 8 / math.pi

        mix = torch.ones((5, 5), dtype=torch.float32)
        self.mix_conv = nn.Conv2d(1, 1, 5, padding=2, bias=False)
        self.mix_conv.weight = nn.Parameter(mix.view(1, 1, 5, 5), False)
        self.mix_conv = self.mix_conv.cuda()
        self.gauss_filter = nn.Conv2d(1, 1, 5, padding=2, bias=False)
        self.gauss_filter.weight = nn.Parameter(self.gauss.view(1, 1, 5, 5), False)
        self.gauss_filter = self.gauss_filter.cuda()

        if self.log_dir is not None:
            self.writer = SummaryWriter(log_dir)

    def get_click(self, target: torch.Tensor, predict: torch.Tensor):
        '''
        -1：background, 1: foreground, 0: same
        :param target: tensor(x, y), dtype=torch.int
        :param predict: tensor(x, y), dtype=torch.float32
        :return: (-1/0/1, x, y)
        '''
        graph = target - (predict > self.threshold).to(dtype=torch.int)
        if graph.min() == 0 and graph.max() == 0:
            return 0, 0, 0
        foreground = (graph == 1).to(dtype=torch.float32)  # 需点击的部分为1
        background = (graph == -1).to(dtype=torch.float32)  # 需点击的部分为1
        row, col = target.shape[0], target.shape[1]

        if foreground.sum() > background.sum():  # foreground的1更多
            res = 1
            dis = foreground
        else:
            res = -1
            dis = background

        dis.unsqueeze_(0)
        dis.unsqueeze_(0)
        dis = torch.min(self.gauss_filter(self.gauss_filter(dis)), dis)
        xy = torch.multinomial(dis.view((-1,)), 1)[0]
        x = xy / col
        y = xy - col * x
        return res, x, y

    @staticmethod
    def loss(target: torch.Tensor, predict: torch.Tensor):
        '''
        Dice Loss
        :param target: tensor(batch, 1, x, y), dtype=torch.int
        :param predict: tensor(batch, 1, x, y), dtype=torch.float32
        :return: tensor(1)
        '''
        n = target.shape[0]
        predict = predict.view(n, -1)
        target = target.view(n, -1).to(dtype=torch.float32)
        inter = predict * target
        predict = predict * predict
        # target全为0/1，无需平方
        loss = 2 * inter.sum(1) / (target.sum(1) + predict.sum(1))
        return -loss.sum() / n

    def Iterate(self, epoch, data_loader: DataLoader, train, tag: str):
        tot_loss, tot_dice = 0, 0
        for i, data in enumerate(data_loader):
            target = data[0].unsqueeze(1).cuda()  # (batch, 1, x, y)
            in_data = data[1].unsqueeze(1).cuda()  # (batch, 1, x, y)
            foreground = torch.zeros_like(in_data).cuda()
            background = torch.zeros_like(in_data).cuda()
            with torch.no_grad():
                tmp = torch.zeros_like(target[0, 0])

                def do_clicks(cur_batch, cur):
                    res, x, y = self.get_click(target[cur_batch, 0], cur)
                    if res == -1:
                        background[cur_batch, 0, x, y] = 1.0
                    elif res == 1:
                        foreground[cur_batch, 0, x, y] = 1.0

                threads = []

                for batch in range(in_data.shape[0]):
                    t = MyThread(do_clicks, batch, tmp)
                    threads.append(t)
                    t.start()

                for batch in range(in_data.shape[0]):
                    threads[batch].join()

                for it in range(1, self.max_iter):
                    prob = it / self.max_iter * 0.9
                    if random.random() < prob:  # 结束迭代
                        break

                    fore = self.gauss_filter(foreground)
                    back = self.gauss_filter(background)
                    clicks = torch.cat((fore, back), dim=1)
                    clicks /= clicks.max()
                    out_data = self.model(
                        torch.cat((in_data, clicks), dim=1)
                    )  # (batch, 1, x, y)

                    for batch in range(in_data.shape[0]):
                        threads[batch] = MyThread(do_clicks, batch, out_data[batch, 0])
                        threads[batch].start()

                    for batch in range(in_data.shape[0]):
                        threads[batch].join()

                fore = self.gauss_filter(foreground)
                back = self.gauss_filter(background)
                clicks = torch.cat((fore, back), dim=1)
                clicks /= clicks.max()

            out_data = self.model(
                torch.cat((in_data, clicks), dim=1)
            )
            # Train or Evaluate
            if not train:
                loss = self.loss(target, out_data)
                tot_loss += loss.item()
                ans = (out_data > self.threshold).to(dtype=torch.float32)
                dice = self.loss(target, ans)
                tot_dice += dice.item()
                print('loss:{}, dice:{}'.format(loss.item(), dice.item()))

            else:
                # 利用当前的foreground和background训练
                loss = self.loss(target, out_data)
                tot_loss += loss.item()
                loss.backward()

                self.optimizer.step()
                self.optimizer.zero_grad()

                with torch.no_grad():
                    self.writer.add_scalar('train/train_loss', loss.item()+1, self.i_acc + i + 1)
                    ans = (out_data > self.threshold).to(dtype=torch.float32)
                    dice = self.loss(target, ans)
                    tot_dice += dice.item()
                    self.writer.add_scalar('train/train_dice', dice.item()+1, self.i_acc + i + 1)
                    log_str = 'epoch {0:d}, step {1:d}: train_loss {2:.3f}; train_dice {3:.3f}'.format(
                        epoch + 1, i + 1, loss.item()+1, dice.item()+1
                    )
                    print(log_str)

            # Draw prediction
            with torch.no_grad():
                fore = self.mix_conv(fore)
                back = self.mix_conv(back)
                for batch in range(data[1].shape[0]):
                    if not train and (data[2][batch] != 'img0001.nii.gz' or data[3][batch].item() != 120):
                        continue

                    if train and (data[2][batch] != 'img0002.nii.gz' or data[3][batch].item() != 120):
                        continue

                    for i in range(in_data[batch,0].shape[0]):
                        for j in range(in_data[batch,0].shape[1]):
                            print('{0:.1f}'.format(in_data[batch, 0, i, j].item()), end='')
                        print('')

                    print('')

                    for i in range(out_data[batch,0].shape[0]):
                        for j in range(out_data[batch,0].shape[1]):
                            print('{0:.1f}'.format(out_data[batch, 0, i, j].item()), end='')
                        print('')

                    image = (in_data[batch] * 255).to(torch.uint8).cpu().numpy()
                    area = (ans[batch] * 200).to(torch.uint8).cpu().numpy()
                    fore_img = (fore[batch] * 255).to(torch.uint8).cpu().numpy()
                    back_img = (back[batch] * 255).to(torch.uint8).cpu().numpy()
                    image = np.concatenate((image, image, image))
                    area_img = np.concatenate((area, fore_img, back_img))
                    image = cv.addWeighted(image, 0.7, area_img, 0.4, 0.0)
                    self.writer.add_image(
                        '{}-imgs-ans-'.format(tag) + data[2][batch] + '-' + str(data[3][batch].item()) + '/' + str(
                            data[4][batch].item()),
                        image, global_step=epoch, dataformats='CHW'
                    )

                    area = (out_data[batch] * 200).to(torch.uint8).cpu().numpy()
                    area_img = np.concatenate((area, fore_img, back_img))
                    image = cv.addWeighted(image, 0.7, area_img, 0.4, 0.0)
                    self.writer.add_image(
                        '{}-imgs-pred-'.format(tag) + data[2][batch] + '-' + str(data[3][batch].item()) + '/' + str(
                            data[4][batch].item()),
                        image, global_step=epoch, dataformats='CHW'
                    )

        tot_dice /= len(data_loader)
        tot_loss /= len(data_loader)
        return tot_loss, tot_dice

    def train(self, n_epoch, begin=0):

        self.i_acc = begin * len(self.train_loader)

        self.model.train()

        for epoch in range(begin, n_epoch):
            out_data = None
            in_data = None

            self.Iterate(epoch, self.train_loader, True, 'train')

            self.i_acc += len(self.train_loader)

            # evaluation
            with torch.no_grad():
                val_loss, val_dice = self.evaluate(epoch)
                self.writer.add_scalar('val/val_loss', val_loss, self.i_acc)
                self.writer.add_scalar('val/val_dice', val_dice, self.i_acc)

            # save model
            if 1 - val_dice > self.best_dice:
                self.best_dice = 1 - val_dice

            self.model.save(self.log_dir, epoch)

    def evaluate(self, epoch):
        self.model.eval()
        tot_data = 0
        with torch.no_grad():
            loss, dice = self.Iterate(epoch, self.val_loader, False, 'val')

        loss = loss / len(self.val_loader) + 1
        dice = dice / len(self.val_loader) + 1

        print('Total #: ', tot_data)
        print('val loss: ', loss)
        print('val dice: ', dice)

        self.model.train()

        return loss, dice
