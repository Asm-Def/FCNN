import torch.utils.data
import torch.nn as nn
import torch.optim as optim
import torch.utils
import torch.utils.data
from Trainer.trainer import Trainer
from Data.DataSet import ImgDataset
from Models.FCNN import FCNN

import os, shutil

def create_folder(log_dir):
    if log_dir.find('/', 1) != -1 or log_dir.find('\\', 1) != -1:
        lst = os.path.split(log_dir)
        create_folder(lst[0])
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)


if __name__ == '__main__':
    log_dir = os.path.join('log', 'FCNN')
    create_folder(log_dir)
    fcnn = FCNN()
    begin = fcnn.load(log_dir)  # 载入当前最新模型
    fcnn = fcnn.cuda()
    optimizer = optim.Adam(fcnn.parameters(), lr=1e-4)
    train_dataset = ImgDataset(
        os.path.join('Data', 'Training'),
        (1, 2, 3, 4, 6, 7, 8, 9, 11, 12, 13, 14, 16, 17, 18, 19, 21,22,23,24,26,27,28,29)
    )
    print('train_dataset', len(train_dataset))
    test_dataset = ImgDataset(os.path.join('Data', 'Training'), (0, 5, 10, 15, 20, 25))
    print('test_dataset', len(test_dataset))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=True)
    print('num_train_imgs: ', len(train_dataset))
    print('num_test_imgs: ', len(test_dataset))
    trainer = Trainer(fcnn, train_loader, test_loader, optimizer, 5, 'FCNN', log_dir)
    trainer.train(30, begin=begin)
