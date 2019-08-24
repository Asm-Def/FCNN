import torch
import torch.nn as nn
import os
import glob


class Model(nn.Module):
    '''
    方便存取参数
    '''
    def __init__(self, name):
        '''
        :param name: 参数文件目录名
        '''
        super(Model, self).__init__()
        self.name = name

    def save(self, path, epoch=0):
        '''
        保存当前参数
        :param path: 参数文件所在根目录
        :param epoch: 产生当前参数的epoch编号
        :return: None
        '''
        complete_path = os.path.join(path, self.name)
        if not os.path.exists(complete_path):
            os.makedirs(complete_path)
        torch.save(self.state_dict(),
                   os.path.join(complete_path,
                                "model-{}.pth".format(str(epoch).zfill(5))))

    def save_results(self, path, data):
        '''
        保存特定数据
        virtual method
        '''
        raise NotImplementedError("Model subclass must implement this method.")

    def load(self, path, modelfile=None):
        '''
        加载参数
        :param path: 参数所在根目录
        :param modelfile: 参数文件名
        :return: None
        '''
        complete_path = os.path.join(path, self.name)
        if not os.path.exists(complete_path):
            os.makedirs(complete_path)

        if modelfile is None:
            model_files = glob.glob(complete_path+"/*")
            if len(model_files) == 0:  # no model_file to load
                return 0
            mf = max(model_files)
        else:
            mf = os.path.join(complete_path, modelfile)

        if torch.cuda.is_available():
            dev = torch.device('cuda')
        else:
            dev = torch.device('cpu')
        self.load_state_dict(torch.load(mf, map_location=dev))

        return int(os.path.split(mf)[1][6:-4]) + 1
