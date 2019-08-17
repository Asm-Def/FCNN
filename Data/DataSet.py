import numpy as np
import glob
import torch.utils.data
import os
import nibabel as nib
import matplotlib.pyplot as plt
import random

class ImgDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, fileset):
        tmp = glob.glob(os.path.join(root_dir, 'img', '*.nii.gz'))
        random.shuffle(tmp)
        tmp.sort()
        self.filepaths = []
        for t in fileset:
            self.filepaths.append(tmp[t])
        self.files = []
        self.labels = []
        self.indx = []
        self.fileid = []
        self.channel_id = []
        mn, mx = 65535, -65536
        for i in range(len(self.filepaths)):
            img = nib.load(self.filepaths[i]).get_data().transpose((2, 1, 0))
            label = nib.load(self.filepaths[i].replace('img', 'label')).get_data().transpose((2, 1, 0))

            order = [j for j in range(img.shape[0])]
            random.shuffle(order)
            for j in order:
                unique = np.unique(label[j])[1:]
                if len(unique) == 0:  # only 0
                    continue
                tensor = torch.tensor(img[j], dtype=torch.float32)
                for t in unique:
                    self.indx.append((len(self.files), t))  # (第几个图片, 分类标记）
                self.files.append(tensor)
                self.labels.append(label[j])
                self.fileid.append(i)
                self.channel_id.append(j)  # 当前CT的第几层
                mn = min(mn, tensor.min())
                mx = max(mx, tensor.max())

        for i in range(len(self.files)):
            self.files[i] = (self.files[i] - mn) / (mx - mn)

        print(mn, mx)

    def __len__(self):
        return len(self.indx)

    def __getitem__(self, idx):
        ind = self.indx[idx]
        cur = ind[0]
        lb = ind[1]
        label = torch.tensor(self.labels[cur] == lb, dtype=torch.int)
        return label, self.files[cur], os.path.split(self.filepaths[self.fileid[cur]])[1], self.channel_id[cur], lb

