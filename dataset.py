import numpy as np
import torch
import os
from torch.utils.data import Dataset
from tqdm.notebook import tqdm


class audioDataSet(Dataset):
  def __init__(self,root_pth,test=False,transform = None):
    print("Dataset initializing...")
    class_num=4
    self.audio_pth = os.path.join(root_pth, 'audios', 'mfcc')
    filling_type = np.load(os.path.join(root_pth, 'audios', 'filling_type.npy'))
    pouring_or_shaking = np.load(os.path.join(root_pth,  'audios', 'pouring_or_shaking.npy'))
    self.label = filling_type * pouring_or_shaking
    self.is_test=test
    self.each_class_size = []
    for i in range(class_num):
        self.each_class_size.append(np.count_nonzero(self.label==i))
    mx=0
    mn=1000
    for idx in tqdm(range(self.label.shape[0])):
        data=np.load(os.path.join(self.audio_pth, "{0:06d}".format(idx+1) + '.npy'), allow_pickle=True)
        tmp_max=np.max(data)
        tmp_min=np.min(data)
        if mx<tmp_max:
            mx=tmp_max
        if mn>tmp_min:
            mn=tmp_min
    self.mn=mn
    self.mx=mx
  def __len__(self):
    return self.label.shape[0]
  def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        lbl = -1

        if self.is_test is False:
            lbl = self.label[idx]
        data=np.load(os.path.join(self.audio_pth, "{0:06d}".format(idx+1) + '.npy'), allow_pickle=True)
        data= (data-self.mn)/(self.mx-self.mn)
        data=data.transpose(2,0,1)
        data=torch.from_numpy(data.astype(np.float32))
        return data , lbl
            
  def get_each_class_size(self):
    return np.array(self.each_class_size)