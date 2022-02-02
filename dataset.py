from logging.config import valid_ident
import numpy as np
import cv2
import torch
import os
import scipy.stats as stats
from torch.utils.data import Dataset
from tqdm import tqdm
from PIL import Image
from torchvision import datasets, transforms

def randomlyAug(crop, depth, label, max_val=640, square=False, normal=False, depth_aug=False):
  h, w, c = crop.shape

  if h >= w:
    max_dim = h
  else:
    max_dim = w
  max_rand = max_val / max_dim

  lower, upper = 0.5, 1.5
  if upper > max_rand:
      upper = max_rand
  mu, sigma = 1, 0.5
  X = stats.truncnorm(
    (lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)

  if not normal:
      rand_num = np.random.uniform(0.5,max_rand,1).item()
  else:
      rand_num = X.rvs(1).item()

      while rand_num < lower or rand_num > upper:
          rand_num = X.rvs(1).item()


      

  width = int(w * rand_num)
  height = int(h * rand_num)
  dim = (width, height)
    
  # resize image
  crop = cv2.resize(crop, dim, interpolation = cv2.INTER_AREA)
  depth = cv2.resize(depth, dim, interpolation = cv2.INTER_NEAREST)[:, :, np.newaxis]

  if square:
      label *= (height / h)**2
  else:
      label *= (height / h)
  
  # depth aug

  lower, upper = 0.8, 1.2

  mu, sigma = 1, 0.5
  X = stats.truncnorm(
    (lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)

  if depth_aug:
      rand_num = X.rvs(1).item()

      while rand_num < lower or rand_num > upper:
              rand_num = X.rvs(1).item()
      
      depth = (depth * rand_num).astype(np.uint8)
    
      if square:
          label *= rand_num**2
      else:
          label *= rand_num


  
  
  return crop, depth, label
 



def get_annotation(id,input,anno_path='/content/labels'):
    anno = np.load(os.path.join(anno_path,'{:06d}.npy'.format(id)),allow_pickle=True).item()
    return anno.get(input)
    

class MiniDataset(Dataset):
    def __init__(self, base_p, label_f, depth, crops_rgb_f, aug=False, square=False, normal=False, depth_aug=False, label_name=['container capacity']):
      self.label_f = label_f #_f = folder
      self.depth = depth
      self.base = base_p
      self.label_name = label_name
      self.crops_rgb_f = crops_rgb_f
      self.samples = os.listdir(crops_rgb_f)
      self.ids = [ int(x.split('.')[0]) for x in self.samples]
      self.transform = transforms.Compose([
                                             transforms.Resize((320, 320)),
                                             transforms.ToTensor(),
                                             transforms.ConvertImageDtype(torch.float),
                                             ])
      self.aug = aug
      self.square = square
      self.normal = normal
      self.depth_aug = depth_aug
    def __len__(self):
      return len(self.ids)

    def __getitem__(self, idx):
      id_ = self.ids[idx]
        
      # depth
      depth = np.asarray(Image.open(os.path.join(self.depth,'{:06d}.png'.format(id_))))[:,:,np.newaxis]
      
      # rgb_cropped
      crop = np.asarray(Image.open(os.path.join(self.crops_rgb_f,'{:06d}.png'.format(id_))))
      # label
      label = np.array([get_annotation(id_,name,os.path.join(self.base, 'labels')) for name in self.label_name]).astype(np.float)

      if self.aug:
          crop, depth, label = randomlyAug(crop, depth, label, max_val=640, square=self.square, normal=self.normal, depth_aug=self.depth_aug)

      h, w, c = crop.shape

      resX = 640 - h
      resY = 640 - w

      up = resX // 2
      down = up
      if resX % 2 != 0:
        down +=1

      left = resY // 2
      right = left

      if resY % 2 != 0:
        left += 1

      padding = transforms.Pad((left, up, right, down))

    
      image = Image.fromarray(np.concatenate((crop, depth), axis=2))
      image = padding(image)
      image = self.transform(image)
      

      return image, label


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



class Padding(object):
    def __init__(self, seq_len):
        self.seq_len = seq_len

    def __call__(self, sample, pred):
        #np.clip(pred, 0,1,out=pred)
        sample_len, input_dim = sample.shape
        #for i in range(sample_len):
        #    sample[i, :] *= pred[i]

        if sample_len >= self.seq_len:
            features = sample[:self.seq_len, :]
            return features
        else:
            start_seq = np.random.randint(0, self.seq_len - sample_len+1)
            #ini=[1]+[0]*(input_dim-1)
            ini=[0]*(input_dim)
            features = np.full((self.seq_len, input_dim),ini, dtype = float)
            features[start_seq:start_seq+sample_len, :] = sample
            return features

class MyLSTMDataset(torch.utils.data.Dataset):
    def __init__(self,root_pth,label=None, test=False,transform = None, padding_size = 100):
        class_num=3
        self.mid_pth = os.path.join(root_pth,'features', 'T2_mid_test')
        self.pred_pth = os.path.join(root_pth,'features', 'T2_pred_test')
        self.label = label  # gt['filling_level'].to_numpy()
        self.is_test=test
        self.each_class_size = []
        self.each_class_sum = [0]*class_num
        for i in range(class_num):
            self.each_class_size.append(np.count_nonzero(self.label==i))
        mx=0
        mn=1000
        len_mx = 0
        
        if label is None:
            self.label = np.zeros((len(os.listdir(self.mid_pth))))
        
        for idx in range(len(os.listdir(self.mid_pth))):
            data=np.load(os.path.join(self.mid_pth, "{0:06d}".format(idx) + '.npy'), allow_pickle=True)
            self.each_class_sum[self.label[idx]]+=data.shape[0]
            if data.shape[0] > len_mx:
                len_mx=data.shape[0]
            tmp_max=np.max(data)
            tmp_min=np.min(data)
            if mx<tmp_max:
                mx=tmp_max
            if mn>tmp_min:
                mn=tmp_min
        self.mn=mn
        self.mx=mx
        self.pad = Padding(padding_size)
        print(len_mx)
            
    def __len__(self):
        return self.label.shape[0]
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        lbl = -1

        if self.is_test is False:
            lbl = self.label[idx]
            
        data=np.load(os.path.join(self.mid_pth, "{0:06d}".format(idx) + '.npy'), allow_pickle=True)
        pred=np.load(os.path.join(self.pred_pth, "{0:06d}".format(idx) + '.npy'), allow_pickle=True)
        data = (data-self.mn)/(self.mx-self.mn)
        data = self.pad(data, pred)

        #np.clip(data, 0,1,out=data)
        data=torch.from_numpy(data.astype(np.float32))
        return data , lbl
            
    def get_each_class_size(self):
        return np.array(self.each_class_size)

    def get_each_class_avg_len(self):
        each_class_avg_len =  np.array(self.each_class_sum)/np.array(self.each_class_size)
        all_class_avg_len = np.sum(np.array(self.each_class_sum))/np.sum(np.array(self.each_class_size))
        return each_class_avg_len, all_class_avg_len
    
class BatchProcess(Dataset):
    """
    type  = train or test   
    """
    def __init__(self,rgbs,depths):
        
        self.rgbs = rgbs
        self.depths = depths
        self.transform = transforms.Compose([transforms.Resize((320, 320)),
                                             transforms.ToTensor(),
                                             transforms.ConvertImageDtype(torch.float),
                                             ])

    def __len__(self):
        return len(self.rgbs)

    def __getitem__(self, idx):
    
        depth = np.asarray(self.depths[idx])[:,:,np.newaxis]

        # rgb_cropped
        crop = np.asarray(self.rgbs[idx])
        h, w, c = crop.shape
        resX = 640 - h
        resY = 640 - w
        up = resX // 2
        down = up
        if resX % 2 != 0:
            down +=1
        left = resY // 2
        right = left
        if resY % 2 != 0:
            left += 1
        padding = transforms.Pad((left, up, right, down))
        image = Image.fromarray(np.concatenate((crop, depth), axis=2))
        image = padding(image)
        image = self.transform(image)

        return image

class MyLSTMDataset_combine(torch.utils.data.Dataset):
    def __init__(self,root_pth,label=None, test=False,transform = None, padding_size = 100):
        class_num=3
        self.mid_pth = os.path.join(root_pth,'features', 'T2_mid_test')
        self.pred_pth = os.path.join(root_pth,'features', 'T2_pred_test')
        self.video_pth = os.path.join(root_pth,'features_video_test')
        self.label = label  # gt['filling_level'].to_numpy()
        self.is_test=test
        self.each_class_size = []
        self.each_class_sum = [0]*class_num
        for i in range(class_num):
            self.each_class_size.append(np.count_nonzero(self.label==i))
        mx=0
        mn=1000
        len_mx = 0
        
        if label is None:
            self.label = np.zeros((len(os.listdir(self.mid_pth))))

        for idx in range(len(os.listdir(self.mid_pth))):
            data=np.load(os.path.join(self.mid_pth, "{0:06d}".format(idx) + '.npy'), allow_pickle=True)
            #self.each_class_sum[self.label[idx]]+=data.shape[0]
            if data.shape[0] > len_mx:
                len_mx=data.shape[0]
            tmp_max=np.max(data)
            tmp_min=np.min(data)
            if mx<tmp_max:
                mx=tmp_max
            if mn>tmp_min:
                mn=tmp_min
        self.mn=mn
        self.mx=mx
        self.pad = Padding(padding_size)
        
    def __len__(self):
        return self.label.shape[0]
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        lbl = -1

        if self.is_test is False:
            lbl = self.label[idx]
          
        data=np.load(os.path.join(self.mid_pth, "{0:06d}".format(idx) + '.npy'), allow_pickle=True)
        data_video=np.load(os.path.join(self.video_pth, "{0:06d}".format(idx) + '.npy'), allow_pickle=True)
        pred=np.load(os.path.join(self.pred_pth, "{0:06d}".format(idx) + '.npy'), allow_pickle=True)
        data = (data-self.mn)/(self.mx-self.mn)
        data = self.pad(data, pred)
        data_video =self.pad(self.pad(data, pred), pred)
        data_combine=np.concatenate((data, data_video),axis=0)
        data_combine=torch.from_numpy(data_combine.astype(np.float32))
        return data_combine , lbl
              
    def get_each_class_size(self):
        return np.array(self.each_class_size)

    def get_each_class_avg_len(self):
        each_class_avg_len =  np.array(self.each_class_sum)/np.array(self.each_class_size)
        all_class_avg_len = np.sum(np.array(self.each_class_sum))/np.sum(np.array(self.each_class_size))
        return each_class_avg_len, all_class_avg_len
