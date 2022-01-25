import scipy
#import librosa
import pandas as pd
import os
import numpy as np
from tqdm.notebook import tqdm
import scipy.io.wavfile
import time
import IPython
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import Subset
import json
from my_utils import *
from dataset import *
from my_models import *
from helper import train_audio, evaluate_audio
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset',help='folder containt datasets (test_pub)',default = '/jmain02/home/J2AD007/txk47/cxz00-txk47/corsmal/datasets/corsmal_all/test_pub')
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

public_test_set = pd.read_csv('public_test_set.csv')
model_pth = 'weights/task2.pth'

model_pretrained = mbv2_ca(in_c=8, num_classes=4)
model_pretrained.load_state_dict(torch.load(model_pth))
model_pretrained.to(device)
model_pretrained.eval()

voting_dir = './results'
os.makedirs(voting_dir, exist_ok=True)
audio_folder = os.path.join(args.dataset,'audio')
#'/jmain02/home/J2AD007/txk47/cxz00-txk47/corsmal/datasets/corsmal_all/test_pub/audio'

tsk2_list = voting(audio_folder, voting_dir, model_pretrained, device, save_size=64)

# Save to pandas
tsk2_np = np.array(tsk2_list)
public_test_set.iloc[:, 4] = (tsk2_np==0).astype(np.int)
public_test_set.iloc[:, 5] = (tsk2_np==1).astype(np.int)
public_test_set.iloc[:, 6] = (tsk2_np==2).astype(np.int)
public_test_set.iloc[:, 7] = (tsk2_np==3).astype(np.int)
public_test_set.iloc[:, 8] = tsk2_np
public_test_set.head()
public_test_set.to_csv('public_test_set.csv',index=False)

os.makedirs('video_frames_test',exist_ok=True)
video_folder = os.path.join(args.dataset,'view3','rgb')
#'/jmain02/home/J2AD007/txk47/cxz00-txk47/corsmal/datasets/corsmal_all/test_pub/view3/rgb'
videoPreprocessing_t1(audio_folder, video_folder)

mobileNet = 'weights/task1_ve.pth'
video_folder = 'video_frames_test'

model = MBV2_CA(in_c=3, num_classes=3)
model.load_state_dict(torch.load(mobileNet),strict=False)
model.to(device)
model.eval()

os.makedirs('features_video_test',exist_ok=True)
videoPreprocessing_feature(video_folder, model, device)

mobileNet = 'weights/task1_ae.pth'
base_path = 'features'
T2_mid_dir = os.path.join(base_path, 'T2_mid_test')
T2_pred_dir = os.path.join(base_path, 'T2_pred_test')
os.makedirs(T2_mid_dir,exist_ok=True)
os.makedirs(T2_pred_dir,exist_ok=True)

model = MBV2_CA(in_c=8, num_classes=4)
model.load_state_dict(torch.load(mobileNet))
model.to(device)
model.eval()

audioPreprocessing_t1(audio_folder,T2_mid_dir, T2_pred_dir, model, device)

base_path = ''
path = 'weights/task1_combine.pth'
model = CNN_LSTM(input_size=1280).to(device)
test_set = MyLSTMDataset_combine(base_path)
test_loader = DataLoader(test_set,batch_size=1,shuffle=False)

model.load_state_dict(torch.load(path),strict=False)
model.to(device)
model.eval()

pred_list = voting_t1(model, test_loader, device)
pred_list = np.array(pred_list)
pred_list[tsk2_np==0] = 0
public_test_set.iloc[:, 9] = (pred_list==0).astype(np.int)
public_test_set.iloc[:, 10] = (pred_list==1).astype(np.int)
public_test_set.iloc[:, 11] = (pred_list==2).astype(np.int)
public_test_set.iloc[:, 12] = pred_list
public_test_set.head()
public_test_set.to_csv('public_test_set.csv',index=False)