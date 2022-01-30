import scipy
import librosa
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
from utils import AudioProcessing, audioPreprocessing, voting, voting_t1
from dataset import *
from models import *
from helper import train_audio, evaluate_audio

parser = argparse.ArgumentParser()
parser.add_argument('--dataset',help='folder containt datasets (test_pub)',default = '/jmain02/home/J2AD007/txk47/cxz00-txk47/corsmal/datasets/corsmal_all/test_pub')
parser.add_argument('--csv',help='csv file to write into',default = 'task1_audio.csv')
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

public_test_set = pd.read_csv(args.csv)

model_pth = './models/task2.pth'

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


base_path = ''
path = 'weight/task1_audio.pth'
model = CNN_LSTM(input_size=1280).to(device)
test_set = MyLSTMDataset(base_path)
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
public_test_set.to_csv(args.csv,index=False)
