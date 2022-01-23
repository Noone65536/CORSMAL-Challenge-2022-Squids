'''
generate testing result of Task3,Task4,Task5 of the test_pub datset
'''
import argparse
import os
import torch
from torch.utils.data import DataLoader
from dataset import BatchProcess
from my_utils import extract_frames,extract_depths_all
from my_utils import crop_images
from my_models import mbv2_ca
import numpy as np
import pandas as pd
#from yolov5.utils.autobatch import autobatch

OBJECT_LIST = ['vase','cup','book','bottle','laptop','wine glass','surfboard','skateboard']

parser = argparse.ArgumentParser()
parser.add_argument('--dataset',help='folder containt datasets (test_pub)',default = 'datasets/corsmal_all/test_pub')
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

rgb_path = os.path.join(args.dataset,'view3','rgb')
depth_path = os.path.join(args.dataset,'view3','depth')

yolo = torch.hub.load('yolov5','custom', path='yolov5/yolov5s6.pt', source='local', force_reload=True)
yolo.eval()
#yolo = autobatch(yolo)
videos = os.listdir(rgb_path)
videos.sort()
model = mbv2_ca().to(device)
model.eval()

print(f'found {len(videos)} videos')

est_capacity = []
est_mass= []
est_wt= []
est_wb= []
est_height= []

def get_est(weight,model,dataloader):
    weights = torch.load(weight)
    model.load_state_dict(weights, strict=True)
    
    est_list=[]
    for i,data in enumerate(dataloader,0):
        data = data.to(device)
        est = model(data)
        est_list.append(est.cpu().item())
    
    seq = np.array(est_list)
    mean = np.mean(seq)
    std = np.std(seq)
    finalseq = [x for x in seq if (x>mean-2*std and x<mean+2*std)]
    est_mean = np.mean(finalseq)
    est_mid = np.median(est_list)
    return est_mean,est_mid

with torch.no_grad():
    
    for video in videos:
        
        print(f'processing video: {video}')
        frames,nums = extract_frames(os.path.join(rgb_path,video))
        depths = extract_depths_all(video.split('.')[0],nums,depth_path) 
        
        results = yolo(frames)
        cropped_rgbs,cropped_depths = crop_images(results,frames,depths,OBJECT_LIST)
        assert(len(cropped_rgbs) == len(cropped_depths))
        
        if len(cropped_rgbs) == 0:
            print(f'No object detected in video {video}, use average value')
            est_capacity.append(1600) # 3200 1239.84 471
            est_mass.append(35) # 59  31.0 15
            est_wt.append(136) # 193 135.0, 80
            est_wb.append(136) # 193 135.0, 80 
            est_height.append(178)  # 241 164.0 131
        
        else:
            batch = BatchProcess(cropped_rgbs,cropped_depths)
            dataloader =  DataLoader (batch, batch_size=1, shuffle=False)

            print('estimating Task 3: Capacity ...')
            est_mean,est_mid = get_est('weights/task3.pth',model,dataloader)
            est_capacity.append(est_mean)
            
            print('estimating Task 4: mass ...')
            est_mean,est_mid = get_est('weights/task4.pth',model,dataloader)
            est_mass.append(est_mean)

            print('estimating Task 5: wideth top ...')
            est_mean,est_mid = get_est('weights/task5_wt.pth',model,dataloader)
            est_wt.append(est_mean)
                 
            print('estimating Task 5: width bottom ...')
            est_mean,est_mid = get_est('weights/task5_wb.pth',model,dataloader)
            est_wb.append(est_mean)

            print('estimating Task 5: height ...')
            est_mean,est_mid = get_est('weights/task5_h.pth',model,dataloader)
            est_height.append(est_mean)

public_test_set = pd.read_csv('public_test_set.csv')

est_capacity = np.array(est_capacity)
est_mass= np.array(est_mass)
est_wt= np.array(est_wt)
est_wb= np.array(est_wb)
est_height= np.array(est_height)

public_test_set.iloc[:, 1] = est_capacity.astype(np.int)
public_test_set.iloc[:, 2] = est_mass.astype(np.int)
public_test_set.iloc[:, 13] = est_wt.astype(np.int)
public_test_set.iloc[:, 14] = est_wb.astype(np.int)
public_test_set.iloc[:, 15] = est_height.astype(np.int)

    #     
    #     model = torch.hub.load('ultralytics/yolov5', 'yolov5s', autoshape=False)
    #     print(autobatch(model))
