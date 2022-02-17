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
from tqdm import tqdm
#from yolov5.utils.autobatch import autobatch

OBJECT_LIST = ['vase','cup','book','bottle','laptop','wine glass','surfboard','skateboard']

parser = argparse.ArgumentParser()
parser.add_argument('--dataset',help='folder containt datasets (test_pub)',default = '/jmain02/home/J2AD007/txk47/cxz00-txk47/corsmal/datasets/corsmal_all/test_pub')
parser.add_argument('--csv',help='csv file to write into',default = 'public_test_set.csv')
parser.add_argument('--bs_yolo',help='batch_size of yolo',default = 128)
parser.add_argument('--bs_model',help='batch_size of the models',default = 128)
args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device1 = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
device2 = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
print('Using device:', device)
print('Using device:', device1)
print('Using device:', device2)

rgb_path = os.path.join(args.dataset,'view3','rgb')
depth_path = os.path.join(args.dataset,'view3','depth')

yolo = torch.hub.load('yolov5','custom', path='yolov5/yolov5s6.pt', source='local', force_reload=True)
yolo.eval()
#yolo = autobatch(yolo)
videos = os.listdir(rgb_path)
videos.sort()

model3 = mbv2_ca().to(device).eval()
model3.load_state_dict(torch.load('weights-new/task3.pth'), strict=True)
#'/jmain02/home/J2AD007/txk47/hxw74-txk47/corsmal_challenge/task3/view3_cap10_pre/mobile-ca70.62.pth'

model4 = mbv2_ca().to(device1).eval()
model4.load_state_dict(torch.load('weights-new/task4.pth'), strict=True)

model5t = mbv2_ca().to(device1).eval()
model5t.load_state_dict(torch.load('weights-new/task5-wt.pth'), strict=True)
#/jmain02/home/J2AD007/txk47/hxw74-txk47/corsmal_challenge/task5/wt_Hey_nodepth/mobile-ca90.03.pth'

model5b = mbv2_ca().to(device2).eval()
model5b.load_state_dict(torch.load('weights-new/task5-wb.pth'), strict=True)
#'/jmain02/home/J2AD007/txk47/hxw74-txk47/corsmal_challenge/task5/wb_Hey_nodepth/mobile-ca88.16.pth'

model5h = mbv2_ca().to(device2).eval()
model5h.load_state_dict(torch.load('weights-new/task5-h.pth'), strict=True)
#'/jmain02/home/J2AD007/txk47/hxw74-txk47/corsmal_challenge/task5/view3_height/mobile-ca89.04.pth'

print(f'found {len(videos)} videos')
public_test_set = pd.read_csv(args.csv)

est_capacity = []
est_mass= []
est_wt= []
est_wb= []
est_height= []

def get_est(est_list):
    seq = np.array(est_list)
    mean = np.mean(seq)
    std = np.std(seq)
    finalseq = [x for x in seq if (x>=mean-2*std and x<=mean+2*std)]
    est_mean = np.mean(finalseq)
    est_mid = np.median(est_list)

    return est_mean

with torch.no_grad():
    
    for video in tqdm(videos):
        
        #print(f'processing video: {video}')
        frames,nums = extract_frames(os.path.join(rgb_path,video))
        depths = extract_depths_all(video.split('.')[0],nums,depth_path) 
        results =[] 
        for i in range(0,len(frames),int(args.bs_yolo)):
            frame_batch = frames[i:i+int(args.bs_yolo)]
            result = yolo(frame_batch)
            result = result.pandas().xyxy
            results.extend(result) 
        cropped_rgbs,cropped_depths = crop_images(results,frames,depths,OBJECT_LIST)
        assert(len(cropped_rgbs) == len(cropped_depths))
        
        if len(cropped_rgbs) == 0:
            print(f'No object detected in video {video}, use average value')
            est_capacity.append(1600) # 3200 1239.84 471
            est_mass.append(35) # 59  31.0 15
            est_wt.append(136) # 193 135.0, 80
            est_wb.append(136) # 193 135.0, 80 
            est_height.append(178)  # 241 164.0 131
            #print(f'{video},1600,35,136,136,178',file=open('test_final.txt', 'a'))
        
        else:
            batch = BatchProcess(cropped_rgbs,cropped_depths)
            dataloader =  DataLoader (batch, int(args.bs_model), shuffle=False)
            ca_l = []
            mass_l = []
            wt_l = []
            wb_l = []
            h_l = []
            
            for i,data in enumerate(dataloader,0):
                #data = data.to(device)
                ca = model3(data.to(device))
                mass = model4(data.to(device1))
                wt = model5t(data.to(device1))
                wb = model5b(data.to(device2))
                h = model5h(data.to(device2))

                ca_l.extend(ca.squeeze(1).cpu())
                mass_l.extend(mass.squeeze(1).cpu())
                wt_l.extend(wt.squeeze(1).cpu())
                wb_l.extend(wb.squeeze(1).cpu())
                h_l.extend(h.squeeze(1).cpu())
            
            est_capacity.append(get_est(ca_l))
            est_mass.append(get_est(mass_l))
            est_wt.append(get_est(wt_l))
            est_wb.append(get_est(wb_l))
            est_height.append(get_est(h_l))

            #print(f'{video},{get_est(ca_l)},{get_est(mass_l)},{get_est(wt_l)},{get_est(wb_l)},{get_est(h_l)}',file=open('test_final.txt', 'a'))

est_capacity = np.array(est_capacity)
est_mass= np.array(est_mass)
est_wt= np.array(est_wt)
est_wb= np.array(est_wb)
est_height= np.array(est_height)

public_test_set.iloc[:, 1] = est_capacity.astype(np.int32)
public_test_set.iloc[:, 2] = est_mass.astype(np.int32)
public_test_set.iloc[:, 13] = est_wt.astype(np.int32)
public_test_set.iloc[:, 14] = est_wb.astype(np.int32)
public_test_set.iloc[:, 15] = est_height.astype(np.int32)
public_test_set.to_csv(args.csv,index=False)
print(f'writing results to {args.csv}')
print("Done!")

#print(torch.cuda.max_memory_reserved(0)/1024/1024)
#print(torch.cuda.max_memory_reserved(1)/1024/1024)
#print(torch.cuda.max_memory_reserved(2)/1024/1024)
#print(torch.cuda.max_memory_reserved(3)/1024/1024)

    #     
    #     model = torch.hub.load('ultralytics/yolov5', 'yolov5s', autoshape=False)
    #     print(autobatch(model))
