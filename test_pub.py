'''
generate testing result of Task3,Task4,Task5 of the test_pub datset
'''
import argparse
import os
import torch
import cv2
import glob
from torch.utils.data import DataLoader
from dataset import BatchProcess
from my_utils import extract_frames,extract_depths_all,read_first_yolo
from my_utils import crop_images
from icecream import ic
from models import MBV2_CA

OBJECT_LIST = ['vase','cup','book','bottle','laptop','wine glass','surfboard','skateboard']

parser = argparse.ArgumentParser()
parser.add_argument('--dataset',help='folder containt datasets (test_pub)',default = 'datasets/corsmal_all/test_pub')
args = parser.parse_args()

rgb_path = os.path.join(args.dataset,'view3','rgb')
depth_path = os.path.join(args.dataset,'view3','depth')

yolo = torch.hub.load('yolov5','custom', path='yolov5/yolov5s6.pt', source='local',force_reload=True)
yolo.eval()
videos = os.listdir(rgb_path)

model = MBV2_CA().to('cuda')
model.eval()

print(f'find {len(videos)} videos')

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
            perdict = 1600
        
        else:
            batch = BatchProcess(cropped_rgbs,cropped_depths)
            dataloader =  DataLoader (batch, batch_size=8, shuffle=False)
            
            est_capacity = 0
            est_mass=0
            est_wt=0
            est_wb=0
            est_heght=0

            print('estimating Task 3: Capacity ...')
            weights = torch.load('mbv2_ca.pth')
            model.load_state_dict(weights, strict=False)
            samples = len(cropped_rgbs)
            
            for i,data in enumerate(dataloader,0):
                est = model(data)
                est_capacity += est