import numpy as np
import os
import scipy
from tqdm.notebook import tqdm
import time
import torch
import json
import cv2

def voting(audio_folder, voting_dir, model_pretrained, device, save_size=64):
    mfcc_MAX_VALUE=194.19187653405487
    mfcc_MIN_VALUE=-313.07119549054045

    t2_MAX_VALUE = 57.464638
    t2_MIN_VALUE = -1.1948369
    start = time.time()

    audio_paths = [os.path.join(audio_folder, path) for path in sorted(os.listdir(audio_folder))]
    save_data = {}
    data_num = 0
    for i, path in enumerate(audio_paths):
      count_pred = [0] * 4
      pred_list = []
      sample_rate, signal = scipy.io.wavfile.read(path)
      ap = AudioProcessing(sample_rate,signal)
      mfcc = ap.calc_MFCC()
      mfcc_length=mfcc.shape[0]
      f_step=int(mfcc.shape[1]*0.25)
      f_length=mfcc.shape[1]
      save_mfcc_num=int(np.ceil(float(np.abs(mfcc_length - save_size)) /f_step))
      for i in range(save_mfcc_num):
          tmp_mfcc = mfcc[i*f_step:save_size+i*f_step,: ,:]
          tmp_mfcc= (tmp_mfcc-mfcc_MIN_VALUE)/(mfcc_MAX_VALUE-mfcc_MIN_VALUE)
          tmp_mfcc=tmp_mfcc.transpose(2,0,1)
          audio=torch.from_numpy(tmp_mfcc.astype(np.float32))
          audio=torch.unsqueeze(audio, 0)
          audio = audio.to(device)
    
          with torch.no_grad():
            pred_T2 = model_pretrained.forward(audio)
            _,pred_T2=torch.max(pred_T2,1)
            count_pred[pred_T2.item()]+=1
            pred_list.append(pred_T2.item())
      if  count_pred[1]>5 or count_pred[2]>5 or count_pred[3]>5:
        final_pred_T2=count_pred[1:4].index(max(count_pred[1:4]))+1
      else:
        final_pred_T2=0
      
      file_name = path.split(os.path.sep)[-1].replace('.wav', '')
      #print("sequence:{}, frequency:{}".format(file_name, count_pred))
      to_save_data = {"data_num":data_num,
                      "file":file_name,
                      "count_pred":count_pred,
                      "final_pred":final_pred_T2,
                      'pred':pred_list}
      save_data["{}".format(file_name)] = to_save_data
      data_num+=1
    
    with open (os.path.join(voting_dir, "voting.json"), 'w') as f:
      json.dump(save_data, f, indent=2, ensure_ascii=False)
      elapsed_time = time.time() - start
      print("elapsed_time:{}".format(elapsed_time) + "sec")


def audioPreprocessing_t1(audio_folder, gt,T2_mid_dir, T2_pred_dir, model, device):
    audio_paths = [os.path.join(audio_folder, path) for path in sorted(os.listdir(audio_folder))]
    save_size=64
    ratio_step = 0.25
    count = 0
    MAX_VALUE=194.19187653405487
    MIN_VALUE=-313.07119549054045
    
    pbar = tqdm(total=len(audio_paths))
    
    for i, path in enumerate(audio_paths):
      id = i
      start_time = gt[gt.id==id]['start'].item()
      end_time = gt[gt.id==id]['end'].item()
      filling_type = gt[gt.id==id]['filling_type'].item()
      datalist = []
      predlist = []
      sample_rate, signal = scipy.io.wavfile.read(path)
      ap = AudioProcessing(sample_rate,signal,nfilt=save_size)
      mfcc = ap.calc_MFCC()
      mfcc_length=mfcc.shape[0]
    
      if mfcc_length < save_size:
        print("file {} is too short".format(id))
      else:
        f_step=int(mfcc.shape[1]*ratio_step)
        f_length=mfcc.shape[1]
        save_mfcc_num=int(np.ceil(float(np.abs(mfcc_length - save_size)) / f_step))
    
        for i in range(save_mfcc_num):
          tmp_mfcc = mfcc[i*f_step:save_size+i*f_step,: ,:]
          tmp_mfcc= (tmp_mfcc-MIN_VALUE)/(MAX_VALUE-MIN_VALUE)
          tmp_mfcc=tmp_mfcc.transpose(2,0,1)
          audio=torch.from_numpy(tmp_mfcc.astype(np.float32))
          audio=torch.unsqueeze(audio, 0)
          audio = audio.to(device) 
          feature, pred=model.extract(audio)
          _,pred=torch.max(pred,1)
          datalist.append(feature.to('cpu').detach().numpy().copy())
          predlist.append(pred.item())
        datalist = np.squeeze(np.array(datalist))
        predlist = np.squeeze(np.array(predlist))
        np.save(os.path.join(T2_mid_dir, "{0:06d}".format(id)), datalist)
        np.save(os.path.join(T2_pred_dir, "{0:06d}".format(id)), predlist)
    
      pbar.update()


def audioPreprocessing(audio_folder, gt, base_path, mfcc_path):
    audio_paths = [os.path.join(audio_folder, path) for path in sorted(os.listdir(audio_folder))]
    save_size=64
    ratio_step = 0.25
    count = 0
    pouring_or_shaking_list = []
    file_idx_list = []
    filling_type_list = []
    pbar = tqdm(total=len(audio_paths))
    
    for i, path in enumerate(audio_paths):
      id = i
      start_time = gt[gt.id==id]['start'].item()
      end_time = gt[gt.id==id]['end'].item()
      filling_type = gt[gt.id==id]['filling_type'].item()
      sample_rate, signal = scipy.io.wavfile.read(path)
      ap = AudioProcessing(sample_rate,signal,nfilt=save_size)
      mfcc = ap.calc_MFCC()
      raw_frames = ap.cal_frames()
      mfcc_length=mfcc.shape[0]
      
      video_name = os.path.join(video_folder,'{:06d}.mp4'.format(id)) # modify the video name here
      video_frame_list = find_corres_video_frame(video_name,mfcc_length,ap.frame_step,ap.frame_length,signal.shape[0])
    
      if mfcc_length < save_size:
        print("file {} is too short".format(id))
      else:
        f_step=int(mfcc.shape[1]*ratio_step)
        f_length=mfcc.shape[1]
        save_mfcc_num=int(np.ceil(float(np.abs(mfcc_length - save_size)) / f_step))
    
        for i in range(save_mfcc_num):
          count += 1
          frame = video_frame_list[i] # extract the corresponding frame here
          tmp_mfcc = mfcc[i*f_step:save_size+i*f_step,: ,:]
          if start_time == -1:
              pouring_or_shaking_list.append(0)
          elif start_time/ap.signal_length_t*mfcc_length<i*f_step+f_length*0.75 and end_time/ap.signal_length_t*mfcc_length>i*f_step+f_length*0.25:
              pouring_or_shaking_list.append(1) 
          else:
              pouring_or_shaking_list.append(0)
          
          filling_type_list.append(filling_type)
          file_idx_list.append(id)
    
          np.save(os.path.join(mfcc_path, "{0:06d}".format(count)), tmp_mfcc)
      pbar.update()
    
    
    np.save(os.path.join(base_path, 'audios', 'pouring_or_shaking'), np.array(pouring_or_shaking_list) )
    np.save(os.path.join(base_path, 'audios', 'filling_type'), np.array(filling_type_list))
    
    
class AudioProcessing():
    def __init__(self,sample_rate,signal,frame_length_t=0.025,frame_stride_t=0.01,nfilt =64):
        
        self.sample_rate=sample_rate
        self.signal = signal
        self.frame_length_t=frame_length_t
        self.frame_stride_t=frame_stride_t
        self.signal_length_t=float(signal.shape[0]/sample_rate)
        self.frame_length=int(round(frame_length_t * sample_rate)) #number of samples
        self.frame_step=int(round(frame_stride_t * sample_rate))
        self.signal_length = signal.shape[0]
        self.nfilt=nfilt
        self.num_frames = int(np.ceil(float(np.abs(self.signal_length - self.frame_length)) / self.frame_step))
        self.pad_signal_length=self.num_frames * self.frame_step + self.frame_length
        self.NFFT=512
        
    def cal_frames(self):
        z = np.zeros([self.pad_signal_length - self.signal_length,8])
        pad_signal = np.concatenate([self.signal, z], 0)
        indices = np.tile(np.arange(0, self.frame_length), (self.num_frames, 1)) + np.tile(np.arange(0, self.num_frames * self.frame_step, self.frame_step), (self.frame_length, 1)).T
        frames = pad_signal[indices.astype(np.int32, copy=False)]
        return frames
        
    def calc_MFCC(self):
        # 291576
        pre_emphasis=0.97

        # (n,8)
        emphasized_signal=np.concatenate([self.signal[0,:].reshape([1,-1]),  self.signal[1:,:] - pre_emphasis * self.signal[:-1,:]], 0)
        z = np.zeros([self.pad_signal_length - self.signal_length,8])
        pad_signal = np.concatenate([emphasized_signal, z], 0)
        indices = np.tile(np.arange(0, self.frame_length), (self.num_frames, 1)) + np.tile(np.arange(0, self.num_frames * self.frame_step, self.frame_step), (self.frame_length, 1)).T
        frames = pad_signal[indices.astype(np.int32, copy=False)]
        frames=frames*np.hamming(self.frame_length).reshape(1,-1,1)
        frames=frames.transpose(0,2,1)
        mag_frames = np.absolute(np.fft.rfft(frames,self.NFFT))
        pow_frames = ((1.0 / self.NFFT) * ((mag_frames) ** 2))
        filter_banks = np.dot(pow_frames, self.cal_fbank().T)
        filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)
        filter_banks = 20 * np.log10(filter_banks)  # dB
        filter_banks =filter_banks.transpose(0,2,1)
        
        return filter_banks
           
    def cal_fbank(self):
        
        low_freq_mel = 0
        high_freq_mel = (2595 * np.log10(1 + (self.sample_rate / 2) / 700))  
        mel_points = np.linspace(low_freq_mel, high_freq_mel, self.nfilt + 2)  
        hz_points = (700 * (10**(mel_points / 2595) - 1)) 
        bin = np.floor((self.NFFT + 1) * hz_points / self.sample_rate)
        fbank = np.zeros((self.nfilt, int(np.floor(self.NFFT / 2 + 1))))
        for m in range(1, self.nfilt + 1):
            f_m_minus = int(bin[m - 1])   # left
            f_m = int(bin[m])             # center
            f_m_plus = int(bin[m + 1])    # right

            for k in range(f_m_minus, f_m):
                fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
            for k in range(f_m, f_m_plus):
                fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])
        return fbank


def find_corres_video_frame(video_name,mfcc_length,frame_step,frame_length,signal_length):
    '''
    Args:
        video name
        mfcc_length : mfcc.shape[0]
        frame_step : 441 if stride_t = 0.01s
        frame_length : 1102 if frame_length_t = 0.25s
        signal_length: signal.shape[0]

    '''
    capture = cv2.VideoCapture(video_name)
    total_frame = capture.get(cv2.CAP_PROP_FRAME_COUNT)
    video_frame_list = []
    for n in range(mfcc_length):
        mid_point = total_frame * (frame_length/2 + frame_step*n) / signal_length
        video_frame_num = round(mid_point)
        capture.set(1, video_frame_num)
        success, image = capture.read()
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        video_frame_list.append(image)
    return np.array(video_frame_list)