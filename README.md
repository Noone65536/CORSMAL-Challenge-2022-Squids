# CORSMAL-Challenge-2022-Squids

The CORSMAL challenge focuses on the estimation of the capacity, dimensions, and mass of containers, the type, mass, and filling (percentage of the container with content), and the overall mass of the container and filling. The specific containers and fillings are unknown to the robot: the only prior is a set of object categories (drinking glasses, cups, food boxes) and a set of filling types (water, pasta, rice). [Click here](https://corsmal.eecs.qmul.ac.uk/challenge.html)

![TaskOverview](https://corsmal.eecs.qmul.ac.uk/images/challenge/diagram_tasks.png)

## Installation

0. Clone repository

    <code>git clone https://github.com/Noone65536/CORSMAL-Challenge-2022-Squids.git</code>

1. From a terminal or an Anaconda Prompt, go to project's root directory and run:

    <code>conda create --name corsmal-squids python=3.7</code> 

    <code>conda activate corsmal-squids</code>

    <code>pip install -r requirements.txt</code>

    This will create a new conda environment and install all software dependencies. We also provide the .yaml file if you'd like to install packages using conda.

2. Install manually  
If you could not insall packages using requiremens.txt through pip or using environments.yaml through conda,
you can install packages manually using following instructions:

  -  `conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch`  
    you need torchvision>=0.8.  
  -  `pip install opencv-python`  
    Do not use "conda" here as "conda" installs opencv3.4 which conflicts with torchvision>0.8  
  -   Other packages can be installed using:  
    `conda install scipy tqdm pandas pyyaml requests seaborn matplotlib ipywidgets ipython`

## Testing 

### Test the Task1 and Task2

![Task12](./Images/Task12.png)

<code>python test_task12.py --dataset datapath</code>

Arguments:
- `--datapath`: Absolute or relative path to dataset you would like to test. Notice that this is the path to the sub-dataset that directly contains the files to be tested, such as `crosmal_datset/train` or `crosmal_datset/test_pub `

- `--csv`: (optional) Absolute or relative path to the csv file. Default output directory is this project's root directory. Please noted that the program reads the csv files and write testing results to the csv file directly. We hope that the indexes and columns of the csv file is the same as the  `public_test_set.csv`

For example: <code>python CORSMAL_test.py --dataset crosmal_datasets/test_pub --csv public_test_set.csv</code>

Noted that this will generate some folders in the project's root directory: 
- features : audio features (.npy)
- features_video_test : frames' features (.npy)
- features_frames_test: frames (.npy)
- results : results to be calculated as votes (.json)

### Test the Task1 (audio only)

<code>python test_task1_audio.py --dataset datapath</code>

Arguments:
- `--datapath`: Absolute or relative path to dataset you would like to test. Notice that this is the path to the sub-dataset that directly contains the files to be tested, such as `crosmal_datset/train` or `crosmal_datset/test_pub `

- `--csv`: (optional) Absolute or relative path to the csv file. Default output directory is this project's root directory. Please noted that the program reads the csv files and write testing results to the csv file directly. We hope that the indexes and columns of the csv file is the same as the  `task1_audio.csv`

### Test the Task3, Task4 and Task5

![Task12](./Images/Task345.png)



<code>python test_task345.py --dataset datapath</code>

Arguments: (The same as the Task1 and Task2)

- `--datapath`: Absolute or relative path to dataset you would like to test. Notice that this is the path to the sub-dataset that directly contains the files to be tested, such as `corsmal_datset/train` or `corsmal_datset/test_pub `
- `--csv`: (optional) Absolute or relative path to the csv file. Default output directory is this project's root directory.



Please note that you may like to try following code for downloading the YOLOv5 weights

```
import torch
yolo = torch.hub.load('yolov5','custom', path='yolov5/yolov5s6.pt', source='local', force_reload=True)
```



## Complexity

#### Task1: 

Audio encoder: Params: 2,674,931 size: 10.7MB

RGB encoder: Params: 2,670,929 size: 10.7MB

LSTM: Params: 6,369,795 size: 24.8MB 

#### Task2: 

Params: 2,674,931 size: 10.7MB

#### Task345: 

params: 2,671,217 size: 10.7MB

## Our Environments

GPU: Tesla-V100 32G

Operating system: Linux and Colab

