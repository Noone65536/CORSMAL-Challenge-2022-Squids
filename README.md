# CORSMAL-Challenge-2022-Squids

## Installation

0. Clone repository

<code>git clone https://github.com/Noone65536/CORSMAL-Challenge-2022-Squids.git</code>

1. From a terminal or an Anaconda Prompt, go to project's root directory and run:

<code>conda create --name <code>corsmal-squids</code> 

<code>conda activate corsmal-squids</code>
 
<code>pip install -r requirements</code>
 

This will create a new conda environment and install all software dependencies.

## Testing 

### Test the Task1 and Task2

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

### Test the Task3, Task4 and Task5

<code>python test_task345.py --dataset datapath</code>

Arguments: (The same as the Task1 and Task2)

- `--datapath`: Absolute or relative path to dataset you would like to test. Notice that this is the path to the sub-dataset that directly contains the files to be tested, such as `crosmal_datset/train` or `crosmal_datset/test_pub `

- `--csv`: (optional) Absolute or relative path to the csv file. Default output directory is this project's root directory.

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

