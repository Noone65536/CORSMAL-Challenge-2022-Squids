#!/bin/bash

DATASET=/import/smartcameras-002/Datasets/CCM-2.0/test_priv/
OUTFILE=squids.csv
YOLO_BATCH_SIZE=24
MODEL_BATCH_SIZE=24

python test_task345_4gpus.py --dataset $DATASET --csv $OUTFILE --bs_yolo $YOLO_BATCH_SIZE	--bs_model $MODEL_BATCH_SIZE
