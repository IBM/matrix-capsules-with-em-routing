#!/bin/sh
python3 train_val.py --dataset mnist --batch_size 64 --num_gpus 1 --epoch 1 --logdir test
# python3 train_val.py --batch_size 8 --num_gpus 1 --epoch 1 --logdir test
