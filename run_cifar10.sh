#!/bin/sh
python3 train_val.py --dataset cifar10 --batch_size=64 --num_gpus=4 --A=128 --B=8 --C=32 --D=32 --epoch=100 --drop_rate=0 --logdir largedropconn
# python3 train_val.py --batch_size 8 --num_gpus 1 --epoch 1 --logdir test
