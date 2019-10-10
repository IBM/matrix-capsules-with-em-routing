#!/bin/sh
python3 train_val.py --dataset cifar10 --batch_size=32 --num_gpus=2 --A=32 --B=32 --C=32 --D=32 --epoch=50 --drop_rate=0.5 --dropout=True --dropconnect=False --logdir dropout
# python3 train_val.py --batch_size 8 --num_gpus 1 --epoch 1 --logdir test
