#!/bin/sh
python3 train_val.py --dataset cifar10 --batch_size 64 --A 16 --B 4 --C 8 --D 8 --num_gpus 1 --epoch 20 --logdir test
# python3 train_val.py --batch_size 8 --num_gpus 1 --epoch 1 --logdir test
