#!/bin/sh
python3 train_val.py --dataset=mnist --batch_size=64 --A=16 --B=4 --C=8 --D=8 --num_gpus=1 --epoch=5 --load_dir=logs/mnist/test/20191001_15\:37\:50\:422185_/ --logdir=test
# python3 train_val.py --batch_size 8 --num_gpus 1 --epoch 1 --logdir test
