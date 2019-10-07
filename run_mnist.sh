#!/bin/sh
python3 train_val.py --dataset=mnist --batch_size=64 --A=16 --B=8 --C=8 --D=8 --drop_rate=0.5 --affine_voting=True --num_gpus=1 --epoch=15 --logdir=dropconn
# python3 train_val.py --batch_size 8 --num_gpus 1 --epoch 1 --logdir test
# python3 test.py --dataset=mnist --batch_size=256 --num_gpus=1 --load_dir=logs/mnist/test/20191003_14\:05\:09\:653069_/ --ckpt_name=all
