#!/bin/sh
python3 train_val.py --dataset=mnist --batch_size=128 --A=64 --B=8 --C=16 --D=16 --weight_reg=True --drop_rate=0.5 --dropout=False --dropconnect=False --recon_loss=True --num_gpus=1 --epoch=30 --logdir=recon_custom_weightreg
# python3 train_val.py --batch_size 8 --num_gpus 1 --epoch 1 --logdir test
# python3 test.py --dataset=mnist --batch_size=256 --num_gpus=2 --load_dir=logs/mnist/recon/20191011_14:03:27:491712_/ --ckpt_name=all
