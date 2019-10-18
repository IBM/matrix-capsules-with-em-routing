#!/bin/sh
python3 train_val.py --num_gpus=4 --dataset=cifar10 --batch_size=16 --rescap=True --A=128 --B=24 --C=24 --D=32 --E=24 --F=24 --G=32 --epoch=50 --weight_reg=True --nn_weight_reg_lambda=0.00000002 --capsule_weight_reg_lambda=0.00000002 --drop_rate=0.5 --dropout=True --dropconnect=False --logdir rescap
# python3 train_val.py --batch_size 8 --num_gpus 1 --epoch 1 --logdir test
