#!/usr/bin/env bash

python -u train.py --gpu_id 0 --dataset cifar --prefix resnet50_hashnet --hash_bit 48 --net ResNet50 --lr 0.0003 --class_num 1.0

python test.py --gpu_id 0 --dataset cifar --prefix resnet50_hashnet --hash_bit 48 --snapshot iter_09000
