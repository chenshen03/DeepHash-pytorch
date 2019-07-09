import sys
import argparse
import os
import os.path as osp
import shutil

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as util_data
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from tensorboardX import SummaryWriter

import loss
import preprocess as prep
from network import load_model
from datalist import ImageDataset
from util import Logger
from pprint import pprint
from datetime import datetime

# https://github.com/microsoft/ptvsd/issues/943#issuecomment-480782087
import multiprocessing
multiprocessing.set_start_method('spawn', True)


def train(args):
    ## tensorboardX
    tflog_path = osp.join(args.output_path, "tflog")
    if os.path.exists(tflog_path):
        shutil.rmtree(tflog_path)
    writer = SummaryWriter(logdir=tflog_path)

    ## prepare data
    train_transform = prep.image_train(resize_size=256, crop_size=224)
    train_set = ImageDataset(args.train_path, transform=train_transform)
    train_loader = util_data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ## set base network
    model = load_model(args.net, args.bit)
    writer.add_graph(model, input_to_model=(torch.rand(2, 3, 224, 224),))
    model.to(device)
    if device == 'cuda':
        cudnn.benchmark = True
    model.train()
        
    ## set optimizer and scheduler
    parameter_list = [{"params":model.feature_layers.parameters(), "lr":args.lr}, \
                      {"params":model.hash_layer.parameters(), "lr":args.lr*10}]
    optimizer = optim.SGD(parameter_list, lr=args.lr, momentum=0.9, weight_decay=0.005, nesterov=True)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3000, gamma=0.5)

    ## train
    for i in range(args.num_iter):
        scheduler.step()
        optimizer.zero_grad()

        if i % (len(train_loader)-1) == 0:
            train_iter = iter(train_loader)
        inputs, labels = train_iter.next()
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = model(inputs)
        s_loss = loss.pairwise_loss(outputs, labels, alpha=args.alpha, class_num=args.class_num)
        q_loss = loss.quantization_loss(outputs)
        total_loss = s_loss + 0.01 * q_loss
        total_loss.backward()
        optimizer.step()

        writer.add_scalar('similarity loss', s_loss, i)
        writer.add_scalar('quantization loss', q_loss, i)
        writer.add_scalar('lr', optimizer.param_groups[0]['lr'], i)
        if i % 10 == 0:
            print("{} #train# Iter: {:05d}, loss: {:.3f} quantizaion loss: {:.3f}".format(
                datetime.now(), i, s_loss.item(), q_loss.item()))
            
    writer.close()
    torch.save(model,  osp.join(args.output_path, "model.pth"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='HashNet')
    parser.add_argument('--gpus', type=str, default='0', help="device id to run")
    parser.add_argument('--dataset', type=str, default='cifar', help="dataset name")
    parser.add_argument('--prefix', type=str, default='1', help="save path prefix")
    parser.add_argument('--bit', type=int, default=32, help="number of hash code bits")
    parser.add_argument('--lr', type=float, default=0.0005, help="learning rate")
    parser.add_argument('--net', type=str, default='AlexNet', help="base network type")
    parser.add_argument('--num_iter', type=int, default=10000, help="train iters")
    parser.add_argument('--snapshot_interval', type=int, default=3000, help="snapshot interval")
    parser.add_argument('--batch_size', type=int, default=128, help="training batch size")
    parser.add_argument('--alpha', type=float, default=10.0, help="loss parameter")
    parser.add_argument('--class_num', type=float, default=5.0, help="positive negative pairs balance weight")
    args = parser.parse_args()
    args.output_path =  f'./snapshot/{args.dataset}_{str(args.bit)}bit_{args.net}_{args.prefix}'
    args.train_path = f'./data/{args.dataset}/train.txt'

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    
    if not osp.exists(args.output_path):
        os.mkdir(args.output_path)
    sys.stdout = Logger(osp.join(args.output_path, "train.log"))

    pprint(args)
    train(args)

    testcmd = f'python test.py --gpus {args.gpus} --dataset {args.dataset} ' + \
              f'--bit {args.bit} --net {args.net} --prefix {args.prefix}'
    os.system(testcmd)
