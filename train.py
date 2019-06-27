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
from torch.autograd import Variable
from tensorboardX import SummaryWriter

import network
import loss
import preprocess as prep
from datalist import ImageList
from util import Logger
from pprint import pprint
from datetime import datetime

# https://github.com/microsoft/ptvsd/issues/943#issuecomment-480782087
import multiprocessing
multiprocessing.set_start_method('spawn', True)


optim_dict = {"SGD": optim.SGD, "Adam": optim.Adam}
scheduler_dict = {"step": optim.lr_scheduler.StepLR}


def train(config):
    ## tensorboardX
    tflog_path = osp.join(config["output_path"], "tflog")
    if os.path.exists(tflog_path):
        shutil.rmtree(tflog_path)
    writer = SummaryWriter(logdir=tflog_path)

    ## set pre-process
    prep_dict = {}
    prep_config = config["prep"]
    prep_dict["train"] = prep.image_train(
                            resize_size=prep_config["resize_size"], \
                            crop_size=prep_config["crop_size"])

    ## prepare data
    train_data_config = config["data"]["train"]
    train_set = ImageList(open(train_data_config["list_path"]).readlines(), \
                    transform=prep_dict["train"])
    train_loader = util_data.DataLoader(train_set, \
                    batch_size=config["batch_size"], \
                    shuffle=True, num_workers=4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ## set base network
    net_config = config["network"]
    base_network = net_config["type"](**net_config["params"])
    writer.add_graph(base_network, input_to_model=(torch.rand(2, 3, 224, 224),))
    base_network.to(device)
    base_network.train()
        
    ## set optimizer and scheduler
    optimizer_config = config["optimizer"]
    lr = optimizer_config["optim_params"]["lr"]
    parameter_list = [{"params":base_network.feature_layers.parameters(), "lr":lr}, \
                      {"params":base_network.hash_layer.parameters(), "lr":lr*10}]
    optimizer = optim_dict[optimizer_config["type"]](parameter_list, \
                **optimizer_config["optim_params"])
    scheduler = scheduler_dict[optimizer_config["lr_type"]](optimizer, \
                ** optimizer_config["lr_param"])

    ## train
    for i in range(config["num_iter"]):
        scheduler.step()
        optimizer.zero_grad()

        if i % (len(train_loader)-1) == 0:
            train_iter = iter(train_loader)
        inputs, labels = train_iter.next()
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = base_network(inputs)
        s_loss = loss.pairwise_loss(outputs, labels, **config["loss"])
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
    torch.save(nn.Sequential(base_network), osp.join(config["output_path"], \
        "iter_{:05d}_model.pth.tar".format(i+1)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='HashNet')
    parser.add_argument('--gpus', type=str, default='0', help="device id to run")
    parser.add_argument('--dataset', type=str, default='cifar', help="dataset name")
    parser.add_argument('--hash_bit', type=int, default=32, help="number of hash code bits")
    parser.add_argument('--net', type=str, default='AlexNet', help="base network type")
    parser.add_argument('--prefix', type=str, default='debug', help="save path prefix")
    parser.add_argument('--lr', type=float, default=0.0005, help="learning rate")
    parser.add_argument('--batch_size', type=int, default=128, help="training batch size")
    parser.add_argument('--alpha', type=float, default=10.0, help="loss parameter")
    parser.add_argument('--class_num', type=float, default=5.0, help="positive negative pairs balance weight")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    # train config  
    config = {}
    config["num_iter"] = 10000
    config["snapshot_interval"] = 3000
    config["batch_size"] = args.batch_size
    config["dataset"] = args.dataset
    config["hash_bit"] = args.hash_bit
    config["output_path"] = "./snapshot/"+args.dataset+"_"+ \
                            str(args.hash_bit)+"bit_"+args.net+"_"+args.prefix

    config["prep"] = {"test_10crop":True, "resize_size":256, "crop_size":224}
    config["loss"] = {"l_threshold":15.0, "alpha":args.alpha, "class_num":args.class_num}
    config["optimizer"] = {"type":"SGD", "optim_params":{"lr":args.lr, "momentum":0.9, \
                            "weight_decay":0.0005, "nesterov":True}, 
                           "lr_type":"step", "lr_param":{"step_size":3000, "gamma":0.5} }

    # network config
    config["network"] = {}
    if "ResNet" in args.net:
        config["network"]["type"] = network.ResNetFc
        config["network"]["params"] = {"name":args.net, "hash_bit":args.hash_bit}
    elif "VGG" in args.net:
        config["network"]["type"] = network.VGGFc
        config["network"]["params"] = {"name":args.net, "hash_bit":args.hash_bit}
    elif "AlexNet" in args.net:
        config["network"]["type"] = network.AlexNetFc
        config["network"]["params"] = {"hash_bit":args.hash_bit}

    # dataset config
    if config["dataset"] == "imagenet":
        config["data"] = {"train":{"list_path":"./data/imagenet/train.txt"}}
    elif config["dataset"] == "nus_wide":
        config["data"] = {"train":{"list_path":"./data/nus_wide/train.txt"}}
    elif config["dataset"] == "coco":
        config["data"] = {"train":{"list_path":"./data/coco/train.txt"}}
    elif config["dataset"] == "cifar":
        config["data"] = {"train":{"list_path":"./data/cifar/train.txt"}}
    
    if not osp.exists(config["output_path"]):
        os.mkdir(config["output_path"])
    sys.stdout = Logger(osp.join(config["output_path"], "train.log"))

    pprint(config)
    train(config)

    testcmd = f'python test.py --gpus {args.gpus} --dataset {args.dataset} ' + \
              f'--hash_bit {args.hash_bit} --net {args.net} --prefix {args.prefix}'
    os.system(testcmd)
