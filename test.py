import sys
import argparse
import os
import os.path as osp

import numpy as np
import torch
import torch.utils.data as util_data
from torch.autograd import Variable

import network
import loss
import preprocess as prep
from datalist import ImageList
from pprint import pprint
from util import Logger, sign
from util.evaluation import *
from util.visualize import *


def save_code_and_label(params, path):
    np.save(path + "_code_and_label.npy", params)


def load_code_and_label(path):
    return np.load(path + "_code_and_label.npy").item()


def code_predict(dataloder, model, test_10crop=True, device=torch.device('cuda')):
    all_output = []
    all_label = []
    for i, (inputs, labels) in enumerate(dataloder):
        inputs = inputs.to(device)
        if test_10crop:
            bs, ncrops, c, h, w = inputs.size()
            result = model(inputs.view(-1, c, h, w))
            outputs = result.view(bs, ncrops, -1).mean(1)
        else:
            outputs = model(inputs)
        all_output.extend(outputs.data.cpu().float().numpy())
        all_label.extend(labels.cpu().float().numpy())
    return np.array(all_output), np.array(all_label)


def predict(config):
    ## set pre-process
    test_10crop, resize_size, crop_size = config['prep'].values()
    if test_10crop:
        database_transform = prep.image_test_10crop(resize_size, crop_size)
        test_transform = prep.image_test_10crop(resize_size, crop_size)
    else:
        database_transform = prep.image_test(resize_size, crop_size)
        test_transform = prep.image_test(resize_size, crop_size)
               
    ## prepare data
    database_set = ImageList(open(config['data']['database']).readlines(), transform=database_transform)
    database_loder = util_data.DataLoader(database_set, \
                            batch_size=config["batch_size"], \
                            shuffle=False, num_workers=4)
    test_set = ImageList(open(config['data']['test']).readlines(), transform=test_transform)
    test_loder = util_data.DataLoader(test_set, \
                            batch_size=config["batch_size"], \
                            shuffle=False, num_workers=4)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load(config["snapshot_path"]).to(device)
    model.eval()

    db_feats, db_labels = code_predict(database_loder, model, test_10crop=test_10crop, device=device)
    test_feats, test_labels = code_predict(test_loder, model, test_10crop=test_10crop, device=device)

    return {"db_feats":db_feats, "db_codes":sign(db_feats), "db_labels":db_labels, \
            "test_feats":test_feats, "test_codes":sign(test_feats), "test_labels":test_labels}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Transfer Learning')
    parser.add_argument('--gpus', type=str, default='0', help="device id to run")
    parser.add_argument('--dataset', type=str, default='cifar', help="dataset name")
    parser.add_argument('--hash_bit', type=int, default=32, help="number of hash code bits")
    parser.add_argument('--net', type=str, default='AlexNet', help="base network type")
    parser.add_argument('--prefix', type=str, default='hashnet', help="save path prefix")
    parser.add_argument('--snapshot', type=str, default='iter_10000', help="model path prefix")
    parser.add_argument('--batch_size', type=int, default=16, help="testing batch size")
    parser.add_argument('--preload', default=False, action='store_true')

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    # train config  
    config = {}
    config["batch_size"] = args.batch_size
    config["dataset"] = args.dataset
    config["output_path"] = "./snapshot/"+config["dataset"]+"_"+str(args.hash_bit)+ \
                            "bit_"+ args.net+"_"+args.prefix
    config["snapshot_path"] = config["output_path"]+"/"+args.snapshot+"_model.pth.tar"
    sys.stdout = Logger(osp.join(config["output_path"], "test.log"))

    config["prep"] = {"test_10crop":True, "resize_size":256, "crop_size":224}
    if config["dataset"] == "imagenet":
        config["data"] = {"database":"./data/imagenet/database.txt", \
                          "test":"./data/imagenet/test.txt"}
        config["R"] = 1000
    elif config["dataset"] == "nus_wide":
        config["data"] = {"database":"./data/nus_wide/database.txt", \
                          "test":"./data/nus_wide/test.txt"}
        config["R"] = 5000
    elif config["dataset"] == "coco":
        config["data"] = {"database":"./data/coco/database.txt", \
                          "test":"./data/coco/test.txt"}
        config["R"] = 5000
    elif config["dataset"] == "cifar":
        config["data"] = {"database":"./data/cifar/database.txt", \
                          "test":"./data/cifar/test.txt"}
        config["R"] = 54000

    print("test start")
    pprint(config)
    if args.preload == True:
        print("loading code and label ...")
        code_and_label = load_code_and_label(osp.join(config["output_path"], args.snapshot))
    else:
        print("calculating code and label ...")
        code_and_label = predict(config)
        print("saving code and label ...")
        save_code_and_label(code_and_label, osp.join(config["output_path"], args.snapshot))
        print("saving done")
    
    db_feats = code_and_label['db_feats']
    db_codes = code_and_label['db_codes']
    db_labels = code_and_label['db_labels']
    test_feats = code_and_label['test_feats']
    test_codes = code_and_label['test_codes']
    test_labels = code_and_label['test_labels']

    print("visualizing data ...")
    plot_distribution(db_feats, config["output_path"])
    plot_distance(db_feats, db_labels, test_feats, test_labels, config["output_path"])
    plot_tsne(db_codes, db_labels, config["output_path"])

    mAP_feat = get_mAP(db_feats, db_labels, test_feats, test_labels, config["R"])
    mAP = get_mAP(db_codes, db_labels, test_codes, test_labels, config["R"])
    print(f"mAP@feats: {mAP_feat}\nmAP@codes: {mAP}")
    
    print("test finished")
