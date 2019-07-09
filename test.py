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
from datalist import ImageDataset
from pprint import pprint
from util import Logger, sign
from util.evaluation import *
from util.visualize import *


R = {'cifar': 54000, 'coco': 5000, 'nuswide': 5000, 'imagenet': 1000}

def save_code_and_label(params, path):
    np.save(osp.join(path, "code_and_label.npy"), params)


def load_code_and_label(path):
    return np.load(osp.join(path, "code_and_label.npy")).item()


def code_predict(dataloder, model, test_10crop=True, device='cuda'):
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


def predict(args):
    ## set pre-process
    if args.test_10crop:
        database_transform = prep.image_test_10crop(resize_size=256, crop_size=224)
        test_transform = prep.image_test_10crop(resize_size=256, crop_size=224)
    else:
        database_transform = prep.image_test(resize_size=256, crop_size=224)
        test_transform = prep.image_test(resize_size=256, crop_size=224)
               
    ## prepare data
    database_set = ImageDataset(args.database_path, transform=database_transform)
    database_loder = util_data.DataLoader(database_set, batch_size=args.batch_size, shuffle=False, num_workers=4)
    test_set = ImageDataset(args.test_path, transform=test_transform)
    test_loder = util_data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = torch.load(args.snapshot_path).to(device)
    model.eval()

    db_feats, db_labels = code_predict(database_loder, model, test_10crop=args.test_10crop, device=device)
    test_feats, test_labels = code_predict(test_loder, model, test_10crop=args.test_10crop, device=device)

    return {"db_feats":db_feats, "db_codes":sign(db_feats), "db_labels":db_labels, \
            "test_feats":test_feats, "test_codes":sign(test_feats), "test_labels":test_labels}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Transfer Learning')
    parser.add_argument('--gpus', type=str, default='0', help="device id to run")
    parser.add_argument('--dataset', type=str, default='cifar', help="dataset name")
    parser.add_argument('--bit', type=int, default=32, help="number of hash code bits")
    parser.add_argument('--net', type=str, default='AlexNet', help="base network type")
    parser.add_argument('--prefix', type=str, default='hashnet', help="save path prefix")
    parser.add_argument('--snapshot', type=str, default='iter_10000', help="model path prefix")
    parser.add_argument('--batch_size', type=int, default=16, help="testing batch size")
    parser.add_argument('--test_10crop', default=True, help='use TenCrop transform')
    parser.add_argument('--preload', default=False, action='store_true')
    args = parser.parse_args()
    args.output_path = f'./snapshot/{args.dataset}_{str(args.bit)}bit_{args.net}_{args.prefix}'
    args.snapshot_path =  f'{args.output_path}/model.pth'
    args.database_path = f'./data/{args.dataset}/database.txt'
    args.test_path = f'./data/{args.dataset}/test.txt'
    args.R = R[args.dataset]

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    sys.stdout = Logger(osp.join(args.output_path, "test.log"))

    print("test start")
    pprint(args)
    if args.preload == True:
        print("loading code and label ...")
        code_and_label = load_code_and_label(args.output_path)
    else:
        print("calculating code and label ...")
        code_and_label = predict(args)
        print("saving code and label ...")
        save_code_and_label(code_and_label, args.output_path)
        print("saving done")
    
    db_feats = code_and_label['db_feats']
    db_codes = code_and_label['db_codes']
    db_labels = code_and_label['db_labels']
    test_feats = code_and_label['test_feats']
    test_codes = code_and_label['test_codes']
    test_labels = code_and_label['test_labels']

    print("visualizing data ...")
    plot_distribution(db_feats, args.output_path)
    plot_distance(db_feats, db_labels, test_feats, test_labels, args.output_path)
    plot_tsne(db_codes, db_labels, args.output_path)

    mAP_feat = get_mAP(db_feats, db_labels, test_feats, test_labels, args.R)
    mAP = get_mAP(db_codes, db_labels, test_codes, test_labels, args.R)
    print(f"mAP@feats: {mAP_feat}\nmAP@codes: {mAP}")
    
    print("test finished")
