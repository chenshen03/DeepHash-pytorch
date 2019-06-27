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
from util import Logger
from util.evaluation import get_mAP
from util.visualize import *


def save_code_and_label(params, path):
    np.save(path + "_code_and_label.npy", params)


def load_code_and_label(path):
    return np.load(path + "_code_and_label.npy").item()

        
def code_predict(loader, model, name, test_10crop=True, device=torch.device('cuda')):
    start_test = True
    if test_10crop:
        iter_test = [iter(loader[name+str(i)]) for i in range(10)]
        for i in range(len(loader[name+'0'])):
            data = [iter_test[j].next() for j in range(10)]
            inputs = [data[j][0] for j in range(10)]
            labels = data[0][1]
            for j in range(10):
                inputs[j] = inputs[j].to(device)

            outputs = []
            for j in range(10):
                outputs.append(model(inputs[j]))
            outputs = sum(outputs) / 10.0
            if start_test:
                all_output = outputs.data.float()
                all_label = labels.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.data.float()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)
    else:
        iter_val = iter(loader[name])
        for i in range(len(loader[name])):
            data = iter_val.next()
            inputs = data[0]
            labels = data[1]
            inputs = inputs.to(device)

            outputs = model(inputs)
            if start_test:
                all_output = outputs.data.cpu().float()
                all_label = labels.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.data.cpu().float()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)
    return all_output, torch.sign(all_output), all_label


def predict(config):
    ## set pre-process
    prep_dict = {}
    prep_config = config["prep"]
    if prep_config["test_10crop"]:
        prep_dict["database"] = prep.image_test_10crop(
                                resize_size=prep_config["resize_size"], \
                                crop_size=prep_config["crop_size"])
        prep_dict["test"] = prep.image_test_10crop(
                                resize_size=prep_config["resize_size"], \
                                crop_size=prep_config["crop_size"])
    else:
        prep_dict["database"] = prep.image_test(
                                resize_size=prep_config["resize_size"], \
                                crop_size=prep_config["crop_size"])
        prep_dict["test"] = prep.image_test(
                                resize_size=prep_config["resize_size"], \
                                crop_size=prep_config["crop_size"])
               
    ## prepare data
    dsets = {}
    dset_loaders = {}
    data_config = config["data"]
    if prep_config["test_10crop"]:
        for i in range(10):
            dsets["database"+str(i)] = ImageList(open(data_config["database"]["list_path"]).readlines(), \
                                transform=prep_dict["database"]["val"+str(i)])
            dset_loaders["database"+str(i)] = util_data.DataLoader(dsets["database"+str(i)], \
                                batch_size=config["batch_size"], \
                                shuffle=False, num_workers=4)
            dsets["test"+str(i)] = ImageList(open(data_config["test"]["list_path"]).readlines(), \
                                transform=prep_dict["test"]["val"+str(i)])
            dset_loaders["test"+str(i)] = util_data.DataLoader(dsets["test"+str(i)], \
                                batch_size=config["batch_size"], \
                                shuffle=False, num_workers=4)
    else:
        dsets["database"] = ImageList(open(data_config["database"]["list_path"]).readlines(), \
                                transform=prep_dict["database"])
        dset_loaders["database"] = util_data.DataLoader(dsets["database"], \
                                batch_size=config["batch_size"], \
                                shuffle=False, num_workers=4)
        dsets["test"] = ImageList(open(data_config["test"]["list_path"]).readlines(), \
                                transform=prep_dict["test"])
        dset_loaders["test"] = util_data.DataLoader(dsets["test"], \
                                batch_size=config["batch_size"], \
                                shuffle=False, num_workers=4)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ## set base network
    base_network = torch.load(config["snapshot_path"])
    base_network = base_network.to(device)
    base_network.eval()

    db_feats, db_codes, db_labels = code_predict(dset_loaders, base_network, "database", 
                                                        test_10crop=prep_config["test_10crop"], device=device)
    test_feats, test_codes, test_labels = code_predict(dset_loaders, base_network, "test", 
                                                        test_10crop=prep_config["test_10crop"], device=device)

    return {"db_feats":db_feats.numpy(), "db_codes":db_codes.numpy(), "db_labels":db_labels.numpy(), \
            "test_feats":test_feats.numpy(), "test_codes":test_codes.numpy(), "test_labels":test_labels.numpy()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Transfer Learning')
    parser.add_argument('--gpus', type=str, default='0', help="device id to run")
    parser.add_argument('--dataset', type=str, default='nus_wide', help="dataset name")
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
    config["output_path"] = "./snapshot/"+config["dataset"]+"_"+str(args.hash_bit)+"bit_"+ \
                            args.net+"_"+args.prefix
    config["snapshot_path"] = config["output_path"]+"/"+args.snapshot+"_model.pth.tar"
    sys.stdout = Logger(osp.join(config["output_path"], "test.log"))

    config["prep"] = {"test_10crop":False, "resize_size":256, "crop_size":224}
    if config["dataset"] == "imagenet":
        config["data"] = {"database":{"list_path":"./data/imagenet/database.txt"}, \
                          "test":{"list_path":"./data/imagenet/test.txt"}}
        config["R"] = 1000
    elif config["dataset"] == "nus_wide":
        config["data"] = {"database":{"list_path":"./data/nus_wide/database.txt"}, \
                          "test":{"list_path":"./data/nus_wide/test.txt"}}
        config["R"] = 5000
    elif config["dataset"] == "coco":
        config["data"] = {"database":{"list_path":"./data/coco/database.txt"}, \
                          "test":{"list_path":"./data/coco/test.txt"}}
        config["R"] = 5000
    elif config["dataset"] == "cifar":
        config["data"] = {"database":{"list_path":"./data/cifar/database.txt"}, \
                          "test":{"list_path":"./data/cifar/test.txt"}}
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
    mAP_feat = get_mAP(db_feats, db_labels, test_feats, test_labels, config["R"])
    mAP = get_mAP(db_codes, db_labels, test_codes, test_labels, config["R"])
    print(f"mAP@feats: {mAP_feat}\nmAP@codes: {mAP}")
    print("visualizing data ...")
    plot_distribution(db_feats, config["output_path"])
    plot_distance(db_feats, db_labels, test_feats, test_labels, config["output_path"])
    plot_tsne(db_codes, db_labels, config["output_path"])

    print("test finished")
