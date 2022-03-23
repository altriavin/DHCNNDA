import argparse
import pickle
import time
from util import Data, split_validation
from model import *
import os


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='ncRNA-disease', help='dataset name: diginetica/Nowplaying/sample')
parser.add_argument('--epoch', type=int, default=30, help='number of epochs to train for')
parser.add_argument('--batchSize', type=int, default=100, help='input batch size')
parser.add_argument('--embSize', type=int, default=100, help='embedding size')
parser.add_argument('--l2', type=float, default=1e-5, help='l2 penalty')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--layer', type=float, default=3, help='the number of layer used')
parser.add_argument('--beta', type=float, default=0.01, help='ssl task maginitude')
parser.add_argument('--filter', type=bool, default=False, help='filter incidence matrix')

opt = parser.parse_args()
print(opt)

def main():
    train_data = pickle.load(open("datasets/ncRNA-disease/train_pos.txt", "rb"))
    test_data_pos = pickle.load(open("datasets/ncRNA-disease/test_pos.txt", "rb"))
    test_data_neg = pickle.load(open("datasets/ncRNA-disease/test_neg.txt", "rb"))

    if opt.dataset == "ncRNA-disease":
        n_node = 556
    elif opt.dataset == "ncRNA-drug":
        n_node = 1816
    train_data = Data(train_data, shuffle=True, n_node=n_node)
    test_data_pos = Data(test_data_pos, shuffle=True, n_node=n_node)
    test_data_neg = Data(test_data_neg, shuffle=True, n_node=n_node)
    model = trans_to_cuda(DHCN(adjacency=train_data.adjacency,n_node=n_node,lr=opt.lr, l2=opt.l2, beta=opt.beta, layers=opt.layer,emb_size=opt.embSize, batch_size=opt.batchSize,dataset=opt.dataset))

    for epoch in range(opt.epoch):
        print('-------------------------------------------------------')
        print('epoch: ', epoch)
        train_test(model, train_data, test_data_pos, test_data_neg)


if __name__ == '__main__':
    main()
