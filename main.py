import argparse
import networkx as nx
import numpy as np
import time
import scipy.sparse as sp
from math import ceil
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder

import torch
import torch.nn as nn
import torch.nn.init as init 
from torch.autograd import Variable 
import torch.nn.functional as F
from torch import optim

from model import k_hop_GraphNN 
from utils import load_data, process_node_labels, generate_batches, accuracy, AverageMeter

# Argument parser
parser = argparse.ArgumentParser(description='k-hop-GNN')

parser.add_argument('--device', type=int, default=0, help='which gpu to use if any (default: 0)')
parser.add_argument('--dataset', default='IMDB-BINARY', help='Dataset name')
parser.add_argument('--use-node-labels', action='store_true', default=False, help='Use node labels')
parser.add_argument('--lr', type=float, default=1e-3, metavar='LR', help='Initial learning rate')
parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate (1 - keep probability).')
parser.add_argument('--batch-size', type=int, default=128, metavar='N', help='Input batch size for training')
parser.add_argument('--epochs', type=int, default=200, metavar='N', help='Number of epochs to train')
parser.add_argument('--hidden_dim', type=int, default=64, metavar='N', help='Size of hidden layer of graph NN')
parser.add_argument('--radius', type=int, default=2, metavar='N', help='Diameter of ego networks')

def main():
    args = parser.parse_args()
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")

    Gs, class_labels = load_data(args.dataset, args.use_node_labels)

    if args.use_node_labels:
        Gs, feat_dim = process_node_labels(Gs)
    else:
        feat_dim = 1
    
    enc = LabelEncoder()
    class_labels = enc.fit_transform(class_labels)
    unique_labels = np.unique(class_labels)
    n_classes = unique_labels.size
    y = [np.array(class_labels[i]) for i in range(class_labels.size)]

    kf = KFold(n_splits=10, shuffle=True)
    it = 0
    accs = list()
    for train_index, test_index in kf.split(y):
        it += 1
        
        idx = np.random.permutation(train_index)
        train_index = idx[:int(idx.size*0.9)].tolist()
        val_index = idx[int(idx.size*0.9):].tolist()
        n_train = len(train_index)
        n_val = len(val_index)
        n_test = len(test_index)

        Gs_train = [Gs[i] for i in train_index]
        y_train = [y[i] for i in train_index]

        Gs_val = [Gs[i] for i in val_index]
        y_val = [y[i] for i in val_index]

        Gs_test = [Gs[i] for i in test_index]
        y_test = [y[i] for i in test_index]

        adj_train, features_train, idx_train, y_train = generate_batches(Gs_train, args.use_node_labels, feat_dim, y_train, args.batch_size, args.radius, device, shuffle=True)
        adj_val, features_val, idx_val, y_val = generate_batches(Gs_val, args.use_node_labels, feat_dim, y_val, args.batch_size, args.radius, device)
        adj_test, features_test, idx_test, y_test = generate_batches(Gs_test, args.use_node_labels, feat_dim, y_test, args.batch_size, args.radius, device)

        n_train_batches = ceil(n_train/args.batch_size)
        n_val_batches = ceil(n_val/args.batch_size)
        n_test_batches = ceil(n_test/args.batch_size)

        model = k_hop_GraphNN(feat_dim, args.hidden_dim, n_classes, args.dropout, args.radius, device).to(device)

        optimizer = optim.Adam(model.parameters(), lr=args.lr)

        def train(epoch, adj, features, idx, y):
            optimizer.zero_grad()
            output = model(adj, features, idx)
            loss_train = F.cross_entropy(output, y)
            loss_train.backward()
            model.clip_grad(5)
            optimizer.step()
            return output, loss_train

        def test(adj, features, idx, y):
            output = model(adj, features, idx)
            loss_test = F.cross_entropy(output, y)
            return output, loss_test

        best_val_acc = 0
        for epoch in range(args.epochs):    
            start = time.time()

            model.train()
            train_loss = AverageMeter()
            train_acc = AverageMeter()

            # Train for one epoch
            for i in range(n_train_batches):
                output, loss = train(epoch, adj_train[i], features_train[i], idx_train[i], y_train[i])
                train_loss.update(loss.data.item(), output.size(0))
                train_acc.update(accuracy(output.data, y_train[i].data), output.size(0))

            # Evaluate on validation set
            model.eval()
            val_loss = AverageMeter()
            val_acc = AverageMeter()

            for i in range(n_val_batches):
                output, loss = test(adj_val[i], features_val[i], idx_val[i], y_val[i])
                val_loss.update(loss.data.item(), output.size(0))
                val_acc.update(accuracy(output.data, y_val[i].data), output.size(0))

            # Print results
            print("Cross-val iter:", '%02d' % it, "epoch:", '%03d' % (epoch + 1), "train_loss=", "{:.5f}".format(train_loss.avg),
                "train_acc=", "{:.5f}".format(train_acc.avg), "val_loss=", "{:.5f}".format(val_loss.avg),
                "val_acc=", "{:.5f}".format(val_acc.avg), "time=", "{:.5f}".format(time.time() - start))

        
            # Remember best accuracy and save checkpoint
            is_best = val_acc.avg >= best_val_acc
            best_val_acc = max(val_acc.avg, best_val_acc)
            if is_best:
                torch.save({
                    'state_dict': model.state_dict(),
                    'optimizer' : optimizer.state_dict(),
                }, 'model_best.pth.tar')

        print("Optimization finished!")

        # Testing
        test_loss = AverageMeter()
        test_acc = AverageMeter()

        print("Loading checkpoint!")
        checkpoint = torch.load('model_best.pth.tar')
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        model.eval()
        print("epoch:", epoch)
        
        for i in range(n_test_batches):
            output, loss = test(adj_test[i], features_test[i], idx_test[i], y_test[i])
            test_loss.update(loss.data.item(), output.size(0))
            test_acc.update(accuracy(output.data, y_test[i].data), output.size(0))
        
        accs.append(test_acc.avg.cpu().numpy())

        # Print results
        print("test_loss=", "{:.5f}".format(test_loss.avg), "test_acc=", "{:.5f}".format(test_acc.avg))
        print()
    print("avg_test_acc=", "{:.5f}".format(np.mean(accs)))

    return np.mean(accs)

if __name__ == '__main__':
    main()