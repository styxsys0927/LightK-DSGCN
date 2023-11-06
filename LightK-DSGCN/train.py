import argparse
from utils import *
from loaddata.load_mat import load_mat
import torch
import random
import os
from trains.train_DSGCN import TrainDSGCN, TestDSGCN
from networks.LightK_DSGCN import LightK_DSGCN
from networks.DSGCN import DSGCN
from networks.ncDSGCN import ncDSGCN
from networks.ntDSGCN import ntDSGCN
from networks.nsDSGCN import nsDSGCN

# Settings
parser = argparse.ArgumentParser(description='traffic prediction')

# data
parser.add_argument('-dataset', type=str, default='EDRA', help='choose dataset to run [options: EDRA, MODMA]')

# model
parser.add_argument('-model', type=str, default='LightK_DSGCN', help='choose model to train and test [options: LightK_DSGCN, DSGCN, ncDSGCN, ntDSGCN, nsDSGCN]')
parser.add_argument('-train', type=str, default='False', help='True if the model needs to be trained')
parser.add_argument('-model_path', type=str, default='./models/sdkfn_EDRA_3-10', help='choose model parameters to load')
parser.add_argument('-save_path', type=str, default='./models/sdkfn_EDRA_3-10', help='location to save the new model')
parser.add_argument('-adj_path', type=str, default='EDRA_dist_adj4.csv', help='choose adjacency matrix to load')
parser.add_argument('-learning_rate', type=float, default=0.0002, help='Initial learning rate.')  # 0.01
parser.add_argument('-epochs', type=int, default=500, help='Number of epochs to train.')  # 300
parser.add_argument('-hidden1', type=int, default=512, help='Number of units in hidden layer 1.')  # 16
parser.add_argument('-dropout', type=float, default=0.5, help='Dropout rate (1 - keep probability).')  # 0.5
parser.add_argument('-weight_decay', type=float, default=1e-4, help='Weight for L2 loss on embedding matrix.')  # 5e-4
parser.add_argument('-early_stopping', type=int, default=2000, help='Tolerance for early stopping (# of steps).')
parser.add_argument('-max_degree', type=int, default=3, help='Maximum Chebyshev polynomial degree.')  # 3
args = parser.parse_args()

# data_name = 'C:\\Users\\yansh\\Downloads\\backups\\codes\\LightK-DSGCN\\data/MODMA'
train_loader, val_loader, test_loader, adj = load_mat(args.dataset, os.path.join('./data/', args.dataset), args.max_degree, adj_path=args.adj_path, BATCH_SIZE=32)

if args.model == 'LightK_DSGCN':
    dsgcn = LightK_DSGCN(args.max_degree, torch.Tensor(adj), adj.shape[0])
elif args.model == 'DSGCN': # no kalman filter
    dsgcn = DSGCN(args.max_degree, torch.Tensor(adj), adj.shape[0])
elif args.model == 'ncDSGCN': # no tconv
    dsgcn = ncDSGCN(args.max_degree, torch.Tensor(adj), adj.shape[0])
elif args.model == 'ntDSGCN': # no tLSTM
    dsgcn = ntDSGCN(args.max_degree, torch.Tensor(adj), adj.shape[0])
elif args.model == 'nsDSGCN': # no sLSTM
    dsgcn = nsDSGCN(args.max_degree, torch.Tensor(adj), adj.shape[0])
else:
    print('Model is not found.')
    quit()

if args.train == 'True':
    if os.path.exists(args.model_path):
        dsgcn.load_state_dict(torch.load(args.model_path))
    print("\nTraining {} model...".format(args.model))
    dsgcn, sdkfn_loss = TrainDSGCN(dsgcn, train_loader, val_loader, adj, lr=args.learning_rate, weight_decay=args.weight_decay, save_path=args.save_path,
                                   K=args.max_degree, num_epochs=args.epochs, early_stop=args.early_stopping)

else:
    dsgcn.load_state_dict(torch.load(args.model_path))

print("\nTesting {} model...".format(args.model))
results = TestDSGCN(dsgcn, test_loader, scaler=None)

