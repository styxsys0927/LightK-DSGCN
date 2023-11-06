import scipy.io as sio
import os, glob, random
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse import lil_matrix
from utils import *
import torch.utils.data as utils
import torch

def load_mat(dataset, data_dir, max_degree, adj_path, BATCH_SIZE=32):
    if dataset == 'MODMA':
        adj_dist = pd.read_csv(os.path.join(data_dir, adj_path), index_col=0, header=0).to_numpy()
        data_x, data_y = [], []
        train_dir = os.path.join(data_dir, 'Train/train_data/**')
        # labels = pd.read_csv(os.path.join(data_dir, 'Train/train_data/train_label.csv'), header=0)
        test_labels = pd.read_csv(os.path.join(data_dir, 'Val/sample.csv'), header=0, index_col=0)
        # print(test_labels.head())
        for file in glob.glob(train_dir, recursive=True):
            if file.endswith(".csv"):
                data_x.append(pd.read_csv(file, header=None).to_numpy().T) # 128*500 --> 500*128
                if ('D' in file.split('/')[-1])and('N' in file.split('/')[-1]):
                    print(file)
                    quit()
                if 'D' in file.split('/')[-1]:
                    data_y.append(np.ones(1))
                else:
                    data_y.append(np.zeros(1))

        for file in os.listdir(os.path.join(data_dir, 'Val/data')):
            if file.endswith(".csv"):
                data_x.append(pd.read_csv(os.path.join(data_dir, 'Val/data', file), header=None).to_numpy().T) # 128*500 --> 500*128
                data_y.append(test_labels.loc[int(file[:-4])].to_numpy())
        data_x, data_y = np.stack(data_x, axis=0), np.concatenate(data_y)

        # data_x = np.load('./data/MODMA/data_x.npy')
        # data_y = np.load('./data/MODMA/data_y.npy')

    elif dataset == 'EDRA':
        chs = ['Fpz', 'Fp2', 'AF7', 'AF3', 'Afz', 'AF4', 'AF8', 'F7', 'F5', 'F3', 'F1', 'Fz', 'F2', 'F4', 'F6', 'F8', 'FT7', 'FC5',
               'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6', 'FT8', 'T7 (T3)', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'T8 (T4)',
               'P9', 'TP7', 'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6', 'TP8', 'P10', 'P7', 'P5', 'P3', 'P1', 'Pz',
               'P2', 'P4', 'P6', 'P8', 'PO7', 'PO3', 'POz', 'PO4', 'PO8', 'O1', 'Oz', 'O2'] # P9 and P10 should actually be TP9 and TP10.. but the locations seem ok
        adj_dist = pd.read_csv(os.path.join(data_dir, adj_path), index_col=0, header=0)
        adj_dist = adj_dist.loc[chs, chs]
        adj_dist = adj_dist.to_numpy()
        mat_contents = sio.loadmat(os.path.join(data_dir, 'Segmentation_EEG.mat')) # 500*(62+1)*sample
        data_x = np.concatenate([mat_contents['LSG'], mat_contents['HSG']], axis=-1).transpose(2, 0, 1)[:, :, :-1] # 500*62 last channel is eye open or not
        data_y = np.concatenate([np.zeros(mat_contents['LSG'].shape[-1]), np.ones(mat_contents['HSG'].shape[-1])])
    else:
        print('Dataset is not allowed')
        quit()

    random.seed(4)  # shuffle and cross validation
    idx = [i for i in range(data_x.shape[0])]
    random.shuffle(idx)
    # train_idx, val_idx, test_idx = idx[int(data_x.shape[0]*0.1):int(data_x.shape[0]*0.9)], idx[int(data_x.shape[0]*0.9):], idx[:int(data_x.shape[0]*0.1)] # 1-10
    train_idx, val_idx, test_idx = idx[int(data_x.shape[0] * 0.2):], idx[:int(data_x.shape[0] * 0.1)], idx[int(
        data_x.shape[0] * 0.1):int(data_x.shape[0] * 0.2)]  # 2-10
    # train_idx, val_idx, test_idx = idx[:int(data_x.shape[0]*0.1)]+idx[int(data_x.shape[0]*0.3):], idx[int(data_x.shape[0]*0.1):int(data_x.shape[0]*0.2)], \
    #                                idx[int(data_x.shape[0]*0.2):int(data_x.shape[0]*0.3)] # 3-10
    # train_idx, val_idx, test_idx = idx[:int(data_x.shape[0]*0.2)]+idx[int(data_x.shape[0]*0.4):], idx[int(data_x.shape[0]*0.2):int(data_x.shape[0]*0.3)], \
    #                                idx[int(data_x.shape[0]*0.3):int(data_x.shape[0]*0.4)] # 4-10
    # train_idx, val_idx, test_idx = idx[:int(data_x.shape[0]*0.3)]+idx[int(data_x.shape[0]*0.5):], idx[int(data_x.shape[0]*0.3):int(data_x.shape[0]*0.4)], \
    #                                idx[int(data_x.shape[0]*0.4):int(data_x.shape[0]*0.5)] # 5-10
    # train_idx, val_idx, test_idx = idx[:int(data_x.shape[0]*0.4)]+idx[int(data_x.shape[0]*0.6):], idx[int(data_x.shape[0]*0.4):int(data_x.shape[0]*0.5)], \
    #                                idx[int(data_x.shape[0]*0.5):int(data_x.shape[0]*0.6)] # 6-10
    # train_idx, val_idx, test_idx = idx[:int(data_x.shape[0]*0.5)]+idx[int(data_x.shape[0]*0.7):], idx[int(data_x.shape[0]*0.5):int(data_x.shape[0]*0.6)], \
    #                                idx[int(data_x.shape[0]*0.6):int(data_x.shape[0]*0.7)] # 7-10
    # train_idx, val_idx, test_idx = idx[:int(data_x.shape[0]*0.6)]+idx[int(data_x.shape[0]*0.8):], idx[int(data_x.shape[0]*0.6):int(data_x.shape[0]*0.7)], \
    #                                idx[int(data_x.shape[0]*0.7):int(data_x.shape[0]*0.8)] # 8-10
    # train_idx, val_idx, test_idx = idx[:int(data_x.shape[0]*0.7)]+idx[int(data_x.shape[0]*0.9):], idx[int(data_x.shape[0]*0.7):int(data_x.shape[0]*0.8)], \
    #                                idx[int(data_x.shape[0]*0.8):int(data_x.shape[0]*0.9)] # 9-10
    # train_idx, val_idx, test_idx = idx[:int(data_x.shape[0]*0.8)], idx[int(data_x.shape[0]*0.8):int(data_x.shape[0]*0.9)], idx[int(data_x.shape[0]*0.9):] # 10-10

    train_x, train_y, val_x, val_y, test_x, test_y = data_x[train_idx], data_y[train_idx], \
                                                     data_x[val_idx], data_y[val_idx], \
                                                     data_x[test_idx], data_y[test_idx]
    print('case distribution:', data_y.sum()/data_y.shape[0], train_y.sum()/train_y.shape[0], val_y.sum()/val_y.shape[0], test_y.sum()/test_y.shape[0])


    scaler = StandardScaler(train_x.mean(), train_x.std())
    train_x, val_x, test_x = scaler.transform(train_x), scaler.transform(val_x), scaler.transform(test_x)

    train_x, train_y = torch.Tensor(train_x), torch.Tensor(train_y)
    val_x, val_y = torch.Tensor(val_x), torch.Tensor(val_y)
    test_x, test_y = torch.Tensor(test_x), torch.Tensor(test_y)

    train_dataset = utils.TensorDataset(train_x, train_y)
    valid_dataset = utils.TensorDataset(val_x, val_y)
    test_dataset = utils.TensorDataset(test_x, test_y)

    train_loader = utils.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = utils.DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    test_loader = utils.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)

    return train_loader, val_loader, test_loader, adj_dist
