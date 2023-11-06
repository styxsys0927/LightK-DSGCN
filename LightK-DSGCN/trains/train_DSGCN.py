import time
import numpy as np
from networks.LightK_DSGCN import LightK_DSGCN
import torch
import torch.nn as nn
from torch.autograd import Variable
from utils import show_results
import copy

# choose device
USE_CUDA = torch.cuda.is_available()
if not USE_CUDA:
    DEVICE = torch.device('cpu')
else:
    DEVICE = torch.device('cuda:0')
print("CUDA:", USE_CUDA, DEVICE)

def TrainDSGCN(model, train_dataloader, valid_dataloader, A, lr, weight_decay, save_path, K=3, num_epochs=100, early_stop=15):
    torch.autograd.set_detect_anomaly(True)
    inputs, labels = next(iter(train_dataloader))
    # print('input', inputs.size(), 'label', labels.size())
    [batch_size, step_size, fea_size] = inputs.size()

    # T_0, T_mult = 4000, 4000
    optimizer = torch.optim.RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0, T_mult)

    ###################### print model parameter states ######################
    print('Net\'s state_dict:')
    total_param = 0
    for param_tensor in model.state_dict():
        print(param_tensor, '\t', model.state_dict()[param_tensor].size())
        total_param += np.prod(model.state_dict()[param_tensor].size())
    print('Net\'s total params:', total_param)

    print('Optimizer\'s state_dict:')
    for var_name in optimizer.state_dict():
        print(var_name, '\t', optimizer.state_dict()[var_name])
    ###########################################################################

    model.to(DEVICE)

    loss = nn.CrossEntropyLoss()

    interval = 20
    losses_train = []
    losses_interval_train = []
    losses_valid = []
    losses_interval_valid = []

    losses_epoch = []
    cur_time = time.time()
    pre_time = time.time()

    best_loss, best_epoch, best_model = 1e4, -1, None
    stop_cnt = 0
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        trained_number = 0

        # validation data loader iterator init
        valid_dataloader_iter = iter(valid_dataloader)

        for data in train_dataloader:
            model.train()
            inputs, labels = data

            if inputs.shape[0] != batch_size:
                continue

            inputs, labels = Variable(inputs.to(DEVICE)), Variable(labels.to(DEVICE))
            # model.zero_grad()
            optimizer.zero_grad()

            pred = model.loop(inputs)
            # L1_loss = 0
            # for param_tensor in model.state_dict():
            #     if 'w_dynamic' in param_tensor:
            #        L1_loss += model.state_dict()[param_tensor].abs().sum()
            # print('regularize', L1_loss)
            loss_train = loss(pred, labels.long())# + L1_loss
            loss_train.backward()

            clipping_value = 5  # arbitrary value of your choosing
            torch.nn.utils.clip_grad_norm_(model.parameters(), clipping_value)
            optimizer.step()
            # scheduler.step(epoch + trained_number / len(train_dataloader))

            losses_train.append(loss_train.data)

            # validation
            try:
                inputs_val, labels_val = next(valid_dataloader_iter)
            except StopIteration:
                valid_dataloader_iter = iter(valid_dataloader)
                inputs_val, labels_val = next(valid_dataloader_iter)

            inputs_val, labels_val = Variable(inputs_val.to(DEVICE)), Variable(labels_val.to(DEVICE))

            labels_val = labels_val

            model.eval()
            pred = model.loop(inputs_val)
            loss_valid = loss(pred, labels_val.long())
            # record here
            losses_valid.append(loss_valid.data)

            # output
            trained_number += 1

            if trained_number % interval == 0:
                cur_time = time.time()
                loss_interval_train = np.around(sum(losses_train[-interval:]).cpu().numpy() / interval, decimals=8)
                losses_interval_train.append(loss_interval_train)
                loss_interval_valid = np.around(sum(losses_valid[-interval:]).cpu().numpy() / interval, decimals=8)
                losses_interval_valid.append(loss_interval_valid)


                print('Iteration #: {}, train_loss: {}, valid_loss: {}, have not updated for:{}, time: {}'.format(
                    trained_number * batch_size,
                    loss_interval_train,
                    loss_interval_valid,
                    stop_cnt,
                    np.around([cur_time - pre_time], decimals=8)))
                torch.save(best_model.state_dict(), save_path)
                pre_time = cur_time

            # update every step
            loss_epoch = loss_valid.cpu().data.numpy()  # np.around(sum(losses_valid[-interval:]).cpu().numpy() / interval, decimals=8)#loss_valid.cpu().data.numpy() # np.around(sum(losses_valid[-trained_number:]).cpu().numpy() / trained_number, decimals=8)
            losses_epoch.append(loss_epoch)
            if best_loss >= loss_epoch:  # .mean():
                stop_cnt = 0
                best_loss, best_epoch = loss_epoch, epoch
                best_model = copy.deepcopy(model)
                print('Current best:', best_loss)
            else:
                stop_cnt += 1

        if stop_cnt > early_stop:
            break

    print('best epoch:', best_epoch)
    return best_model, [losses_train, losses_interval_train, losses_valid, losses_interval_valid]


def TestDSGCN(model, test_dataloader, scaler=None):
    inputs, labels = next(iter(test_dataloader))
    [batch_size, step_size, fea_size] = inputs.size()

    use_gpu = torch.cuda.is_available()
    tested_batch = 0

    predictions = []
    ground_truths = []

    model.to(DEVICE)
    model.eval()
    for data in test_dataloader:
        inputs, labels = data

        if inputs.shape[0] != batch_size:
            continue

        if use_gpu:
            inputs, labels = Variable(inputs.to(DEVICE)), Variable(labels.to(DEVICE))
        else:
            inputs, labels = Variable(inputs), Variable(labels)

        pred = model.loop(inputs)
        pred = torch.max(pred, 1)[1].view(labels.size())
        predictions.append(pred.cpu().data.numpy())
        ground_truths.append(labels.long().cpu().data.numpy())

        tested_batch += 1

    predictions, ground_truths = np.concatenate(predictions, axis=0), np.concatenate(ground_truths, axis=0)
    results = show_results(ground_truths, predictions)

    print('Tested: Acc: {}, F1: {}, PRC: {}, RCL: {}'.format(results[0], results[1], results[2], results[3]))
    return results
