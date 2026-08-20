from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch
from sklearn.model_selection import KFold
import sklearn.metrics as skmetrics
from sklearn.metrics import confusion_matrix
import numpy as np
import timeit
import matplotlib.pyplot as plt
import glob
from dataset import EdfDataset_2EEG, EdfDataset_1EEG, EdfDataset_EOG_3Stages, EdfDataset_EOG_5Stages, EdfDataset_2EEG_1EOG
from network import MySleepNet_1Chan, MySleepNet_2Chan
from loss import Kl_loss
import os
import argparse
import torch.nn.functional as F
import pandas as pd
device = "cuda" if torch.cuda.is_available() else "cpu"
# wandb用于在线追溯实验，方便实验结果保存和调参，如若需要解开注释即可
# import wandb

# run-settings
modal = 'EEG'      # EEG, EOG, Fused3, Fused5
model_save_path = "./models/with_val/EEG/"

# 命令行传参
parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=150)
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--seq_len", type=int, default=20)
parser.add_argument("--network", type=str, default="LSTM", help="GRU | LSTM | Attention")
args = parser.parse_args()


# data_path = "G:/Data/Temp"
data_path = "G:/Data/SleepEdf39/pre_processing"
#data_path = "G:/Data/SleepEdf153/pre_processing"
KF = KFold(n_splits=20)

#定义超参数
n_epochs = 40               # 迭代次数,每个epoch会对整个训练集遍历一遍
batch_size = args.batch_size   # 一次加载的数据量，对一个epoch中的样本数的拆分
learning_rate = 0.001          # 学习率，或者说步长
seq_len = args.seq_len
network = args.network

seed = 100
np.random.seed(seed)
torch.manual_seed(seed)  # 为CPU设置随机种子
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)  # 为当前GPU设置随机种子
    torch.cuda.manual_seed_all(seed)  # 为所有GPU设置随


def get_model(modal):
    if modal == 'EEG':
        model = MySleepNet_1Chan(stages=5)
    elif modal == 'EOG':
        model = MySleepNet_1Chan(stages=3)
    elif modal == 'Fused3':
        model = MySleepNet_2Chan(stages=3)
    elif modal == 'Fused5':
        model = MySleepNet_2Chan(stages=5)
    else:
        exit()
    return model



def evaluate_model(model_, loader_data_):
    evaluate_loss = []
    evaluate_trues = []
    evaluate_preds = []

    if modal == 'EEG' or modal == 'Fused5':
        criterion = Kl_loss(stage=5)
    else:
        criterion = Kl_loss(stage=3)
        assert modal == 'EOG' or modal == 'Fused3'

    with torch.no_grad():  # 不计算梯度，加快运算速度
        for batch_idx, (X, Y) in enumerate(loader_data_):
            if modal == 'Fused3':
                X_in, Y_in = X[:, [0, 2], :, :], Y[:, 1, :]  # X_eeg:[16, 2, 20, 3000]
            elif modal == 'Fused5':
                X_in, Y_in = X[:, [0, 2], :, :], Y[:, 0, :]
            elif modal == 'EEG':
                X_in, Y_in = X[:, 0, :, :], Y[:, 0, :]
            elif modal == 'EOG':
                X_in, Y_in = X[:, 2, :, :], Y[:, 1, :]

            X_in, Y_in = X_in.to(device), Y_in.reshape(-1, ).to(device, dtype=torch.long)

            pred = model_(X_in)
            y = F.one_hot(Y_in, pred.size(-1))
            loss = criterion(y, pred + 1, epoch_idx + 1, 10 * (batch_idx + 1))

            evaluate_trues.append(Y_in.cpu())
            evaluate_preds.append(pred.argmax(dim=1).cpu())
            train_loss.append(loss.item())
    evaluate_trues = np.hstack(evaluate_trues)
    evaluate_preds = np.hstack(evaluate_preds)
    evaluate_acc = skmetrics.accuracy_score(y_true=evaluate_trues, y_pred=evaluate_preds)
    evaluate_f1_score = skmetrics.f1_score(evaluate_trues, evaluate_preds, average="macro")

    return evaluate_acc, evaluate_loss, evaluate_f1_score


def train_model(model_, loader_data_, optimizer_, epoch_idx):
    train_loss = []
    train_trues = []
    train_preds = []

    if modal == 'EEG' or modal == 'Fused5':
        criterion = Kl_loss(stage=5)
    else:
        criterion = Kl_loss(stage=3)
        assert modal == 'EOG' or modal == 'Fused3'

    for batch_idx, (X, Y) in enumerate(loader_data_):
        optimizer_.zero_grad()

        if modal == 'Fused3':
            X_in, Y_in = X[:, [0, 2], :, :], Y[:, 1, :]       # X_eeg:[16, 2, 20, 3000]
        elif modal == 'Fused5':
            X_in, Y_in = X[:, [0, 2], :, :], Y[:, 0, :]
        elif modal == 'EEG':
            X_in, Y_in = X[:, 0, :, :], Y[:, 0, :]
        elif modal == 'EOG':
            X_in, Y_in = X[:, 2, :, :], Y[:, 1, :]

        X_in, Y_in = X_in.to(device), Y_in.reshape(-1, ).to(device, dtype=torch.long)

        pred = model_(X_in)
        y = F.one_hot(Y_in, pred.size(-1))

        loss = criterion(y, pred + 1, epoch_idx + 1, 10 * (batch_idx + 1))

        train_trues.append(Y_in.cpu())
        train_preds.append(pred.argmax(dim=1).cpu())
        train_loss.append(loss.item())
        loss.backward()
        optimizer_.step()
    train_trues = np.hstack(train_trues)
    train_preds = np.hstack(train_preds)
    train_acc = skmetrics.accuracy_score(y_true=train_trues, y_pred=train_preds)
    train_f1_score = skmetrics.f1_score(train_trues, train_preds, average="macro")

    return train_acc, train_loss, train_f1_score



#加载数据
files = glob.glob(os.path.join(data_path, "*.npz"))
files_arr = np.array(files)
print('len(files):', len(files_arr))

fold = 0
for tr_val_index, test_index in KF.split(files_arr):
    fold = fold + 1
    print('--------------------------------------------------------------------------------- fold:', fold)

    # 数据集划分
    val_files = files_arr[tr_val_index[-2:]]
    train_files = files_arr[tr_val_index[:-2]]
    test_files = files_arr[test_index]

    model = get_model(modal=modal)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)  # 设置优化器

    train_data = EdfDataset_2EEG_1EOG(files=train_files, seq_len=seq_len)
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_data = EdfDataset_2EEG_1EOG(files=val_files, seq_len=seq_len)
    val_dataloader = DataLoader(val_data, batch_size=batch_size)
    test_data = EdfDataset_2EEG_1EOG(files=test_files, seq_len=seq_len)
    test_dataloader = DataLoader(test_data, batch_size=batch_size)

    best_acc = -1
    factor = 0.015
    for epoch_idx in range(n_epochs):
        # 训练模型
        print('train the model')
        model.train()
        train_acc, train_loss, train_f1_score = train_model(model, train_dataloader, optimizer, epoch_idx)

        # 验证模型
        print('validate the model')
        model.eval()
        val_acc, val_loss, val_f1_score = evaluate_model(model, val_dataloader)

        print(f"[epoch: {epoch_idx + 1:3}/{n_epochs:3}] || train_loss:{np.sum(train_loss):6.2f} || train_acc:{train_acc * 100:5.2f}% || train_mf1:{train_f1_score:4.2f}\
        || val_loss:{np.sum(val_loss):6.2f} || val_acc:{val_acc * 100:5.2f}% || val_mf1:{val_f1_score:4.2f}")

        if best_acc < (factor * train_acc + (0.999 - factor) * val_acc):
            best_acc = factor * train_acc + (0.999 - factor) * val_acc
            torch.save(model.state_dict(), os.path.join(model_save_path, f"{fold}.pt"))
            print(f"[epoch: {epoch_idx + 1:3}/{n_epochs:3}] .................. save the best model ..................")

    del train_data, train_dataloader, val_data, val_dataloader, model

    # 测试 last-model
    print('test the last model')

    # 测试 best-model
    model_para = os.path.join(model_save_path, f"{fold}.pt")
    print('test the best model:')
    #best_model = MySleepNet_2EEG(network=network, seq_len=seq_len)
    best_model = get_model(modal=modal)
    best_model.load_state_dict(torch.load(model_para, map_location=device))
    best_model.to(device)
    best_model.eval()
    test_acc, test_loss, test_f1_score = evaluate_model(best_model, test_dataloader)
    print(f"[---------- fold: {fold:3} / {test_acc:7}]")

    del test_data, test_dataloader, best_model




