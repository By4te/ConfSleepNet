from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch
from sklearn.model_selection import KFold
import sklearn.metrics as skmetrics
from sklearn.metrics import confusion_matrix, cohen_kappa_score
import numpy as np
import timeit
import matplotlib.pyplot as plt
import glob
from dataset import EdfDataset_2EEG_1EOG, EdfDataset_2EEG, EdfDataset_1EEG, EdfDataset_EOG_3Stages, EdfDataset_EOG_5Stages
from network import MySleepNet_1Chan, MySleepNet_2Chan
from loss import Kl_loss
import os
import argparse
import torch.nn.functional as F
import pandas as pd


# 命令行传参
parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=150)
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--seq_len", type=int, default=20)
parser.add_argument("--network", type=str, default="LSTM", help="GRU | LSTM | Attention")
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
seed = 100
np.random.seed(seed)
torch.manual_seed(seed)  # 为CPU设置随机种子
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)  # 为当前GPU设置随机种子
    torch.cuda.manual_seed_all(seed)  # 为所有GPU设置随

#定义超参数
n_epochs = 40 #args.n_epochs       # 迭代次数,每个epoch会对整个训练集遍历一遍
batch_size = args.batch_size   # 一次加载的数据量，对一个epoch中的样本数的拆分
learning_rate = 0.001          # 学习率，或者说步长
seq_len = args.seq_len
network = args.network

# data_path = "G:/Data/Temp"
data_path = "G:/Data/SleepEdf39/pre_processing"
# data_path = "G:/Data/SleepEdf153/pre_processing"
KF = KFold(n_splits=20)



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



# alpha1, alpha2: [N, classes]
def Conflict_Aware_Combin_two(classes, alpha1, alpha2):
    # Calculate the merger of two DS evidences
    alpha = dict()
    alpha[0], alpha[1] = alpha1, alpha2
    b, S, E, u = dict(), dict(), dict(), dict()
    for v in range(2):
        S[v] = torch.sum(alpha[v], dim=1, keepdim=True)  # [N, 1]
        E[v] = alpha[v] - 1
        b[v] = E[v] / (S[v].expand(E[v].shape))  # [N, classes]
        u[v] = classes / S[v]  # [N, 1]

    # calculate the conflict: C
    mul_b01 = torch.mul(b[0], b[1])
    sum_b01 = torch.sum(mul_b01, dim=1, keepdim=True)      # [N, 1]
    sum_b0 = torch.sum(b[0], dim=1, keepdim=True)          # [N, 1]
    sum_b1 = torch.sum(b[1], dim=1, keepdim=True)          # [N, 1]
    C = 1 - sum_b01 / (torch.mul(sum_b0, sum_b1))          # [N, 1]

    # calculate new_u
    # assert C >= 0 and C <= 1
    mul_u01 = torch.mul(u[0], u[1])    # [N, 1]
    sum_u01 = u[0] + u[1]
    term1 = torch.mul(1 - C, mul_u01)  # [N, 1]
    new_u = term1 + 2 * torch.mul(C, mul_u01) / sum_u01    # [N, 1]

    # calculate new_b
    u0_expand = u[0].expand(b[0].shape)     # [N, classes]
    u1_expand = u[1].expand(b[1].shape)     # [N, classes]
    sum_u01_expand = sum_u01.expand(b[1].shape)
    mul_u0_b1 = torch.mul(u0_expand, b[1])
    mul_u1_b0 = torch.mul(u1_expand, b[0])
    term1_expand = term1.expand(b[0].shape)
    term2 = torch.mul(term1_expand, b[0] + b[1])
    new_b = (mul_u0_b1 + mul_u1_b0 + term2) / sum_u01_expand

    # calculate new S
    new_S = classes / new_u     # [N, 1]
    # calculate new e_k
    e_new = torch.mul(new_b, new_S.expand(new_b.shape))
    alpha_a = e_new + 1
    return alpha_a


# alpha1, alpha2: [N, classes]     proposed by 'Xidian University'
def Conflict_Combin_two(classes, alpha1, alpha2):
    # Calculate the merger of two DS evidences
    alpha = dict()
    alpha[0], alpha[1] = alpha1, alpha2
    b, S, E, u = dict(), dict(), dict(), dict()
    for v in range(2):
        S[v] = torch.sum(alpha[v], dim=1, keepdim=True)  # [N, 1]
        E[v] = alpha[v] - 1
        b[v] = E[v] / (S[v].expand(E[v].shape))  # [N, classes]
        u[v] = classes / S[v]  # [N, 1]

    # calculate new_u
    sum_u01 = u[0] + u[1]
    new_u = 2 * torch.mul(u[0], u[1]) / sum_u01   # [N, 1]

    # calculate new_b
    u0_expand = u[0].expand(b[0].shape)     # [N, classes]
    u1_expand = u[1].expand(b[1].shape)     # [N, classes]
    sum_u01_expand = sum_u01.expand(b[1].shape)
    mul_u0_b1 = torch.mul(u0_expand, b[1])
    mul_u1_b0 = torch.mul(u1_expand, b[0])
    new_b = (mul_u0_b1 + mul_u1_b0) / sum_u01_expand

    # calculate new S
    new_S = classes / new_u     # [N, 1]
    # calculate new e_k
    e_new = torch.mul(new_b, new_S.expand(new_b.shape))
    alpha_a = e_new + 1
    return alpha_a



# alpha1, alpha2: [N, classes]
def DS_Combin_two(classes, alpha1, alpha2):
        # Calculate the merger of two DS evidences
        alpha = dict()
        alpha[0], alpha[1] = alpha1, alpha2
        b, S, E, u = dict(), dict(), dict(), dict()
        for v in range(2):
            S[v] = torch.sum(alpha[v], dim=1, keepdim=True)
            E[v] = alpha[v] - 1
            b[v] = E[v] / (S[v].expand(E[v].shape))
            u[v] = classes / S[v]

        # b^0 @ b^(0+1)
        bb = torch.bmm(b[0].view(-1, classes, 1), b[1].view(-1, 1, classes))
        # b^0 * u^1
        uv1_expand = u[1].expand(b[0].shape)
        bu = torch.mul(b[0], uv1_expand)
        # b^1 * u^0
        uv_expand = u[0].expand(b[0].shape)
        ub = torch.mul(b[1], uv_expand)
        # calculate K
        bb_sum = torch.sum(bb, dim=(1, 2), out=None)
        bb_diag = torch.diagonal(bb, dim1=-2, dim2=-1).sum(-1)
        # bb_diag1 = torch.diag(torch.mm(b[v], torch.transpose(b[v+1], 0, 1)))
        K = bb_sum - bb_diag

        # calculate b^a
        b_a = (torch.mul(b[0], b[1]) + bu + ub) / ((1 - K).view(-1, 1).expand(b[0].shape))
        # calculate u^a
        u_a = torch.mul(u[0], u[1]) / ((1 - K).view(-1, 1).expand(u[0].shape))
        # test = torch.sum(b_a, dim = 1, keepdim = True) + u_a #Verify programming errors

        # calculate new S
        S_a = classes / u_a
        # calculate new e_k
        e_a = torch.mul(b_a, S_a.expand(b_a.shape))
        alpha_a = e_a + 1
        return alpha_a



def Compute_Uncertainty(alpha):
    S = torch.sum(alpha, dim=1, keepdim=True)  # [N, 1]
    E = alpha - 1
    b = E / (S.expand(E.shape))                # [N, classes]
    u = 5 / S
    assert (u.shape[0] == alpha.shape[0])
    assert (u.shape[1] == 1)

    return u


# alpha[0 ... num-1][N][classes]
def Conflict_Aware_Combin_Plus(alpha, num):
    c = dict()

    #---------------------------------------------
    # 计算 alpha[0] 和 alpha[1, ..., num-1] 的冲突
    c[0] = Compute_Uncertainty(alpha[0])   # [N, 1]
    c[1] = Compute_Uncertainty(alpha[1])
    c[2] = Compute_Uncertainty(alpha[2])
    c[3] = Compute_Uncertainty(alpha[3])

    u = torch.cat((c[0], c[1]), 1)              # [N, 2]
    u = torch.cat((u, c[2]), 1)                 # [N, 3]
    u = torch.cat((u, c[3]), 1)                 # [N, 4]
    value, index = torch.max(u, 1)        # [N]

    for x in range(index.shape[0]):
        if index[x] == 1:
            temp = alpha[0][x].clone()
            alpha[0][x] = alpha[1][x]
            alpha[1][x] = temp
        elif index[x] == 2:
            temp = alpha[0][x].clone()
            alpha[0][x] = alpha[2][x]
            alpha[2][x] = temp
        elif index[x] == 3:
            temp = alpha[0][x].clone()
            alpha[0][x] = alpha[3][x]
            alpha[3][x] = temp
        else:
            assert (index[x] == 0)

    # ---------------------------------------------
    cc = dict()
    cc[0] = Compute_Uncertainty(alpha[1])
    cc[1] = Compute_Uncertainty(alpha[2])
    cc[2] = Compute_Uncertainty(alpha[3])

    uu = torch.cat((cc[0], cc[1]), 1)  # [N, 2]
    uu = torch.cat((uu, cc[2]), 1)  # [N, 3]
    value1, index1 = torch.max(uu, 1)

    for x in range(index1.shape[0]):
        if index1[x] == 1:
            temp = alpha[1][x].clone()
            alpha[1][x] = alpha[2][x]
            alpha[2][x] = temp
        elif index1[x] == 2:
            temp = alpha[1][x].clone()
            alpha[1][x] = alpha[3][x]
            alpha[3][x] = temp
        else:
            assert (index1[x] == 0)

    # ---------------------------------------------
    c3 = dict()
    c3[0] = Compute_Uncertainty(alpha[2])
    c3[1] = Compute_Uncertainty(alpha[3])

    u3 = torch.cat((c3[0], c3[1]), 1)  # [N, 2]
    value3, index3 = torch.max(u3, 1)
    for x in range(index3.shape[0]):
        if index3[x] == 1:
            temp = alpha[2][x].clone()
            alpha[2][x] = alpha[3][x]
            alpha[3][x] = temp
        else:
            assert (index3[x] == 0)

    # ---------------------------------------------
    alpha01 = Conflict_Aware_Combin_two(5, alpha[0], alpha[1])
    alpha02 = Conflict_Aware_Combin_two(5, alpha01,  alpha[2])
    alpha03 = Conflict_Aware_Combin_two(5, alpha02,  alpha[3])

    return alpha03



# Conflict_Combin_two    DS_Combin_two    Conflict_Aware_Combin_two
def evaluate_model(model_eeg, model_eog, model_fused3, model_fused5, loader_data, test=True):
    Trues3 = []
    Trues5 = []
    preds_eeg = []
    preds_eog = []
    preds_fused3 = []
    preds_fused5 = []
    conflict_aware_preds = []
    conflict_preds = []
    DS_preds = []
    matrix = []

    with torch.no_grad():  # 不计算梯度，加快运算速度
        for batch_idx, (X, y) in enumerate(loader_data):    # X:[16, 3, 20, 3000]  y:[16, 2, 20]
            X_eeg, X_eog, X_fused = X[:, 0, :, :], X[:, 2, :, :], X[:, [0, 2], :, :]  # X_fused:[16, 2, 20, 3000], X_eog:[16, 20, 3000]
            y5, y3 = y[:, 0, :], y[:, 1, :]  # [16, 20]
            X_eeg, y5 = X_eeg.to(device), y5.reshape(-1, ).to(device, dtype=torch.long)
            X_eog, y3 = X_eog.to(device), y3.reshape(-1, ).to(device, dtype=torch.long)
            X_fused = X_fused.to(device)

            res_eeg = model_eeg(X_eeg)
            tmp_eog = model_eog(X_eog)
            tmp_fused3 = model_fused3(X_fused)
            res_fused5 = model_fused5(X_fused)
            decompose = torch.Tensor([[1, 0, 0, 0, 0], [0, 1 / 3, 1 / 3, 1 / 3, 0], [0, 0, 0, 0, 1]]).to(device)
            res_eog = torch.mm(tmp_eog, decompose)
            res_fused3 = torch.mm(tmp_fused3, decompose)

            Trues3.append(y3.cpu())
            Trues5.append(y5.cpu())
            preds_eeg.append(res_eeg.argmax(dim=1).cpu())
            preds_eog.append(tmp_eog.argmax(dim=1).cpu())
            preds_fused3.append(tmp_fused3.argmax(dim=1).cpu())
            preds_fused5.append(res_fused5.argmax(dim=1).cpu())

            # Compute_Conflict_degree(alpha1, alpha2): [N, 1]
            # e: res_eeg, res_eog, res_fused3, res_fused5
            alpha = dict()
            alpha[0], alpha[1], alpha[2], alpha[3] = res_eeg + 1, res_eog + 1, res_fused3 + 1, res_fused5 + 1
            comb_eeg_eog_f3_f5 = Conflict_Aware_Combin_Plus(alpha, 4)

            """
            comb_eeg_eog = Conflict_Aware_Combin_two(5, res_eeg + 1, res_eog + 1)
            comb_eeg_eog_f3 = Conflict_Aware_Combin_two(5, comb_eeg_eog, res_fused3 + 1)
            comb_eeg_eog_f3_f5 = Conflict_Aware_Combin_two(5, comb_eeg_eog_f3, res_fused5 + 1)
            """
            conflict_aware_preds.append(comb_eeg_eog_f3_f5.argmax(dim=1).cpu())


            comb_eeg_eog1 = Conflict_Combin_two(5, res_eeg + 1, res_eog + 1)
            comb_eeg_eog1_f3 = Conflict_Combin_two(5, comb_eeg_eog1, res_fused3 + 1)
            comb_eeg_eog1_f3_f5 = Conflict_Combin_two(5, comb_eeg_eog1_f3, res_fused5 + 1)
            conflict_preds.append(comb_eeg_eog1_f3_f5.argmax(dim=1).cpu())

            comb_eeg_eog2 = DS_Combin_two(5, res_eeg + 1, res_eog + 1)
            comb_eeg_eog2_f3 = DS_Combin_two(5, comb_eeg_eog2, res_fused3 + 1)
            comb_eeg_eog2_f3_f5 = DS_Combin_two(5, comb_eeg_eog2_f3, res_fused5 + 1)
            DS_preds.append(comb_eeg_eog2_f3_f5.argmax(dim=1).cpu())

    Trues5, preds_eeg = np.hstack(Trues5), np.hstack(preds_eeg)
    acc_eeg = skmetrics.accuracy_score(y_true=Trues5, y_pred=preds_eeg)
    Trues3, preds_eog = np.hstack(Trues3), np.hstack(preds_eog)
    acc_eog = skmetrics.accuracy_score(y_true=Trues3, y_pred=preds_eog)
    preds_fused3, preds_fused5 = np.hstack(preds_fused3), np.hstack(preds_fused5)
    acc_fused3 = skmetrics.accuracy_score(y_true=Trues3, y_pred=preds_fused3)
    acc_fused5 = skmetrics.accuracy_score(y_true=Trues5, y_pred=preds_fused5)

    conflict_aware_preds = np.hstack(conflict_aware_preds)
    conflict_aware_acc = skmetrics.accuracy_score(y_true=Trues5, y_pred=conflict_aware_preds)

    conflict_preds = np.hstack(conflict_preds)
    conflict_acc = skmetrics.accuracy_score(y_true=Trues5, y_pred=conflict_preds)

    DS_preds = np.hstack(DS_preds)
    DS_acc = skmetrics.accuracy_score(y_true=Trues5, y_pred=DS_preds)

    #if test == True:
    """
    kappa_eeg = cohen_kappa_score(Trues5, preds_eeg)
    kappa_eog = cohen_kappa_score(Trues3, preds_eog)
    kappa_fused3 = cohen_kappa_score(Trues3, preds_fused3)
    kappa_fused5 = cohen_kappa_score(Trues5, preds_fused5)
    """
    kappa_DS = cohen_kappa_score(Trues5, DS_preds)
    kappa_conflict = cohen_kappa_score(Trues5, conflict_preds)
    kappa_conflict_aware = cohen_kappa_score(Trues5, conflict_aware_preds)
    """
    f1_eeg = skmetrics.f1_score(Trues5, preds_eeg, average="macro")
    f1_eog = skmetrics.f1_score(Trues3, preds_eog, average="macro")
    f1_fused3 = skmetrics.f1_score(Trues3, preds_fused3, average="macro")
    f1_fused5 = skmetrics.f1_score(Trues5, preds_fused5, average="macro")
    """
    f1_DS = skmetrics.f1_score(Trues5, DS_preds, average="macro")
    f1_conflict = skmetrics.f1_score(Trues5, conflict_preds, average="macro")
    f1_conflict_aware = skmetrics.f1_score(Trues5, conflict_aware_preds, average="macro")

    print(f"[ fold: {fold:3} / EEG: {acc_eeg:6.4f} / EOG: {acc_eog:6.4f} / Fused3: {acc_fused3:6.4f}  / Fused5: {acc_fused5:6.4f} ]")
    print(f"[ F1-score: {f1_DS:6.3}, {f1_conflict:6.3}, {f1_conflict_aware:6.3} ]")
    print(f"[ kappa:  {kappa_DS:6.3}, {kappa_conflict:6.3}, {kappa_conflict_aware:6.3} ]")

    if test == True:
        matrix_eeg = confusion_matrix(Trues5, preds_eeg)
        matrix_eog = confusion_matrix(Trues3, preds_eog)
        matrix_fused3 = confusion_matrix(Trues3, preds_fused3)
        matrix_fused5 = confusion_matrix(Trues5, preds_fused5)
        matrix_DS = confusion_matrix(Trues5, DS_preds)
        matrix_conflict = confusion_matrix(Trues5, conflict_preds)
        matrix_conflict_aware = confusion_matrix(Trues5, conflict_aware_preds)
        matrix.append(matrix_eeg)
        matrix.append(matrix_eog)
        matrix.append(matrix_fused3)
        matrix.append(matrix_fused5)
        matrix.append(matrix_DS)
        matrix.append(matrix_conflict)
        matrix.append(matrix_conflict_aware)
        #print('matrix_conflict_aware:', matrix_conflict_aware)

    return DS_acc, conflict_acc, conflict_aware_acc, f1_DS, f1_conflict, f1_conflict_aware, kappa_DS, kappa_conflict, kappa_conflict_aware, matrix


#加载数据
files = glob.glob(os.path.join(data_path, "*.npz"))
files_arr = np.array(files)
print('len(files):', len(files_arr))

fold = 0
# ******************** 以下为全局变量 ********************
gl_DS_acc = 0.
gl_conflict_acc = 0.
gl_conflict_aware_acc = 0.
gl_kappa_DS = 0.
gl_kappa_conflict = 0.
gl_kappa_conflict_aware = 0.
gl_f1_DS = 0.
gl_f1_conflict = 0.
gl_f1_conflict_aware = 0.


for tr_val_index, test_index in KF.split(files_arr):
    fold = fold + 1
    print('--------------------------------------------------------------------------------- fold:', fold)

    # 数据集划分
    val_files = files_arr[tr_val_index[-2:]]
    train_files = files_arr[tr_val_index[:-2]]
    test_files = files_arr[test_index]

    test_dataset = EdfDataset_2EEG_1EOG(test_files, seq_len=seq_len)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

    model_eeg = get_model(modal='EEG')
    model_eog = get_model(modal='EOG')
    model_fused3 = get_model(modal='Fused3')
    model_fused5 = get_model(modal='Fused5')

    para_eeg = os.path.join("./models/edf39/EEG/", f"{fold}.pt")
    para_eog = os.path.join("./models/edf39/EOG/", f"{fold}.pt")
    para_fused3 = os.path.join("./models/edf39/Fused3/", f"{fold}.pt")
    para_fused5 = os.path.join("./models/edf39/Fused5/", f"{fold}.pt")

    print('test the best model:')
    model_eeg.load_state_dict(torch.load(para_eeg, map_location=device))
    model_eeg.to(device), model_eeg.eval()
    model_eog.load_state_dict(torch.load(para_eog, map_location=device))
    model_eog.to(device), model_eog.eval()
    model_fused3.load_state_dict(torch.load(para_fused3, map_location=device))
    model_fused3.to(device), model_fused3.eval()
    model_fused5.load_state_dict(torch.load(para_fused5, map_location=device))
    model_fused5.to(device), model_fused5.eval()

    DS_acc, conflict_acc, conflict_aware_acc, f1_DS, f1_conflict, f1_conflict_aware, kappa_DS, kappa_conflict, kappa_conflict_aware, matrix = evaluate_model(
        model_eeg,
        model_eog,
        model_fused3,
        model_fused5,
        test_dataloader)

    gl_DS_acc = gl_DS_acc + DS_acc
    gl_conflict_acc = gl_conflict_acc + conflict_acc
    gl_conflict_aware_acc = gl_conflict_aware_acc + conflict_aware_acc
    gl_kappa_DS = gl_kappa_DS + kappa_DS
    gl_kappa_conflict = gl_kappa_conflict + kappa_conflict
    gl_kappa_conflict_aware = gl_kappa_conflict_aware + kappa_conflict_aware
    gl_f1_DS = gl_f1_DS + f1_DS
    gl_f1_conflict = gl_f1_conflict + f1_conflict
    gl_f1_conflict_aware = gl_f1_conflict_aware + f1_conflict_aware

    if fold == 1:
        gl_matrix = matrix
    else:
        for i in range(7):
            gl_matrix[i] = gl_matrix[i] + matrix[i]

    print(f"[ ----- DS acc: {DS_acc:6.3f}  / Conflict acc: {conflict_acc:6.3f}   / Conflict_aware acc: {conflict_aware_acc:6.3f}]")
    print(f"[ ----- gl_F1-score: {gl_f1_DS:6.3}, {gl_f1_conflict:6.3}, {gl_f1_conflict_aware:6.3} ]")
    print(f"[ ----- gl_kappa:  {gl_kappa_DS:6.3}, {gl_kappa_conflict:6.3}, {gl_kappa_conflict_aware:6.3} ]")

    del test_dataloader

print('  matrix_eeg:\n', gl_matrix[0])
print('  matrix_eog:\n', gl_matrix[1])
print('  matrix_fused3:\n', gl_matrix[2])
print('  matrix_fused5:\n', gl_matrix[3])
print('  matrix_DS:\n', gl_matrix[4])
print('  matrix_conflict:\n', gl_matrix[5])
print('  matrix_conflict_aware:\n', gl_matrix[6])

print(f"[ ***** Avg_acc ***** DS_acc: {gl_DS_acc/20:6.3f}  / Conflict acc: {gl_conflict_acc/20:6.3f}   / Conflict_aware acc: {gl_conflict_aware_acc/20:6.3f}]")
print(f"[ ***** Avg_F1 ***** gl_F1-score: {gl_f1_DS/20:6.3}, {gl_f1_conflict/20:6.3}, {gl_f1_conflict_aware/20:6.3} ]")
print(f"[ ***** Avg_kappa ***** gl_kappa:  {gl_kappa_DS/20:6.3}, {gl_kappa_conflict/20:6.3}, {gl_kappa_conflict_aware/20:6.3} ]")

