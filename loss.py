import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

class FocalLoss(nn.Module):

    def __init__(self, gamma=0, eps=1e-7):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps

    def forward(self, input, target):            # EOG 3 stages, input.shape, target.shape: [320, 3], [320]
        y = F.one_hot(target, input.size(-1))    # [320, 3]
        logit = F.softmax(input, dim=-1)         # input.shape: [320, 3]
        logit = logit.clamp(self.eps, 1. - self.eps)

        loss = -1 * y * torch.log(logit) # cross entropy
        loss = loss * (1 - logit) ** self.gamma # focal loss

        return loss.mean()

class Kl_loss(nn.Module):
    def __init__(self, stage=5): #  python中的构造函数
        super(Kl_loss, self).__init__()
        self.stages = stage

    def kl_fun(self, alpha):
        beta = torch.tensor(np.ones((1, self.stages)), dtype=torch.float32)
        # beta = torch.tensor(np.ones((1, 5)), dtype=np.float32)
        S_alpha = torch.sum(alpha, dim=1, keepdim=True)
        S_beta = torch.sum(beta, dim=1, keepdim=True)
        lnB = torch.lgamma(S_alpha) - torch.sum(torch.lgamma(alpha), dim=1, keepdim=True)
        lnB_uni = torch.sum(torch.lgamma(beta), dim=1, keepdim=True) - torch.lgamma(S_beta)
        dg0 = torch.digamma(S_alpha)
        dg1 = torch.digamma(alpha)
        alpha = alpha.to("cuda")
        beta = beta.to("cuda")
        dg0 = dg0.to("cuda")
        dg1 = dg1.to("cuda")
        lnB = lnB.to("cuda")
        lnB_uni = lnB_uni.to("cuda")
        kl = torch.sum((alpha - beta) * (dg1 - dg0), dim=1, keepdim=True) + lnB + lnB_uni
        return kl

    # Calculate the loss
    def forward(self, y_new, alpha, global_step, annealing_step):
        S = torch.sum(alpha, dim=1, keepdim=True)
        # E = alpha - 1
        m = alpha / S
        A = torch.sum((y_new - m) ** 2, dim=1, keepdim=True)
        B = torch.sum(alpha * (S - alpha) / (S * S * (S + 1)), dim=1, keepdim=True)
        num1 = 1.0
        num2 = global_step / annealing_step
        annealing_coef = min(num1, num2)   #退火系数
        alp = alpha * (1 - y_new) + y_new
        C = annealing_coef * self.kl_fun(alp)
        loss = (A + B) + C
        return loss.mean()

