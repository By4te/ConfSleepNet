# 融合后加卷积

import torch
import torch.nn as nn
from collections import OrderedDict
import torch.nn.functional as F
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"


class Fusion_layer(nn.Module):
    def __init__(self):
        super(Fusion_layer, self).__init__()
        self.W = nn.Parameter(torch.rand(1, 5), requires_grad=True)
        self.decompose = torch.Tensor([[1, 0, 0, 0, 0], [0, 1/3, 1/3, 1/3, 0], [0, 0, 0, 0, 1]])  # (3, 5)

    def forward(self, x_eeg, x_eog):
        assert x_eeg.shape[-1] == 5 and x_eog.shape[-1] == 3
        x_eeg = torch.mul(x_eeg, self.W)           # [N, 5]

        x_eog = torch.mm(x_eog, self.decompose.to(device))    # [N, 3] --> [N, 5]
        W = 1 - self.W
        W = W.clamp(0, 1.0)
        x_eog = torch.mul(x_eog, W)       # [N, 5]

        return x_eeg + x_eog

class Fusion_layer_3(nn.Module):
    def __init__(self):
        super(Fusion_layer_3, self).__init__()
        self.W_eeg = nn.Parameter(torch.rand(1, 5), requires_grad=True)
        self.W_eog = nn.Parameter(torch.rand(1, 5), requires_grad=True)
        self.decompose = torch.Tensor([[1, 0, 0, 0, 0], [0, 1/3, 1/3, 1/3, 0], [0, 0, 0, 0, 1]])  # (3, 5)

    def forward(self, x_eeg, x_eog, x_fused):
        assert x_eeg.shape[-1] == 5 and x_eog.shape[-1] == 3 and x_fused.shape[-1] == 3
        x_eeg = torch.mul(x_eeg, self.W_eeg)           # [N, 5]

        x_eog = torch.mm(x_eog, self.decompose.to(device))    # [N, 3] --> [N, 5]
        x_eog = torch.mul(x_eog, self.W_eog)

        x_fused = torch.mm(x_fused, self.decompose.to(device))
        W_fused = 1 - self.W_eeg - self.W_eog
        W_fused = W_fused.clamp(0, 1.0)
        x_fused = torch.mul(x_fused, W_fused)       # [N, 5]

        return x_eeg + x_eog + x_fused

# GCNet 注意力
class GlobalContextBlock(nn.Module):
    def __init__(self, in_channels, scale=16):
        super(GlobalContextBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = self.in_channels // scale
        self.Conv_key = nn.Conv1d(self.in_channels, 1, 1)
        self.SoftMax = nn.Softmax(dim=1)
        self.Conv_value = nn.Sequential(
            nn.Conv1d(self.in_channels, self.out_channels, 1),
            nn.LayerNorm([self.out_channels, 1]),
            nn.ReLU(),
            nn.Conv1d(self.out_channels, self.in_channels, 1),
        )

    def forward(self, x):
        b, c, s = x.size()  # b: batch size, c: channel, s: signal length
        # key -> [b, 1, s] -> [b, 1, s] -> [b, s, 1]
        key = self.SoftMax(self.Conv_key(x).view(b, 1, -1).permute(0, 2, 1).view(b, -1, 1).contiguous())
        query = x.view(b, c, s)
        # [b, c, s] * [b, s, 1]
        concate_QK = torch.matmul(query, key)       #[b,c,1]
        concate_QK = concate_QK.view(b, c, 1).contiguous()
        value = self.Conv_value(concate_QK)
        out = x + value
        return out

# CBAM注意力

# （1）通道注意力机制
class channel_attention(nn.Module):
    def __init__(self, in_channel, ratio=4):
        super(channel_attention, self).__init__()

        # 全局最大池化 [b,c,s]==>[b,c,1]
        self.max_pool = nn.AdaptiveMaxPool1d(output_size=1)
        # 全局平均池化 [b,c,s]==>[b,c,1]
        self.avg_pool = nn.AdaptiveAvgPool1d(output_size=1)

        # 第一个全连接层, 通道数下降4倍
        self.fc1 = nn.Linear(in_features=in_channel, out_features=in_channel // ratio, bias=False)
        # 第二个全连接层, 恢复通道数
        self.fc2 = nn.Linear(in_features=in_channel // ratio, out_features=in_channel, bias=False)

        # relu激活函数
        self.relu = nn.ReLU()
        # sigmoid激活函数
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):
        b, c, s = inputs.shape
        # 输入信号做全局最大池化 [b,c,s]==>[b,c,1]
        max_pool = self.max_pool(inputs)
        # 输入信号的全局平均池化 [b,c,s]==>[b,c,1]
        avg_pool = self.avg_pool(inputs)

        # 调整池化结果的维度 [b,c,1]==>[b,c]
        max_pool = max_pool.view([b, c])
        avg_pool = avg_pool.view([b, c])

        # 第一个全连接层下降通道数 [b,c]==>[b,c//4]
        x_maxpool = self.fc1(max_pool)
        x_avgpool = self.fc1(avg_pool)

        # 激活函数
        x_maxpool = self.relu(x_maxpool)
        x_avgpool = self.relu(x_avgpool)

        # 第二个全连接层恢复通道数 [b,c//4]==>[b,c]
        x_maxpool = self.fc2(x_maxpool)
        x_avgpool = self.fc2(x_avgpool)

        # 将这两种池化结果相加 [b,c]==>[b,c]
        x = x_maxpool + x_avgpool
        # sigmoid函数权值归一化
        x = self.sigmoid(x)
        # 调整维度 [b,c]==>[b,c,1]
        x = x.view([b, c, 1])
        # 输入特征图和通道权重相乘 [b,c,s]
        outputs = inputs * x

        return outputs

# （2）空间注意力机制
class spatial_attention(nn.Module):
    def __init__(self, kernel_size=7):
        super(spatial_attention, self).__init__()

        padding = kernel_size // 2
        # 7*7卷积融合通道信息 [b,2,s]==>[b,1,s]
        self.conv = nn.Conv1d(in_channels=2, out_channels=1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):
        # 在通道维度上最大池化 [b,1,s]  keepdim保留原有深度
        x_maxpool, _ = torch.max(inputs, dim=1, keepdim=True)

        # 在通道维度上平均池化 [b,1,s]
        x_avgpool = torch.mean(inputs, dim=1, keepdim=True)
        # 池化后的结果在通道维度上堆叠 [b,2,s]
        x = torch.cat([x_maxpool, x_avgpool], dim=1)

        # 卷积融合通道信息 [b,2,s]==>[b,1,s]
        x = self.conv(x)
        # 空间权重归一化
        x = self.sigmoid(x)
        # 输入特征图和空间权重相乘
        outputs = inputs * x
        return outputs

class cbam(nn.Module):
    def __init__(self, in_channel, ratio=4, kernel_size=7):
        super(cbam, self).__init__()

        # 实例化通道注意力机制
        self.channel_attention = channel_attention(in_channel=in_channel, ratio=ratio)
        # 实例化空间注意力机制
        self.spatial_attention = spatial_attention(kernel_size=kernel_size)

    def forward(self, inputs):
        # 先将输入信号经过通道注意力机制
        x = self.channel_attention(inputs)
        # 然后经过空间注意力机制
        x = self.spatial_attention(x)
        return x



class SKConv(nn.Module):
    def __init__(self, in_ch, M=3, G=1, r=4, stride=1, L=32) -> None:
        super().__init__()
        """ Constructor
        Args:
        in_ch: input channel dimensionality.
        M: the number of branches.
        G: num of convolution groups.
        r: the ratio for computing d, the length of z.
        stride: stride, default 1.
        L: the minimum dim of the vector z in paper, default 32.
        """
        d = max(int(in_ch / r), L)  # 用来进行线性层的输出通道，当输入数据In_ch很大时，用L就有点丢失数据了。
        self.M = M
        self.in_ch = in_ch
        self.convs = nn.ModuleList([])
        for i in range(M):
            self.convs.append(
                nn.Sequential(
                    nn.Conv1d(in_ch, in_ch, kernel_size=3 + i * 2, stride=stride, padding=1 + i, groups=G),
                    nn.BatchNorm1d(in_ch),
                    nn.ReLU(inplace=True)
                )
            )
        self.fc = nn.Linear(in_ch, d)
        self.fcs = nn.ModuleList([])
        for i in range(M):
            self.fcs.append(nn.Linear(d, in_ch))
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        for i, conv in enumerate(self.convs):  # 第一部分，每个分支的数据进行相加,虽然这里使用的是torch.cat，但是后面又用了unsqueeze和sum进行升维和降维
            fea = conv(x).clone().unsqueeze_(dim=1).clone()  # 这里在1这个地方新增了一个维度  16*1*64*s
            if i == 0:
                feas = fea
            else:
                feas = torch.cat([feas.clone(), fea], dim=1)  # feas.shape  batch*M*in_ch*s
        fea_U = torch.sum(feas.clone(), dim=1)  # batch*in_ch*s
        fea_s = fea_U.clone().mean(-1)  # Batch*in_ch
        fea_z = self.fc(fea_s)  # batch*in_ch-> batch*d
        for i, fc in enumerate(self.fcs):
            vector = fc(fea_z).clone().unsqueeze_(dim=1)  # batch*d->batch*in_ch->batch*1*in_ch
            if i == 0:
                attention_vectors = vector
            else:
                attention_vectors = torch.cat([attention_vectors.clone(), vector], dim=1)  # 同样的相加操作 # batch*M*in_ch
        attention_vectors = self.softmax(attention_vectors.clone())  # 对每个分支的数据进行softmax操作
        attention_vectors = attention_vectors.clone().unsqueeze(-1)  # ->batch*M*in_ch*1
        fea_v = (feas * attention_vectors).clone().sum(dim=1)  # ->batch*in_ch*s
        return fea_v


class NonLocalBlockND(nn.Module):
    def __init__(self, in_channels, inter_channels=None, sub_sample=True, bn_layer=True) -> None:
        super().__init__()
        """
        in_channels: 输入通道
        inter_channels: 中间数据通道
        sub_sample: 是否进行最大池化 一般是True
        bn_layer: 一般是True
        """
        self.sub_sample = sub_sample
        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = self.in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1


        conv_nd = nn.Conv1d
        max_pool_layer = nn.MaxPool1d(kernel_size=2)
        bn = nn.BatchNorm1d

        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1)

        if bn_layer:
            self.W = nn.Sequential(
                conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels, kernel_size=1),
                bn(self.in_channels)
            )
            nn.init.constant_(self.W[1].weight, 0)  # 使用 0 对 参数进行赋初值
            nn.init.constant_(self.W[1].bias, 0)  # 使用 0 对参数进行赋初值
        else:
            self.W = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels, kernel_size=1)
            nn.init.constant_(self.W.weight, 0)
            nn.init.constant_(self.W.bias, 0)

        self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1)
        self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1)

        if sub_sample:
            self.g = nn.Sequential(self.g, max_pool_layer)
            self.phi = nn.Sequential(self.phi, max_pool_layer)

    def forward(self, x):
        batch_size = x.size(0)

        g_x = self.g(x).view(batch_size, self.inter_channels, -1)  # b c s 这里经过了maxpool的操作
        g_x = g_x.permute(0, 2, 1)  # 维度变化 b s c

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)  # b c s 这里没有经过maxpool操作
        theta_x = theta_x.permute(0, 2, 1)  # b s c

        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)  # b c s 这里经过了maxpool
        f = torch.matmul(theta_x, phi_x)  # b s s

        f_div_C = F.softmax(f, dim=-1)  # 对最后一维做softmax

        y = torch.matmul(f_div_C, g_x)  # b s c
        y = y.permute(0, 2, 1).contiguous()  # 得到 batch_size*c*w
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])  # 恢复数据维度
        W_y = self.W(y)  # b c s -> b c s
        z = W_y + x  # 进行残差连接
        return z


class EMA(nn.Module):
    def __init__(self, channels, c2=None, factor=32):
        super(EMA, self).__init__()
        self.groups = factor  # 定义组的数量为 factor，默认值为 32
        assert channels // self.groups > 0  # 确保通道数可以被组数整除
        self.softmax = nn.Softmax(-1)  # 定义 softmax 层，用于最后一个维度
        self.agp = nn.AdaptiveAvgPool1d(1)  # 定义自适应平均池化层，输出大小为 1
        self.pool = nn.AdaptiveAvgPool1d(1)  # 定义自适应平均池化层，在长度上池化
        self.gn = nn.GroupNorm(channels // self.groups, channels // self.groups)  # 定义组归一化层
        self.conv1x1 = nn.Conv1d(channels // self.groups, channels // self.groups, kernel_size=1, stride=1,
                                 padding=0)  # 定义 1x1 卷积层
        self.conv1x3 = nn.Conv1d(channels // self.groups, channels // self.groups, kernel_size=3, stride=1,
                                 padding=1)  # 定义 1x3 卷积层

    def forward(self, x):
        b, c, s = x.size()  # 获取输入张量的大小：批次、通道、长度
        group_x = x.reshape(b * self.groups, -1, s)  # 将输入张量重新形状为 (b * 组数, c // 组数, 长度)
        x_pool = self.pool(group_x)  # 在长度上进行池化
        hw = self.conv1x1(x_pool)  # 将池化结果通过 1x1 卷积层
        x1 = self.gn(group_x * hw.sigmoid())  # 进行组归一化，并结合激活结果
        x2 = self.conv1x3(group_x)  # 通过 1x3 卷积层
        x11 = self.softmax(self.agp(x1).reshape(b * self.groups, -1, 1).permute(0, 2, 1))  # 对 x1 进行池化、形状变换、并应用 softmax
        x12 = x2.reshape(b * self.groups, c // self.groups, -1)  # 将 x2 重新形状为 (b * 组数, c // 组数, 长度)
        x21 = self.softmax(self.agp(x2).reshape(b * self.groups, -1, 1).permute(0, 2, 1))  # 对 x2 进行池化、形状变换、并应用 softmax
        x22 = x1.reshape(b * self.groups, c // self.groups, -1)  # 将 x1 重新形状为 (b * 组数, c // 组数, 长度)
        weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(b * self.groups, 1, s)  # 计算权重
        return (group_x * weights.sigmoid()).reshape(b, c, s)  # 应用权重并将形状恢复为原始大小




# --------------------------------------------------------------------------------------------------
# ----------------------------------------- 1 Channel  ---------------------------------------------
# --------------------------------------------------------------------------------------------------

# input: x: [B, C, samples]
class SimAM(torch.nn.Module):
    def __init__(self, channels=None, out_channels=None, e_lambda=1e-4):
        super(SimAM, self).__init__()
        self.activaton = nn.Sigmoid()
        self.e_lambda = e_lambda

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += ('lambda=%f)' % self.e_lambda)
        return s

    def forward(self, x):
        b, c, s = x.size()
        n = s - 1
        x_minus_mu_square = (x - x.mean(dim=2, keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=2, keepdim=True) / n + self.e_lambda)) + 0.5

        return x * self.activaton(y)


class SELayer_1D(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer_1D, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()  # b为batch size，c为通道数，_是长度
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)



# 1 channel, simple concat                     output: [320, 2, 121]
class Feature_1Chan_Concat(nn.Module):
    def __init__(self, hidden_size=128, seq_len=20, is_bidirectional=False, network="LSTM"):
        super(Feature_1Chan_Concat, self).__init__()
        # Branch 1
        self.block1 = nn.Sequential(  # [320, 1, 3000] -->[320, 100, 124]
            nn.Conv1d(in_channels=1, out_channels=100, kernel_size=20, stride=4, padding=0),
            # nn.BatchNorm1d(100, affine=True),
            nn.ReLU(True),
            nn.MaxPool1d(6, stride=6),
            nn.Conv1d(in_channels=100, out_channels=50, kernel_size=8, stride=1, padding=0),
            nn.Dropout(p=0.2),
            #nn.BatchNorm1d(50, affine=True),
            nn.ReLU(True),
            nn.MaxPool1d(kernel_size=4, stride=3),
        )
        self.att = SimAM()
        self.rnn1 = nn.LSTM(input_size=1900, hidden_size=512, num_layers=1, batch_first=True, bidirectional=False)

        # Branch 2
        self.block3 = nn.Sequential(  # [320, 1, 3000] -->[320, 100, 599]
            nn.Conv1d(in_channels=1, out_channels=100, kernel_size=5, stride=1, padding=0),
            # nn.BatchNorm1d(100, affine=True),
            nn.ReLU(True),
            nn.MaxPool1d(5, stride=5),
            nn.Conv1d(in_channels=100, out_channels=50, kernel_size=3, stride=1, padding=0),
            nn.Dropout(p=0.2),
            #nn.BatchNorm1d(50, affine=True),
            nn.ReLU(True),
            nn.MaxPool1d(kernel_size=5, stride=5),
        )
        self.rnn2 = nn.LSTM(input_size=5950, hidden_size=512, num_layers=1, batch_first=True, bidirectional=False)

        # After Fusion
        self.block5 = nn.Sequential(      # [320, 2, 512] --> [320, 8, 252]
            nn.Conv1d(in_channels=2, out_channels=16, kernel_size=5, stride=2, padding=0),
            #nn.Dropout(p=0.2),
            nn.BatchNorm1d(16, affine=True),
            nn.ReLU(True),
            nn.Conv1d(in_channels=16, out_channels=8, kernel_size=3, stride=1, padding=0),
            # nn.Dropout(p=0.2),
            nn.BatchNorm1d(8, affine=True),
            nn.ReLU(True),
        )
        self.rnn3 = nn.LSTM(input_size=2016, hidden_size=512, num_layers=1, batch_first=True, bidirectional=False)

        self.fc = nn.Linear(1024, 5)
        self.rnn_dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        x = x.reshape(-1, 1, 3000)       # [320, 1, 3000]
        # Branch 1
        x1 = self.block1(x)              # [320, 1, 3000] -->[320, 100, 124]
        x1 = x1.reshape(-1, 20, x1.shape[-1] * x1.shape[-2])  # [16, 20, 1900]
        w1 = self.att(x1)                 # [16, 20, 1900]
        att_x1 = torch.mul(x1, w1)       # [16, 20, 1900]
        x1, _ = self.rnn1(att_x1)        # [16, 20, 512]
        x1 = self.rnn_dropout(x1)
        x1_ = x1.reshape(-1, 1, x1.shape[-1])   # [320, 1, 512]

        # Branch 2
        x2 = self.block3(x)              # [320, 1, 3000] -->[320, 100, 599]
        x2 = x2.reshape(-1, 20, x2.shape[-1] * x2.shape[-2])  # [16, 20, 5950]
        w2 = self.att(x2)                # [16, 20, 5950]
        att_x2 = torch.mul(x2, w2)       # [16, 20, 5950]
        x2, _ = self.rnn2(att_x2)        # [16, 20, 512]
        x2 = self.rnn_dropout(x2)
        x2_ = x2.reshape(-1, 1, x2.shape[-1])   # [320, 1, 512]

        # Fuse Branch 1 and 2
        x12 = torch.cat((x1_, x2_), 1)     # [320, 2, 512]
        x12 = self.block5(x12)           # [320, 2, 512] --> [320, 8, 252]
        x12 = x12.reshape(-1, 20, x12.shape[-1] * x12.shape[-2])  # [16, 20, 2016]
        x12, _ = self.rnn3(x12)             # [16, 20, 512]
        x12 = self.rnn_dropout(x12)
        #x12 = x12.reshape(-1, 1, x12.shape[-1])  # [320, 1, 512]
        x_all = torch.cat((x1, x2), 2)
        x_all = torch.cat((x_all, x12), 2)  # [16, 20, 512*3]

        #x12 = self.rnn_dropout(x12)
        #w3 = self.att(x12)
        #x12 = torch.mul(x12, w3)     # [16, 20, 1024]

        return x_all


class MySleepNet_1Chan_Concat(nn.Module):
    def __init__(self, stages=5, hidden_size=128, seq_len=20, is_bidirectional=False, network="LSTM"):
        super(MySleepNet_1Chan_Concat, self).__init__()
        self.stages = stages
        self.Feature = Feature_1Chan_Concat()
        self.fc = nn.Linear(512*3, self.stages)
        self.output = nn.Softplus()

    def forward(self, x):
        x = self.Feature(x)   # [16, 20, 512*3]
        #print('********* 1 x.shape: ', x.shape)
        x = x.reshape(-1, x.shape[-1])  # [320, 1024]
        #print('********* 2 x.shape: ', x.shape)
        x = self.fc(x)
        x = self.output(x)
        return x




class Feature_1Chan_original(nn.Module):
    def __init__(self, hidden_size=128, seq_len=20, is_bidirectional=False, network="LSTM"):
        super(Feature_1Chan, self).__init__()
        self.flatten = nn.Flatten()
        self.simAM = SimAM()
        # Branch 1
        self.block1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=128, kernel_size=30, stride=4, padding=0),
            nn.BatchNorm1d(128, affine=True),
            nn.ReLU(True),
            nn.MaxPool1d(8, stride=8),
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=8, stride=1, padding=0),
            nn.BatchNorm1d(128, affine=True),
            nn.ReLU(True),
            nn.MaxPool1d(kernel_size=4, stride=4),
            nn.Dropout(p=0.2),
        )
        self.block2 = nn.Sequential(  # [320, 128, 21] -->[320, 32, 21]
            nn.Conv1d(in_channels=128, out_channels=32, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(32, affine=True),
            nn.ReLU(True),  # [320, 128, 743]
        )

        self.rnn1 = nn.LSTM(input_size=672, hidden_size=256, num_layers=1, batch_first=True, bidirectional=False)

        # Branch 1: cross attention
        self.qcross1 = nn.Sequential(nn.Linear(256, 128), nn.ReLU(inplace=True))
        self.kcross1 = nn.Sequential(nn.Linear(256, 128), nn.ReLU(inplace=True))
        self.vcross1 = nn.Sequential(nn.Linear(256, 128), nn.ReLU(inplace=True))

        # Branch 2
        self.block3 = nn.Sequential(  # [320, 1, 3000] -->[320, 128, 187]
            nn.Conv1d(in_channels=1, out_channels=128, kernel_size=8, stride=2, padding=0),
            nn.BatchNorm1d(128, affine=True),
            nn.ReLU(True),
            nn.MaxPool1d(8, stride=8),
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=8, stride=1, padding=0),
            nn.BatchNorm1d(128, affine=True),
            nn.ReLU(True),
            nn.MaxPool1d(kernel_size=4, stride=4),
            nn.Dropout(p=0.2),
        )
        self.block4 = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=32, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(32, affine=True),
            nn.ReLU(True),  # [320, 128, 743]
        )

        self.rnn2 = nn.LSTM(input_size=1440, hidden_size=256, num_layers=1, batch_first=True, bidirectional=False)

        # Branch 2: cross attention
        self.qcross2 = nn.Sequential(nn.Linear(256, 128), nn.ReLU(inplace=True))
        self.kcross2 = nn.Sequential(nn.Linear(256, 128), nn.ReLU(inplace=True))
        self.vcross2 = nn.Sequential(nn.Linear(256, 128), nn.ReLU(inplace=True))

        # after concat                 [320, 2, 128] --> [320, 32, 120]
        self.block5 = nn.Sequential(
            nn.Conv1d(in_channels=2, out_channels=32, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm1d(32, affine=True),
            nn.ReLU(True),
            nn.Conv1d(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm1d(32, affine=True),
            nn.ReLU(True),
            nn.Dropout(p=0.15),
        )

        self.block6 = nn.Sequential(   # [320, 128, 21] --> [320, 32, 17]
            nn.Conv1d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm1d(64, affine=True),
            nn.ReLU(True),
            nn.Conv1d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm1d(32, affine=True),
            nn.ReLU(True),
        )

        self.block7 = nn.Sequential(  # [320, 128, 45] --> [320, 32, 41]
            nn.Conv1d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm1d(64, affine=True),
            nn.ReLU(True),
            nn.Conv1d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm1d(32, affine=True),
            nn.ReLU(True),
            nn.Dropout(p=0.15),
        )

        self.block8 = nn.Sequential(  # [320, 32, 178] --> [320, 6, 54]
            nn.Conv1d(in_channels=32, out_channels=16, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm1d(16, affine=True),
            nn.ReLU(True),                     # [320, 16, 174]
            nn.MaxPool1d(kernel_size=3, stride=3),
            nn.Conv1d(in_channels=16, out_channels=6, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm1d(6, affine=True),    # [320, 6, 170]
            nn.ReLU(True),
            nn.Dropout(p=0.15),
            #nn.Flatten(),
        )

        self.rnn_dropout = nn.Dropout(p=0.2)
        self.fc_dropout = nn.Dropout(p=0.2)
        self.cross_attn1 = nn.Linear(128 + 128, 128)
        self.cross_attn2 = nn.Linear(128 + 128, 128)
        # self.fc = nn.Linear(400, self.stages)
        # self.output = nn.Softplus()

    def forward(self, x):
        x = x.reshape(-1, 1, 3000)   # [320, 1, 3000]
        # Branch 1
        x1 = self.block1(x)          # [320, 128, 21]

        x1_bra = self.block2(x1)     # [320, 32, 21]
        x1_bra = self.flatten(x1_bra)  # [320， 672]
        x1_bra = x1_bra.reshape(-1, 20, x1_bra.shape[-1])  # [16, 20, 672]
        x1_bra, _ = self.rnn1(x1_bra)         # [16, 20, 256]
        x1_bra = self.rnn_dropout(x1_bra)
        x1_bra = x1_bra.reshape(-1, x1_bra.shape[-1])    # [320, 256]
        #x1_ori = self.block6(x1_ori)                    # [320, 4, 1010]

        # Branch 2
        x2 = self.block3(x)          # [320, 128, 45]

        x2_bra = self.block4(x2)     # [320, 32, 45]
        x2_bra = self.flatten(x2_bra)        # [320， 1440]
        x2_bra = x2_bra.reshape(-1, 20, x2_bra.shape[-1])    # [16, 20， 1440]
        x2_bra, _ = self.rnn2(x2_bra)        # [16, 20, 256]
        x2_bra = self.rnn_dropout(x2_bra)
        x2_bra = x2_bra.reshape(-1, x2_bra.shape[-1])    # [320, 256]
        # x2_ori = self.block7(x2_ori)             # [320, 4, 1010]

        # cross attention
        cro_q1, cro_k1, cro_v1 = self.qcross1(x1_bra), self.kcross1(x1_bra), self.vcross1(x1_bra)
        cro_q2, cro_k2, cro_v2 = self.qcross2(x2_bra), self.kcross2(x2_bra), self.vcross2(x2_bra)

        q2_k1 = F.softmax(self.cross_attn2(torch.cat((cro_q2, cro_k1), 1)), dim=1)
        q2_kv1 = torch.mul(q2_k1, cro_v1)         # [320, 128]
        q2_kv1_ = torch.unsqueeze(q2_kv1, dim=1)  # [320, 1, 128]

        q1_k2 = F.softmax(self.cross_attn1(torch.cat((cro_q1, cro_k2), 1)), dim=1)
        q1_kv2 = torch.mul(q1_k2, cro_v2)         # [320, 128]
        q1_kv2_ = torch.unsqueeze(q1_kv2, dim=1)  # [320, 1, 128]

        x3 = torch.cat((q2_kv1_, q1_kv2_), 1)      # [320,  2, 128]
        x3 = self.block5(x3)                       # [320, 16, 120]

        # x1: [320, 128, 21]    x2: [320, 128, 45]    x3: [320, 32, 120]
        x1 = self.block6(x1)              # [320, 128, 21] --> [320, 32, 17]
        x2 = self.block7(x2)              # [320, 128, 45] --> [320, 32, 41]
        x12 = torch.cat((x1, x2), 2)               # [320, 32, 58]
        x123 = torch.cat((x12, x3), 2)             # [320, 32, 178]

        x_feature = self.block8(x123)              # [320, 32, 178] --> [320, 6, 54]
        # x_feature = self.simAM(x_feature)

        return x_feature


# ***************** 当前最好的 *******************
class _Feature_1Chan(nn.Module):
    def __init__(self, hidden_size=128, seq_len=20, is_bidirectional=False, network="LSTM"):
        super(Feature_1Chan, self).__init__()
        self.flatten = nn.Flatten()
        self.simAM = SimAM()
        self.se1 = SELayer_1D(128)
        self.se2 = SELayer_1D(128)
        # Branch 1
        self.block1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=128, kernel_size=30, stride=4, padding=0),
            nn.BatchNorm1d(128, affine=True),
            nn.ReLU(True),
            nn.MaxPool1d(8, stride=8),
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=8, stride=1, padding=0),
            nn.BatchNorm1d(128, affine=True),
            nn.ReLU(True),
            nn.MaxPool1d(kernel_size=4, stride=4),
            nn.Dropout(p=0.2),
        )
        self.block2 = nn.Sequential(  # [320, 128, 21] -->[320, 32, 21]
            nn.Conv1d(in_channels=128, out_channels=32, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(32, affine=True),
            nn.ReLU(True),  # [320, 128, 743]
        )

        self.rnn1 = nn.LSTM(input_size=672, hidden_size=256, num_layers=3, batch_first=True, bidirectional=True)

        # Branch 1: cross attention
        self.qcross1 = nn.Sequential(nn.Linear(512, 128), nn.ReLU(inplace=True))
        self.kcross1 = nn.Sequential(nn.Linear(512, 128), nn.ReLU(inplace=True))
        self.vcross1 = nn.Sequential(nn.Linear(512, 128), nn.ReLU(inplace=True))

        # Branch 2
        self.block3 = nn.Sequential(  # [320, 1, 3000] -->[320, 128, 187]
            nn.Conv1d(in_channels=1, out_channels=128, kernel_size=8, stride=2, padding=0),
            nn.BatchNorm1d(128, affine=True),
            nn.ReLU(True),
            nn.MaxPool1d(8, stride=8),
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=8, stride=1, padding=0),
            nn.BatchNorm1d(128, affine=True),
            nn.ReLU(True),
            nn.MaxPool1d(kernel_size=4, stride=4),
            nn.Dropout(p=0.2),
        )
        self.block4 = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=32, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(32, affine=True),
            nn.ReLU(True),  # [320, 128, 743]
        )

        self.rnn2 = nn.LSTM(input_size=1440, hidden_size=256, num_layers=3, batch_first=True, bidirectional=True)

        # Branch 2: cross attention
        self.qcross2 = nn.Sequential(nn.Linear(512, 128), nn.ReLU(inplace=True))
        self.kcross2 = nn.Sequential(nn.Linear(512, 128), nn.ReLU(inplace=True))
        self.vcross2 = nn.Sequential(nn.Linear(512, 128), nn.ReLU(inplace=True))

        # after concat                 [320, 2, 128] --> [320, 2, 120]      和[320, 2, 2048]合并
        self.block5_0 = nn.Sequential(
            nn.Conv1d(in_channels=2, out_channels=6, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm1d(6, affine=True),
            nn.ReLU(True),
        )
        self.block5_1 = nn.Sequential(
            nn.Conv1d(in_channels=6, out_channels=2, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm1d(2, affine=True),
            nn.ReLU(True),
            nn.Dropout(p=0.15),
        )

        self.block6 = nn.Sequential(   # [320, 128, 21] --> [320, 32, 17]
            nn.Conv1d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm1d(64, affine=True),
            nn.ReLU(True),
            nn.Conv1d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm1d(32, affine=True),
            nn.ReLU(True),
        )

        self.block7 = nn.Sequential(  # [320, 128, 45] --> [320, 32, 41]
            nn.Conv1d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm1d(64, affine=True),
            nn.ReLU(True),
            nn.Conv1d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm1d(32, affine=True),
            nn.ReLU(True),
            nn.Dropout(p=0.15),
        )

        self.block8 = nn.Sequential(  # [320, 2, 2168] --> [320, 6, 537]
            nn.Conv1d(in_channels=2, out_channels=16, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm1d(16, affine=True),
            nn.ReLU(True),
            nn.MaxPool1d(kernel_size=4, stride=4),
            nn.Conv1d(in_channels=16, out_channels=6, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm1d(6, affine=True),
            nn.ReLU(True),
            nn.Dropout(p=0.15),
            #nn.Flatten(),
        )

        self.rnn_dropout = nn.Dropout(p=0.2)
        self.fc_dropout = nn.Dropout(p=0.2)
        self.cross_attn1 = nn.Linear(128 + 128, 128)
        self.cross_attn2 = nn.Linear(128 + 128, 128)
        # self.fc = nn.Linear(400, self.stages)
        # self.output = nn.Softplus()
        self.rnn3 = nn.LSTM(input_size=544, hidden_size=1024, num_layers=3, batch_first=True, bidirectional=True)
        self.rnn4 = nn.LSTM(input_size=1312, hidden_size=1024, num_layers=3, batch_first=True, bidirectional=True)

    def forward(self, x):
        x = x.reshape(-1, 1, 3000)   # [320, 1, 3000]
        # Branch 1
        x1 = self.block1(x)          # [320, 128, 21]
        x1 = self.se1(x1)

        x1_bra = self.block2(x1)     # [320, 32, 21]
        x1_bra = self.flatten(x1_bra)  # [320， 672]
        x1_bra = x1_bra.reshape(-1, 20, x1_bra.shape[-1])  # [16, 20, 672]
        x1_bra, _ = self.rnn1(x1_bra)         # [16, 20, 512]
        x1_bra = self.rnn_dropout(x1_bra)
        x1_bra = x1_bra.reshape(-1, x1_bra.shape[-1])    # [320, 512]

        # Branch 2
        x2 = self.block3(x)          # [320, 128, 45]
        x2 = self.se2(x2)

        x2_bra = self.block4(x2)     # [320, 32, 45]
        x2_bra = self.flatten(x2_bra)        # [320， 1440]
        x2_bra = x2_bra.reshape(-1, 20, x2_bra.shape[-1])    # [16, 20， 1440]
        x2_bra, _ = self.rnn2(x2_bra)        # [16, 20, 512]
        x2_bra = self.rnn_dropout(x2_bra)
        x2_bra = x2_bra.reshape(-1, x2_bra.shape[-1])    # [320, 512]

        # cross attention
        cro_q1, cro_k1, cro_v1 = self.qcross1(x1_bra), self.kcross1(x1_bra), self.vcross1(x1_bra)
        cro_q2, cro_k2, cro_v2 = self.qcross2(x2_bra), self.kcross2(x2_bra), self.vcross2(x2_bra)

        q2_k1 = F.softmax(self.cross_attn2(torch.cat((cro_q2, cro_k1), 1)), dim=1)
        q2_kv1 = torch.mul(q2_k1, cro_v1)         # [320, 128]
        q2_kv1_ = torch.unsqueeze(q2_kv1, dim=1)  # [320, 1, 128]

        q1_k2 = F.softmax(self.cross_attn1(torch.cat((cro_q1, cro_k2), 1)), dim=1)
        q1_kv2 = torch.mul(q1_k2, cro_v2)         # [320, 128]
        q1_kv2_ = torch.unsqueeze(q1_kv2, dim=1)  # [320, 1, 128]

        x3 = torch.cat((q2_kv1_, q1_kv2_), 1)      # [320,  2, 128]
        x3 = self.block5_0(x3)
        x3 = self.block5_1(x3)                     # [320, 2, 120]

        # x1: [320, 128, 21]    x2: [320, 128, 45]    x3: [320, 32, 120]
        x1 = self.block6(x1)              # [320, 128, 21] --> [320, 32, 17]
        x2 = self.block7(x2)              # [320, 128, 45] --> [320, 32, 41]
        x1 = self.flatten(x1)             # [320, 544]
        x2 = self.flatten(x2)             # [320, 1312]
        x1 = x1.reshape(-1, 20, x1.shape[-1])      # [16, 20, 544]
        x2 = x2.reshape(-1, 20, x2.shape[-1])      # [16, 20, 1312]
        x1, _ = self.rnn3(x1)                      # [16, 20, 2048]
        x2, _ = self.rnn4(x2)                      # [16, 20, 2048]
        x1 = x1.reshape(-1, 1, x1.shape[-1])       # [320, 1, 2048]
        x2 = x2.reshape(-1, 1, x2.shape[-1])       # [320, 1, 2048]


        x12 = torch.cat((x1, x2), 1)               # [320, 2, 2048]
        x123 = torch.cat((x12, x3), 2)             # [320, 2, 2168]

        x_feature = self.block8(x123)              # [320, 2, 2168] --> [320, 6, 537]
        # x_feature = self.simAM(x_feature)

        return x_feature


class _MySleepNet_1Chan(nn.Module):
    def __init__(self, stages=5, hidden_size=128, seq_len=20, is_bidirectional=False, network="LSTM"):
        super(MySleepNet_1Chan, self).__init__()
        self.stages = stages
        self.Feature = Feature_1Chan()
        self.fc = nn.Linear(536, self.stages)
        self.output = nn.Softplus()

        self.block = nn.Sequential(  # [320, 6, 537]
            nn.Conv1d(in_channels=6, out_channels=2, kernel_size=3, stride=2, padding=0),
            nn.BatchNorm1d(2, affine=True),
            nn.ReLU(True),
            nn.Flatten(),
        )

    def forward(self, x):
        x = self.Feature(x)        # [320, 6, 537]
        x = self.block(x)          # [320, 536]
        x = self.fc(x)
        x = self.output(x)
        return x


class _MySleepNet_2Chan(nn.Module):
    def __init__(self, stages=3, hidden_size=128, seq_len=20, is_bidirectional=False, network="LSTM"):
        super(MySleepNet_2Chan, self).__init__()
        self.stages = stages
        self.Feature1 = Feature_1Chan()
        self.Feature2 = Feature_1Chan()
        self.fc = nn.Linear(100, self.stages)
        self.output = nn.Softplus()

        self.block = nn.Sequential(  # [320, 12, 54] --> [320, 2, 50]
            nn.Conv1d(in_channels=12, out_channels=8, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm1d(8, affine=True),
            nn.ReLU(True),
            nn.Conv1d(in_channels=8, out_channels=2, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm1d(2, affine=True),
            nn.ReLU(True),
            nn.Flatten(),
        )

    def forward(self, x):
        x1 = x[:, 0, :, :]               # channel 1: [16, 20, 3000]
        x1 = x1.reshape(-1, 1, 3000)     # [320, 1, 3000]
        x2 = x[:, 1, :, :]
        x2 = x2.reshape(-1, 1, 3000)

        x1 = self.Feature1(x1)        # [320, 6, 54]
        x2 = self.Feature2(x2)
        x = torch.cat((x1, x2), 1)     # [320, 12, 54]
        x = self.block(x)          # [320, 104]
        x = self.fc(x)
        x = self.output(x)
        return x


# ***************** 继续提升 2024.6.25 -v0  *******************  2*SENet + SENet
class _Feature_1Chan(nn.Module):
    def __init__(self, hidden_size=128, seq_len=20, is_bidirectional=False, network="LSTM"):
        super(Feature_1Chan, self).__init__()
        self.flatten = nn.Flatten()
        self.simAM = SimAM()
        self.se1 = SELayer_1D(128)
        self.se2 = SELayer_1D(128)
        # Branch 1
        self.block1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=128, kernel_size=30, stride=4, padding=0),
            nn.BatchNorm1d(128, affine=True),
            nn.ReLU(True),
            nn.MaxPool1d(8, stride=8),
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=8, stride=1, padding=0),
            nn.BatchNorm1d(128, affine=True),
            nn.ReLU(True),
            nn.MaxPool1d(kernel_size=4, stride=4),
            nn.Dropout(p=0.2),
        )
        self.block2 = nn.Sequential(  # [320, 128, 21] -->[320, 32, 21]
            nn.Conv1d(in_channels=128, out_channels=32, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(32, affine=True),
            nn.ReLU(True),  # [320, 128, 743]
        )

        self.rnn1 = nn.LSTM(input_size=672, hidden_size=256, num_layers=1, batch_first=True, bidirectional=True)

        # Branch 1: cross attention
        self.qcross1 = nn.Sequential(nn.Linear(512, 128), nn.ReLU(inplace=True))
        self.kcross1 = nn.Sequential(nn.Linear(512, 128), nn.ReLU(inplace=True))
        self.vcross1 = nn.Sequential(nn.Linear(512, 128), nn.ReLU(inplace=True))

        # Branch 2
        self.block3 = nn.Sequential(  # [320, 1, 3000] -->[320, 128, 187]
            nn.Conv1d(in_channels=1, out_channels=128, kernel_size=8, stride=2, padding=0),
            nn.BatchNorm1d(128, affine=True),
            nn.ReLU(True),
            nn.MaxPool1d(8, stride=8),
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=8, stride=1, padding=0),
            nn.BatchNorm1d(128, affine=True),
            nn.ReLU(True),
            nn.MaxPool1d(kernel_size=4, stride=4),
            nn.Dropout(p=0.2),
        )
        self.block4 = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=32, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(32, affine=True),
            nn.ReLU(True),  # [320, 128, 743]
        )

        self.rnn2 = nn.LSTM(input_size=1440, hidden_size=256, num_layers=1, batch_first=True, bidirectional=True)

        # Branch 2: cross attention
        self.qcross2 = nn.Sequential(nn.Linear(512, 128), nn.ReLU(inplace=True))
        self.kcross2 = nn.Sequential(nn.Linear(512, 128), nn.ReLU(inplace=True))
        self.vcross2 = nn.Sequential(nn.Linear(512, 128), nn.ReLU(inplace=True))

        # after concat                 [320, 2, 128] --> [320, 32, 42]
        self.block5 = nn.Sequential(
            nn.Conv1d(in_channels=2, out_channels=32, kernel_size=5, stride=1, padding=1),
            nn.BatchNorm1d(32, affine=True),
            nn.ReLU(True),
            nn.MaxPool1d(kernel_size=3, stride=3),
            nn.Dropout(p=0.15),
        )

        self.block6 = nn.Sequential(   # [320, 128, 21] --> [320, 32, 17]
            nn.Conv1d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm1d(64, affine=True),
            nn.ReLU(True),
            nn.Conv1d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm1d(32, affine=True),
            nn.ReLU(True),
        )

        self.block7 = nn.Sequential(  # [320, 128, 45] --> [320, 32, 41]
            nn.Conv1d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm1d(64, affine=True),
            nn.ReLU(True),
            nn.Conv1d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm1d(32, affine=True),
            nn.ReLU(True),
            nn.Dropout(p=0.15),
        )

        self.block8 = nn.Sequential(  # [320, 32, 100] --> [320, 32, 33]
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(64, affine=True),
            nn.ReLU(True),
            nn.MaxPool1d(kernel_size=3, stride=3),
            nn.Conv1d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(32, affine=True),
            nn.ReLU(True),
            nn.Dropout(p=0.15),
            # nn.Flatten(),
        )

        self.rnn_dropout = nn.Dropout(p=0.2)
        self.fc_dropout = nn.Dropout(p=0.2)
        self.cross_attn1 = nn.Linear(128 + 128, 128)
        self.cross_attn2 = nn.Linear(128 + 128, 128)
        # self.fc = nn.Linear(400, self.stages)
        # self.output = nn.Softplus()
        # self.rnn3 = nn.LSTM(input_size=544, hidden_size=1024, num_layers=3, batch_first=True, bidirectional=True)
        # self.rnn4 = nn.LSTM(input_size=1312, hidden_size=1024, num_layers=3, batch_first=True, bidirectional=True)

    def forward(self, x):
        x = x.reshape(-1, 1, 3000)   # [320, 1, 3000]
        # Branch 1
        x1 = self.block1(x)          # [320, 128, 21]
        x1 = self.se1(x1)

        x1_bra = self.block2(x1)     # [320, 32, 21]
        x1_bra = self.flatten(x1_bra)  # [320， 672]
        x1_bra = x1_bra.reshape(-1, 20, x1_bra.shape[-1])  # [16, 20, 672]
        x1_bra, _ = self.rnn1(x1_bra)         # [16, 20, 512]
        x1_bra = self.rnn_dropout(x1_bra)
        x1_bra = x1_bra.reshape(-1, x1_bra.shape[-1])    # [320, 512]

        # Branch 2
        x2 = self.block3(x)          # [320, 128, 45]
        x2 = self.se2(x2)

        x2_bra = self.block4(x2)     # [320, 32, 45]
        x2_bra = self.flatten(x2_bra)        # [320， 1440]
        x2_bra = x2_bra.reshape(-1, 20, x2_bra.shape[-1])    # [16, 20， 1440]
        x2_bra, _ = self.rnn2(x2_bra)        # [16, 20, 512]
        x2_bra = self.rnn_dropout(x2_bra)
        x2_bra = x2_bra.reshape(-1, x2_bra.shape[-1])    # [320, 512]

        # cross attention
        cro_q1, cro_k1, cro_v1 = self.qcross1(x1_bra), self.kcross1(x1_bra), self.vcross1(x1_bra)
        cro_q2, cro_k2, cro_v2 = self.qcross2(x2_bra), self.kcross2(x2_bra), self.vcross2(x2_bra)

        q2_k1 = F.softmax(self.cross_attn2(torch.cat((cro_q2, cro_k1), 1)), dim=1)
        q2_kv1 = torch.mul(q2_k1, cro_v1)         # [320, 128]
        q2_kv1_ = torch.unsqueeze(q2_kv1, dim=1)  # [320, 1, 128]

        q1_k2 = F.softmax(self.cross_attn1(torch.cat((cro_q1, cro_k2), 1)), dim=1)
        q1_kv2 = torch.mul(q1_k2, cro_v2)         # [320, 128]
        q1_kv2_ = torch.unsqueeze(q1_kv2, dim=1)  # [320, 1, 128]

        x3 = torch.cat((q2_kv1_, q1_kv2_), 1)      # [320,  2, 128]
        x3 = self.block5(x3)                       # [320, 32, 42]

        # x1: [320, 128, 21]    x2: [320, 128, 45]    x3: [320, 32, 120]
        x1 = self.block6(x1)              # [320, 128, 21] --> [320, 32, 17]
        x2 = self.block7(x2)              # [320, 128, 45] --> [320, 32, 41]

        #x1 = self.flatten(x1)             # [320, 544]
        #x1 = x1.reshape(-1, 20, x1.shape[-1])      # [16, 20, 544]
        #x1, _ = self.rnn3(x1)                      # [16, 20, 2048]
        #x1 = x1.reshape(-1, 1, x1.shape[-1])       # [320, 1, 2048]


        x12 = torch.cat((x1, x2), 2)               # [320, 32, 58]
        x123 = torch.cat((x12, x3), 2)             # [320, 32, 100]

        x_feature = self.block8(x123)              # [320, 32, 100] --> [320, 32, 33]
        # x_feature = self.simAM(x_feature)

        return x_feature

class _MySleepNet_1Chan(nn.Module):
    def __init__(self, stages=5, hidden_size=128, seq_len=20, is_bidirectional=False, network="LSTM"):
        super(MySleepNet_1Chan, self).__init__()
        self.stages = stages
        self.flatten = nn.Flatten()
        self.Feature = Feature_1Chan()
        self.fc = nn.Linear(2000, self.stages)
        self.se = SELayer_1D(32)
        self.output = nn.Softplus()
        self.rnn = nn.LSTM(input_size=1056, hidden_size=1000, num_layers=1, batch_first=True, bidirectional=True)

        self.block = nn.Sequential(  # [320, 32, 33]
            nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(32, affine=True),
            nn.ReLU(True),
            nn.Flatten(),
        )

    def forward(self, x):
        x = self.Feature(x)        # [320, 32, 33]
        x = self.se(x)
        x = self.block(x)          # [320, 1056]
        x = x.reshape(-1, 20, x.shape[-1])      # [16, 20, 1056]
        x, _ = self.rnn(x)                      # [16, 20, 2000]
        x = x.reshape(-1, x.shape[-1])          # [320, 2000]

        x = self.fc(x)
        x = self.output(x)
        return x


# ***************** 继续提升 2024.6.26 -v1  ******************* 2*SENet + GCNet
class _Feature_1Chan(nn.Module):
    def __init__(self, hidden_size=128, seq_len=20, is_bidirectional=False, network="LSTM"):
        super(Feature_1Chan, self).__init__()
        self.flatten = nn.Flatten()
        self.simAM = SimAM()
        self.se1 = SELayer_1D(128)
        self.se2 = SELayer_1D(128)
        # Branch 1
        self.block1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=128, kernel_size=30, stride=4, padding=0),
            nn.BatchNorm1d(128, affine=True),
            nn.ReLU(True),
            nn.MaxPool1d(8, stride=8),
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=8, stride=1, padding=0),
            nn.BatchNorm1d(128, affine=True),
            nn.ReLU(True),
            nn.MaxPool1d(kernel_size=4, stride=4),
            nn.Dropout(p=0.2),
        )
        self.block2 = nn.Sequential(  # [320, 128, 21] -->[320, 32, 21]
            nn.Conv1d(in_channels=128, out_channels=32, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(32, affine=True),
            nn.ReLU(True),  # [320, 128, 743]
        )

        self.rnn1 = nn.LSTM(input_size=672, hidden_size=256, num_layers=1, batch_first=True, bidirectional=True)

        # Branch 1: cross attention
        self.qcross1 = nn.Sequential(nn.Linear(512, 128), nn.ReLU(inplace=True))
        self.kcross1 = nn.Sequential(nn.Linear(512, 128), nn.ReLU(inplace=True))
        self.vcross1 = nn.Sequential(nn.Linear(512, 128), nn.ReLU(inplace=True))

        # Branch 2
        self.block3 = nn.Sequential(  # [320, 1, 3000] -->[320, 128, 187]
            nn.Conv1d(in_channels=1, out_channels=128, kernel_size=8, stride=2, padding=0),
            nn.BatchNorm1d(128, affine=True),
            nn.ReLU(True),
            nn.MaxPool1d(8, stride=8),
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=8, stride=1, padding=0),
            nn.BatchNorm1d(128, affine=True),
            nn.ReLU(True),
            nn.MaxPool1d(kernel_size=4, stride=4),
            nn.Dropout(p=0.2),
        )
        self.block4 = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=32, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(32, affine=True),
            nn.ReLU(True),  # [320, 128, 743]
        )

        self.rnn2 = nn.LSTM(input_size=1440, hidden_size=256, num_layers=1, batch_first=True, bidirectional=True)

        # Branch 2: cross attention
        self.qcross2 = nn.Sequential(nn.Linear(512, 128), nn.ReLU(inplace=True))
        self.kcross2 = nn.Sequential(nn.Linear(512, 128), nn.ReLU(inplace=True))
        self.vcross2 = nn.Sequential(nn.Linear(512, 128), nn.ReLU(inplace=True))

        # after concat                 [320, 2, 128] --> [320, 32, 42]
        self.block5 = nn.Sequential(
            nn.Conv1d(in_channels=2, out_channels=32, kernel_size=5, stride=1, padding=1),
            nn.BatchNorm1d(32, affine=True),
            nn.ReLU(True),
            nn.MaxPool1d(kernel_size=3, stride=3),
            nn.Dropout(p=0.15),
        )

        self.block6 = nn.Sequential(   # [320, 128, 21] --> [320, 32, 17]
            nn.Conv1d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm1d(64, affine=True),
            nn.ReLU(True),
            nn.Conv1d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm1d(32, affine=True),
            nn.ReLU(True),
        )

        self.block7 = nn.Sequential(  # [320, 128, 45] --> [320, 32, 41]
            nn.Conv1d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm1d(64, affine=True),
            nn.ReLU(True),
            nn.Conv1d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm1d(32, affine=True),
            nn.ReLU(True),
            nn.Dropout(p=0.15),
        )

        self.block8 = nn.Sequential(  # [320, 32, 100] --> [320, 32, 33]
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(64, affine=True),
            nn.ReLU(True),
            nn.MaxPool1d(kernel_size=3, stride=3),
            nn.Conv1d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(32, affine=True),
            nn.ReLU(True),
            nn.Dropout(p=0.15),
            # nn.Flatten(),
        )

        self.rnn_dropout = nn.Dropout(p=0.2)
        self.fc_dropout = nn.Dropout(p=0.2)
        self.cross_attn1 = nn.Linear(128 + 128, 128)
        self.cross_attn2 = nn.Linear(128 + 128, 128)
        # self.fc = nn.Linear(400, self.stages)
        # self.output = nn.Softplus()
        # self.rnn3 = nn.LSTM(input_size=544, hidden_size=1024, num_layers=3, batch_first=True, bidirectional=True)
        # self.rnn4 = nn.LSTM(input_size=1312, hidden_size=1024, num_layers=3, batch_first=True, bidirectional=True)

    def forward(self, x):
        x = x.reshape(-1, 1, 3000)   # [320, 1, 3000]
        # Branch 1
        x1 = self.block1(x)          # [320, 128, 21]
        x1 = self.se1(x1)

        x1_bra = self.block2(x1)     # [320, 32, 21]
        x1_bra = self.flatten(x1_bra)  # [320， 672]
        x1_bra = x1_bra.reshape(-1, 20, x1_bra.shape[-1])  # [16, 20, 672]
        x1_bra, _ = self.rnn1(x1_bra)         # [16, 20, 512]
        x1_bra = self.rnn_dropout(x1_bra)
        x1_bra = x1_bra.reshape(-1, x1_bra.shape[-1])    # [320, 512]

        # Branch 2
        x2 = self.block3(x)          # [320, 128, 45]
        x2 = self.se2(x2)

        x2_bra = self.block4(x2)     # [320, 32, 45]
        x2_bra = self.flatten(x2_bra)        # [320， 1440]
        x2_bra = x2_bra.reshape(-1, 20, x2_bra.shape[-1])    # [16, 20， 1440]
        x2_bra, _ = self.rnn2(x2_bra)        # [16, 20, 512]
        x2_bra = self.rnn_dropout(x2_bra)
        x2_bra = x2_bra.reshape(-1, x2_bra.shape[-1])    # [320, 512]

        # cross attention
        cro_q1, cro_k1, cro_v1 = self.qcross1(x1_bra), self.kcross1(x1_bra), self.vcross1(x1_bra)
        cro_q2, cro_k2, cro_v2 = self.qcross2(x2_bra), self.kcross2(x2_bra), self.vcross2(x2_bra)

        q2_k1 = F.softmax(self.cross_attn2(torch.cat((cro_q2, cro_k1), 1)), dim=1)
        q2_kv1 = torch.mul(q2_k1, cro_v1)         # [320, 128]
        q2_kv1_ = torch.unsqueeze(q2_kv1, dim=1)  # [320, 1, 128]

        q1_k2 = F.softmax(self.cross_attn1(torch.cat((cro_q1, cro_k2), 1)), dim=1)
        q1_kv2 = torch.mul(q1_k2, cro_v2)         # [320, 128]
        q1_kv2_ = torch.unsqueeze(q1_kv2, dim=1)  # [320, 1, 128]

        x3 = torch.cat((q2_kv1_, q1_kv2_), 1)      # [320,  2, 128]
        x3 = self.block5(x3)                       # [320, 32, 42]

        # x1: [320, 128, 21]    x2: [320, 128, 45]    x3: [320, 32, 120]
        x1 = self.block6(x1)              # [320, 128, 21] --> [320, 32, 17]
        x2 = self.block7(x2)              # [320, 128, 45] --> [320, 32, 41]

        #x1 = self.flatten(x1)             # [320, 544]
        #x1 = x1.reshape(-1, 20, x1.shape[-1])      # [16, 20, 544]
        #x1, _ = self.rnn3(x1)                      # [16, 20, 2048]
        #x1 = x1.reshape(-1, 1, x1.shape[-1])       # [320, 1, 2048]


        x12 = torch.cat((x1, x2), 2)               # [320, 32, 58]
        x123 = torch.cat((x12, x3), 2)             # [320, 32, 100]

        x_feature = self.block8(x123)              # [320, 32, 100] --> [320, 32, 33]
        # x_feature = self.simAM(x_feature)

        return x_feature

class _MySleepNet_1Chan(nn.Module):
    def __init__(self, stages=5, hidden_size=128, seq_len=20, is_bidirectional=False, network="LSTM"):
        super(MySleepNet_1Chan, self).__init__()
        self.stages = stages
        self.flatten = nn.Flatten()
        self.Feature = Feature_1Chan()
        self.fc = nn.Linear(2000, self.stages)
        self.se = SELayer_1D(32)
        self.GCNet = GlobalContextBlock(32)
        self.output = nn.Softplus()
        self.rnn = nn.LSTM(input_size=1056, hidden_size=1000, num_layers=1, batch_first=True, bidirectional=True)

        self.block = nn.Sequential(  # [320, 32, 33]
            nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(32, affine=True),
            nn.ReLU(True),
            nn.Flatten(),
        )

    def forward(self, x):
        x = self.Feature(x)        # [320, 32, 33]
        # x = self.se(x)
        x = self.GCNet(x)
        x = self.block(x)          # [320, 1056]
        x = x.reshape(-1, 20, x.shape[-1])      # [16, 20, 1056]
        x, _ = self.rnn(x)                      # [16, 20, 2000]
        x = x.reshape(-1, x.shape[-1])          # [320, 2000]

        x = self.fc(x)
        x = self.output(x)
        return x


# ***************** 继续提升 2024.6.26 -v2  ******************* 2*GCNet + GCNet
class _Feature_1Chan(nn.Module):
    def __init__(self, hidden_size=128, seq_len=20, is_bidirectional=False, network="LSTM"):
        super(Feature_1Chan, self).__init__()
        self.flatten = nn.Flatten()
        self.simAM = SimAM()
        #self.se1 = SELayer_1D(128)
        #self.se2 = SELayer_1D(128)
        self.GCNet1 = GlobalContextBlock(128)
        self.GCNet2 = GlobalContextBlock(128)
        # Branch 1
        self.block1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=128, kernel_size=30, stride=4, padding=0),
            nn.BatchNorm1d(128, affine=True),
            nn.ReLU(True),
            nn.MaxPool1d(8, stride=8),
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=8, stride=1, padding=0),
            nn.BatchNorm1d(128, affine=True),
            nn.ReLU(True),
            nn.MaxPool1d(kernel_size=4, stride=4),
            nn.Dropout(p=0.2),
        )
        self.block2 = nn.Sequential(  # [320, 128, 21] -->[320, 32, 21]
            nn.Conv1d(in_channels=128, out_channels=32, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(32, affine=True),
            nn.ReLU(True),  # [320, 128, 743]
        )

        self.rnn1 = nn.LSTM(input_size=672, hidden_size=256, num_layers=1, batch_first=True, bidirectional=True)

        # Branch 1: cross attention
        self.qcross1 = nn.Sequential(nn.Linear(512, 128), nn.ReLU(inplace=True))
        self.kcross1 = nn.Sequential(nn.Linear(512, 128), nn.ReLU(inplace=True))
        self.vcross1 = nn.Sequential(nn.Linear(512, 128), nn.ReLU(inplace=True))

        # Branch 2
        self.block3 = nn.Sequential(  # [320, 1, 3000] -->[320, 128, 187]
            nn.Conv1d(in_channels=1, out_channels=128, kernel_size=8, stride=2, padding=0),
            nn.BatchNorm1d(128, affine=True),
            nn.ReLU(True),
            nn.MaxPool1d(8, stride=8),
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=8, stride=1, padding=0),
            nn.BatchNorm1d(128, affine=True),
            nn.ReLU(True),
            nn.MaxPool1d(kernel_size=4, stride=4),
            nn.Dropout(p=0.2),
        )
        self.block4 = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=32, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(32, affine=True),
            nn.ReLU(True),  # [320, 128, 743]
        )

        self.rnn2 = nn.LSTM(input_size=1440, hidden_size=256, num_layers=1, batch_first=True, bidirectional=True)

        # Branch 2: cross attention
        self.qcross2 = nn.Sequential(nn.Linear(512, 128), nn.ReLU(inplace=True))
        self.kcross2 = nn.Sequential(nn.Linear(512, 128), nn.ReLU(inplace=True))
        self.vcross2 = nn.Sequential(nn.Linear(512, 128), nn.ReLU(inplace=True))

        # after concat                 [320, 2, 128] --> [320, 32, 42]
        self.block5 = nn.Sequential(
            nn.Conv1d(in_channels=2, out_channels=32, kernel_size=5, stride=1, padding=1),
            nn.BatchNorm1d(32, affine=True),
            nn.ReLU(True),
            nn.MaxPool1d(kernel_size=3, stride=3),
            nn.Dropout(p=0.15),
        )

        self.block6 = nn.Sequential(   # [320, 128, 21] --> [320, 32, 17]
            nn.Conv1d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm1d(64, affine=True),
            nn.ReLU(True),
            nn.Conv1d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm1d(32, affine=True),
            nn.ReLU(True),
        )

        self.block7 = nn.Sequential(  # [320, 128, 45] --> [320, 32, 41]
            nn.Conv1d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm1d(64, affine=True),
            nn.ReLU(True),
            nn.Conv1d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm1d(32, affine=True),
            nn.ReLU(True),
            nn.Dropout(p=0.15),
        )

        self.block8 = nn.Sequential(  # [320, 32, 100] --> [320, 32, 33]
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(64, affine=True),
            nn.ReLU(True),
            nn.MaxPool1d(kernel_size=3, stride=3),
            nn.Conv1d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(32, affine=True),
            nn.ReLU(True),
            nn.Dropout(p=0.15),
            # nn.Flatten(),
        )

        self.rnn_dropout = nn.Dropout(p=0.2)
        self.fc_dropout = nn.Dropout(p=0.2)
        self.cross_attn1 = nn.Linear(128 + 128, 128)
        self.cross_attn2 = nn.Linear(128 + 128, 128)
        # self.fc = nn.Linear(400, self.stages)
        # self.output = nn.Softplus()
        # self.rnn3 = nn.LSTM(input_size=544, hidden_size=1024, num_layers=3, batch_first=True, bidirectional=True)
        # self.rnn4 = nn.LSTM(input_size=1312, hidden_size=1024, num_layers=3, batch_first=True, bidirectional=True)

    def forward(self, x):
        x = x.reshape(-1, 1, 3000)   # [320, 1, 3000]
        # Branch 1
        x1 = self.block1(x)          # [320, 128, 21]
        x1 = self.GCNet1(x1)

        x1_bra = self.block2(x1)     # [320, 32, 21]
        x1_bra = self.flatten(x1_bra)  # [320， 672]
        x1_bra = x1_bra.reshape(-1, 20, x1_bra.shape[-1])  # [16, 20, 672]
        x1_bra, _ = self.rnn1(x1_bra)         # [16, 20, 512]
        x1_bra = self.rnn_dropout(x1_bra)
        x1_bra = x1_bra.reshape(-1, x1_bra.shape[-1])    # [320, 512]

        # Branch 2
        x2 = self.block3(x)          # [320, 128, 45]
        x2 = self.GCNet2(x2)

        x2_bra = self.block4(x2)     # [320, 32, 45]
        x2_bra = self.flatten(x2_bra)        # [320， 1440]
        x2_bra = x2_bra.reshape(-1, 20, x2_bra.shape[-1])    # [16, 20， 1440]
        x2_bra, _ = self.rnn2(x2_bra)        # [16, 20, 512]
        x2_bra = self.rnn_dropout(x2_bra)
        x2_bra = x2_bra.reshape(-1, x2_bra.shape[-1])    # [320, 512]

        # cross attention
        cro_q1, cro_k1, cro_v1 = self.qcross1(x1_bra), self.kcross1(x1_bra), self.vcross1(x1_bra)
        cro_q2, cro_k2, cro_v2 = self.qcross2(x2_bra), self.kcross2(x2_bra), self.vcross2(x2_bra)

        q2_k1 = F.softmax(self.cross_attn2(torch.cat((cro_q2, cro_k1), 1)), dim=1)
        q2_kv1 = torch.mul(q2_k1, cro_v1)         # [320, 128]
        q2_kv1_ = torch.unsqueeze(q2_kv1, dim=1)  # [320, 1, 128]

        q1_k2 = F.softmax(self.cross_attn1(torch.cat((cro_q1, cro_k2), 1)), dim=1)
        q1_kv2 = torch.mul(q1_k2, cro_v2)         # [320, 128]
        q1_kv2_ = torch.unsqueeze(q1_kv2, dim=1)  # [320, 1, 128]

        x3 = torch.cat((q2_kv1_, q1_kv2_), 1)      # [320,  2, 128]
        x3 = self.block5(x3)                       # [320, 32, 42]

        # x1: [320, 128, 21]    x2: [320, 128, 45]    x3: [320, 32, 120]
        x1 = self.block6(x1)              # [320, 128, 21] --> [320, 32, 17]
        x2 = self.block7(x2)              # [320, 128, 45] --> [320, 32, 41]

        #x1 = self.flatten(x1)             # [320, 544]
        #x1 = x1.reshape(-1, 20, x1.shape[-1])      # [16, 20, 544]
        #x1, _ = self.rnn3(x1)                      # [16, 20, 2048]
        #x1 = x1.reshape(-1, 1, x1.shape[-1])       # [320, 1, 2048]


        x12 = torch.cat((x1, x2), 2)               # [320, 32, 58]
        x123 = torch.cat((x12, x3), 2)             # [320, 32, 100]

        x_feature = self.block8(x123)              # [320, 32, 100] --> [320, 32, 33]
        # x_feature = self.simAM(x_feature)

        return x_feature

class _MySleepNet_1Chan(nn.Module):
    def __init__(self, stages=5, hidden_size=128, seq_len=20, is_bidirectional=False, network="LSTM"):
        super(MySleepNet_1Chan, self).__init__()
        self.stages = stages
        self.flatten = nn.Flatten()
        self.Feature = Feature_1Chan()
        self.fc = nn.Linear(2000, self.stages)
        self.se = SELayer_1D(32)
        self.GCNet = GlobalContextBlock(32)
        self.output = nn.Softplus()
        self.rnn = nn.LSTM(input_size=1056, hidden_size=1000, num_layers=1, batch_first=True, bidirectional=True)

        self.block = nn.Sequential(  # [320, 32, 33]
            nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(32, affine=True),
            nn.ReLU(True),
            nn.Flatten(),
        )

    def forward(self, x):
        x = self.Feature(x)        # [320, 32, 33]
        # x = self.se(x)
        x = self.GCNet(x)
        x = self.block(x)          # [320, 1056]
        x = x.reshape(-1, 20, x.shape[-1])      # [16, 20, 1056]
        x, _ = self.rnn(x)                      # [16, 20, 2000]
        x = x.reshape(-1, x.shape[-1])          # [320, 2000]

        x = self.fc(x)
        x = self.output(x)
        return x



# ***************** 继续提升 2024.6.26 -v3  ******************* : 2*cbam + GCNet
class _Feature_1Chan(nn.Module):
    def __init__(self, hidden_size=128, seq_len=20, is_bidirectional=False, network="LSTM"):
        super(Feature_1Chan, self).__init__()
        self.flatten = nn.Flatten()
        self.simAM = SimAM()
        self.se1 = cbam(128)
        self.se2 = cbam(128)
        # Branch 1
        self.block1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=128, kernel_size=30, stride=4, padding=0),
            nn.BatchNorm1d(128, affine=True),
            nn.ReLU(True),
            nn.MaxPool1d(8, stride=8),
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=8, stride=1, padding=0),
            nn.BatchNorm1d(128, affine=True),
            nn.ReLU(True),
            nn.MaxPool1d(kernel_size=4, stride=4),
            nn.Dropout(p=0.2),
        )
        self.block2 = nn.Sequential(  # [320, 128, 21] -->[320, 32, 21]
            nn.Conv1d(in_channels=128, out_channels=32, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(32, affine=True),
            nn.ReLU(True),  # [320, 128, 743]
        )

        self.rnn1 = nn.LSTM(input_size=672, hidden_size=256, num_layers=1, batch_first=True, bidirectional=True)

        # Branch 1: cross attention
        self.qcross1 = nn.Sequential(nn.Linear(512, 128), nn.ReLU(inplace=True))
        self.kcross1 = nn.Sequential(nn.Linear(512, 128), nn.ReLU(inplace=True))
        self.vcross1 = nn.Sequential(nn.Linear(512, 128), nn.ReLU(inplace=True))

        # Branch 2
        self.block3 = nn.Sequential(  # [320, 1, 3000] -->[320, 128, 187]
            nn.Conv1d(in_channels=1, out_channels=128, kernel_size=8, stride=2, padding=0),
            nn.BatchNorm1d(128, affine=True),
            nn.ReLU(True),
            nn.MaxPool1d(8, stride=8),
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=8, stride=1, padding=0),
            nn.BatchNorm1d(128, affine=True),
            nn.ReLU(True),
            nn.MaxPool1d(kernel_size=4, stride=4),
            nn.Dropout(p=0.2),
        )
        self.block4 = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=32, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(32, affine=True),
            nn.ReLU(True),  # [320, 128, 743]
        )

        self.rnn2 = nn.LSTM(input_size=1440, hidden_size=256, num_layers=1, batch_first=True, bidirectional=True)

        # Branch 2: cross attention
        self.qcross2 = nn.Sequential(nn.Linear(512, 128), nn.ReLU(inplace=True))
        self.kcross2 = nn.Sequential(nn.Linear(512, 128), nn.ReLU(inplace=True))
        self.vcross2 = nn.Sequential(nn.Linear(512, 128), nn.ReLU(inplace=True))

        # after concat                 [320, 2, 128] --> [320, 32, 42]
        self.block5 = nn.Sequential(
            nn.Conv1d(in_channels=2, out_channels=32, kernel_size=5, stride=1, padding=1),
            nn.BatchNorm1d(32, affine=True),
            nn.ReLU(True),
            nn.MaxPool1d(kernel_size=3, stride=3),
            nn.Dropout(p=0.15),
        )

        self.block6 = nn.Sequential(   # [320, 128, 21] --> [320, 32, 17]
            nn.Conv1d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm1d(64, affine=True),
            nn.ReLU(True),
            nn.Conv1d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm1d(32, affine=True),
            nn.ReLU(True),
        )

        self.block7 = nn.Sequential(  # [320, 128, 45] --> [320, 32, 41]
            nn.Conv1d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm1d(64, affine=True),
            nn.ReLU(True),
            nn.Conv1d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm1d(32, affine=True),
            nn.ReLU(True),
            nn.Dropout(p=0.15),
        )

        self.block8 = nn.Sequential(  # [320, 32, 100] --> [320, 32, 33]
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(64, affine=True),
            nn.ReLU(True),
            nn.MaxPool1d(kernel_size=3, stride=3),
            nn.Conv1d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(32, affine=True),
            nn.ReLU(True),
            nn.Dropout(p=0.15),
            # nn.Flatten(),
        )

        self.rnn_dropout = nn.Dropout(p=0.2)
        self.fc_dropout = nn.Dropout(p=0.2)
        self.cross_attn1 = nn.Linear(128 + 128, 128)
        self.cross_attn2 = nn.Linear(128 + 128, 128)
        # self.fc = nn.Linear(400, self.stages)
        # self.output = nn.Softplus()
        # self.rnn3 = nn.LSTM(input_size=544, hidden_size=1024, num_layers=3, batch_first=True, bidirectional=True)
        # self.rnn4 = nn.LSTM(input_size=1312, hidden_size=1024, num_layers=3, batch_first=True, bidirectional=True)

    def forward(self, x):
        x = x.reshape(-1, 1, 3000)   # [320, 1, 3000]
        # Branch 1
        x1 = self.block1(x)          # [320, 128, 21]
        x1 = self.se1(x1)

        x1_bra = self.block2(x1)     # [320, 32, 21]
        x1_bra = self.flatten(x1_bra)  # [320， 672]
        x1_bra = x1_bra.reshape(-1, 20, x1_bra.shape[-1])  # [16, 20, 672]
        x1_bra, _ = self.rnn1(x1_bra)         # [16, 20, 512]
        x1_bra = self.rnn_dropout(x1_bra)
        x1_bra = x1_bra.reshape(-1, x1_bra.shape[-1])    # [320, 512]

        # Branch 2
        x2 = self.block3(x)          # [320, 128, 45]
        x2 = self.se2(x2)

        x2_bra = self.block4(x2)     # [320, 32, 45]
        x2_bra = self.flatten(x2_bra)        # [320， 1440]
        x2_bra = x2_bra.reshape(-1, 20, x2_bra.shape[-1])    # [16, 20， 1440]
        x2_bra, _ = self.rnn2(x2_bra)        # [16, 20, 512]
        x2_bra = self.rnn_dropout(x2_bra)
        x2_bra = x2_bra.reshape(-1, x2_bra.shape[-1])    # [320, 512]

        # cross attention
        cro_q1, cro_k1, cro_v1 = self.qcross1(x1_bra), self.kcross1(x1_bra), self.vcross1(x1_bra)
        cro_q2, cro_k2, cro_v2 = self.qcross2(x2_bra), self.kcross2(x2_bra), self.vcross2(x2_bra)

        q2_k1 = F.softmax(self.cross_attn2(torch.cat((cro_q2, cro_k1), 1)), dim=1)
        q2_kv1 = torch.mul(q2_k1, cro_v1)         # [320, 128]
        q2_kv1_ = torch.unsqueeze(q2_kv1, dim=1)  # [320, 1, 128]

        q1_k2 = F.softmax(self.cross_attn1(torch.cat((cro_q1, cro_k2), 1)), dim=1)
        q1_kv2 = torch.mul(q1_k2, cro_v2)         # [320, 128]
        q1_kv2_ = torch.unsqueeze(q1_kv2, dim=1)  # [320, 1, 128]

        x3 = torch.cat((q2_kv1_, q1_kv2_), 1)      # [320,  2, 128]
        x3 = self.block5(x3)                       # [320, 32, 42]

        # x1: [320, 128, 21]    x2: [320, 128, 45]    x3: [320, 32, 120]
        x1 = self.block6(x1)              # [320, 128, 21] --> [320, 32, 17]
        x2 = self.block7(x2)              # [320, 128, 45] --> [320, 32, 41]

        #x1 = self.flatten(x1)             # [320, 544]
        #x1 = x1.reshape(-1, 20, x1.shape[-1])      # [16, 20, 544]
        #x1, _ = self.rnn3(x1)                      # [16, 20, 2048]
        #x1 = x1.reshape(-1, 1, x1.shape[-1])       # [320, 1, 2048]


        x12 = torch.cat((x1, x2), 2)               # [320, 32, 58]
        x123 = torch.cat((x12, x3), 2)             # [320, 32, 100]

        x_feature = self.block8(x123)              # [320, 32, 100] --> [320, 32, 33]
        # x_feature = self.simAM(x_feature)

        return x_feature

class _MySleepNet_1Chan(nn.Module):
    def __init__(self, stages=5, hidden_size=128, seq_len=20, is_bidirectional=False, network="LSTM"):
        super(MySleepNet_1Chan, self).__init__()
        self.stages = stages
        self.flatten = nn.Flatten()
        self.Feature = Feature_1Chan()
        self.fc = nn.Linear(2000, self.stages)
        self.se = SELayer_1D(32)
        self.GCNet = GlobalContextBlock(32)
        self.output = nn.Softplus()
        self.rnn = nn.LSTM(input_size=1056, hidden_size=1000, num_layers=1, batch_first=True, bidirectional=True)

        self.block = nn.Sequential(  # [320, 32, 33]
            nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(32, affine=True),
            nn.ReLU(True),
            nn.Flatten(),
        )

    def forward(self, x):
        x = self.Feature(x)        # [320, 32, 33]
        # x = self.se(x)
        x = self.GCNet(x)
        x = self.block(x)          # [320, 1056]
        x = x.reshape(-1, 20, x.shape[-1])      # [16, 20, 1056]
        x, _ = self.rnn(x)                      # [16, 20, 2000]
        x = x.reshape(-1, x.shape[-1])          # [320, 2000]

        x = self.fc(x)
        x = self.output(x)
        return x



# ***************** 继续提升 2024.6.26 -v4  ******************* : 2*SKconv + GCNet
class _Feature_1Chan(nn.Module):
    def __init__(self, hidden_size=128, seq_len=20, is_bidirectional=False, network="LSTM"):
        super(Feature_1Chan, self).__init__()
        self.flatten = nn.Flatten()
        self.simAM = SimAM()
        self.se1 = SKConv(in_ch=128, M=3, G=1, r=2)
        self.se2 = SKConv(in_ch=128, M=3, G=1, r=2)
        # Branch 1
        self.block1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=128, kernel_size=30, stride=4, padding=0),
            nn.BatchNorm1d(128, affine=True),
            nn.ReLU(True),
            nn.MaxPool1d(8, stride=8),
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=8, stride=1, padding=0),
            nn.BatchNorm1d(128, affine=True),
            nn.ReLU(True),
            nn.MaxPool1d(kernel_size=4, stride=4),
            nn.Dropout(p=0.2),
        )
        self.block2 = nn.Sequential(  # [320, 128, 21] -->[320, 32, 21]
            nn.Conv1d(in_channels=128, out_channels=32, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(32, affine=True),
            nn.ReLU(True),  # [320, 128, 743]
        )

        self.rnn1 = nn.LSTM(input_size=672, hidden_size=256, num_layers=1, batch_first=True, bidirectional=True)

        # Branch 1: cross attention
        self.qcross1 = nn.Sequential(nn.Linear(512, 128), nn.ReLU(inplace=True))
        self.kcross1 = nn.Sequential(nn.Linear(512, 128), nn.ReLU(inplace=True))
        self.vcross1 = nn.Sequential(nn.Linear(512, 128), nn.ReLU(inplace=True))

        # Branch 2
        self.block3 = nn.Sequential(  # [320, 1, 3000] -->[320, 128, 187]
            nn.Conv1d(in_channels=1, out_channels=128, kernel_size=8, stride=2, padding=0),
            nn.BatchNorm1d(128, affine=True),
            nn.ReLU(True),
            nn.MaxPool1d(8, stride=8),
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=8, stride=1, padding=0),
            nn.BatchNorm1d(128, affine=True),
            nn.ReLU(True),
            nn.MaxPool1d(kernel_size=4, stride=4),
            nn.Dropout(p=0.2),
        )
        self.block4 = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=32, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(32, affine=True),
            nn.ReLU(True),  # [320, 128, 743]
        )

        self.rnn2 = nn.LSTM(input_size=1440, hidden_size=256, num_layers=1, batch_first=True, bidirectional=True)

        # Branch 2: cross attention
        self.qcross2 = nn.Sequential(nn.Linear(512, 128), nn.ReLU(inplace=True))
        self.kcross2 = nn.Sequential(nn.Linear(512, 128), nn.ReLU(inplace=True))
        self.vcross2 = nn.Sequential(nn.Linear(512, 128), nn.ReLU(inplace=True))

        # after concat                 [320, 2, 128] --> [320, 32, 42]
        self.block5 = nn.Sequential(
            nn.Conv1d(in_channels=2, out_channels=32, kernel_size=5, stride=1, padding=1),
            nn.BatchNorm1d(32, affine=True),
            nn.ReLU(True),
            nn.MaxPool1d(kernel_size=3, stride=3),
            nn.Dropout(p=0.15),
        )

        self.block6 = nn.Sequential(   # [320, 128, 21] --> [320, 32, 17]
            nn.Conv1d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm1d(64, affine=True),
            nn.ReLU(True),
            nn.Conv1d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm1d(32, affine=True),
            nn.ReLU(True),
        )

        self.block7 = nn.Sequential(  # [320, 128, 45] --> [320, 32, 41]
            nn.Conv1d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm1d(64, affine=True),
            nn.ReLU(True),
            nn.Conv1d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm1d(32, affine=True),
            nn.ReLU(True),
            nn.Dropout(p=0.15),
        )

        self.block8 = nn.Sequential(  # [320, 32, 100] --> [320, 32, 33]
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(64, affine=True),
            nn.ReLU(True),
            nn.MaxPool1d(kernel_size=3, stride=3),
            nn.Conv1d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(32, affine=True),
            nn.ReLU(True),
            nn.Dropout(p=0.15),
            # nn.Flatten(),
        )

        self.rnn_dropout = nn.Dropout(p=0.2)
        self.fc_dropout = nn.Dropout(p=0.2)
        self.cross_attn1 = nn.Linear(128 + 128, 128)
        self.cross_attn2 = nn.Linear(128 + 128, 128)
        # self.fc = nn.Linear(400, self.stages)
        # self.output = nn.Softplus()
        # self.rnn3 = nn.LSTM(input_size=544, hidden_size=1024, num_layers=3, batch_first=True, bidirectional=True)
        # self.rnn4 = nn.LSTM(input_size=1312, hidden_size=1024, num_layers=3, batch_first=True, bidirectional=True)

    def forward(self, x):
        x = x.reshape(-1, 1, 3000)   # [320, 1, 3000]
        # Branch 1
        x1 = self.block1(x)          # [320, 128, 21]
        x1 = self.se1(x1)

        x1_bra = self.block2(x1)     # [320, 32, 21]
        x1_bra = self.flatten(x1_bra)  # [320， 672]
        x1_bra = x1_bra.reshape(-1, 20, x1_bra.shape[-1])  # [16, 20, 672]
        x1_bra, _ = self.rnn1(x1_bra)         # [16, 20, 512]
        x1_bra = self.rnn_dropout(x1_bra)
        x1_bra = x1_bra.reshape(-1, x1_bra.shape[-1])    # [320, 512]

        # Branch 2
        x2 = self.block3(x)          # [320, 128, 45]
        x2 = self.se2(x2)

        x2_bra = self.block4(x2)     # [320, 32, 45]
        x2_bra = self.flatten(x2_bra)        # [320， 1440]
        x2_bra = x2_bra.reshape(-1, 20, x2_bra.shape[-1])    # [16, 20， 1440]
        x2_bra, _ = self.rnn2(x2_bra)        # [16, 20, 512]
        x2_bra = self.rnn_dropout(x2_bra)
        x2_bra = x2_bra.reshape(-1, x2_bra.shape[-1])    # [320, 512]

        # cross attention
        cro_q1, cro_k1, cro_v1 = self.qcross1(x1_bra), self.kcross1(x1_bra), self.vcross1(x1_bra)
        cro_q2, cro_k2, cro_v2 = self.qcross2(x2_bra), self.kcross2(x2_bra), self.vcross2(x2_bra)

        q2_k1 = F.softmax(self.cross_attn2(torch.cat((cro_q2, cro_k1), 1)), dim=1)
        q2_kv1 = torch.mul(q2_k1, cro_v1)         # [320, 128]
        q2_kv1_ = torch.unsqueeze(q2_kv1, dim=1)  # [320, 1, 128]

        q1_k2 = F.softmax(self.cross_attn1(torch.cat((cro_q1, cro_k2), 1)), dim=1)
        q1_kv2 = torch.mul(q1_k2, cro_v2)         # [320, 128]
        q1_kv2_ = torch.unsqueeze(q1_kv2, dim=1)  # [320, 1, 128]

        x3 = torch.cat((q2_kv1_, q1_kv2_), 1)      # [320,  2, 128]
        x3 = self.block5(x3)                       # [320, 32, 42]

        # x1: [320, 128, 21]    x2: [320, 128, 45]    x3: [320, 32, 120]
        x1 = self.block6(x1)              # [320, 128, 21] --> [320, 32, 17]
        x2 = self.block7(x2)              # [320, 128, 45] --> [320, 32, 41]

        #x1 = self.flatten(x1)             # [320, 544]
        #x1 = x1.reshape(-1, 20, x1.shape[-1])      # [16, 20, 544]
        #x1, _ = self.rnn3(x1)                      # [16, 20, 2048]
        #x1 = x1.reshape(-1, 1, x1.shape[-1])       # [320, 1, 2048]


        x12 = torch.cat((x1, x2), 2)               # [320, 32, 58]
        x123 = torch.cat((x12, x3), 2)             # [320, 32, 100]

        x_feature = self.block8(x123)              # [320, 32, 100] --> [320, 32, 33]
        # x_feature = self.simAM(x_feature)

        return x_feature

class _MySleepNet_1Chan(nn.Module):
    def __init__(self, stages=5, hidden_size=128, seq_len=20, is_bidirectional=False, network="LSTM"):
        super(MySleepNet_1Chan, self).__init__()
        self.stages = stages
        self.flatten = nn.Flatten()
        self.Feature = Feature_1Chan()
        self.fc = nn.Linear(2000, self.stages)
        #self.se = SELayer_1D(32)
        self.GCNet = GlobalContextBlock(32)
        self.cbam = cbam(32)
        self.output = nn.Softplus()
        self.rnn = nn.LSTM(input_size=1056, hidden_size=1000, num_layers=1, batch_first=True, bidirectional=True)

        self.block = nn.Sequential(  # [320, 32, 33]
            nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(32, affine=True),
            nn.ReLU(True),
            nn.Flatten(),
        )

    def forward(self, x):
        x = self.Feature(x)        # [320, 32, 33]
        # x = self.se(x)
        x = self.GCNet(x)
        x = self.block(x)          # [320, 1056]
        x = x.reshape(-1, 20, x.shape[-1])      # [16, 20, 1056]
        x, _ = self.rnn(x)                      # [16, 20, 2000]
        x = x.reshape(-1, x.shape[-1])          # [320, 2000]

        x = self.fc(x)
        x = self.output(x)
        return x


# ***************** 继续提升 2024.6.27 -v6  ******************* : 2*EMA + GCNet
class _Feature_1Chan(nn.Module):
    def __init__(self, hidden_size=128, seq_len=20, is_bidirectional=False, network="LSTM"):
        super(Feature_1Chan, self).__init__()
        self.flatten = nn.Flatten()
        self.simAM = SimAM()
        self.se1 = EMA(128)
        self.se2 = EMA(128)
        # Branch 1
        self.block1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=128, kernel_size=30, stride=4, padding=0),
            nn.BatchNorm1d(128, affine=True),
            nn.ReLU(True),
            nn.MaxPool1d(8, stride=8),
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=8, stride=1, padding=0),
            nn.BatchNorm1d(128, affine=True),
            nn.ReLU(True),
            nn.MaxPool1d(kernel_size=4, stride=4),
            nn.Dropout(p=0.2),
        )
        self.block2 = nn.Sequential(  # [320, 128, 21] -->[320, 32, 21]
            nn.Conv1d(in_channels=128, out_channels=32, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(32, affine=True),
            nn.ReLU(True),  # [320, 128, 743]
        )

        self.rnn1 = nn.LSTM(input_size=672, hidden_size=256, num_layers=1, batch_first=True, bidirectional=True)

        # Branch 1: cross attention
        self.qcross1 = nn.Sequential(nn.Linear(512, 128), nn.ReLU(inplace=True))
        self.kcross1 = nn.Sequential(nn.Linear(512, 128), nn.ReLU(inplace=True))
        self.vcross1 = nn.Sequential(nn.Linear(512, 128), nn.ReLU(inplace=True))

        # Branch 2
        self.block3 = nn.Sequential(  # [320, 1, 3000] -->[320, 128, 187]
            nn.Conv1d(in_channels=1, out_channels=128, kernel_size=8, stride=2, padding=0),
            nn.BatchNorm1d(128, affine=True),
            nn.ReLU(True),
            nn.MaxPool1d(8, stride=8),
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=8, stride=1, padding=0),
            nn.BatchNorm1d(128, affine=True),
            nn.ReLU(True),
            nn.MaxPool1d(kernel_size=4, stride=4),
            nn.Dropout(p=0.2),
        )
        self.block4 = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=32, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(32, affine=True),
            nn.ReLU(True),  # [320, 128, 743]
        )

        self.rnn2 = nn.LSTM(input_size=1440, hidden_size=256, num_layers=1, batch_first=True, bidirectional=True)

        # Branch 2: cross attention
        self.qcross2 = nn.Sequential(nn.Linear(512, 128), nn.ReLU(inplace=True))
        self.kcross2 = nn.Sequential(nn.Linear(512, 128), nn.ReLU(inplace=True))
        self.vcross2 = nn.Sequential(nn.Linear(512, 128), nn.ReLU(inplace=True))

        # after concat                 [320, 2, 128] --> [320, 32, 42]
        self.block5 = nn.Sequential(
            nn.Conv1d(in_channels=2, out_channels=32, kernel_size=5, stride=1, padding=1),
            nn.BatchNorm1d(32, affine=True),
            nn.ReLU(True),
            nn.MaxPool1d(kernel_size=3, stride=3),
            nn.Dropout(p=0.15),
        )

        self.block6 = nn.Sequential(   # [320, 128, 21] --> [320, 32, 17]
            nn.Conv1d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm1d(64, affine=True),
            nn.ReLU(True),
            nn.Conv1d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm1d(32, affine=True),
            nn.ReLU(True),
        )

        self.block7 = nn.Sequential(  # [320, 128, 45] --> [320, 32, 41]
            nn.Conv1d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm1d(64, affine=True),
            nn.ReLU(True),
            nn.Conv1d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm1d(32, affine=True),
            nn.ReLU(True),
            nn.Dropout(p=0.15),
        )

        self.block8 = nn.Sequential(  # [320, 32, 100] --> [320, 32, 33]
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(64, affine=True),
            nn.ReLU(True),
            nn.MaxPool1d(kernel_size=3, stride=3),
            nn.Conv1d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(32, affine=True),
            nn.ReLU(True),
            nn.Dropout(p=0.15),
            # nn.Flatten(),
        )

        self.rnn_dropout = nn.Dropout(p=0.2)
        self.fc_dropout = nn.Dropout(p=0.2)
        self.cross_attn1 = nn.Linear(128 + 128, 128)
        self.cross_attn2 = nn.Linear(128 + 128, 128)
        # self.fc = nn.Linear(400, self.stages)
        # self.output = nn.Softplus()
        # self.rnn3 = nn.LSTM(input_size=544, hidden_size=1024, num_layers=3, batch_first=True, bidirectional=True)
        # self.rnn4 = nn.LSTM(input_size=1312, hidden_size=1024, num_layers=3, batch_first=True, bidirectional=True)

    def forward(self, x):
        x = x.reshape(-1, 1, 3000)   # [320, 1, 3000]
        # Branch 1
        x1 = self.block1(x)          # [320, 128, 21]
        x1 = self.se1(x1)

        x1_bra = self.block2(x1)     # [320, 32, 21]
        x1_bra = self.flatten(x1_bra)  # [320， 672]
        x1_bra = x1_bra.reshape(-1, 20, x1_bra.shape[-1])  # [16, 20, 672]
        x1_bra, _ = self.rnn1(x1_bra)         # [16, 20, 512]
        x1_bra = self.rnn_dropout(x1_bra)
        x1_bra = x1_bra.reshape(-1, x1_bra.shape[-1])    # [320, 512]

        # Branch 2
        x2 = self.block3(x)          # [320, 128, 45]
        x2 = self.se2(x2)

        x2_bra = self.block4(x2)     # [320, 32, 45]
        x2_bra = self.flatten(x2_bra)        # [320， 1440]
        x2_bra = x2_bra.reshape(-1, 20, x2_bra.shape[-1])    # [16, 20， 1440]
        x2_bra, _ = self.rnn2(x2_bra)        # [16, 20, 512]
        x2_bra = self.rnn_dropout(x2_bra)
        x2_bra = x2_bra.reshape(-1, x2_bra.shape[-1])    # [320, 512]

        # cross attention
        cro_q1, cro_k1, cro_v1 = self.qcross1(x1_bra), self.kcross1(x1_bra), self.vcross1(x1_bra)
        cro_q2, cro_k2, cro_v2 = self.qcross2(x2_bra), self.kcross2(x2_bra), self.vcross2(x2_bra)

        q2_k1 = F.softmax(self.cross_attn2(torch.cat((cro_q2, cro_k1), 1)), dim=1)
        q2_kv1 = torch.mul(q2_k1, cro_v1)         # [320, 128]
        q2_kv1_ = torch.unsqueeze(q2_kv1, dim=1)  # [320, 1, 128]

        q1_k2 = F.softmax(self.cross_attn1(torch.cat((cro_q1, cro_k2), 1)), dim=1)
        q1_kv2 = torch.mul(q1_k2, cro_v2)         # [320, 128]
        q1_kv2_ = torch.unsqueeze(q1_kv2, dim=1)  # [320, 1, 128]

        x3 = torch.cat((q2_kv1_, q1_kv2_), 1)      # [320,  2, 128]
        x3 = self.block5(x3)                       # [320, 32, 42]

        # x1: [320, 128, 21]    x2: [320, 128, 45]    x3: [320, 32, 120]
        x1 = self.block6(x1)              # [320, 128, 21] --> [320, 32, 17]
        x2 = self.block7(x2)              # [320, 128, 45] --> [320, 32, 41]

        #x1 = self.flatten(x1)             # [320, 544]
        #x1 = x1.reshape(-1, 20, x1.shape[-1])      # [16, 20, 544]
        #x1, _ = self.rnn3(x1)                      # [16, 20, 2048]
        #x1 = x1.reshape(-1, 1, x1.shape[-1])       # [320, 1, 2048]


        x12 = torch.cat((x1, x2), 2)               # [320, 32, 58]
        x123 = torch.cat((x12, x3), 2)             # [320, 32, 100]

        x_feature = self.block8(x123)              # [320, 32, 100] --> [320, 32, 33]
        # x_feature = self.simAM(x_feature)

        return x_feature

class _MySleepNet_1Chan(nn.Module):
    def __init__(self, stages=5, hidden_size=128, seq_len=20, is_bidirectional=False, network="LSTM"):
        super(MySleepNet_1Chan, self).__init__()
        self.stages = stages
        self.flatten = nn.Flatten()
        self.Feature = Feature_1Chan()
        self.fc = nn.Linear(2000, self.stages)
        self.se = SELayer_1D(32)
        self.GCNet = GlobalContextBlock(32)
        self.output = nn.Softplus()
        self.rnn = nn.LSTM(input_size=1056, hidden_size=1000, num_layers=1, batch_first=True, bidirectional=True)

        self.block = nn.Sequential(  # [320, 32, 33]
            nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(32, affine=True),
            nn.ReLU(True),
            nn.Flatten(),
        )

    def forward(self, x):
        x = self.Feature(x)        # [320, 32, 33]
        # x = self.se(x)
        x = self.GCNet(x)
        x = self.block(x)          # [320, 1056]
        x = x.reshape(-1, 20, x.shape[-1])      # [16, 20, 1056]
        x, _ = self.rnn(x)                      # [16, 20, 2000]
        x = x.reshape(-1, x.shape[-1])          # [320, 2000]

        x = self.fc(x)
        x = self.output(x)
        return x


# ***************** 继续提升 2024.6.27 -v7  ******************* : 2*NonLocalBlockND + NonLocalBlockND
class _Feature_1Chan(nn.Module):
    def __init__(self, hidden_size=128, seq_len=20, is_bidirectional=False, network="LSTM"):
        super(Feature_1Chan, self).__init__()
        self.flatten = nn.Flatten()
        self.simAM = SimAM()
        self.se1 = NonLocalBlockND(128)
        self.se2 = NonLocalBlockND(128)
        # Branch 1
        self.block1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=128, kernel_size=30, stride=4, padding=0),
            nn.BatchNorm1d(128, affine=True),
            nn.ReLU(True),
            nn.MaxPool1d(8, stride=8),
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=8, stride=1, padding=0),
            nn.BatchNorm1d(128, affine=True),
            nn.ReLU(True),
            nn.MaxPool1d(kernel_size=4, stride=4),
            nn.Dropout(p=0.2),
        )
        self.block2 = nn.Sequential(  # [320, 128, 21] -->[320, 32, 21]
            nn.Conv1d(in_channels=128, out_channels=32, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(32, affine=True),
            nn.ReLU(True),  # [320, 128, 743]
        )

        self.rnn1 = nn.LSTM(input_size=672, hidden_size=256, num_layers=1, batch_first=True, bidirectional=True)

        # Branch 1: cross attention
        self.qcross1 = nn.Sequential(nn.Linear(512, 128), nn.ReLU(inplace=True))
        self.kcross1 = nn.Sequential(nn.Linear(512, 128), nn.ReLU(inplace=True))
        self.vcross1 = nn.Sequential(nn.Linear(512, 128), nn.ReLU(inplace=True))

        # Branch 2
        self.block3 = nn.Sequential(  # [320, 1, 3000] -->[320, 128, 187]
            nn.Conv1d(in_channels=1, out_channels=128, kernel_size=8, stride=2, padding=0),
            nn.BatchNorm1d(128, affine=True),
            nn.ReLU(True),
            nn.MaxPool1d(8, stride=8),
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=8, stride=1, padding=0),
            nn.BatchNorm1d(128, affine=True),
            nn.ReLU(True),
            nn.MaxPool1d(kernel_size=4, stride=4),
            nn.Dropout(p=0.2),
        )
        self.block4 = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=32, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(32, affine=True),
            nn.ReLU(True),  # [320, 128, 743]
        )

        self.rnn2 = nn.LSTM(input_size=1440, hidden_size=256, num_layers=1, batch_first=True, bidirectional=True)

        # Branch 2: cross attention
        self.qcross2 = nn.Sequential(nn.Linear(512, 128), nn.ReLU(inplace=True))
        self.kcross2 = nn.Sequential(nn.Linear(512, 128), nn.ReLU(inplace=True))
        self.vcross2 = nn.Sequential(nn.Linear(512, 128), nn.ReLU(inplace=True))

        # after concat                 [320, 2, 128] --> [320, 32, 42]
        self.block5 = nn.Sequential(
            nn.Conv1d(in_channels=2, out_channels=32, kernel_size=5, stride=1, padding=1),
            nn.BatchNorm1d(32, affine=True),
            nn.ReLU(True),
            nn.MaxPool1d(kernel_size=3, stride=3),
            nn.Dropout(p=0.15),
        )

        self.block6 = nn.Sequential(   # [320, 128, 21] --> [320, 32, 17]
            nn.Conv1d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm1d(64, affine=True),
            nn.ReLU(True),
            nn.Conv1d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm1d(32, affine=True),
            nn.ReLU(True),
        )

        self.block7 = nn.Sequential(  # [320, 128, 45] --> [320, 32, 41]
            nn.Conv1d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm1d(64, affine=True),
            nn.ReLU(True),
            nn.Conv1d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm1d(32, affine=True),
            nn.ReLU(True),
            nn.Dropout(p=0.15),
        )

        self.block8 = nn.Sequential(  # [320, 32, 100] --> [320, 32, 33]
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(64, affine=True),
            nn.ReLU(True),
            nn.MaxPool1d(kernel_size=3, stride=3),
            nn.Conv1d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(32, affine=True),
            nn.ReLU(True),
            nn.Dropout(p=0.15),
            # nn.Flatten(),
        )

        self.rnn_dropout = nn.Dropout(p=0.2)
        self.fc_dropout = nn.Dropout(p=0.2)
        self.cross_attn1 = nn.Linear(128 + 128, 128)
        self.cross_attn2 = nn.Linear(128 + 128, 128)
        # self.fc = nn.Linear(400, self.stages)
        # self.output = nn.Softplus()
        # self.rnn3 = nn.LSTM(input_size=544, hidden_size=1024, num_layers=3, batch_first=True, bidirectional=True)
        # self.rnn4 = nn.LSTM(input_size=1312, hidden_size=1024, num_layers=3, batch_first=True, bidirectional=True)

    def forward(self, x):
        x = x.reshape(-1, 1, 3000)   # [320, 1, 3000]
        # Branch 1
        x1 = self.block1(x)          # [320, 128, 21]
        x1 = self.se1(x1)

        x1_bra = self.block2(x1)     # [320, 32, 21]
        x1_bra = self.flatten(x1_bra)  # [320， 672]
        x1_bra = x1_bra.reshape(-1, 20, x1_bra.shape[-1])  # [16, 20, 672]
        x1_bra, _ = self.rnn1(x1_bra)         # [16, 20, 512]
        x1_bra = self.rnn_dropout(x1_bra)
        x1_bra = x1_bra.reshape(-1, x1_bra.shape[-1])    # [320, 512]

        # Branch 2
        x2 = self.block3(x)          # [320, 128, 45]
        x2 = self.se2(x2)

        x2_bra = self.block4(x2)     # [320, 32, 45]
        x2_bra = self.flatten(x2_bra)        # [320， 1440]
        x2_bra = x2_bra.reshape(-1, 20, x2_bra.shape[-1])    # [16, 20， 1440]
        x2_bra, _ = self.rnn2(x2_bra)        # [16, 20, 512]
        x2_bra = self.rnn_dropout(x2_bra)
        x2_bra = x2_bra.reshape(-1, x2_bra.shape[-1])    # [320, 512]

        # cross attention
        cro_q1, cro_k1, cro_v1 = self.qcross1(x1_bra), self.kcross1(x1_bra), self.vcross1(x1_bra)
        cro_q2, cro_k2, cro_v2 = self.qcross2(x2_bra), self.kcross2(x2_bra), self.vcross2(x2_bra)

        q2_k1 = F.softmax(self.cross_attn2(torch.cat((cro_q2, cro_k1), 1)), dim=1)
        q2_kv1 = torch.mul(q2_k1, cro_v1)         # [320, 128]
        q2_kv1_ = torch.unsqueeze(q2_kv1, dim=1)  # [320, 1, 128]

        q1_k2 = F.softmax(self.cross_attn1(torch.cat((cro_q1, cro_k2), 1)), dim=1)
        q1_kv2 = torch.mul(q1_k2, cro_v2)         # [320, 128]
        q1_kv2_ = torch.unsqueeze(q1_kv2, dim=1)  # [320, 1, 128]

        x3 = torch.cat((q2_kv1_, q1_kv2_), 1)      # [320,  2, 128]
        x3 = self.block5(x3)                       # [320, 32, 42]

        # x1: [320, 128, 21]    x2: [320, 128, 45]    x3: [320, 32, 120]
        x1 = self.block6(x1)              # [320, 128, 21] --> [320, 32, 17]
        x2 = self.block7(x2)              # [320, 128, 45] --> [320, 32, 41]

        #x1 = self.flatten(x1)             # [320, 544]
        #x1 = x1.reshape(-1, 20, x1.shape[-1])      # [16, 20, 544]
        #x1, _ = self.rnn3(x1)                      # [16, 20, 2048]
        #x1 = x1.reshape(-1, 1, x1.shape[-1])       # [320, 1, 2048]


        x12 = torch.cat((x1, x2), 2)               # [320, 32, 58]
        x123 = torch.cat((x12, x3), 2)             # [320, 32, 100]

        x_feature = self.block8(x123)              # [320, 32, 100] --> [320, 32, 33]
        # x_feature = self.simAM(x_feature)

        return x_feature

class _MySleepNet_1Chan(nn.Module):
    def __init__(self, stages=5, hidden_size=128, seq_len=20, is_bidirectional=False, network="LSTM"):
        super(MySleepNet_1Chan, self).__init__()
        self.stages = stages
        self.flatten = nn.Flatten()
        self.Feature = Feature_1Chan()
        self.fc = nn.Linear(2000, self.stages)
        self.GCNet = NonLocalBlockND(32)
        self.output = nn.Softplus()
        self.rnn = nn.LSTM(input_size=1056, hidden_size=1000, num_layers=1, batch_first=True, bidirectional=True)

        self.block = nn.Sequential(  # [320, 32, 33]
            nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(32, affine=True),
            nn.ReLU(True),
            nn.Flatten(),
        )

    def forward(self, x):
        x = self.Feature(x)        # [320, 32, 33]
        # x = self.se(x)
        x = self.GCNet(x)
        x = self.block(x)          # [320, 1056]
        x = x.reshape(-1, 20, x.shape[-1])      # [16, 20, 1056]
        x, _ = self.rnn(x)                      # [16, 20, 2000]
        x = x.reshape(-1, x.shape[-1])          # [320, 2000]

        x = self.fc(x)
        x = self.output(x)
        return x



# ***************** 继续提升 2024.6.27 -v8  ******************* : 2*EMA + EMA
class _Feature_1Chan(nn.Module):
    def __init__(self, hidden_size=128, seq_len=20, is_bidirectional=False, network="LSTM"):
        super(Feature_1Chan, self).__init__()
        self.flatten = nn.Flatten()
        self.simAM = SimAM()
        self.se1 = EMA(128)
        self.se2 = EMA(128)
        # Branch 1
        self.block1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=128, kernel_size=30, stride=4, padding=0),
            nn.BatchNorm1d(128, affine=True),
            nn.ReLU(True),
            nn.MaxPool1d(8, stride=8),
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=8, stride=1, padding=0),
            nn.BatchNorm1d(128, affine=True),
            nn.ReLU(True),
            nn.MaxPool1d(kernel_size=4, stride=4),
            nn.Dropout(p=0.2),
        )
        self.block2 = nn.Sequential(  # [320, 128, 21] -->[320, 32, 21]
            nn.Conv1d(in_channels=128, out_channels=32, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(32, affine=True),
            nn.ReLU(True),  # [320, 128, 743]
        )

        self.rnn1 = nn.LSTM(input_size=672, hidden_size=256, num_layers=1, batch_first=True, bidirectional=True)

        # Branch 1: cross attention
        self.qcross1 = nn.Sequential(nn.Linear(512, 128), nn.ReLU(inplace=True))
        self.kcross1 = nn.Sequential(nn.Linear(512, 128), nn.ReLU(inplace=True))
        self.vcross1 = nn.Sequential(nn.Linear(512, 128), nn.ReLU(inplace=True))

        # Branch 2
        self.block3 = nn.Sequential(  # [320, 1, 3000] -->[320, 128, 187]
            nn.Conv1d(in_channels=1, out_channels=128, kernel_size=8, stride=2, padding=0),
            nn.BatchNorm1d(128, affine=True),
            nn.ReLU(True),
            nn.MaxPool1d(8, stride=8),
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=8, stride=1, padding=0),
            nn.BatchNorm1d(128, affine=True),
            nn.ReLU(True),
            nn.MaxPool1d(kernel_size=4, stride=4),
            nn.Dropout(p=0.2),
        )
        self.block4 = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=32, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(32, affine=True),
            nn.ReLU(True),  # [320, 128, 743]
        )

        self.rnn2 = nn.LSTM(input_size=1440, hidden_size=256, num_layers=1, batch_first=True, bidirectional=True)

        # Branch 2: cross attention
        self.qcross2 = nn.Sequential(nn.Linear(512, 128), nn.ReLU(inplace=True))
        self.kcross2 = nn.Sequential(nn.Linear(512, 128), nn.ReLU(inplace=True))
        self.vcross2 = nn.Sequential(nn.Linear(512, 128), nn.ReLU(inplace=True))

        # after concat                 [320, 2, 128] --> [320, 32, 42]
        self.block5 = nn.Sequential(
            nn.Conv1d(in_channels=2, out_channels=32, kernel_size=5, stride=1, padding=1),
            nn.BatchNorm1d(32, affine=True),
            nn.ReLU(True),
            nn.MaxPool1d(kernel_size=3, stride=3),
            nn.Dropout(p=0.15),
        )

        self.block6 = nn.Sequential(   # [320, 128, 21] --> [320, 32, 17]
            nn.Conv1d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm1d(64, affine=True),
            nn.ReLU(True),
            nn.Conv1d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm1d(32, affine=True),
            nn.ReLU(True),
        )

        self.block7 = nn.Sequential(  # [320, 128, 45] --> [320, 32, 41]
            nn.Conv1d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm1d(64, affine=True),
            nn.ReLU(True),
            nn.Conv1d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm1d(32, affine=True),
            nn.ReLU(True),
            nn.Dropout(p=0.15),
        )

        self.block8 = nn.Sequential(  # [320, 32, 100] --> [320, 32, 33]
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(64, affine=True),
            nn.ReLU(True),
            nn.MaxPool1d(kernel_size=3, stride=3),
            nn.Conv1d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(32, affine=True),
            nn.ReLU(True),
            nn.Dropout(p=0.15),
            # nn.Flatten(),
        )

        self.rnn_dropout = nn.Dropout(p=0.2)
        self.fc_dropout = nn.Dropout(p=0.2)
        self.cross_attn1 = nn.Linear(128 + 128, 128)
        self.cross_attn2 = nn.Linear(128 + 128, 128)
        # self.fc = nn.Linear(400, self.stages)
        # self.output = nn.Softplus()
        # self.rnn3 = nn.LSTM(input_size=544, hidden_size=1024, num_layers=3, batch_first=True, bidirectional=True)
        # self.rnn4 = nn.LSTM(input_size=1312, hidden_size=1024, num_layers=3, batch_first=True, bidirectional=True)

    def forward(self, x):
        x = x.reshape(-1, 1, 3000)   # [320, 1, 3000]
        # Branch 1
        x1 = self.block1(x)          # [320, 128, 21]
        x1 = self.se1(x1)

        x1_bra = self.block2(x1)     # [320, 32, 21]
        x1_bra = self.flatten(x1_bra)  # [320， 672]
        x1_bra = x1_bra.reshape(-1, 20, x1_bra.shape[-1])  # [16, 20, 672]
        x1_bra, _ = self.rnn1(x1_bra)         # [16, 20, 512]
        x1_bra = self.rnn_dropout(x1_bra)
        x1_bra = x1_bra.reshape(-1, x1_bra.shape[-1])    # [320, 512]

        # Branch 2
        x2 = self.block3(x)          # [320, 128, 45]
        x2 = self.se2(x2)

        x2_bra = self.block4(x2)     # [320, 32, 45]
        x2_bra = self.flatten(x2_bra)        # [320， 1440]
        x2_bra = x2_bra.reshape(-1, 20, x2_bra.shape[-1])    # [16, 20， 1440]
        x2_bra, _ = self.rnn2(x2_bra)        # [16, 20, 512]
        x2_bra = self.rnn_dropout(x2_bra)
        x2_bra = x2_bra.reshape(-1, x2_bra.shape[-1])    # [320, 512]

        # cross attention
        cro_q1, cro_k1, cro_v1 = self.qcross1(x1_bra), self.kcross1(x1_bra), self.vcross1(x1_bra)
        cro_q2, cro_k2, cro_v2 = self.qcross2(x2_bra), self.kcross2(x2_bra), self.vcross2(x2_bra)

        q2_k1 = F.softmax(self.cross_attn2(torch.cat((cro_q2, cro_k1), 1)), dim=1)
        q2_kv1 = torch.mul(q2_k1, cro_v1)         # [320, 128]
        q2_kv1_ = torch.unsqueeze(q2_kv1, dim=1)  # [320, 1, 128]

        q1_k2 = F.softmax(self.cross_attn1(torch.cat((cro_q1, cro_k2), 1)), dim=1)
        q1_kv2 = torch.mul(q1_k2, cro_v2)         # [320, 128]
        q1_kv2_ = torch.unsqueeze(q1_kv2, dim=1)  # [320, 1, 128]

        x3 = torch.cat((q2_kv1_, q1_kv2_), 1)      # [320,  2, 128]
        x3 = self.block5(x3)                       # [320, 32, 42]

        # x1: [320, 128, 21]    x2: [320, 128, 45]    x3: [320, 32, 120]
        x1 = self.block6(x1)              # [320, 128, 21] --> [320, 32, 17]
        x2 = self.block7(x2)              # [320, 128, 45] --> [320, 32, 41]

        #x1 = self.flatten(x1)             # [320, 544]
        #x1 = x1.reshape(-1, 20, x1.shape[-1])      # [16, 20, 544]
        #x1, _ = self.rnn3(x1)                      # [16, 20, 2048]
        #x1 = x1.reshape(-1, 1, x1.shape[-1])       # [320, 1, 2048]


        x12 = torch.cat((x1, x2), 2)               # [320, 32, 58]
        x123 = torch.cat((x12, x3), 2)             # [320, 32, 100]

        x_feature = self.block8(x123)              # [320, 32, 100] --> [320, 32, 33]
        # x_feature = self.simAM(x_feature)

        return x_feature

class _MySleepNet_1Chan(nn.Module):
    def __init__(self, stages=5, hidden_size=128, seq_len=20, is_bidirectional=False, network="LSTM"):
        super(MySleepNet_1Chan, self).__init__()
        self.stages = stages
        self.flatten = nn.Flatten()
        self.Feature = Feature_1Chan()
        self.fc = nn.Linear(2000, self.stages)
        self.GCNet = EMA(32)
        self.output = nn.Softplus()
        self.rnn = nn.LSTM(input_size=1056, hidden_size=1000, num_layers=1, batch_first=True, bidirectional=True)

        self.block = nn.Sequential(  # [320, 32, 33]
            nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(32, affine=True),
            nn.ReLU(True),
            nn.Flatten(),
        )

    def forward(self, x):
        x = self.Feature(x)        # [320, 32, 33]
        # x = self.se(x)
        x = self.GCNet(x)
        x = self.block(x)          # [320, 1056]
        x = x.reshape(-1, 20, x.shape[-1])      # [16, 20, 1056]
        x, _ = self.rnn(x)                      # [16, 20, 2000]
        x = x.reshape(-1, x.shape[-1])          # [320, 2000]

        x = self.fc(x)
        x = self.output(x)
        return x



# ***************** 继续提升 2024.6.27 -v9  ******************* : 2*NonLocalBlockND + NonLocalBlockND + GCNet
class _Feature_1Chan(nn.Module):
    def __init__(self, hidden_size=128, seq_len=20, is_bidirectional=False, network="LSTM"):
        super(Feature_1Chan, self).__init__()
        self.flatten = nn.Flatten()
        self.simAM = SimAM()
        self.se1 = NonLocalBlockND(128)
        self.se2 = NonLocalBlockND(128)
        self.se3 = NonLocalBlockND(32)
        # Branch 1
        self.block1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=128, kernel_size=30, stride=4, padding=0),
            nn.BatchNorm1d(128, affine=True),
            nn.ReLU(True),
            nn.MaxPool1d(8, stride=8),
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=8, stride=1, padding=0),
            nn.BatchNorm1d(128, affine=True),
            nn.ReLU(True),
            nn.MaxPool1d(kernel_size=4, stride=4),
            nn.Dropout(p=0.2),
        )
        self.block2 = nn.Sequential(  # [320, 128, 21] -->[320, 32, 21]
            nn.Conv1d(in_channels=128, out_channels=32, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(32, affine=True),
            nn.ReLU(True),  # [320, 128, 743]
        )

        self.rnn1 = nn.LSTM(input_size=672, hidden_size=256, num_layers=1, batch_first=True, bidirectional=True)

        # Branch 1: cross attention
        self.qcross1 = nn.Sequential(nn.Linear(512, 128), nn.ReLU(inplace=True))
        self.kcross1 = nn.Sequential(nn.Linear(512, 128), nn.ReLU(inplace=True))
        self.vcross1 = nn.Sequential(nn.Linear(512, 128), nn.ReLU(inplace=True))

        # Branch 2
        self.block3 = nn.Sequential(  # [320, 1, 3000] -->[320, 128, 187]
            nn.Conv1d(in_channels=1, out_channels=128, kernel_size=8, stride=2, padding=0),
            nn.BatchNorm1d(128, affine=True),
            nn.ReLU(True),
            nn.MaxPool1d(8, stride=8),
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=8, stride=1, padding=0),
            nn.BatchNorm1d(128, affine=True),
            nn.ReLU(True),
            nn.MaxPool1d(kernel_size=4, stride=4),
            nn.Dropout(p=0.2),
        )
        self.block4 = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=32, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(32, affine=True),
            nn.ReLU(True),  # [320, 128, 743]
        )

        self.rnn2 = nn.LSTM(input_size=1440, hidden_size=256, num_layers=1, batch_first=True, bidirectional=True)

        # Branch 2: cross attention
        self.qcross2 = nn.Sequential(nn.Linear(512, 128), nn.ReLU(inplace=True))
        self.kcross2 = nn.Sequential(nn.Linear(512, 128), nn.ReLU(inplace=True))
        self.vcross2 = nn.Sequential(nn.Linear(512, 128), nn.ReLU(inplace=True))

        # after concat                 [320, 2, 128] --> [320, 32, 42]
        self.block5 = nn.Sequential(
            nn.Conv1d(in_channels=2, out_channels=32, kernel_size=5, stride=1, padding=1),
            nn.BatchNorm1d(32, affine=True),
            nn.ReLU(True),
            nn.MaxPool1d(kernel_size=3, stride=3),
            nn.Dropout(p=0.15),
        )

        self.block6 = nn.Sequential(   # [320, 128, 21] --> [320, 32, 17]
            nn.Conv1d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm1d(64, affine=True),
            nn.ReLU(True),
            nn.Conv1d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm1d(32, affine=True),
            nn.ReLU(True),
        )

        self.block7 = nn.Sequential(  # [320, 128, 45] --> [320, 32, 41]
            nn.Conv1d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm1d(64, affine=True),
            nn.ReLU(True),
            nn.Conv1d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm1d(32, affine=True),
            nn.ReLU(True),
            nn.Dropout(p=0.15),
        )

        self.block8 = nn.Sequential(  # [320, 32, 100] --> [320, 32, 33]
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(64, affine=True),
            nn.ReLU(True),
            nn.MaxPool1d(kernel_size=3, stride=3),
            nn.Conv1d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(32, affine=True),
            nn.ReLU(True),
            nn.Dropout(p=0.15),
            # nn.Flatten(),
        )

        self.rnn_dropout = nn.Dropout(p=0.2)
        self.fc_dropout = nn.Dropout(p=0.2)
        self.cross_attn1 = nn.Linear(128 + 128, 128)
        self.cross_attn2 = nn.Linear(128 + 128, 128)
        # self.fc = nn.Linear(400, self.stages)
        # self.output = nn.Softplus()
        # self.rnn3 = nn.LSTM(input_size=544, hidden_size=1024, num_layers=3, batch_first=True, bidirectional=True)
        # self.rnn4 = nn.LSTM(input_size=1312, hidden_size=1024, num_layers=3, batch_first=True, bidirectional=True)

    def forward(self, x):
        x = x.reshape(-1, 1, 3000)   # [320, 1, 3000]
        # Branch 1
        x1 = self.block1(x)          # [320, 128, 21]
        x1 = self.se1(x1)

        x1_bra = self.block2(x1)     # [320, 32, 21]
        x1_bra = self.flatten(x1_bra)  # [320， 672]
        x1_bra = x1_bra.reshape(-1, 20, x1_bra.shape[-1])  # [16, 20, 672]
        x1_bra, _ = self.rnn1(x1_bra)         # [16, 20, 512]
        x1_bra = self.rnn_dropout(x1_bra)
        x1_bra = x1_bra.reshape(-1, x1_bra.shape[-1])    # [320, 512]

        # Branch 2
        x2 = self.block3(x)          # [320, 128, 45]
        x2 = self.se2(x2)

        x2_bra = self.block4(x2)     # [320, 32, 45]
        x2_bra = self.flatten(x2_bra)        # [320， 1440]
        x2_bra = x2_bra.reshape(-1, 20, x2_bra.shape[-1])    # [16, 20， 1440]
        x2_bra, _ = self.rnn2(x2_bra)        # [16, 20, 512]
        x2_bra = self.rnn_dropout(x2_bra)
        x2_bra = x2_bra.reshape(-1, x2_bra.shape[-1])    # [320, 512]

        # cross attention
        cro_q1, cro_k1, cro_v1 = self.qcross1(x1_bra), self.kcross1(x1_bra), self.vcross1(x1_bra)
        cro_q2, cro_k2, cro_v2 = self.qcross2(x2_bra), self.kcross2(x2_bra), self.vcross2(x2_bra)

        q2_k1 = F.softmax(self.cross_attn2(torch.cat((cro_q2, cro_k1), 1)), dim=1)
        q2_kv1 = torch.mul(q2_k1, cro_v1)         # [320, 128]
        q2_kv1_ = torch.unsqueeze(q2_kv1, dim=1)  # [320, 1, 128]

        q1_k2 = F.softmax(self.cross_attn1(torch.cat((cro_q1, cro_k2), 1)), dim=1)
        q1_kv2 = torch.mul(q1_k2, cro_v2)         # [320, 128]
        q1_kv2_ = torch.unsqueeze(q1_kv2, dim=1)  # [320, 1, 128]

        x3 = torch.cat((q2_kv1_, q1_kv2_), 1)      # [320,  2, 128]
        x3 = self.block5(x3)                       # [320, 32, 42]
        x3 = self.se3(x3)

        # x1: [320, 128, 21]    x2: [320, 128, 45]    x3: [320, 32, 120]
        x1 = self.block6(x1)              # [320, 128, 21] --> [320, 32, 17]
        x2 = self.block7(x2)              # [320, 128, 45] --> [320, 32, 41]

        #x1 = self.flatten(x1)             # [320, 544]
        #x1 = x1.reshape(-1, 20, x1.shape[-1])      # [16, 20, 544]
        #x1, _ = self.rnn3(x1)                      # [16, 20, 2048]
        #x1 = x1.reshape(-1, 1, x1.shape[-1])       # [320, 1, 2048]


        x12 = torch.cat((x1, x2), 2)               # [320, 32, 58]
        x123 = torch.cat((x12, x3), 2)             # [320, 32, 100]

        x_feature = self.block8(x123)              # [320, 32, 100] --> [320, 32, 33]
        # x_feature = self.simAM(x_feature)

        return x_feature

class _MySleepNet_1Chan(nn.Module):
    def __init__(self, stages=5, hidden_size=128, seq_len=20, is_bidirectional=False, network="LSTM"):
        super(MySleepNet_1Chan, self).__init__()
        self.stages = stages
        self.flatten = nn.Flatten()
        self.Feature = Feature_1Chan()
        self.fc = nn.Linear(2000, self.stages)
        self.se = SELayer_1D(32)
        self.GCNet = GlobalContextBlock(32)
        self.output = nn.Softplus()
        self.rnn = nn.LSTM(input_size=1056, hidden_size=1000, num_layers=1, batch_first=True, bidirectional=True)

        self.block = nn.Sequential(  # [320, 32, 33]
            nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(32, affine=True),
            nn.ReLU(True),
            nn.Flatten(),
        )

    def forward(self, x):
        x = self.Feature(x)        # [320, 32, 33]
        # x = self.se(x)
        x = self.GCNet(x)
        x = self.block(x)          # [320, 1056]
        x = x.reshape(-1, 20, x.shape[-1])      # [16, 20, 1056]
        x, _ = self.rnn(x)                      # [16, 20, 2000]
        x = x.reshape(-1, x.shape[-1])          # [320, 2000]

        x = self.fc(x)
        x = self.output(x)
        return x



# ***************** 继续提升 2024.6.27 -v10  ******************* : 2*SE + SE + GCNet
class _Feature_1Chan(nn.Module):
    def __init__(self, hidden_size=128, seq_len=20, is_bidirectional=False, network="LSTM"):
        super(Feature_1Chan, self).__init__()
        self.flatten = nn.Flatten()
        self.simAM = SimAM()
        self.se1 = SELayer_1D(128)
        self.se2 = SELayer_1D(128)
        self.se3 = SELayer_1D(32)
        # Branch 1
        self.block1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=128, kernel_size=30, stride=4, padding=0),
            nn.BatchNorm1d(128, affine=True),
            nn.ReLU(True),
            nn.MaxPool1d(8, stride=8),
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=8, stride=1, padding=0),
            nn.BatchNorm1d(128, affine=True),
            nn.ReLU(True),
            nn.MaxPool1d(kernel_size=4, stride=4),
            nn.Dropout(p=0.2),
        )
        self.block2 = nn.Sequential(  # [320, 128, 21] -->[320, 32, 21]
            nn.Conv1d(in_channels=128, out_channels=32, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(32, affine=True),
            nn.ReLU(True),  # [320, 128, 743]
        )

        self.rnn1 = nn.LSTM(input_size=672, hidden_size=256, num_layers=1, batch_first=True, bidirectional=True)

        # Branch 1: cross attention
        self.qcross1 = nn.Sequential(nn.Linear(512, 128), nn.ReLU(inplace=True))
        self.kcross1 = nn.Sequential(nn.Linear(512, 128), nn.ReLU(inplace=True))
        self.vcross1 = nn.Sequential(nn.Linear(512, 128), nn.ReLU(inplace=True))

        # Branch 2
        self.block3 = nn.Sequential(  # [320, 1, 3000] -->[320, 128, 187]
            nn.Conv1d(in_channels=1, out_channels=128, kernel_size=8, stride=2, padding=0),
            nn.BatchNorm1d(128, affine=True),
            nn.ReLU(True),
            nn.MaxPool1d(8, stride=8),
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=8, stride=1, padding=0),
            nn.BatchNorm1d(128, affine=True),
            nn.ReLU(True),
            nn.MaxPool1d(kernel_size=4, stride=4),
            nn.Dropout(p=0.2),
        )
        self.block4 = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=32, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(32, affine=True),
            nn.ReLU(True),  # [320, 128, 743]
        )

        self.rnn2 = nn.LSTM(input_size=1440, hidden_size=256, num_layers=1, batch_first=True, bidirectional=True)

        # Branch 2: cross attention
        self.qcross2 = nn.Sequential(nn.Linear(512, 128), nn.ReLU(inplace=True))
        self.kcross2 = nn.Sequential(nn.Linear(512, 128), nn.ReLU(inplace=True))
        self.vcross2 = nn.Sequential(nn.Linear(512, 128), nn.ReLU(inplace=True))

        # after concat                 [320, 2, 128] --> [320, 32, 42]
        self.block5 = nn.Sequential(
            nn.Conv1d(in_channels=2, out_channels=32, kernel_size=5, stride=1, padding=1),
            nn.BatchNorm1d(32, affine=True),
            nn.ReLU(True),
            nn.MaxPool1d(kernel_size=3, stride=3),
            nn.Dropout(p=0.15),
        )

        self.block6 = nn.Sequential(   # [320, 128, 21] --> [320, 32, 17]
            nn.Conv1d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm1d(64, affine=True),
            nn.ReLU(True),
            nn.Conv1d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm1d(32, affine=True),
            nn.ReLU(True),
        )

        self.block7 = nn.Sequential(  # [320, 128, 45] --> [320, 32, 41]
            nn.Conv1d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm1d(64, affine=True),
            nn.ReLU(True),
            nn.Conv1d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm1d(32, affine=True),
            nn.ReLU(True),
            nn.Dropout(p=0.15),
        )

        self.block8 = nn.Sequential(  # [320, 32, 100] --> [320, 32, 33]
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(64, affine=True),
            nn.ReLU(True),
            nn.MaxPool1d(kernel_size=3, stride=3),
            nn.Conv1d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(32, affine=True),
            nn.ReLU(True),
            nn.Dropout(p=0.15),
            # nn.Flatten(),
        )

        self.rnn_dropout = nn.Dropout(p=0.2)
        self.fc_dropout = nn.Dropout(p=0.2)
        self.cross_attn1 = nn.Linear(128 + 128, 128)
        self.cross_attn2 = nn.Linear(128 + 128, 128)
        # self.fc = nn.Linear(400, self.stages)
        # self.output = nn.Softplus()
        # self.rnn3 = nn.LSTM(input_size=544, hidden_size=1024, num_layers=3, batch_first=True, bidirectional=True)
        # self.rnn4 = nn.LSTM(input_size=1312, hidden_size=1024, num_layers=3, batch_first=True, bidirectional=True)

    def forward(self, x):
        x = x.reshape(-1, 1, 3000)   # [320, 1, 3000]
        # Branch 1
        x1 = self.block1(x)          # [320, 128, 21]
        x1 = self.se1(x1)

        x1_bra = self.block2(x1)     # [320, 32, 21]
        x1_bra = self.flatten(x1_bra)  # [320， 672]
        x1_bra = x1_bra.reshape(-1, 20, x1_bra.shape[-1])  # [16, 20, 672]
        x1_bra, _ = self.rnn1(x1_bra)         # [16, 20, 512]
        x1_bra = self.rnn_dropout(x1_bra)
        x1_bra = x1_bra.reshape(-1, x1_bra.shape[-1])    # [320, 512]

        # Branch 2
        x2 = self.block3(x)          # [320, 128, 45]
        x2 = self.se2(x2)

        x2_bra = self.block4(x2)     # [320, 32, 45]
        x2_bra = self.flatten(x2_bra)        # [320， 1440]
        x2_bra = x2_bra.reshape(-1, 20, x2_bra.shape[-1])    # [16, 20， 1440]
        x2_bra, _ = self.rnn2(x2_bra)        # [16, 20, 512]
        x2_bra = self.rnn_dropout(x2_bra)
        x2_bra = x2_bra.reshape(-1, x2_bra.shape[-1])    # [320, 512]

        # cross attention
        cro_q1, cro_k1, cro_v1 = self.qcross1(x1_bra), self.kcross1(x1_bra), self.vcross1(x1_bra)
        cro_q2, cro_k2, cro_v2 = self.qcross2(x2_bra), self.kcross2(x2_bra), self.vcross2(x2_bra)

        q2_k1 = F.softmax(self.cross_attn2(torch.cat((cro_q2, cro_k1), 1)), dim=1)
        q2_kv1 = torch.mul(q2_k1, cro_v1)         # [320, 128]
        q2_kv1_ = torch.unsqueeze(q2_kv1, dim=1)  # [320, 1, 128]

        q1_k2 = F.softmax(self.cross_attn1(torch.cat((cro_q1, cro_k2), 1)), dim=1)
        q1_kv2 = torch.mul(q1_k2, cro_v2)         # [320, 128]
        q1_kv2_ = torch.unsqueeze(q1_kv2, dim=1)  # [320, 1, 128]

        x3 = torch.cat((q2_kv1_, q1_kv2_), 1)      # [320,  2, 128]
        x3 = self.block5(x3)                       # [320, 32, 42]
        x3 = self.se3(x3)

        # x1: [320, 128, 21]    x2: [320, 128, 45]    x3: [320, 32, 120]
        x1 = self.block6(x1)              # [320, 128, 21] --> [320, 32, 17]
        x2 = self.block7(x2)              # [320, 128, 45] --> [320, 32, 41]

        #x1 = self.flatten(x1)             # [320, 544]
        #x1 = x1.reshape(-1, 20, x1.shape[-1])      # [16, 20, 544]
        #x1, _ = self.rnn3(x1)                      # [16, 20, 2048]
        #x1 = x1.reshape(-1, 1, x1.shape[-1])       # [320, 1, 2048]


        x12 = torch.cat((x1, x2), 2)               # [320, 32, 58]
        x123 = torch.cat((x12, x3), 2)             # [320, 32, 100]

        x_feature = self.block8(x123)              # [320, 32, 100] --> [320, 32, 33]
        # x_feature = self.simAM(x_feature)

        return x_feature

class _MySleepNet_1Chan(nn.Module):
    def __init__(self, stages=5, hidden_size=128, seq_len=20, is_bidirectional=False, network="LSTM"):
        super(MySleepNet_1Chan, self).__init__()
        self.stages = stages
        self.flatten = nn.Flatten()
        self.Feature = Feature_1Chan()
        self.fc = nn.Linear(2000, self.stages)
        self.se = SELayer_1D(32)
        self.GCNet = GlobalContextBlock(32)
        self.output = nn.Softplus()
        self.rnn = nn.LSTM(input_size=1056, hidden_size=1000, num_layers=1, batch_first=True, bidirectional=True)

        self.block = nn.Sequential(  # [320, 32, 33]
            nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(32, affine=True),
            nn.ReLU(True),
            nn.Flatten(),
        )

    def forward(self, x):
        x = self.Feature(x)        # [320, 32, 33]
        # x = self.se(x)
        x = self.GCNet(x)
        x = self.block(x)          # [320, 1056]
        x = x.reshape(-1, 20, x.shape[-1])      # [16, 20, 1056]
        x, _ = self.rnn(x)                      # [16, 20, 2000]
        x = x.reshape(-1, x.shape[-1])          # [320, 2000]

        x = self.fc(x)
        x = self.output(x)
        return x



# ***************** 继续提升 2024.6.27 -v5  ******************* : 2*NonLocalBlockND + GCNet
class Feature_1Chan(nn.Module):
    def __init__(self, hidden_size=128, seq_len=20, is_bidirectional=False, network="LSTM"):
        super(Feature_1Chan, self).__init__()
        self.flatten = nn.Flatten()
        self.simAM = SimAM()
        self.se1 = NonLocalBlockND(128)
        self.se2 = NonLocalBlockND(128)
        # Branch 1
        self.block1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=128, kernel_size=30, stride=4, padding=0),
            nn.BatchNorm1d(128, affine=True),
            nn.ReLU(True),
            nn.MaxPool1d(8, stride=8),
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=8, stride=1, padding=0),
            nn.BatchNorm1d(128, affine=True),
            nn.ReLU(True),
            nn.MaxPool1d(kernel_size=4, stride=4),
            nn.Dropout(p=0.2),
        )
        self.block2 = nn.Sequential(  # [320, 128, 21] -->[320, 32, 21]
            nn.Conv1d(in_channels=128, out_channels=32, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(32, affine=True),
            nn.ReLU(True),  # [320, 128, 743]
        )

        self.rnn1 = nn.LSTM(input_size=672, hidden_size=256, num_layers=1, batch_first=True, bidirectional=True)

        # Branch 1: cross attention
        self.qcross1 = nn.Sequential(nn.Linear(512, 128), nn.ReLU(inplace=True))
        self.kcross1 = nn.Sequential(nn.Linear(512, 128), nn.ReLU(inplace=True))
        self.vcross1 = nn.Sequential(nn.Linear(512, 128), nn.ReLU(inplace=True))

        # Branch 2
        self.block3 = nn.Sequential(  # [320, 1, 3000] -->[320, 128, 187]
            nn.Conv1d(in_channels=1, out_channels=128, kernel_size=8, stride=2, padding=0),
            nn.BatchNorm1d(128, affine=True),
            nn.ReLU(True),
            nn.MaxPool1d(8, stride=8),
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=8, stride=1, padding=0),
            nn.BatchNorm1d(128, affine=True),
            nn.ReLU(True),
            nn.MaxPool1d(kernel_size=4, stride=4),
            nn.Dropout(p=0.2),
        )
        self.block4 = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=32, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(32, affine=True),
            nn.ReLU(True),  # [320, 128, 743]
        )

        self.rnn2 = nn.LSTM(input_size=1440, hidden_size=256, num_layers=1, batch_first=True, bidirectional=True)

        # Branch 2: cross attention
        self.qcross2 = nn.Sequential(nn.Linear(512, 128), nn.ReLU(inplace=True))
        self.kcross2 = nn.Sequential(nn.Linear(512, 128), nn.ReLU(inplace=True))
        self.vcross2 = nn.Sequential(nn.Linear(512, 128), nn.ReLU(inplace=True))

        # after concat                 [320, 2, 128] --> [320, 32, 42]
        self.block5 = nn.Sequential(
            nn.Conv1d(in_channels=2, out_channels=32, kernel_size=5, stride=1, padding=1),
            nn.BatchNorm1d(32, affine=True),
            nn.ReLU(True),
            nn.MaxPool1d(kernel_size=3, stride=3),
            nn.Dropout(p=0.15),
        )

        self.block6 = nn.Sequential(   # [320, 128, 21] --> [320, 32, 17]
            nn.Conv1d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm1d(64, affine=True),
            nn.ReLU(True),
            nn.Conv1d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm1d(32, affine=True),
            nn.ReLU(True),
        )

        self.block7 = nn.Sequential(  # [320, 128, 45] --> [320, 32, 41]
            nn.Conv1d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm1d(64, affine=True),
            nn.ReLU(True),
            nn.Conv1d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm1d(32, affine=True),
            nn.ReLU(True),
            nn.Dropout(p=0.15),
        )

        self.block8 = nn.Sequential(  # [320, 32, 100] --> [320, 32, 33]
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(64, affine=True),
            nn.ReLU(True),
            nn.MaxPool1d(kernel_size=3, stride=3),
            nn.Conv1d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(32, affine=True),
            nn.ReLU(True),
            nn.Dropout(p=0.15),
            # nn.Flatten(),
        )

        self.rnn_dropout = nn.Dropout(p=0.2)
        self.fc_dropout = nn.Dropout(p=0.2)
        self.cross_attn1 = nn.Linear(128 + 128, 128)
        self.cross_attn2 = nn.Linear(128 + 128, 128)
        # self.fc = nn.Linear(400, self.stages)
        # self.output = nn.Softplus()
        # self.rnn3 = nn.LSTM(input_size=544, hidden_size=1024, num_layers=3, batch_first=True, bidirectional=True)
        # self.rnn4 = nn.LSTM(input_size=1312, hidden_size=1024, num_layers=3, batch_first=True, bidirectional=True)

    def forward(self, x):
        x = x.reshape(-1, 1, 3000)   # [320, 1, 3000]
        # Branch 1
        x1 = self.block1(x)          # [320, 128, 21]
        x1 = self.se1(x1)

        x1_bra = self.block2(x1)     # [320, 32, 21]
        x1_bra = self.flatten(x1_bra)  # [320， 672]
        x1_bra = x1_bra.reshape(-1, 20, x1_bra.shape[-1])  # [16, 20, 672]
        x1_bra, _ = self.rnn1(x1_bra)         # [16, 20, 512]
        x1_bra = self.rnn_dropout(x1_bra)
        x1_bra = x1_bra.reshape(-1, x1_bra.shape[-1])    # [320, 512]

        # Branch 2
        x2 = self.block3(x)          # [320, 128, 45]
        x2 = self.se2(x2)

        x2_bra = self.block4(x2)     # [320, 32, 45]
        x2_bra = self.flatten(x2_bra)        # [320， 1440]
        x2_bra = x2_bra.reshape(-1, 20, x2_bra.shape[-1])    # [16, 20， 1440]
        x2_bra, _ = self.rnn2(x2_bra)        # [16, 20, 512]
        x2_bra = self.rnn_dropout(x2_bra)
        x2_bra = x2_bra.reshape(-1, x2_bra.shape[-1])    # [320, 512]

        # cross attention
        cro_q1, cro_k1, cro_v1 = self.qcross1(x1_bra), self.kcross1(x1_bra), self.vcross1(x1_bra)
        cro_q2, cro_k2, cro_v2 = self.qcross2(x2_bra), self.kcross2(x2_bra), self.vcross2(x2_bra)

        q2_k1 = F.softmax(self.cross_attn2(torch.cat((cro_q2, cro_k1), 1)), dim=1)
        q2_kv1 = torch.mul(q2_k1, cro_v1)         # [320, 128]
        q2_kv1_ = torch.unsqueeze(q2_kv1, dim=1)  # [320, 1, 128]

        q1_k2 = F.softmax(self.cross_attn1(torch.cat((cro_q1, cro_k2), 1)), dim=1)
        q1_kv2 = torch.mul(q1_k2, cro_v2)         # [320, 128]
        q1_kv2_ = torch.unsqueeze(q1_kv2, dim=1)  # [320, 1, 128]

        x3 = torch.cat((q2_kv1_, q1_kv2_), 1)      # [320,  2, 128]
        x3 = self.block5(x3)                       # [320, 32, 42]

        # x1: [320, 128, 21]    x2: [320, 128, 45]    x3: [320, 32, 120]
        x1 = self.block6(x1)              # [320, 128, 21] --> [320, 32, 17]
        x2 = self.block7(x2)              # [320, 128, 45] --> [320, 32, 41]

        #x1 = self.flatten(x1)             # [320, 544]
        #x1 = x1.reshape(-1, 20, x1.shape[-1])      # [16, 20, 544]
        #x1, _ = self.rnn3(x1)                      # [16, 20, 2048]
        #x1 = x1.reshape(-1, 1, x1.shape[-1])       # [320, 1, 2048]


        x12 = torch.cat((x1, x2), 2)               # [320, 32, 58]
        x123 = torch.cat((x12, x3), 2)             # [320, 32, 100]

        x_feature = self.block8(x123)              # [320, 32, 100] --> [320, 32, 33]
        # x_feature = self.simAM(x_feature)

        return x_feature

class MySleepNet_1Chan(nn.Module):
    def __init__(self, stages=5, hidden_size=128, seq_len=20, is_bidirectional=False, network="LSTM"):
        super(MySleepNet_1Chan, self).__init__()
        self.stages = stages
        self.flatten = nn.Flatten()
        self.Feature = Feature_1Chan()
        self.fc = nn.Linear(2000, self.stages)
        self.se = SELayer_1D(32)
        self.GCNet = GlobalContextBlock(32)
        self.output = nn.Softplus()
        self.rnn = nn.LSTM(input_size=1056, hidden_size=1000, num_layers=1, batch_first=True, bidirectional=True)

        self.block = nn.Sequential(  # [320, 32, 33]
            nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(32, affine=True),
            nn.ReLU(True),
            nn.Flatten(),
        )

    def forward(self, x):
        x = self.Feature(x)        # [320, 32, 33]
        # x = self.se(x)
        x = self.GCNet(x)
        x = self.block(x)          # [320, 1056]
        x = x.reshape(-1, 20, x.shape[-1])      # [16, 20, 1056]
        x, _ = self.rnn(x)                      # [16, 20, 2000]
        x = x.reshape(-1, x.shape[-1])          # [320, 2000]

        x = self.fc(x)
        x = self.output(x)
        return x

# ***************** 架构1  *****************
class MySleepNet_2Chan(nn.Module):
    def __init__(self, stages=5, hidden_size=128, seq_len=20, is_bidirectional=False, network="LSTM"):
        super(MySleepNet_2Chan, self).__init__()
        self.stages = stages
        self.flatten = nn.Flatten()
        self.Feature1 = Feature_1Chan()
        self.Feature2 = Feature_1Chan()
        self.fc = nn.Linear(2000, self.stages)
        self.se = SELayer_1D(32)
        self.GCNet = GlobalContextBlock(64)
        self.output = nn.Softplus()
        self.rnn = nn.LSTM(input_size=1056, hidden_size=1000, num_layers=1, batch_first=True, bidirectional=True)

        self.block1 = nn.Sequential(  # [320, 64, 33]
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64, affine=True),
            nn.ReLU(True),
            # nn.Flatten(),
        )
        self.block2 = nn.Sequential(  # [320, 64, 33]
            nn.Conv1d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(32, affine=True),
            nn.ReLU(True),
            nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(32, affine=True),
            nn.ReLU(True),
            nn.Flatten(),
            nn.Dropout(p=0.1),
        )

    def forward(self, x_ori):
        x1 = x_ori[:, 0, :, :]  # channel 1: [16, 20, 3000]
        x1 = x1.reshape(-1, 1, 3000)  # [320, 1, 3000]
        x2 = x_ori[:, 1, :, :]
        x2 = x2.reshape(-1, 1, 3000)

        x1 = self.Feature1(x1)        # [320, 32, 33]
        x2 = self.Feature2(x2)        # [320, 32, 33]
        x = torch.cat((x1, x2), 1)    # [320, 64, 33]
        x = self.block1(x)            # [320, 64, 33]
        # x = self.se(x)
        x = self.GCNet(x)
        x = self.block2(x)          # [320, 1056]

        x = x.reshape(-1, 20, x.shape[-1])      # [16, 20, 1056]
        x, _ = self.rnn(x)                      # [16, 20, 2000]
        x = x.reshape(-1, x.shape[-1])          # [320, 2000]

        x = self.fc(x)
        x = self.output(x)
        return x

# ***************** 架构2 ( 效果不如架构 1 ) *****************
class _MySleepNet_2Chan(nn.Module):
    def __init__(self, stages=5, hidden_size=128, seq_len=20, is_bidirectional=False, network="LSTM"):
        super(MySleepNet_2Chan, self).__init__()
        self.stages = stages
        self.flatten = nn.Flatten()
        self.Feature1 = Feature_1Chan()
        self.Feature2 = Feature_1Chan()
        self.fc = nn.Linear(4000, self.stages)
        self.se = SELayer_1D(32)
        self.GCNet1 = GlobalContextBlock(32)
        self.GCNet2 = GlobalContextBlock(32)
        self.output = nn.Softplus()
        self.rnn1 = nn.LSTM(input_size=1056, hidden_size=1000, num_layers=1, batch_first=True, bidirectional=True)
        self.rnn2 = nn.LSTM(input_size=1056, hidden_size=1000, num_layers=1, batch_first=True, bidirectional=True)

        self.block1 = nn.Sequential(  # [320, 32, 33]
            nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(32, affine=True),
            nn.ReLU(True),
            nn.Flatten(),
        )
        self.block2 = nn.Sequential(  # [320, 32, 33]
            nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(32, affine=True),
            nn.ReLU(True),
            nn.Flatten(),
        )

    def forward(self, x_ori):
        x1 = x_ori[:, 0, :, :]  # channel 1: [16, 20, 3000]
        x1 = x1.reshape(-1, 1, 3000)  # [320, 1, 3000]
        x2 = x_ori[:, 1, :, :]
        x2 = x2.reshape(-1, 1, 3000)

        x1 = self.Feature1(x1)        # [320, 32, 33]
        x1 = self.GCNet1(x1)
        x1 = self.block1(x1)          # [320, 1056]
        x1 = x1.reshape(-1, 20, x1.shape[-1])      # [16, 20, 1056]
        x1, _ = self.rnn1(x1)                      # [16, 20, 2000]
        x1 = x1.reshape(-1, x1.shape[-1])          # [320, 2000]

        x2 = self.Feature2(x2)        # [320, 32, 33]
        x2 = self.GCNet2(x2)
        x2 = self.block2(x2)          # [320, 1056]
        x2 = x2.reshape(-1, 20, x2.shape[-1])      # [16, 20, 1056]
        x2, _ = self.rnn2(x2)                      # [16, 20, 2000]
        x2 = x2.reshape(-1, x2.shape[-1])          # [320, 2000]

        x12 = torch.cat((x1, x2), 1)  # [320, 4000]
        x12 = self.fc(x12)
        x12 = self.output(x12)
        return x12



