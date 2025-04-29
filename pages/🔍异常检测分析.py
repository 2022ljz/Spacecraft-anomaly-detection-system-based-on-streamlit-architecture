import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import time
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.data import DataLoader
import numpy as np
from sklearn.preprocessing import StandardScaler
import os
import torch

'''MyModel部分代码.streamlit部分代码在593行'''
class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None):
        new_x, lm,lh,hm = self.attention(
            x, x, x,
            attn_mask=attn_mask
        )
        x = x + self.dropout(new_x)
        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm2(x + y), lm,lh,hm

class Encoder(nn.Module):
    def __init__(self, attn_layers, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.norm = norm_layer

    def forward(self, x, attn_mask=None):
        # x [B, L, D]
        lm_list = []
        lh_list = []
        hm_list = []
        for attn_layer in self.attn_layers:
            x, lm,lh,hm = attn_layer(x, attn_mask=attn_mask)
            lm_list.append(lm)
            lh_list.append(lh)
            hm_list.append(hm)

        if self.norm is not None:
            x = self.norm(x)

        return x, lm_list, lh_list, hm_list

class MyModel(nn.Module):
    def __init__(self, win_size, enc_in, c_out, d_model=256, n_heads=8, e_layers=3, d_ff=512,
                 dropout=0.0, activation='gelu', output_attention=True):
        super(MyModel, self).__init__()
        self.output_attention = output_attention
        # Encoding
        self.embedding = DataEmbedding(enc_in, d_model, dropout)
        encoder_self_att = FourierBlock(in_channels=d_model,
                                        out_channels=d_model,
                                        n_heads=n_heads,
                                        seq_len=win_size,
                                        modes=32
                                        )

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AutoCorrelationLayer(
                        encoder_self_att,
                        d_model, n_heads),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )

        self.projection = nn.Linear(d_model, c_out, bias=True)

    def forward(self, x):
        enc_out = self.embedding(x)
        enc_out, lm,lh,hm = self.encoder(enc_out)
        enc_out = self.projection(enc_out)
        for u in range(len(lm)):
            lm[u] = self.projection(lm[u])
            lh[u] = self.projection(lh[u])
            hm[u] = self.projection(hm[u])

        if self.output_attention:
            return enc_out,lm,lh,hm
        else:
            return enc_out  # [B, L, D]


def get_frequency_modes(seq_len, modes=64, mode_select_method='random'):
    modes = min(modes, seq_len // 2)
    if mode_select_method == 'random': #打乱
        index = list(range(0, seq_len // 2))
        np.random.shuffle(index)
        index = index[:modes]
    else:
        index = list(range(0, modes))
    index.sort() #保证升序输出
    return index

def split_frequencies(signal: torch.Tensor):
    B, H, E, L = signal.shape
    freq_len = L // 2 + 1
    fft_result = torch.fft.rfft(signal, dim=-1)  # 对 L 轴做傅里叶变换，结果形状: (B, H, E, L//2+1)
    power = fft_result.abs() ** 2  # 形状: (B, H, E, L//2+1)
    total_energy = power.sum(dim=-1, keepdim=True)  # 总能量，形状: (B, H, E, 1)
    cumulative_energy = power.cumsum(dim=-1)  # 按频率累积，形状: (B, H, E, L//2+1)
    energy_target = total_energy / 3 #计算能量划分阈值 (1/3 总能量)
    def find_split_indices(cum_energy, target_energy):
        """
        找到使得累积能量最接近 target_energy 的索引，保证每个部分至少包含一个频率
        """
        indices = (cum_energy >= target_energy).int().argmax(dim=-1)
        return indices.clamp(min=1)  # 最小值保证至少一个频率

    low_end = find_split_indices(cumulative_energy, energy_target)
    mid_end = find_split_indices(cumulative_energy, 2 * energy_target)
    # 生成掩码
    device = signal.device
    freq_indices = torch.arange(freq_len, device=device)[None, None, None, :]  # (1,1,1,L//2+1)

    low_mask = freq_indices < low_end.unsqueeze(-1) #低频设置为True
    mid_mask = (freq_indices >= low_end.unsqueeze(-1)) & (freq_indices < mid_end.unsqueeze(-1)) #中频设置为True
    high_mask = freq_indices >= mid_end.unsqueeze(-1) #高频设置为True

    # 生成三种组合
    low_mid = fft_result * (low_mask | mid_mask)+0.3*fft_result*high_mask #这个地方还是复数，不能单纯乘个0.3？
    low_high = fft_result * (low_mask | high_mask)+0.3*fft_result*mid_mask
    high_mid = fft_result * (high_mask | mid_mask)+0.3*fft_result*low_mask

    return low_mid, low_high, high_mid


class sparseKernelFT1d(nn.Module):
    def __init__(self,in_dim,out_dim,H,L):
        super(sparseKernelFT1d, self).__init__()
        self.scale = (1 / (in_dim * out_dim))
        self.weights1 = nn.Parameter(self.scale * torch.rand(H, in_dim//H, out_dim//H,L//2+1,dtype=torch.float))
        self.weights2 = nn.Parameter(self.scale * torch.rand(H, in_dim//H, out_dim//H,L//2+1,dtype=torch.float))
        self.weights1.requires_grad = True
        self.weights2.requires_grad = True

    def compl_mul1d(self, order, x, weights):
        x_flag = True
        w_flag = True
        if not torch.is_complex(x):
            x_flag = False
            x = torch.complex(x, torch.zeros_like(x).to(x.device))
        if not torch.is_complex(weights):
            w_flag = False
            weights = torch.complex(weights, torch.zeros_like(weights).to(weights.device))
        if x_flag or w_flag:
            return torch.complex(torch.einsum(order, x.real, weights.real) - torch.einsum(order, x.imag, weights.imag),
                                 torch.einsum(order, x.real, weights.imag) + torch.einsum(order, x.imag, weights.real))
        else:
            return torch.einsum(order, x.real, weights.real)

    def forward(self, x):#传入B H E L//2+1
        B,H,E,L = x.shape
        out_ft = self.compl_mul1d("bhil,hiol->bhol", x,torch.complex(self.weights1, self.weights2))
        x = torch.fft.irfft(out_ft, n=(x.size(-1)-1)*2) #B H E L
        return x


class FourierBlock(nn.Module):
    def __init__(self, in_channels, out_channels, n_heads, seq_len, modes=0, mode_select_method='random'):
        super(FourierBlock, self).__init__()
        # get modes on frequency domain
        self.index = get_frequency_modes(seq_len, modes=modes, mode_select_method=mode_select_method) #从中随机抽取频率分量

        self.n_heads = n_heads
        self.scale = (1 / (in_channels * out_channels)) #缩放分子
        #初始化复数的实部和虚部。这个人造复数实际上是为了更好的调整被转化成频域信号的X的幅度和相位，通过这两个可学习的参数实现自动优化。
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(self.n_heads, in_channels // self.n_heads, out_channels // self.n_heads,
                                    len(self.index), dtype=torch.float)) #H D D L
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(self.n_heads, in_channels // self.n_heads, out_channels // self.n_heads,
                                    len(self.index), dtype=torch.float))
        self.A = sparseKernelFT1d(in_channels,out_channels,n_heads,seq_len)
        self.B = sparseKernelFT1d(in_channels, out_channels, n_heads, seq_len)
        self.C = sparseKernelFT1d(in_channels, out_channels, n_heads, seq_len)

    # 复数乘法模块
    def compl_mul1d(self, order, x, weights):
        x_flag = True #用于标记是否是原生复数张量
        w_flag = True
        if not torch.is_complex(x):
            x_flag = False
            x = torch.complex(x, torch.zeros_like(x).to(x.device)) ## 将实数转换为复数：实部=原值，虚部=全0
        if not torch.is_complex(weights):
            w_flag = False
            weights = torch.complex(weights, torch.zeros_like(weights).to(weights.device))
        if x_flag or w_flag: # 任意一个输入原本是复数，复数乘法公式：(a+bi)(c+di) = (ac-bd) + (ad+bc)i
            return torch.complex(torch.einsum(order, x.real, weights.real) - torch.einsum(order, x.imag, weights.imag),
                                 torch.einsum(order, x.real, weights.imag) + torch.einsum(order, x.imag, weights.real))
        else: #都是实数
            return torch.einsum(order, x.real, weights.real)

    def forward(self, q, k, v):
        B, L, H, E = q.shape
        x = q.permute(0, 2, 3, 1) #B H E L
        x_ft = torch.fft.rfft(x, dim=-1) #B, H, E, L // 2 + 1
        lm,lh,hm = split_frequencies(x)
        out_ft = torch.zeros(B, H, E, L // 2 + 1, device=x.device, dtype=torch.cfloat)
        for wi, i in enumerate(self.index): #enumerate 函数会返回一个包含每个元素的索引和该元素本身的元组。
            if i >= x_ft.shape[3] or wi >= out_ft.shape[3]:
                continue
            out_ft[:, :, :, wi] = self.compl_mul1d("bhi,hio->bho", x_ft[:, :, :, i],
                                                   torch.complex(self.weights1, self.weights2)[:, :, :, wi])
        x = torch.fft.irfft(out_ft, n=x.size(-1)) #B H E L
        lm = self.A(lm)
        lh = self.B(lh)
        hm = self.C(hm)
        return x,lm,lh,hm


class AutoCorrelationLayer(nn.Module):
    def __init__(self, correlation, d_model, n_heads, d_keys=None,
                 d_values=None):
        super(AutoCorrelationLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_correlation = correlation
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out,lm,lh,hm = self.inner_correlation(
            queries,
            keys,
            values,
        )
        out = out.view(B, L, -1)
        lm = lm.view(B, L, -1)
        lh = lh.view(B, L, -1)
        hm = hm.view(B, L, -1)

        return self.out_projection(out),self.out_projection(lm),self.out_projection(lh),self.out_projection(hm)

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x


class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, dropout=0.0):
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.value_embedding(x) + self.position_embedding(x)
        return self.dropout(x)


class MSLSegLoader(object):
    def __init__(self, data_path, win_size, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(data_path + "/MSL_train.npy")
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(data_path + "/MSL_test.npy")
        self.test = self.scaler.transform(test_data)

        self.train = data
        self.val = self.test
        self.test_labels = np.load(data_path + "/MSL_test_label.npy")

    def __len__(self):

        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


class SMAPSegLoader(object):
    def __init__(self, data_path, win_size, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(data_path + "/SMAP_train.npy")
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(data_path + "/SMAP_test.npy")
        self.test = self.scaler.transform(test_data)

        self.train = data
        self.val = self.test
        self.test_labels = np.load(data_path + "/SMAP_test_label.npy")

    def __len__(self):

        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


def get_loader_segment(data_path, batch_size, win_size=100, step=100, mode='train', dataset='KDD'):
    if (dataset == 'MSL'):
        dataset = MSLSegLoader(data_path, win_size, 1, mode)
    elif (dataset == 'SMAP'):
        dataset = SMAPSegLoader(data_path, win_size, 1, mode)

    shuffle = False
    if mode == 'train':
        shuffle = True

    data_loader = DataLoader(dataset=dataset,
                             batch_size=batch_size,
                             shuffle=shuffle,
                             num_workers=0)
    return data_loader


def test(dataset):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_loader = get_loader_segment(os.path.join('dataset/', dataset), batch_size=256, win_size=100,
                                      mode='train',
                                      dataset=dataset)
    thre_loader = get_loader_segment(os.path.join('dataset/', dataset), batch_size=256, win_size=100,
                                     mode='thre',
                                     dataset=dataset)
    checkpoint_path = os.path.join(
        'checkpoints',
        f"{dataset}_checkpoint.pth"
    )
    if dataset == 'MSL':
        d = 55
        anormly_ratio = 0.9
    elif dataset == 'SMAP':
        d = 25
        anormly_ratio = 1.1
    model = MyModel(win_size=100, enc_in=d, c_out=d, e_layers=3)
    checkpoint = torch.load(os.path.join(checkpoint_path),map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    l_num = checkpoint['l_num']
    m_num = checkpoint['m_num']
    h_num = checkpoint['h_num']

    model.eval()
    num_list = [l_num, m_num, h_num]
    num_list = sorted(num_list)
    adjust = (num_list[2] + num_list[1] - 2 * num_list[0]) // 2

    criterion = nn.MSELoss(reduction='none')

    # (1) stastic on the train set
    attens_energy = []
    for i, (input_data, labels) in enumerate(train_loader):
        input = input_data.float().to(device)
        output, lm, lh, hm = model(input)

        lm_loss1 = 0.0
        lh_loss1 = 0.0
        hm_loss1 = 0.0

        for u in range(len(lm)):
            lm_loss1 += torch.einsum('bld,bsd->bls', input, lm[u])
            lh_loss1 += torch.einsum('bld,bsd->bls', input, lh[u])
            hm_loss1 += torch.einsum('bld,bsd->bls', input, hm[u])

        lm_loss1 = lm_loss1 / len(lm) * (l_num + m_num) * (1 / h_num) * adjust
        lh_loss1 = lh_loss1 / len(lh) * (l_num + h_num) * (1 / m_num) * adjust
        hm_loss1 = hm_loss1 / len(hm) * (h_num + m_num) * (1 / l_num) * adjust
        loss2 = lm_loss1 + lh_loss1 + hm_loss1

        loss = torch.mean(criterion(input, output), dim=-1)

        metric = torch.softmax(torch.mean(-loss2, dim=-1), dim=-1)
        cri = metric * loss
        cri = cri.detach().cpu().numpy()
        attens_energy.append(cri)

    attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
    train_energy = np.array(attens_energy)

    # (2) find the threshold
    attens_energy = []
    for i, (input_data, labels) in enumerate(thre_loader):
        input = input_data.float().to(device)
        output, lm, lh, hm = model(input)

        lm_loss1 = 0.0
        lh_loss1 = 0.0
        hm_loss1 = 0.0

        for u in range(len(lm)):
            lm_loss1 += torch.einsum('bld,bsd->bls', input, lm[u])
            lh_loss1 += torch.einsum('bld,bsd->bls', input, lh[u])
            hm_loss1 += torch.einsum('bld,bsd->bls', input, hm[u])

        lm_loss1 = lm_loss1 / len(lm) * (l_num + m_num) * (1 / h_num) * adjust
        lh_loss1 = lh_loss1 / len(lh) * (l_num + h_num) * (1 / m_num) * adjust
        hm_loss1 = hm_loss1 / len(hm) * (h_num + m_num) * (1 / l_num) * adjust
        loss2 = lm_loss1 + lh_loss1 + hm_loss1

        loss = torch.mean(criterion(input, output), dim=-1)

        metric = torch.softmax(torch.mean(-loss2, dim=-1), dim=-1)
        cri = metric * loss
        cri = cri.detach().cpu().numpy()
        attens_energy.append(cri)

    attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
    test_energy = np.array(attens_energy)
    combined_energy = np.concatenate([train_energy, test_energy], axis=0)

    thresh = np.percentile(combined_energy, 100 - anormly_ratio)

    # (3) evaluation on the test set
    test_labels = []
    attens_energy = []
    for i, (input_data, labels) in enumerate(thre_loader):
        input = input_data.float().to(device)
        output, lm, lh, hm = model(input)

        lm_loss1 = 0.0
        lh_loss1 = 0.0
        hm_loss1 = 0.0

        for u in range(len(lm)):
            lm_loss1 += torch.einsum('bld,bsd->bls', input, lm[u])
            lh_loss1 += torch.einsum('bld,bsd->bls', input, lh[u])
            hm_loss1 += torch.einsum('bld,bsd->bls', input, hm[u])

        lm_loss1 = lm_loss1 / len(lm) * (l_num + m_num) * (1 / h_num) * adjust
        lh_loss1 = lh_loss1 / len(lh) * (l_num + h_num) * (1 / m_num) * adjust
        hm_loss1 = hm_loss1 / len(hm) * (h_num + m_num) * (1 / l_num) * adjust
        loss2 = lm_loss1 + lh_loss1 + hm_loss1

        loss = torch.mean(criterion(input, output), dim=-1)

        metric = torch.softmax(torch.mean(-loss2, dim=-1), dim=-1)
        cri = metric * loss
        cri = cri.detach().cpu().numpy()
        attens_energy.append(cri)
        test_labels.append(labels)

    attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
    test_labels = np.concatenate(test_labels, axis=0).reshape(-1)
    test_energy = np.array(attens_energy)
    test_labels = np.array(test_labels)

    pred = (test_energy > thresh).astype(int) #1是异常
    gt = test_labels.astype(int)

    anomaly_state = False
    for i in range(len(gt)):
        if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
            anomaly_state = True
            for j in range(i, 0, -1):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
            for j in range(i, len(gt)):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
        elif gt[i] == 0:
            anomaly_state = False
        if anomaly_state:
            pred[i] = 1

    pred = np.array(pred)
    gt = np.array(gt)

    from sklearn.metrics import precision_recall_fscore_support
    from sklearn.metrics import accuracy_score
    accuracy = accuracy_score(gt, pred)
    precision, recall, f_score, support = precision_recall_fscore_support(gt, pred,
                                                                          average='binary')

    return (float("{:.4f}".format(accuracy)),float("{:.4f}".format(precision)),float("{:.4f}".format(recall)),float("{:.4f}".format(f_score)),pred)







'''streamlit部分代码'''

st.set_page_config(page_title="异常检测分析", page_icon="🔍", layout="wide")

st.title('🔍异常检测分析')
cols = st.columns([0.8,0.2])
with cols[0]:
    st.subheader('功能介绍')
    st.markdown('''
    - 在这里用户可以使用模型MyModel对航天数据集SMAP、MSL进行异常检测并得到可视化结果。注意，由于两个航天数据集中数据点个数过多，因此采用窗口展示可视化图像，避免数据点全部挤在一起。用户可以在侧边栏处选择想要异常检测的数据集，然后选择展示窗口的长度，以及需要异常检测的数据范围，点击“应用配置”按钮后稍作等待，便可以得到异常检测结果。
    - 在左边展示的检测指标均为基于真实的数据标签计算得到。用户每一次点击“应用配置”，MyModel模型都会重新对用户所选择的数据集进行异常检测，因此每次得到的检测指标可能会有细微的差别。
    - 在右边展示的可视化图像中已经用红点将检测的异常点标出。用户可以双击图例来更好的观察整体数据和异常点。如：双击“异常点”便可以得到只含有红色异常点的图像。
    - 本页面支持对异常检测后的数据进行下载，用户点击“下载检测结果”即可选择下载。下载的文件为csv文件，文件保留了数据集的全部维度并在最后添加列“label”,标注为1的表示异常点。
    ''')
st.markdown("---")

# 数据集路径配置（保持不变）
DATASET_PATHS = {
    "SMAP": "dataset/SMAP/SMAP_test.npy",
    "MSL": "dataset/MSL/MSL_test.npy"
}

# 初始化session状态
if 'anomaly_plot_created' not in st.session_state:
    st.session_state.anomaly_plot_created = False
if 'current_anomaly_pos' not in st.session_state:
    st.session_state.current_anomaly_pos = 0
if 'anomaly_progress' not in st.session_state:
    st.session_state.anomaly_progress = 0
if 'anomaly_loading' not in st.session_state:
    st.session_state.anomaly_loading = False
if 'anomaly_params' not in st.session_state:
    st.session_state.anomaly_params = {
        "dataset": "SMAP",
        "window_size": 1000,
        "start_idx": 0,
        "end_idx": 10000,
        "detection_start": 0,
        "detection_end": 10000
    }
if 'detection_results' not in st.session_state:
    st.session_state.detection_results = None

# 侧边栏控件（卡片式布局）
with st.sidebar:
    st.markdown("### 🛠️ 配置面板")

    with st.expander("📂 数据源设置", expanded=True):
        selected_dataset = st.selectbox(
            "选择数据集",
            options=["SMAP", "MSL"],
            help="SMAP: 土壤湿度卫星数据 | MSL: 火星科学实验室数据"
        )

    with st.expander("⚙️ 可视化参数", expanded=True):
        try:
            current_data = np.load(DATASET_PATHS[selected_dataset], allow_pickle=True)
            current_real_data = current_data[:, 0]
            current_max_length = len(current_real_data)
        except Exception as e:
            st.error(f"🚨 数据集加载失败: {str(e)}")
            st.stop()

        selected_window_size = st.slider(
            "窗口长度",
            min_value=100,
            max_value=5000,
            value=1000,
            step=100,
            help="可视化窗口包含的数据点数"
        )

        MAX_RANGE = current_max_length - 1
        detection_range = st.slider(
            "数据范围",
            min_value=0,
            max_value=MAX_RANGE,
            value=(
                min(st.session_state.anomaly_params["start_idx"], MAX_RANGE),
                min(st.session_state.anomaly_params["end_idx"], MAX_RANGE)
            ),
            step=1,
            format="%d"
        )

    apply_button = st.button(
        "🚀 应用配置",
        use_container_width=True,
        type="primary"
    )

    # 应用按钮独立显示
    if apply_button:
        if detection_range[0] >= detection_range[1]:
            st.error("起始位置不能大于等于结束位置！")
            st.stop()

        st.session_state.anomaly_params = {
            "dataset": selected_dataset,
            "window_size": selected_window_size,
            "start_idx": detection_range[0],
            "end_idx": detection_range[1],
            "detection_start": detection_range[0],
            "detection_end": detection_range[1]
        }
        st.session_state.current_anomaly_pos = detection_range[0]
        st.session_state.anomaly_loading = True
        st.session_state.anomaly_plot_created = False
        st.rerun()


# 处理加载状态
progress_placeholder = st.empty()
if st.session_state.anomaly_loading:
    with progress_placeholder.container():
        # 显示加载转圈
        with st.spinner("🚀 正在进行异常检测并将结果可视化，请稍候..."):
            try:
                results = test(st.session_state.anomaly_params["dataset"])

                # 获取原始数据集总长度
                original_data = np.load(
                    DATASET_PATHS[st.session_state.anomaly_params["dataset"]],
                    allow_pickle=True
                )
                original_length = original_data.shape[0]

                # 创建全零数组（与原始数据长度对齐）
                aligned_flags = np.zeros(original_length, dtype=int)

                # 将检测结果填入前部
                detected_length = len(results[4])
                aligned_flags[:detected_length] = results[4]

                st.session_state.detection_results = {
                    "metrics": results[:4],
                    "anomaly_flags": aligned_flags
                }
            except Exception as e:
                st.error(f"异常检测失败: {str(e)}")
                st.stop()
            st.session_state.anomaly_plot_created = True
    # 关闭加载状态并清空占位符
    st.session_state.anomaly_loading = False
    progress_placeholder.empty()
    st.rerun()

# 主显示区域布局优化
if st.session_state.anomaly_plot_created and st.session_state.detection_results:
    try:
        # 加载数据（保持不变）
        applied_data = np.load(DATASET_PATHS[st.session_state.anomaly_params["dataset"]], allow_pickle=True)
        applied_real_data = applied_data[:, 0]

        # 获取检测结果（保持不变）
        accuracy, precision, recall, f_score = st.session_state.detection_results["metrics"]
        anomaly_flags = st.session_state.detection_results["anomaly_flags"]

        # 使用3:7的列宽比例
        main_col1, main_col2 = st.columns([3, 7])

        with main_col1:
            # 紧凑型指标布局
            st.subheader("📊 检测指标")

            # 第一行指标
            row1_col1, row1_col2 = st.columns(2)
            with row1_col1:
                st.metric("准确率", f"{accuracy:.2%}", help="正确识别的样本比例")
            with row1_col2:
                st.metric("精确率", f"{precision:.2%}", help="阳性预测准确率")

            # 第二行指标
            row2_col1, row2_col2 = st.columns(2)
            with row2_col1:
                st.metric("召回率", f"{recall:.2%}", help="异常检出率")
            with row2_col2:
                st.metric("F1分数", f"{f_score:.2%}", help="综合评估指标")

            # 统计信息卡片
            with st.expander("📈 异常统计", expanded=True):
                anomaly_count = np.sum(anomaly_flags)
                total_points = len(anomaly_flags)
                st.write(f"""
                **数据集规模**  
                • 总数据点：`{total_points:,}`  
                • 检测范围：`{st.session_state.anomaly_params['start_idx']}-{st.session_state.anomaly_params['end_idx']}`

                **异常分布**  
                • 异常数量：`{anomaly_count:,}`  
                • 异常密度：`{anomaly_count / (st.session_state.anomaly_params['end_idx'] - st.session_state.anomaly_params['start_idx']):.2%}`
                """)

        with main_col2:
            # 图表增强显示
            st.subheader("📈 异常检测结果可视化")

            # 获取当前窗口参数
            start_idx = st.session_state.anomaly_params["start_idx"]
            end_idx = st.session_state.anomaly_params["end_idx"]
            window_size = st.session_state.anomaly_params["window_size"]
            current_pos = st.session_state.current_anomaly_pos

            # 创建响应式图表容器
            with st.container(height=680):
                fig = go.Figure()

                # 动态计算可视范围
                visible_start = max(start_idx, current_pos - window_size // 4)
                visible_end = min(end_idx, current_pos + window_size + window_size // 4)

                # 主数据序列
                fig.add_trace(go.Scattergl(
                    x=np.arange(visible_start, visible_end),
                    y=applied_real_data[visible_start:visible_end],
                    mode='lines',
                    name='传感器数据',
                    line=dict(width=1)
                ))

                # 异常点标注
                anomaly_indices = np.where(anomaly_flags == 1)[0]
                visible_anomalies = anomaly_indices[
                    (anomaly_indices >= visible_start) &
                    (anomaly_indices <= visible_end)
                    ]
                fig.add_trace(go.Scattergl(
                    x=visible_anomalies,
                    y=applied_real_data[visible_anomalies],
                    mode='markers',
                    name='异常点',
                    marker=dict(color='#FF2B2B', size=5, opacity=0.6)
                ))

                # 可视化增强
                fig.update_layout(
                    height=600,
                    template="plotly_dark",
                    margin=dict(l=20, r=20, t=40, b=20),
                    legend=dict(orientation="h", y=1.1),
                    xaxis=dict(
                        range=[current_pos, current_pos + window_size],
                        autorange=False,
                        showspikes=True,
                        spikethickness=1
                    ),
                    yaxis=dict(
                        showspikes=True,
                        spikethickness=1
                    ),
                    hovermode="x unified"
                )

                # 添加辅助线
                fig.add_vline(
                    x=current_pos + window_size / 2,
                    line=dict(color="white", dash="dot", width=1),
                )

                st.plotly_chart(fig, use_container_width=True)

            # 窗口控制条
            min_pos = start_idx
            max_pos = end_idx - window_size  # 理论最大值
            if max_pos < min_pos:  # 处理窗口过大的边界情况
                max_pos = min_pos

            # 确保当前位置不越界
            current_pos = st.session_state.current_anomaly_pos
            if current_pos > max_pos:
                current_pos = max_pos
                st.session_state.current_anomaly_pos = current_pos

            # 更新滑块
            new_pos = st.slider(
                "调整观察窗口位置",
                min_value=min_pos,
                max_value=max_pos,
                value=current_pos,  # 使用校正后的位置
                step=max(window_size // 10, 1),  # 保证步长至少为1
                key='anomaly_window_slider',
                label_visibility="collapsed"
            )
            if new_pos != current_pos:
                st.session_state.current_anomaly_pos = new_pos
                st.rerun()

            # 异常检测页面的数据下载功能
            with st.expander("📥 下载检测结果"):
                col_anomaly1, col_anomaly2 = st.columns(2)

                # 下载完整标注数据
                with col_anomaly1:
                    try:
                        # 加载完整原始数据
                        full_data = np.load(
                            DATASET_PATHS[st.session_state.anomaly_params["dataset"]],
                            allow_pickle=True
                        )

                        # 构建规范列名（自动处理多维度数据）
                        num_columns = full_data.shape[1] if len(full_data.shape) > 1 else 1
                        columns = ["value"] + [f"col_{i}" for i in range(1, num_columns)]

                        # 创建DataFrame
                        if len(full_data.shape) == 1:
                            df_full = pd.DataFrame(full_data, columns=["value"])
                        else:
                            df_full = pd.DataFrame(full_data, columns=columns)

                        # 添加标签列（确保标签长度匹配）
                        df_full["label"] = np.array(anomaly_flags).astype(int)

                        # 生成CSV
                        full_csv = df_full.to_csv(index=False).encode('utf-8')

                        st.download_button(
                            label="下载完整标注数据",
                            data=full_csv,
                            file_name=f"{st.session_state.anomaly_params['dataset']}_FULL_LABELED.csv",
                            mime="text/csv",
                            use_container_width=True,
                            help="包含完整数据及异常标注（最后一列为标签）"
                        )
                    except Exception as e:
                        st.error(f"完整标注数据生成失败: {str(e)}")

                # 下载范围标注数据
                with col_anomaly2:
                    try:
                        original_data = np.load(
                            DATASET_PATHS[st.session_state.anomaly_params["dataset"]],
                            allow_pickle=True
                        )
                        # 获取当前范围数据
                        start_idx = st.session_state.anomaly_params["start_idx"]
                        end_idx = st.session_state.anomaly_params["end_idx"]
                        data_slice = original_data[start_idx:end_idx]

                        # 自动处理数据维度（支持单列和多列）
                        num_columns = data_slice.shape[1] if len(data_slice.shape) > 1 else 1
                        columns = ["value"] + [f"col_{i}" for i in range(1, num_columns)]

                        # 创建带规范列名的DataFrame
                        if len(data_slice.shape) == 1:
                            df_slice = pd.DataFrame(data_slice, columns=["value"])
                        else:
                            df_slice = pd.DataFrame(data_slice, columns=columns)

                        # 添加标签列（确保索引对齐）
                        df_slice["label"] = anomaly_flags[start_idx:end_idx].astype(int)

                        # 生成CSV
                        slice_csv = df_slice.to_csv(index=False).encode('utf-8')

                        st.download_button(
                            label="下载范围标注数据",
                            data=slice_csv,
                            file_name=f"{st.session_state.anomaly_params['dataset']}_LABELED_{start_idx}-{end_idx}.csv",
                            mime="text/csv",
                            use_container_width=True,
                            help=f"标注数据范围：{start_idx}-{end_idx}（含{len(df_slice)}条记录）"
                        )
                    except Exception as e:
                        st.error(f"范围标注数据生成失败: {str(e)}")

    except Exception as e:
        st.error(f"显示错误: {str(e)}")
        st.stop()
elif st.session_state.anomaly_plot_created:
    st.warning("检测结果加载失败，请重新尝试")
else:
    st.info("ℹ️ 请先在侧边栏完成参数配置并点击【应用配置】")

