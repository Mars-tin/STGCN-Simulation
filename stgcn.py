import math
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F


class Align(nn.Module):
    """
    Alignment Layer.
    If the size of input channel is less than the output,
    pad x to the same size of output channel.
    """
    def __init__(self, c_in, c_out):
        """
        :param c_in: int, size of input channel.
        :param c_out: int, size of output channel.
        """
        super(Align, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        if c_in > c_out:
            self.conv1x1 = nn.Conv2d(c_in, c_out, 1)

    def forward(self, x):
        """
        :param x: tensor, [batch_size, c_in, n_his, n_route].
        :return: tensor, [batch_size, c_out, n_his, n_route].
        """
        if self.c_in > self.c_out:
            return self.conv1x1(x)
        if self.c_in < self.c_out:
            return F.pad(x, [0, 0, 0, 0, 0, self.c_out - self.c_in, 0, 0])
        return x


class TemporalConv(nn.Module):
    """
    Temporal convolution layer.
    """
    def __init__(self, kt, c_in, c_out, act="relu"):
        """
        Temporal convolution layer.
        :param kt: int, kernel size of temporal convolution.
        :param c_in: int, size of input channel.
        :param c_out: int, size of output channel.
        :param act: str, activation function.
        """
        super(TemporalConv, self).__init__()
        self.kt = kt
        self.act = act
        self.c_out = c_out
        self.align = Align(c_in, c_out)
        if self.act == "GLU":
            self.conv = nn.Conv2d(c_in, c_out * 2, (kt, 1), 1)
        else:
            self.conv = nn.Conv2d(c_in, c_out, (kt, 1), 1)

    def forward(self, x):
        """
        :param x: tensor, [batch_size, c_in, n_his, n_route].
        :return: tensor, [batch_size, c_out, n_his-kt+1, n_route].
        """
        x_in = self.align(x)[:, :, self.kt - 1:, :]
        if self.act == "GLU":
            x_conv = self.conv(x)
            return (x_conv[:, :self.c_out, :, :] + x_in) * torch.sigmoid(x_conv[:, self.c_out:, :, :])
        if self.act == "sigmoid":
            return torch.sigmoid(self.conv(x) + x_in)
        return torch.relu(self.conv(x) + x_in)


class SpatialConv(nn.Module):
    """
    Spatial graph convolution layer.
    """
    def __init__(self, ks, c, Lk):
        """
        :param ks: int, kernel size of spatial convolution.
        :param c: int, size of input channel and output channel: c = c_in = c_out.
        """
        super(SpatialConv, self).__init__()
        self.Lk = Lk
        self.theta = nn.Parameter(torch.FloatTensor(c, c, ks))
        self.b = nn.Parameter(torch.FloatTensor(1, c, 1, 1))
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.theta, a=math.sqrt(5))
        fan_in, _ = init._calculate_fan_in_and_fan_out(self.theta)
        bound = 1 / math.sqrt(fan_in)
        init.uniform_(self.b, -bound, bound)

    def forward(self, x):
        """
        :param x: tensor, [batch_size, c_in, n_his-kt+1, n_route].
        :return: tensor, [batch_size, c_out, n_his-kt+1, n_route].
        In PeMSD7 dataset, m = n = n_route, i = o = c.
        """
        x_c = torch.einsum("knm,bitm->bitkn", self.Lk, x)
        x_gc = torch.einsum("iok,bitkn->botn", self.theta, x_c) + self.b
        return torch.relu(x_gc + x)


class STConvBlock(nn.Module):
    """
    Spatial-temporal convolutional block.
    Contains two temporal gated convolution layers
    and one spatial graph convolution layer in the middle.
    """
    def __init__(self, ks, kt, n, c, p, Lk):
        """
        :param ks: int, kernel size of spatial convolution.
        :param kt: int, kernel size of temporal convolution.
        :param c: list, channel configs of a single ST-Conv block. [c_in, c_mid, c_out]
        :param n: int, number of monitor stations.
        :param p: placeholder, rate of dropout.
        """
        super(STConvBlock, self).__init__()
        self.tconv1 = TemporalConv(kt, c[0], c[1], "GLU")
        self.sconv = SpatialConv(ks, c[1], Lk)
        self.tconv2 = TemporalConv(kt, c[1], c[2])
        self.ln = nn.LayerNorm([n, c[2]])
        self.dropout = nn.Dropout(p)

    def forward(self, x):
        """
        :param x: tensor, [batch_size, c_in, n_his, n_route].
        :return: tensor, [batch_size, c_out, n_his-2kt+2, n_route].
        """
        x_t1 = self.tconv1(x)
        x_s = self.sconv(x_t1)
        x_t2 = self.tconv2(x_s)
        x_ln = self.ln(x_t2.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return self.dropout(x_ln)


class FullyConv(nn.Module):
    """
    Fully connected layer.
    Maps multi-channels to one.
    """
    def __init__(self, c):
        """
        :param c: channel size of input x.
        """
        super(FullyConv, self).__init__()
        self.conv = nn.Conv2d(c, 1, 1)

    def forward(self, x):
        """
        :param x: tensor, [batch_size, c, 1, n_route].
        :return: tensor, [batch_size, 1, 1, n_route].
        """
        return self.conv(x)


class OutLayer(nn.Module):
    """
    Output layer.
    Temporal convolution layers attach with one fully connected layer,
    which map outputs of the last ST-Conv block to a single-step prediction.
    """
    def __init__(self, c, T, n):
        """
        :param c: int, channel size of input x.
        :param T: int, kernel size of temporal convolution.
        :param n: int, kernel size of temporal convolution.
        """
        super(OutLayer, self).__init__()
        self.tconv1 = TemporalConv(T, c, c, "GLU")
        self.ln = nn.LayerNorm([n, c])
        self.tconv2 = TemporalConv(1, c, c, "sigmoid")
        self.fc = FullyConv(c)

    def forward(self, x):
        """
        :param x: tensor, [batch_size, c_out, n_his-4kt+4, n_route].
        :return: tensor, [batch_size, 1, 1, n_route].
        """
        x_t1 = self.tconv1(x)
        x_ln = self.ln(x_t1.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x_t2 = self.tconv2(x_ln)
        return self.fc(x_t2)


class STGCN(nn.Module):
    """
    Spatio-temporal graph convolutional network.
    https://arxiv.org/abs/1709.04875v3
    Proposed by Yu et al.
    """
    def __init__(self, ks, kt, bs, T, n, Lk, p):
        """
        :param ks: int, kernel size of spatial convolution.
        :param kt: int, kernel size of temporal convolution.
        :param bs: list, channel configs of ST-Conv blocks.
        :param T: int, n_his, size of historical records for training.
        :param n: int, n_route, number of monitor stations.
        :param Lk: [ks, n_route, n_route]. Graph Kernel.
        :param p: float, drop out rate.
        """
        super(STGCN, self).__init__()
        self.st_conv1 = STConvBlock(ks, kt, n, bs[0], p, Lk)
        self.st_conv2 = STConvBlock(ks, kt, n, bs[1], p, Lk)
        self.output = OutLayer(bs[1][2], T - 4 * (kt - 1), n)

    def forward(self, x):
        """
        :param x: tensor, [batch_size, 1, n_his, n_route].
        :return: tensor, [batch_size, 1, 1, n_route].
        """
        x_st1 = self.st_conv1(x)
        x_st2 = self.st_conv2(x_st1)
        return self.output(x_st2)
