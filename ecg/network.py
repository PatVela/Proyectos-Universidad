"""PyTorch port of awni/ecg network.py.

Replicates the deep residual CNN from:
    Cardiologist-Level Arrhythmia Detection and Classification in Ambulatory
    Electrocardiograms Using a Deep Neural Network (Hannun et al., Nature Medicine 2019)

The original Keras network (examples/cinc17/config.json) is mapped 1:1. There is
one important representational difference: Keras uses (batch, time, channels=1)
and Conv1D; PyTorch uses (batch, channels=1, time). Inputs given to this model are
always (batch, 1, time). The output is (batch, time, num_categories) with softmax
over the category axis, matching Keras' TimeDistributed(Dense)+softmax.
"""

import torch
import torch.nn as nn


# ----------------------------------------------------------------------------
# Same ("SAME") padding, replicating Keras/TensorFlow Conv1D & MaxPool1D.
# Keras 'same' pads asymmetrically: output length = ceil(input / stride).
# ----------------------------------------------------------------------------
def _same_padding(n: int, kernel: int, stride: int) -> tuple:
    """Return (pad_left, pad_right) such that out = ceil(n / stride)."""
    out = -(-n // stride)                      # ceil division
    total = max(0, (out - 1) * stride + kernel - n)
    left = total // 2
    right = total - left
    return left, right


class _SamePad1d(nn.Module):
    """Asymmetric padding producing TensorFlow 'SAME' output length."""

    def __init__(self, kernel: int, stride: int):
        super().__init__()
        self.kernel = kernel
        self.stride = stride

    def forward(self, x):
        n = x.shape[-1]
        left, right = _same_padding(n, self.kernel, self.stride)
        if left == 0 and right == 0:
            return x
        return nn.functional.pad(x, (left, right))


# ----------------------------------------------------------------------------
# Building blocks (ports of network.py)
# ----------------------------------------------------------------------------

class _ConvSame1d(nn.Module):
    """Conv1d with SAME padding + optional stride."""

    def __init__(self, in_channels, out_channels, kernel, stride, seed=None):
        super().__init__()
        self.pad = _SamePad1d(kernel, stride)
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size=kernel, stride=stride,
            padding=0, bias=False)
        # Keras default kernel_initializer for Conv1D is 'glorot_uniform'.
        # The config uses 'he_normal' (see config.json), so default to that,
        # but keep glorot available to be maximally faithful to Keras defaults.
        if seed is not None:
            torch.manual_seed(seed)
        nn.init.kaiming_normal_(
            self.conv.weight, a=0, mode='fan_in', nonlinearity='relu')

    def forward(self, x):
        return self.conv(self.pad(x))


class _BNRelu(nn.Module):
    """BatchNorm -> ReLU -> (optional) Dropout, port of _bn_relu().

    Numerical parity with Keras: Keras BatchNormalization uses eps=1e-3,
    momentum=0.99 (a decay), while PyTorch defaults are eps=1e-5, momentum=0.1
    (and PyTorch's momentum is the *inverse* of Keras' decay). To reproduce the
    original training as closely as possible we pin the Keras values here.
    torch.BatchNorm1d(momentum=0.99) == keras(tf) momentum=0.99 decay semantics
    for the running mean/var update.
    """

    def __init__(self, channels, dropout, eps=1e-3, momentum=0.99, seed=None):
        super().__init__()
        self.bn = nn.BatchNorm1d(channels, eps=eps, momentum=momentum)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        return self.dropout(self.act(self.bn(x)))


class _ConvBlock(nn.Module):
    """conv -> bn_relu, port of add_conv_weight + _bn_relu."""

    def __init__(self, in_channels, out_channels, filter_length, subsample, dropout,
                 seed=None):
        super().__init__()
        self.conv = _ConvSame1d(in_channels, out_channels, filter_length, subsample,
                                seed=seed)
        self.bnrelu = _BNRelu(out_channels, dropout, seed=seed)

    def forward(self, x):
        return self.bnrelu(self.conv(x))


class _ResBlock(nn.Module):
    """One residual block, port of resnet_block().

    shortcut = MaxPool1D(subsample) of the input; zero-padded on the channel axis
    whenever the number of filters doubles (block_index % 4 == 0 and block_index > 0).
    """

    def __init__(self, in_filters, out_filters, subsample, block_index,
                 filter_length, num_skip, dropout, seed=0):
        super().__init__()
        self.subsample = subsample
        self.zero_pad = (block_index % 4 == 0) and block_index > 0

        # shortcut branch
        self.pool = nn.MaxPool1d(kernel_size=subsample)  # stride == kernel

        # residual branch: num_skip convs
        layers = []
        for i in range(num_skip):
            # BN+ReLU is applied *before* every conv except the very first
            # conv of block 0 (matches the original `if not (block==0 and i==0)`).
            if not (block_index == 0 and i == 0):
                layers.append(_BNRelu(in_filters, dropout if i > 0 else 0))
            layers.append(
                _ConvSame1d(in_filters, out_filters, filter_length,
                            subsample if i == 0 else 1))
            in_filters = out_filters
        self.residual = nn.Sequential(*layers)
        self.add = nn.Identity()  # (for clarity) element-wise add below

    def forward(self, x):
        # shortcut: maxpool with SAME padding
        n = x.shape[-1]
        pool_pad = _SamePad1d(self.subsample, self.subsample)
        shortcut = pool_pad(x)
        shortcut = self.pool(shortcut)
        if self.zero_pad:
            zeros = torch.zeros_like(shortcut)
            shortcut = torch.cat([shortcut, zeros], dim=1)

        residual = self.residual(x)
        return shortcut + residual


class _ResNet_Layers(nn.Module):
    """Initial conv -> 16 residual blocks -> final BN+ReLU.
    Port of add_resnet_layers()."""

    def __init__(self, chunk_length, filter_length, num_filters_start, subsample_lengths,
                 num_skip, dropout, seed=0):
        super().__init__()
        self.conv0 = _ConvBlock(1, num_filters_start, filter_length, 1, 0, seed=seed)
        blocks = []
        in_filters = num_filters_start
        for index, subsample in enumerate(subsample_lengths):
            out_filters = _num_filters_at(index, num_filters_start)
            blocks.append(_ResBlock(in_filters, out_filters, subsample, index,
                                    filter_length, num_skip, dropout, seed=seed + index))
            in_filters = out_filters
        self.blocks = nn.Sequential(*blocks)
        self.final_bnrelu = _BNRelu(in_filters, 0)

    def forward(self, x):
        x = self.conv0(x)
        x = self.blocks(x)
        x = self.final_bnrelu(x)
        return x


class _PlainCNN_Layers(nn.Module):
    """Conventional CNN matched to the ResNet convolutional schedule.

    This is the ablation used when ``is_regular_conv=True``. It keeps the same
    initial convolution, number of convolutional layers, filter length, temporal
    subsampling pattern, dropout placement and channel-growth schedule as the
    residual model, but removes the shortcut/maxpool additions. Therefore the
    ResNet-vs-CNN experiment isolates the contribution of residual connections
    much better than a shallow plain stack would.
    """

    def __init__(self, chunk_length, filter_length, num_filters_start, subsample_lengths,
                 num_skip, dropout, seed=0):
        super().__init__()
        self.conv0 = _ConvBlock(1, num_filters_start, filter_length, 1, 0, seed=seed)
        blocks = []
        in_filters = num_filters_start
        for index, subsample in enumerate(subsample_lengths):
            out_filters = _num_filters_at(index, num_filters_start)
            layers = []
            for i in range(num_skip):
                # Match the pre-activation pattern used inside the residual
                # branch. The only removed operation is shortcut + addition.
                if not (index == 0 and i == 0):
                    layers.append(_BNRelu(in_filters, dropout if i > 0 else 0))
                layers.append(
                    _ConvSame1d(in_filters, out_filters, filter_length,
                                subsample if i == 0 else 1,
                                seed=seed + index * max(num_skip, 1) + i))
                in_filters = out_filters
            blocks.append(nn.Sequential(*layers))
        self.blocks = nn.Sequential(*blocks)
        self.final_bnrelu = _BNRelu(in_filters, 0)

    def forward(self, x):
        x = self.conv0(x)
        x = self.blocks(x)
        x = self.final_bnrelu(x)
        return x


def _num_filters_at(index, num_start_filters):
    """Port of get_num_filters_at_index(): doubles every `increase_channels_at` (4)."""
    return int(2 ** (index // 4)) * num_start_filters


class _OutputLayer(nn.Module):
    """TimeDistributed(Dense(num_categories)) + softmax, port of add_output_layer()."""

    def __init__(self, in_features, num_categories):
        super().__init__()
        self.dense = nn.Linear(in_features, num_categories, bias=True)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # x: (batch, channels, time) -> (batch, time, channels) -> dense
        x = x.transpose(1, 2)
        return self.softmax(self.dense(x))  # (batch, time, num_categories)


class ECGNetwork(nn.Module):
    """Full network. Mirrors build_network() in network.py.

    Input  : (batch, 1, time)  float32
    Output : (batch, time // 2^8, num_categories) probabilities (softmax)
    """

    def __init__(self, conv_filter_length=16, conv_num_filters_start=32,
                 conv_subsample_lengths=None, conv_num_skip=2, conv_dropout=0.2,
                 num_categories=4, is_regular_conv=False, seed=0, **unused):
        super().__init__()
        if conv_subsample_lengths is None:
            conv_subsample_lengths = [1, 2] * 8
        self.conv_subsample_lengths = list(conv_subsample_lengths)
        self.num_categories = num_categories
        self.is_regular_conv = is_regular_conv

        if self.is_regular_conv:
            self.cnn = _PlainCNN_Layers(
                0, conv_filter_length, conv_num_filters_start,
                self.conv_subsample_lengths, conv_num_skip, conv_dropout, seed=seed)
        else:
            self.resnet = _ResNet_Layers(
                0, conv_filter_length, conv_num_filters_start,
                self.conv_subsample_lengths, conv_num_skip, conv_dropout, seed=seed)

        # number of channels at the final layer
        final_channels = _num_filters_at(len(self.conv_subsample_lengths) - 1,
                                         conv_num_filters_start)

        self.final_reshape = nn.Identity()
        self.output = _OutputLayer(final_channels, num_categories)

    def forward(self, x):
        if self.is_regular_conv:
            h = self.cnn(x)
        else:
            h = self.resnet(x)
        probs = self.output(h)
        return probs

    def logits(self, x):
        """Softmax-free output, for numerically stable training."""
        if self.is_regular_conv:
            h = self.cnn(x)
        else:
            h = self.resnet(x)
        h = h.transpose(1, 2)
        return self.output.dense(h)


def build_network(**params):
    """Port of build_network(): returns an ECGNetwork (input_shape handled by caller)."""
    return ECGNetwork(**params)
