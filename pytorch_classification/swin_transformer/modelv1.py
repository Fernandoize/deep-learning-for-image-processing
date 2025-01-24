import random
from typing import Optional

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from pytorch_classification.swin_transformer.model import swin_base_patch4_window7_224


def drop_path_f(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path_f(x, self.drop_prob, self.training)


def window_partition(x, window_size: int):
    """
     将feature map 按照window_size划分为一个个没有重叠的windows
    :param x: (B, H, W, C)
    :param window_size:
    :return:
    """
    B, W, H, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    # permute [B, H//Wh, Mh, W//Mw, Mw, C] -> [B, H//Mh, W//Mh, Mw, Mw, C]
    # view [B, H//Mh, W//Mh, Mw, Mw, C] -> [B * num_windows, window_size, window_size, C]
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def windows_reverse(windows, window_size: int, H, W):
    """
    :param windows: (B*num_windows, window_size, window_size, C)
    :param window_size:
    :param H:
    :param W:
    :return:
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    # [B*num_windows, window_size, window_size, C] -> [B, H//Wh, W//Mw, Mh, Mw, C]
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    # permute: [B, H//Wh, W//Mw, Mh, Mw, C] -> [B, H//Wh, Mh, W//Mw, Mw, C]
    # view: [B, H//Wh, Mh, W//Mw, Mw, C] -> [B, H, W, C]
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class PatchEmbed(nn.Module):
    """
    将图像转换为feature map并打平
    输入(batch_size, in_channels, 224, 224)
    输出(batch_size, 56 * 56, embed_dim)
    """

    def __init__(self, patch_size=4, in_chan=3, embed_dim=96, norm_layer=None):
        super(PatchEmbed, self).__init__()
        patch_size = (patch_size, patch_size)
        self.patch_size = patch_size
        self.in_chan = in_chan
        self.embed_dim = embed_dim
        self.proj = nn.Conv2d(in_chan, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        _, _, H, W = x.shape
        pad_input = (H % self.patch_size[0] != 0) or (W % self.patch_size[1] != 0)
        if pad_input:
            x = F.pad(x, (0, self.patch_size[1] - W % self.patch_size[1],
                          0, self.patch_size[0] - H % self.patch_size[0],
                          0, 0))

        x = self.proj(x)
        _, _, H, W = x.shape
        # flatten: [B, C, H, W] -> (B, C, HW)
        # transpose: [B, C, HW] -> (B, HW, C)
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, H, W


class PatchMerging(nn.Module):
    """
    借助全连接层动态的融合每个小patch
    可以尝试用PatchMerging替换2 * 2下采样，或许效果会更好
    """

    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super(PatchMerging, self).__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x, H, W):
        # 此时X的shape为(B, HW, C) L = HW
        B, L, C = x.shape
        x = x.view(B, H, W, C)

        # padding
        # 如果输入feature map的H，W不是2的整数倍，需要进行padding
        pad_input = (H % 2 == 1) or (W % 2 == 1)
        if pad_input:
            # to pad the last 3 dimensions, starting from the last dimension and moving forward.
            # (C_front, C_back, W_left, W_right, H_top, H_bottom)
            # 注意这里的Tensor通道是[B, H, W, C]，所以会和官方文档有些不同
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))

        # # [B, H/2, W/2, C]
        # 分别按照H和W取值，step=2, W, H分别从0开始
        x0 = x[:, 0::2, 0::2, :]
        # W从1开始，H从0开始
        x1 = x[:, 1::2, 0::2, :]
        # W从0开始，H从1开始
        x2 = x[:, 0::2, 1::2, :]
        # W, H都从1开始
        x3 = x[:, 1::2, 1::2, :]

        # [B, H/2, W/2, 4*C]
        x = torch.cat([x0, x1, x2, x3], -1)
        # [B, H/2 * W/2, 4*C]
        x = x.view(B, -1, 4 * C)

        # 在进入到线性层之前进行norm处理, 这里处理的维度默认是chanel维度上每一个像素
        x = self.norm(x)
        # [B, H/2*W/2, 2*C]
        x = self.reduction(x)
        return x


class Mlp(nn.Module):
    """
    这个MLP用于 Transformer Encoder Block
    """

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super(Mlp, self).__init__()

        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class WindowAttention(nn.Module):
    """

    VIT中的MSA结构
    1. layer norm
    2. q,k,v计算
    3. match & softmax
    4. 提取信息
    5. drop path


    注意：这里的drop分为三种：
    1. attention系数后的attn_drop
    2. multi head concat结束后的proj_drop
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super(WindowAttention, self).__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads

        # 计算每个head的dim以及注意力权重的缩放系数 根号dim
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # 计算relative position
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # 定义relative position bias
        # relative bias table的大小取[2*window_size-1] * [2*window_size-1]
        # 每个num_head使用单独的position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))

        # 生成坐标
        coords_h = torch.arange(window_size[0])
        corrds_w = torch.arange(window_size[1])
        # [2, Mh, Mw]
        coords = torch.stack(torch.meshgrid([coords_h, corrds_w], indexing="ij"))
        # tensor([[0, 0, 1, 1],
        #         [0, 1, 0, 1]])
        # [2, Mh * Mw]
        coords_flatten = torch.flatten(coords, 1)
        # [2, Mh*Mw, Mh*Mw]
        # 1. 计算相对位置
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # [Mh*Mw, Mh*Mw, 2]
        # 2. 行列标分别+window_size-1
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        # 3. 行标 * 2*window_size-1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        # 4. 行、列标相加，二元转为一元
        # [Mh*Mw, Mh*Mw]
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask: Optional[torch.Tensor] = None):
        """
        :param x:
        :param mask: mask
        :return:
        """
        # [batch_size*num_windows, Mh*Mw, total_embed_dim]
        B, N, C = x.shape
        # qkv(): -> [batch_size*num_windows, Mh*Mw, 3 * total_embed_dim]
        # reshape() -> [batch_size*num_windows, Mh*Mw, 3, num_heads, total_embed_dim]
        # permute() -> [3, batch_size*num_windows, num_heads, Mh*Mw, embed_dim_per_head]
        # 将 Mh*Mw, embed_dim_per_head 移位到最后两列便于计算
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # [batch_size*num_window, num_heads, Mh*Mw, embed_dim_per_head]
        q, k, v = qkv.unbind(0)

        # transpose: -> [batch_size*num_windows, num_heads, embed_dim_per_head, Mh*Mw]
        # @: multiply -> [batch_size*num_windows, num_heads, Mh*Mw, Mh*Mw]
        attn = (q @ k.transpose(-2, -1)) * self.scale

        # relative bias
        # self.relative_position_index.view(-1)  [Mh*Mw,Mh*Mw]
        # self.relative_position_bias_table[self.relative_position_index.view(-1)] -> [Mh*Mw*Mh*Mw, num_heads]
        # [Mh*Mw*Mh*Mw, num_heads] -> [Mh*Mw,Mh*Mw, num_heads]
        relative_position_bias = (self.relative_position_bias_table[self.relative_position_index.view(-1)]
                                  .view(self.window_size[0] * self.window_size[1],
                                        self.window_size[0] * self.window_size[1], -1))
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # [num_heads, Mh*Mw, Mh*Mw]
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            # mask [nW, Mh*Mw, Mh*Mw], nW指的是num_windows
            nW = mask.shape[0]
            # attn.view: [batch_size, num_windows, num_heads, Mh*Mw, Mh*Mw]
            # mask.unsqueeze: [1, nW, 1, Mh*Mw, Mh*Mw] 在batch维度和num_heads维度上进行广播
            attn = attn.view(B // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            # 再转回到原来的形状
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        # @: multiply -> [batch_size*num_windows, num_heads, Mh*Mw, embed_dim_per_head]
        # transpose() -> [batch_size*num_windows, Mh*Mw, num_heads, embed_dim_per_head] 将num_heads还原到拆分前的位置，后续进行合并
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwingTransformerBlock(nn.Module):
    """
    Swing Transformer Block
    1. layer norm
    2. MSA
    3. drop path
    4. residual block

    5. layer norm
    6. mlp
    7. drop path
    8. residual block
    """

    def __init__(self, dim, num_heads,
                 window_size=7,
                 shift_size=0,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        """
        :param dim: 输入dim
        :param num_heads: heads
        :param window_size: window
        :param shift_size: window shift size
        :param mlp_ratio: mlp channel和 dim的比例
        :param qkv_bias:
        :param drop: Dropout rate. Default: 0.0 用于mlp 和 drop path, attn proj drop
        :param attn_drop: attn_drop
        :param drop_path:
        :param act_layer:
        :param norm_layer:
        """
        super(SwingTransformerBlock, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(dim, window_size=(self.window_size, self.window_size), num_heads=num_heads,
                                    attn_drop=attn_drop, proj_drop=drop, qkv_bias=qkv_bias)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(dim, mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, attn_mask):
        H, W = self.H, self.W
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # pad feature maps to multiples of window size
        # 把feature map给pad到window size的整数倍
        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape

        # cyclic shift
        if self.shift_size > 0:
            # shift_size = -1时， dim=0代表将第一行沿y轴向上移动1个单位，也就是移动到了最底部
            # tensor([[0, 1, 2],
            #         [3, 4, 5],
            #         [6, 7, 8]])
            #  tensor([[4, 5, 3],
            #         [7, 8, 6],
            #         [1, 2, 0]])
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x
            attn_mask = None

        # [B*num_windows, Mh, Mw, C]
        x_windows = window_partition(shifted_x, self.window_size)
        # [B*num_windows, Mh*Mw, C] 将window特征打平
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=attn_mask)
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)

        # 将attention后的windows还原为原始特征图
        shifted_x = windows_reverse(attn_windows, self.window_size, Hp, Wp)  # [B, H', W', C]

        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        if pad_r > 0 or pad_b > 0:
            # 把前面pad的数据移除掉
            x = x[:, :H, :W, :].contiguous()

        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class BasicLayer(nn.Module):
    """
    包括两个swing block
    一个为windows MSA
    一个为shift windows MSA
    """

    def __init__(self, dim, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False):
        super(BasicLayer, self).__init__()
        self.dim = dim
        self.depth = depth
        self.window_size = window_size
        self.use_checkpoint = use_checkpoint
        self.shift_size = window_size // 2

        # build blocks: 第一个是MSA, 第二个是shift MSA
        # 这里MSA和WMSA两个总depth=2
        self.blocks = nn.ModuleList([
            SwingTransformerBlock(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else self.shift_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer)
            for i in range(depth)])

        if downsample is not None:
            # 使用patch merging进行下采样,2倍率下采样
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def create_mask(self, x, H, W):
        """
        mask的作用：shift后本来非连续windows会连接在一起，因此计算自注意力时需要将非连续的windos mask掉，防止影响计算结果
        :param H:
        :param W:
        :return:
        """
        # SW-MSA使用的mask

        # 保证Hp和Wp是window_size的整数倍
        Hp = int(np.ceil(H / self.window_size)) * self.window_size
        Wp = int(np.ceil(W / self.window_size)) * self.window_size

        # 拥有和feature map一样的通道排列顺序，方便后续window_partition
        img_mask = torch.zeros((1, Hp, Wp, 1), device=x.device)
        # slice的主要作用是进行切片，将高切为三个部分,
        # 第一行到倒数第一个window的起始行,
        # 倒数第一个window的起始行到shift块的起始行
        # shift块起始行到结束
        h_slice = (slice(0, -self.window_size),
                   slice(-self.window_size, -self.window_size),
                   slice(-self.shift_size, None))
        w_slice = (slice(0, -self.window_size),
                   slice(-self.window_size, -self.shift_size),
                   slice(-self.shift_size, None))

        # mask编号
        cnt = 0
        for h in h_slice:
            for w in w_slice:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        # [B * num_windows, window_size, window_size, C] 此处B为1-> [nW, Mh, Mw, 1]
        mask_windows = window_partition(img_mask, self.window_size)
        # 打平，和x_windows特征大小保持一致 [nW, Mh * Mw]
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        # [nW, 1, Mh * Mw] - # [nW, Mh * Mw, 1] 使用广播机制模拟attn 系数的计算，只要该窗口是由于shift带来的不连续区域，计算差值后都为非0
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        # [nW, Mh*Mw, Mh*Mw]， 非0区域使用-100填充，后续softmax计算时会被忽略
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        return attn_mask

    def forward(self, x, H, W):
        # [nW, Mh*Mw, Mh*Mw]
        attn_mask = self.create_mask(x, H, W)
        for block in self.blocks:
            block.H = H
            block.W = W
            x = block(x, attn_mask)

        if self.downsample is not None:
            x = self.downsample(x, H, W)
            H, W = (H + 1) // 2, (W + 1) // 2

        return x, H, W


class SwinTransformer(nn.Module):
    def __init__(self, patch_size=4, in_chan=3, num_classes=1000,
                 embed_dim=96, depths=(2, 2, 6, 2), num_heads=(3, 6, 12, 24),
                 window_size=7, mlp_ratio=4., qkv_bias=True,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 patch_norm=True, norm_layer=nn.LayerNorm,
                 use_checkpoint=False, **kwargs):
        super(SwinTransformer, self).__init__()

        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.mlp_ratio = mlp_ratio
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))

        # 1. patch embed
        self.patch_embed = PatchEmbed(patch_size=patch_size, in_chan=in_chan, embed_dim=embed_dim,
                                      norm_layer=norm_layer if self.patch_norm else None)
        self.pos_drop = nn.Dropout(p=drop_rate)

        # 每一个 SwingTransformerBlock 使用相同的drop path rate
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        # 2. 4个stage, 每个stage有指定的 transformer block
        self.layers = nn.ModuleList([])
        for i_layer in range(self.num_layers):
            # 注意这里构建的stage和论文图中有些差异
            # 这里的stage不包含该stage的patch_merging层，包含的是下个stage的
            layers = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                                depth=depths[i_layer],
                                num_heads=num_heads[i_layer],
                                window_size=window_size,
                                mlp_ratio=mlp_ratio,
                                qkv_bias=qkv_bias,
                                drop=drop_rate,
                                attn_drop=attn_drop_rate,
                                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])], norm_layer=norm_layer,
                                # 最后一层没有下采样吗
                                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None)
            self.layers.append(layers)

        # 3. mlp head
        self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(self.num_features, self.num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        # 输入 [B, C, H, W], 输出[B, L, C]
        x, H, W = self.patch_embed(x)
        x = self.pos_drop(x)

        # 输入[B, L, C] 输出[B, L, C]
        for layer in self.layers:
            x, H, W = layer(x, H, W)

        x = self.norm(x)

        # x.transpose [B, L, C], [B, C, L]
        x = self.avgpool(x.transpose(1, 2))
        x = torch.flatten(x, 1)
        x = self.head(x)
        return x


def swin_tiny_patch4_window7_224(num_classes: int = 1000, **kwargs):
    # trained ImageNet-1K
    # https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth
    model = SwinTransformer(in_chans=3,
                            patch_size=4,
                            window_size=7,
                            embed_dim=96,
                            depths=(2, 2, 6, 2),
                            num_heads=(3, 6, 12, 24),
                            num_classes=num_classes,
                            **kwargs)
    return model


def swin_small_patch4_window7_224(num_classes: int = 1000, **kwargs):
    # trained ImageNet-1K
    # https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_small_patch4_window7_224.pth
    model = SwinTransformer(in_chans=3,
                            patch_size=4,
                            window_size=7,
                            embed_dim=96,
                            depths=(2, 2, 18, 2),
                            num_heads=(3, 6, 12, 24),
                            num_classes=num_classes,
                            **kwargs)
    return model



def swin_base_patch4_window7_224(num_classes=1000, **kwargs):
    # trained ImageNet-1K
    # https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window7_224.pth
    model = SwinTransformer(in_chan=3,
                            patch_size=4,
                            window_size=7,
                            embed_dim=128,
                            depths=(2, 2, 18, 2),
                            num_heads=(4, 8, 16, 32),
                            num_classes=num_classes,
                            **kwargs)
    return model


def swin_base_patch4_window12_384(num_classes: int = 1000, **kwargs):
    # trained ImageNet-1K
    # https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window12_384.pth
    model = SwinTransformer(in_chans=3,
                            patch_size=4,
                            window_size=12,
                            embed_dim=128,
                            depths=(2, 2, 18, 2),
                            num_heads=(4, 8, 16, 32),
                            num_classes=num_classes,
                            **kwargs)
    return model


def swin_base_patch4_window7_224_in22k(num_classes: int = 21841, **kwargs):
    # trained ImageNet-22K
    # https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window7_224_22k.pth
    model = SwinTransformer(in_chans=3,
                            patch_size=4,
                            window_size=7,
                            embed_dim=128,
                            depths=(2, 2, 18, 2),
                            num_heads=(4, 8, 16, 32),
                            num_classes=num_classes,
                            **kwargs)
    return model


def swin_base_patch4_window12_384_in22k(num_classes: int = 21841, **kwargs):
    # trained ImageNet-22K
    # https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window12_384_22k.pth
    model = SwinTransformer(in_chans=3,
                            patch_size=4,
                            window_size=12,
                            embed_dim=128,
                            depths=(2, 2, 18, 2),
                            num_heads=(4, 8, 16, 32),
                            num_classes=num_classes,
                            **kwargs)
    return model


def swin_large_patch4_window7_224_in22k(num_classes: int = 21841, **kwargs):
    # trained ImageNet-22K
    # https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window7_224_22k.pth
    model = SwinTransformer(in_chans=3,
                            patch_size=4,
                            window_size=7,
                            embed_dim=192,
                            depths=(2, 2, 18, 2),
                            num_heads=(6, 12, 24, 48),
                            num_classes=num_classes,
                            **kwargs)
    return model


def swin_large_patch4_window12_384_in22k(num_classes: int = 21841, **kwargs):
    # trained ImageNet-22K
    # https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window12_384_22k.pth
    model = SwinTransformer(in_chans=3,
                            patch_size=4,
                            window_size=12,
                            embed_dim=192,
                            depths=(2, 2, 18, 2),
                            num_heads=(6, 12, 24, 48),
                            num_classes=num_classes,
                            **kwargs)
    return model


def fix_seed():
    SEED = 42
    random.seed(SEED)  # 设置 Python 内置 random 模块的种子
    np.random.seed(SEED)  # 设置 NumPy 随机数生成器的种子
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == '__main__':
    fix_seed()
    inputs = torch.rand(1, 3, 224, 224)
    model = swin_base_patch4_window7_224(num_classes=5)
    print(model(inputs))
