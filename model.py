import torch
from torch import nn
import torch.nn.functional as F
import settings
from einops import rearrange
import numpy as np
import numbers

def im2col(input_data, filter_h, filter_w, stride=1, pad=0):
    N, C, H, W = input_data.shape
    out_h = (H + 2*pad - filter_h)//stride + 1
    out_w = (W + 2*pad - filter_w)//stride + 1
    ZeroPad2d = nn.ZeroPad2d(padding=(pad, pad, pad, pad))
    img = ZeroPad2d(input_data)
    col = torch.zeros([N, C, filter_h,filter_w, out_h,out_w]).cuda()
    for y in range(filter_h):
        y_max = y + stride*out_h
        for x in range(filter_w):
            x_max = x + stride*out_w
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride,]

    col = col.reshape(N, C, filter_h*filter_w, out_h*out_w)

    return col

def col2im(col,orisize,filter_h, filter_w, stride=1, pad=0):
    N, C, H, W = orisize
    out_h = (H + 2*pad - filter_h)//stride + 1
    out_w = (W + 2*pad - filter_w)//stride + 1
    col = col.reshape(N, C, filter_h, filter_w,out_h, out_w)
    img = torch.zeros([N, C, H + 2*pad + stride - 1, W + 2*pad + stride - 1]).cuda()
    for y in range(filter_h):
        y_max = y + stride*out_h
        for x in range(filter_w):
            x_max = x + stride*out_w
            img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]

    return img[:, :, pad:H + pad, pad:W + pad]

class nlwt(nn.Module):
    def __init__(self):
        super(nlwt, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        return nlwt_init(x)

def nlwt_init(x):
    U1ImulP1I = np.array([[-0.5,   -0.5,   -0.5,    0.5],
                          [0.5,   -0.5,    0.5,    0.5],
                          [0.5,    0.5,   -0.5,    0.5],
                          [0.5,   -0.5,   -0.5,   -0.5]])
    U2ImulP2Imul4 = np.array([[-1,     1,     1,    -1],
                              [-1,    -1,    -1,    -1],
                              [-1,    -1,     1,     1],
                              [1,    -1,     1,    -1]])
    U1ImulP1I = torch.cuda.FloatTensor(U1ImulP1I).unsqueeze(0).unsqueeze(0)
    U2ImulP2Imul4 = torch.cuda.FloatTensor(U2ImulP2Imul4).unsqueeze(0).unsqueeze(0)

    b, c, h, w = x.size()
    orisize = x.size()

    xT_col = im2col(x, 2, 2, stride=2, pad=0);
    x1 = U1ImulP1I @ xT_col;

    h1 = h // 2
    w1 = w // 2
    T2 = x1[:, :, 1, :].reshape(b, c, h1, w1);
    T3 = x1[:, :, 2, :].reshape(b, c, h1, w1);
    T4 = x1[:, :, 3, :].reshape(b, c, h1, w1);

    T22 = torch.roll(T2, shifts=-1, dims=2)
    T32 = torch.roll(T3, shifts=-1, dims=3)
    T42 = torch.roll(T4, shifts=(-1, -1), dims=(2, 3))

    x1[:, :, 1, :] = T22.flatten(2);
    x1[:, :, 2, :] = T32.flatten(2);
    x1[:, :, 3, :] = T42.flatten(2);

    x2 = U2ImulP2Imul4 @ x1;

    A_low0 = x2[:, :, 0, :].reshape(b, c, h1, w1);
    B_high1 = x2[:, :, 1, :].reshape(b, c, h1, w1);
    C_high2 = x2[:, :, 2, :].reshape(b, c, h1, w1);
    D_high3 = x2[:, :, 3, :].reshape(b, c, h1, w1);

    return A_low0, B_high1, C_high2, D_high3, orisize

class inlwt(nn.Module):
    def __init__(self):
        super(inlwt, self).__init__()
        self.requires_grad = False

    def forward(self, A_low0, B_high1, C_high2, D_high3,orisize):
        return inlwt_init(A_low0, B_high1, C_high2, D_high3, orisize)

def inlwt_init(A_low0,B_high1,C_high2,D_high3,orisize):
    P2mulU2 = np.array([[-1,    -1,    -1,     1],
                        [1,    -1,    -1,    -1],
                        [1,    -1,     1,     1],
                        [-1,    -1,     1,    -1]])
    P1mulU1div4 = np.array([[-0.125,    0.125,    0.125,    0.125],
                            [-0.125,   -0.125,    0.125,   -0.125],
                            [-0.125,    0.125,   -0.125,   -0.125],
                            [0.125,    0.125,    0.125,   -0.125]])
    P2mulU2 = torch.cuda.FloatTensor(P2mulU2).unsqueeze(0).unsqueeze(0)
    P1mulU1div4 = torch.cuda.FloatTensor(P1mulU1div4).unsqueeze(0).unsqueeze(0)

    b, c, h1, w1 = A_low0.size()
    A = A_low0.reshape(b, c, 1, h1 * w1);
    B = B_high1.reshape(b, c, 1, h1 * w1);
    C = C_high2.reshape(b, c, 1, h1 * w1);
    D = D_high3.reshape(b, c, 1, h1 * w1);

    Y1 = torch.cat([A, B, C, D], dim=2)
    Y2 = P2mulU2 @ Y1;
    t2 = Y2[:, :, 1, :].reshape(b, c, h1, w1);
    t3 = Y2[:, :, 2, :].reshape(b, c, h1, w1);
    t4 = Y2[:, :, 3, :].reshape(b, c, h1, w1);

    t22 = torch.roll(t2, shifts=1, dims=2)
    t32 = torch.roll(t3, shifts=1, dims=3)
    t42 = torch.roll(t4, shifts=(1, 1), dims=(2, 3))

    Y2[:, :, 1, :] = t22.flatten(2)
    Y2[:, :, 2, :] = t32.flatten(2)
    Y2[:, :, 3, :] = t42.flatten(2)

    Y3 = P1mulU1div4 @ Y2;
    rst = col2im(Y3, orisize, 2, 2, stride=2, pad=0);

    return rst

class nlwt_catone(nn.Module):
    def __init__(self):
        super(nlwt_catone, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        A_low0, B_high1, C_high2, D_high3, orisize = nlwt_init(x)
        out_catone = torch.cat([A_low0, B_high1, C_high2, D_high3], dim=1)

        return out_catone, orisize

class inlwt_catone(nn.Module):
    def __init__(self):
        super(inlwt_catone, self).__init__()
        self.requires_grad = False

    def forward(self, decoder_one,orisize):
        out_channel = orisize[1]
        A_low0 = decoder_one[:, 0:out_channel, :, :]
        B_high1 = decoder_one[:, out_channel:out_channel * 2, :, :]
        C_high2 = decoder_one[:, out_channel * 2:out_channel * 3, :, :]
        D_high3 = decoder_one[:, out_channel * 3:out_channel * 4, :, :]

        rst = inlwt_init(A_low0, B_high1, C_high2, D_high3, orisize)

        return rst


class convd(nn.Module):
    def __init__(self, inputchannel, outchannel, kernel_size, stride):
        super(convd, self).__init__()
        self.padding = nn.ReflectionPad2d(kernel_size // 2)
        self.conv = nn.Sequential(nn.Conv2d(inputchannel, outchannel, kernel_size, stride), nn.LeakyReLU(0.2))

    def forward(self, x):
        x = self.conv(self.padding(x))
        return x

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


##########################################################################
## Gated-Dconv Feed-Forward Network (GDFN)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()
        hidden_features = int(dim * ffn_expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


##########################################################################
## Multi-DConv Head Transposed Self-Attention (MDTA)
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out



class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x

class ChannelAttentionModule(nn.Module):
    def __init__(self, embed_dim, ratio=16):
        super(ChannelAttentionModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.shared_MLP = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(embed_dim // ratio, embed_dim, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = self.shared_MLP(self.avg_pool(x))
        maxout = self.shared_MLP(self.max_pool(x))
        return self.sigmoid(avgout + maxout)


class SpatialAttentionModule(nn.Module):
    def __init__(self):
        super(SpatialAttentionModule, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        out = self.sigmoid(self.conv2d(out))
        return out

class GFM(nn.Module):
    def __init__(self, embed_dim):
        super(GFM, self).__init__()
        self.channel_attention = ChannelAttentionModule(2*embed_dim)
        self.spatial_attention = SpatialAttentionModule()
        self.embed_dim = embed_dim

    def forward(self, x1,x2):
        c_w = self.channel_attention(torch.cat([x1, x2], dim=1))
        c_w = c_w.view(-1, 2, self.embed_dim)[:, :, :, None, None]
        out = c_w[:, 0, ::] * x1 + c_w[:, 1, ::] * x2
        out = self.spatial_attention(out) * out
        return out

class Dual_UFormer_Image_Deraining_Network(nn.Module):
    def __init__(self, in_dim=3, embed_dim=settings.channel):
        super().__init__()
        self.ffn_expansion_factor = 2.66
        self.bias = False
        self.LayerNorm_type = 'WithBias'
        self.num_heads = settings.heads
        self.depth = settings.depth
        self.convert = nn.Sequential(nn.Conv2d(in_dim, embed_dim, 3, 1, 1), nn.LeakyReLU(0.2))
        self.out = nn.Sequential(nn.Conv2d(embed_dim, embed_dim, 3, 1, 1), nn.LeakyReLU(0.2),
                                 nn.Conv2d(embed_dim, 3, 1, 1))
        self.conv_l0_No1 = nn.Sequential(nn.Conv2d(embed_dim, embed_dim, 3, 1, 1), nn.LeakyReLU(0.2))
        self.TB_l0_No1 = nn.Sequential(*[TransformerBlock(dim=embed_dim, num_heads=self.num_heads,
                                                      ffn_expansion_factor=self.ffn_expansion_factor,bias=self.bias,
                                                      LayerNorm_type=self.LayerNorm_type) for i in range(self.depth)])
        self.TB_l0_No2 = nn.Sequential(*[TransformerBlock(dim=embed_dim, num_heads=self.num_heads,
                                                      ffn_expansion_factor=self.ffn_expansion_factor, bias=self.bias,
                                                      LayerNorm_type=self.LayerNorm_type) for i in range(self.depth)])
        self.TB_l0_No3 = nn.Sequential(*[TransformerBlock(dim=embed_dim, num_heads=self.num_heads,
                                                      ffn_expansion_factor=self.ffn_expansion_factor, bias=self.bias,
                                                      LayerNorm_type=self.LayerNorm_type) for i in range(self.depth)])
        self.TB_l0_No4 = nn.Sequential(*[TransformerBlock(dim=embed_dim, num_heads=self.num_heads,
                                                      ffn_expansion_factor=self.ffn_expansion_factor, bias=self.bias,
                                                      LayerNorm_type=self.LayerNorm_type) for i in range(self.depth)])
        self.conv_l1_No1 = nn.Sequential(nn.Conv2d(4*embed_dim, embed_dim, 3, 1, 1), nn.LeakyReLU(0.2))
        self.TB_l1_No1 = nn.Sequential(*[TransformerBlock(dim=embed_dim, num_heads=self.num_heads,
                                                      ffn_expansion_factor=self.ffn_expansion_factor, bias=self.bias,
                                                      LayerNorm_type=self.LayerNorm_type) for i in range(self.depth)])
        self.TB_l1_No2 = nn.Sequential(*[TransformerBlock(dim=embed_dim, num_heads=self.num_heads,
                                                      ffn_expansion_factor=self.ffn_expansion_factor, bias=self.bias,
                                                      LayerNorm_type=self.LayerNorm_type) for i in range(self.depth)])
        self.TB_l1_No3 = nn.Sequential(*[TransformerBlock(dim=embed_dim, num_heads=self.num_heads,
                                                      ffn_expansion_factor=self.ffn_expansion_factor, bias=self.bias,
                                                      LayerNorm_type=self.LayerNorm_type) for i in range(self.depth)])
        self.TB_l1_No4 = nn.Sequential(*[TransformerBlock(dim=embed_dim, num_heads=self.num_heads,
                                                      ffn_expansion_factor=self.ffn_expansion_factor, bias=self.bias,
                                                      LayerNorm_type=self.LayerNorm_type) for i in range(self.depth)])
        self.TB_l1_No5 = nn.Sequential(*[TransformerBlock(dim=embed_dim, num_heads=self.num_heads,
                                                      ffn_expansion_factor=self.ffn_expansion_factor, bias=self.bias,
                                                      LayerNorm_type=self.LayerNorm_type) for i in range(self.depth)])
        self.conv_l2_No1 = nn.Sequential(nn.Conv2d(4*embed_dim, embed_dim, 3, 1, 1), nn.LeakyReLU(0.2))
        self.TB_l2_No1 = nn.Sequential(*[TransformerBlock(dim=embed_dim, num_heads=self.num_heads,
                                                      ffn_expansion_factor=self.ffn_expansion_factor, bias=self.bias,
                                                      LayerNorm_type=self.LayerNorm_type) for i in range(self.depth)])
        self.TB_l2_No2 = nn.Sequential(*[TransformerBlock(dim=embed_dim, num_heads=self.num_heads,
                                                      ffn_expansion_factor=self.ffn_expansion_factor, bias=self.bias,
                                                      LayerNorm_type=self.LayerNorm_type) for i in range(self.depth)])
        self.TB_l2_No3 = nn.Sequential(*[TransformerBlock(dim=embed_dim, num_heads=self.num_heads,
                                                      ffn_expansion_factor=self.ffn_expansion_factor, bias=self.bias,
                                                      LayerNorm_type=self.LayerNorm_type) for i in range(self.depth)])
        self.TB_l2_No4 = nn.Sequential(*[TransformerBlock(dim=embed_dim, num_heads=self.num_heads,
                                                      ffn_expansion_factor=self.ffn_expansion_factor, bias=self.bias,
                                                      LayerNorm_type=self.LayerNorm_type) for i in range(self.depth)])
        self.TB_l2_No5 = nn.Sequential(*[TransformerBlock(dim=embed_dim, num_heads=self.num_heads,
                                                      ffn_expansion_factor=self.ffn_expansion_factor, bias=self.bias,
                                                      LayerNorm_type=self.LayerNorm_type) for i in range(self.depth)])

        self.conv_l3_No1 =  nn.Sequential(nn.Conv2d(4*embed_dim, embed_dim, 3, 1, 1), nn.LeakyReLU(0.2))
        self.TB_l3_No1 = nn.Sequential(*[TransformerBlock(dim=embed_dim, num_heads=self.num_heads,
                                                      ffn_expansion_factor=self.ffn_expansion_factor, bias=self.bias,
                                                      LayerNorm_type=self.LayerNorm_type) for i in range(self.depth)])
        self.TB_l3_No2 = nn.Sequential(*[TransformerBlock(dim=embed_dim, num_heads=self.num_heads,
                                                      ffn_expansion_factor=self.ffn_expansion_factor, bias=self.bias,
                                                      LayerNorm_type=self.LayerNorm_type) for i in range(self.depth)])
        self.TB_l3_No3 = nn.Sequential(*[TransformerBlock(dim=embed_dim, num_heads=self.num_heads,
                                                      ffn_expansion_factor=self.ffn_expansion_factor, bias=self.bias,
                                                      LayerNorm_type=self.LayerNorm_type) for i in range(self.depth)])
        self.TB_l3_No4 = nn.Sequential(*[TransformerBlock(dim=embed_dim, num_heads=self.num_heads,
                                                      ffn_expansion_factor=self.ffn_expansion_factor, bias=self.bias,
                                                      LayerNorm_type=self.LayerNorm_type) for i in range(self.depth)])
        self.TB_l3_No5 = nn.Sequential(*[TransformerBlock(dim=embed_dim, num_heads=self.num_heads,
                                                      ffn_expansion_factor=self.ffn_expansion_factor, bias=self.bias,
                                                      LayerNorm_type=self.LayerNorm_type) for i in range(self.depth)])
        self.conv_l4_No1 = nn.Sequential(nn.Conv2d(4*embed_dim, embed_dim, 3, 1, 1), nn.LeakyReLU(0.2))
        self.TB_l4_No1 = nn.Sequential(*[TransformerBlock(dim=embed_dim, num_heads=self.num_heads,
                                                      ffn_expansion_factor=self.ffn_expansion_factor, bias=self.bias,
                                                      LayerNorm_type=self.LayerNorm_type) for i in range(self.depth)])
        self.TB_l4_No2 = nn.Sequential(*[TransformerBlock(dim=embed_dim, num_heads=self.num_heads,
                                                      ffn_expansion_factor=self.ffn_expansion_factor, bias=self.bias,
                                                      LayerNorm_type=self.LayerNorm_type) for i in range(self.depth)])
        self.TB_l4_No3 = nn.Sequential(*[TransformerBlock(dim=embed_dim, num_heads=self.num_heads,
                                                      ffn_expansion_factor=self.ffn_expansion_factor, bias=self.bias,
                                                      LayerNorm_type=self.LayerNorm_type) for i in range(self.depth)])
        self.GFM_l0_No1 = GFM(embed_dim)
        self.GFM_l0_No2 = GFM(embed_dim)
        self.GFM_l1_No1 = GFM(embed_dim)
        self.GFM_l1_No2 = GFM(embed_dim)
        self.GFM_l1_No3 = GFM(embed_dim)
        self.GFM_l1_No4 = GFM(embed_dim)
        self.GFM_l2_No1 = GFM(embed_dim)
        self.GFM_l2_No2 = GFM(embed_dim)
        self.GFM_l2_No3 = GFM(embed_dim)
        self.GFM_l2_No4 = GFM(embed_dim)
        self.GFM_l3_No1 = GFM(embed_dim)
        self.GFM_l3_No2 = GFM(embed_dim)
        self.GFM_l3_No3 = GFM(embed_dim)
        self.GFM_l3_No4 = GFM(embed_dim)
        self.GFM_l4_No1 = GFM(embed_dim)
        self.GFM_l4_No2 = GFM(embed_dim)
        self.fuse1 = convd(embed_dim * 4, embed_dim, 3, 1)
        self.fuse2 = convd(embed_dim * 4, embed_dim, 3, 1)
        self.fuse3 = convd(embed_dim * 4, embed_dim, 3, 1)
        self.fuse4 = convd(embed_dim * 4, embed_dim, 3, 1)
        self.fuse5 = convd(embed_dim * 4, embed_dim, 3, 1)
        self.fuse6 = convd(embed_dim * 4, embed_dim, 3, 1)
        self.fuse7 = convd(embed_dim * 4, embed_dim, 3, 1)
        self.fuse8 = convd(embed_dim * 4, embed_dim, 3, 1)
        self.ifuse8 = convd(embed_dim, embed_dim * 4, 3, 1)
        self.ifuse7 = convd(embed_dim, embed_dim * 4, 3, 1)
        self.ifuse6 = convd(embed_dim, embed_dim * 4, 3, 1)
        self.ifuse5 = convd(embed_dim, embed_dim * 4, 3, 1)
        self.ifuse4 = convd(embed_dim, embed_dim * 4, 3, 1)
        self.ifuse3 = convd(embed_dim, embed_dim * 4, 3, 1)
        self.ifuse2 = convd(embed_dim, embed_dim * 4, 3, 1)
        self.ifuse1 = convd(embed_dim, embed_dim * 4, 3, 1)

        self.nlwt_catone = nlwt_catone()
        self.inlwt_catone = inlwt_catone()

    def check_image_size(self, x):
        _, _, h, w = x.size()
        size = 16
        mod_pad_h = (size - h % size) % size
        mod_pad_w = (size - w % size) % size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        ori_size = [h, w]
        return x, ori_size

    def restore_image_size(self, x, ori_size):
        return x[:, :, :ori_size[0], :ori_size[1]]

    def forward(self, x):
        x_check, ori_size = self.check_image_size(x)
        I0 = self.convert(x_check)

        #Scale-guide branch
        I1_LH, l0_orisize = self.nlwt_catone(I0)
        S1 = self.conv_l1_No1(I1_LH)
        F1 = self.TB_l1_No1(S1)

        I2_LH, l1_orisize = self.nlwt_catone(S1)
        S2 = self.conv_l2_No1(I2_LH)
        F2 = self.TB_l2_No1(S2)

        I3_LH, l2_orisize = self.nlwt_catone(S2)
        S3 = self.conv_l3_No1(I3_LH)
        F3 = self.TB_l3_No1(S3)

        I4_LH, l3_orisize  = self.nlwt_catone(S3)
        S4 = self.conv_l4_No1(I4_LH)
        F4 = self.TB_l4_No1(S4)

        #Encoding branch
        I0_conv = self.conv_l0_No1(I0)
        EB0 = self.TB_l0_No1(I0_conv)
        EB0_LH, _ = self.nlwt_catone(EB0)
        EB0_lwt = self.fuse1(EB0_LH)

        EB1 = self.TB_l1_No2(self.GFM_l1_No1(EB0_lwt, F1))
        EB1_LH, _ = self.nlwt_catone(EB1)
        EB1_lwt = self.fuse2(EB1_LH)

        EB2 = self.TB_l2_No2(self.GFM_l2_No1(EB1_lwt, F2))
        EB2_LH, _= self.nlwt_catone(EB2)
        EB2_lwt = self.fuse3(EB2_LH)

        EB3 = self.TB_l3_No2(self.GFM_l3_No1(EB2_lwt, F3))
        EB3_LH, _ = self.nlwt_catone(EB3)
        EB3_lwt = self.fuse4(EB3_LH)

        EB4 = self.GFM_l4_No1(EB3_lwt, F4)

        #Shallow decoder
        SD4 = self.TB_l4_No2(EB4)

        SD4_iwt = self.inlwt_catone(self.ifuse4(SD4),l3_orisize)

        SD3 = self.TB_l3_No3(self.GFM_l3_No2(SD4_iwt, EB3))
        SD3_iwt = self.inlwt_catone(self.ifuse3(SD3),l2_orisize)

        SD2 = self.TB_l2_No3(self.GFM_l2_No2(SD3_iwt, EB2))
        SD2_iwt = self.inlwt_catone(self.ifuse2(SD2),l1_orisize)

        SD1 = self.TB_l1_No3(self.GFM_l1_No2(SD2_iwt, EB1))
        SD1_iwt = self.inlwt_catone(self.ifuse1(SD1),l0_orisize)

        SD0 = self.TB_l0_No2(self.GFM_l0_No1(SD1_iwt, EB0))

        x2 = I0 - SD0

        #Serial Guided Encoder
        DE0 = self.TB_l0_No3(x2)
        DE0_LH, _ = self.nlwt_catone(DE0)
        DE0_lwt = self.fuse5(DE0_LH)

        DE1 = self.TB_l1_No4(self.GFM_l1_No3(DE0_lwt, SD1))
        DE1_LH, _ = self.nlwt_catone(DE1)
        DE1_lwt = self.fuse6(DE1_LH)

        DE2 = self.TB_l2_No4(self.GFM_l2_No3(DE1_lwt, SD2))
        DE2_LH, _ = self.nlwt_catone(DE2)
        DE2_lwt = self.fuse7(DE2_LH)

        DE3 = self.TB_l3_No4(self.GFM_l3_No3(DE2_lwt, SD3))
        DE3_LH, _ = self.nlwt_catone(DE3)
        DE3_lwt = self.fuse8(DE3_LH)

        DE4 = self.GFM_l4_No2(DE3_lwt, SD4)

        #Deep Decoder
        DD4 = self.TB_l4_No3(DE4)
        DD4_iwt = self.inlwt_catone(self.ifuse8(DD4),l3_orisize)

        DD3 = self.TB_l3_No5(self.GFM_l3_No4(DD4_iwt, DE3))
        DD3_iwt = self.inlwt_catone(self.ifuse7(DD3), l2_orisize)

        DD2 = self.TB_l2_No5(self.GFM_l2_No4(DD3_iwt, DE2))
        DD2_iwt = self.inlwt_catone(self.ifuse6(DD2), l1_orisize)

        DD1 = self.TB_l1_No5(self.GFM_l1_No4(DD2_iwt, DE1))
        DD1_iwt = self.inlwt_catone(self.ifuse5(DD1),l0_orisize)

        DD0 = self.TB_l0_No4(self.GFM_l0_No2(DD1_iwt, DE0))

        y = x2 - DD0

        out = self.restore_image_size(self.out(y), ori_size)

        return out





