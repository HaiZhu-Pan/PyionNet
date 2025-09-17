import torch.nn as nn
import torch
from einops import rearrange
from einops.layers.torch import Rearrange


def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1, groups=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                     padding=padding, dilation=dilation, groups=groups, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class PyConv4(nn.Module):
    def __init__(self, inplans, planes, out_planes_div=[4, 4, 4, 4],
                 pyconv_kernels=[3, 5, 7, 9], stride=1, pyconv_groups=[1, 4, 8, 16]):
        super(PyConv4, self).__init__()
        self.conv2_1 = conv(inplans, planes // out_planes_div[0], kernel_size=pyconv_kernels[0],
                            padding=pyconv_kernels[0] // 2, stride=stride, groups=pyconv_groups[0])
        self.conv2_2 = conv(inplans, planes // out_planes_div[1], kernel_size=pyconv_kernels[1],
                            padding=pyconv_kernels[1] // 2, stride=stride, groups=pyconv_groups[1])
        self.conv2_3 = conv(inplans, planes // out_planes_div[2], kernel_size=pyconv_kernels[2],
                            padding=pyconv_kernels[2] // 2, stride=stride, groups=pyconv_groups[2])
        self.conv2_4 = conv(inplans, planes // out_planes_div[3], kernel_size=pyconv_kernels[3],
                            padding=pyconv_kernels[3] // 2, stride=stride, groups=pyconv_groups[3])

    def forward(self, x):
        return torch.cat((self.conv2_1(x), self.conv2_2(x), self.conv2_3(x), self.conv2_4(x)), dim=1)


class PyConv3(nn.Module):
    def __init__(self, inplans, planes, out_planes_div=[4, 4, 2],
                 pyconv_kernels=[3, 5, 7], stride=1, pyconv_groups=[1, 4, 8]):
        super(PyConv3, self).__init__()
        self.conv2_1 = conv(inplans, planes // out_planes_div[0], kernel_size=pyconv_kernels[0],
                            padding=pyconv_kernels[0] // 2, stride=stride, groups=pyconv_groups[0])
        self.conv2_2 = conv(inplans, planes // out_planes_div[1], kernel_size=pyconv_kernels[1],
                            padding=pyconv_kernels[1] // 2, stride=stride, groups=pyconv_groups[1])
        self.conv2_3 = conv(inplans, planes // out_planes_div[2], kernel_size=pyconv_kernels[2],
                            padding=pyconv_kernels[2] // 2, stride=stride, groups=pyconv_groups[2])

    def forward(self, x):
        return torch.cat((self.conv2_1(x), self.conv2_2(x), self.conv2_3(x)), dim=1)


class PyConv2(nn.Module):
    def __init__(self, inplans, planes, out_planes_div=[2, 2],
                 pyconv_kernels=[3, 5], stride=1, pyconv_groups=[1, 4]):
        super(PyConv2, self).__init__()
        self.conv2_1 = conv(inplans, planes // out_planes_div[0], kernel_size=pyconv_kernels[0],
                            padding=pyconv_kernels[0] // 2, stride=stride, groups=pyconv_groups[0])
        self.conv2_2 = conv(inplans, planes // out_planes_div[1], kernel_size=pyconv_kernels[1],
                            padding=pyconv_kernels[1] // 2, stride=stride, groups=pyconv_groups[1])

    def forward(self, x):
        return torch.cat((self.conv2_1(x), self.conv2_2(x)), dim=1)


def get_pyconv(inplans, planes, pyconv_kernels, out_planes_div, stride=1, pyconv_groups=[1]):
    if len(pyconv_kernels) == 1:
        return conv(inplans, planes, kernel_size=pyconv_kernels[0], stride=stride, groups=pyconv_groups[0])
    elif len(pyconv_kernels) == 2:
        return PyConv2(inplans, planes, out_planes_div=out_planes_div, pyconv_kernels=pyconv_kernels,
                       stride=stride, pyconv_groups=pyconv_groups)
    elif len(pyconv_kernels) == 3:
        return PyConv3(inplans, planes, out_planes_div=out_planes_div, pyconv_kernels=pyconv_kernels,
                       stride=stride, pyconv_groups=pyconv_groups)
    elif len(pyconv_kernels) == 4:
        return PyConv4(inplans, planes, out_planes_div=out_planes_div, pyconv_kernels=pyconv_kernels,
                       stride=stride, pyconv_groups=pyconv_groups)


class HSI_Encoder(nn.Module):
    def __init__(self, in_channels_3d=1, out_channels_3d=16,
                 in_depth_3d=144, out_channels_2d=64):
        super(HSI_Encoder, self).__init__()
        self.relu = nn.ReLU()

        # 3d
        self.conv1 = nn.Conv3d(in_channels=in_channels_3d, out_channels=out_channels_3d, kernel_size=(11, 3, 3),
                               stride=(3, 1, 1), padding=(5, 1, 1))
        self.bn1 = nn.BatchNorm3d(out_channels_3d)

        # self.conv2_1 = nn.Conv3d(in_channels=out_channels_3d, out_channels=out_channels_3d // 4, kernel_size=(1, 1, 1),
        #                          stride=(1, 1, 1), padding=(0, 0, 0))
        self.conv2_2 = nn.Conv3d(in_channels=out_channels_3d, out_channels=out_channels_3d // 2, kernel_size=(3, 1, 1),
                                 stride=(1, 1, 1), padding=(1, 0, 0))
        self.conv2_3 = nn.Conv3d(in_channels=out_channels_3d, out_channels=out_channels_3d // 2, kernel_size=(5, 1, 1),
                                 stride=(1, 1, 1), padding=(2, 0, 0))
        self.conv2_4 = nn.Conv3d(in_channels=out_channels_3d, out_channels=out_channels_3d // 2, kernel_size=(7, 1, 1),
                                 stride=(1, 1, 1), padding=(3, 0, 0))
        self.bn2 = nn.BatchNorm3d((out_channels_3d//2)*3)

        self.conv3 = nn.Conv3d(in_channels=(out_channels_3d//2)*3, out_channels=out_channels_3d, kernel_size=(3, 3, 3),
                               stride=(1, 1, 1), padding=(1, 1, 1))
        self.bn3 = nn.BatchNorm3d(out_channels_3d)

        # 2d
        self.in_channels_2d = int((in_depth_3d + 2) / 3) * out_channels_3d
        self.conv4 = get_pyconv(inplans=self.in_channels_2d, planes=out_channels_2d, stride=1,
                                pyconv_kernels=[3, 5, 7], out_planes_div=[2, 2, 2], pyconv_groups=[4, 4, 8])
        self.bn4 = nn.BatchNorm2d((out_channels_2d//2)*3)

        self.conv5 = nn.Conv2d(in_channels=(out_channels_2d//2)*3, out_channels=out_channels_2d, kernel_size=1, stride=1)
        self.bn5 = nn.BatchNorm2d(out_channels_2d)

        self.pool = nn.MaxPool2d(2)

    def forward(self, hsi_img):
        # 3d
        x1 = self.conv1(hsi_img)
        x1 = self.bn1(x1)
        x1 = self.relu(x1)

        x2 = torch.cat((self.conv2_2(x1), self.conv2_3(x1), self.conv2_4(x1)), dim=1)  #
        x2 = self.bn2(x2)
        x2 = self.relu(x2)


        x3 = self.conv3(x2)
        x3 = self.bn3(x3)
        out_3d = self.relu(x3)

        # 2d
        x = rearrange(out_3d, 'b c h w y ->b (c h) w y')

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)

        x = self.conv5(x)
        x = self.bn5(x)
        x = torch.relu(x)
        x = self.pool(x)

        return x


class LiDAR_Encoder(nn.Module):
    def __init__(self, in_channels=1, out_channels=64):
        super(LiDAR_Encoder, self).__init__()
        self.relu = nn.ReLU()

        self.conv1 = get_pyconv(inplans=in_channels, planes=out_channels//2, stride=1,
                                pyconv_kernels=[3, 5, 7], out_planes_div=[2, 2, 2], pyconv_groups=[1, 1, 1])
        self.bn1 = nn.BatchNorm2d((out_channels//4)*3)
#         self.conv1_2 = nn.Conv2d(in_channels=(out_channels//4)*3, out_channels=out_channels//2, kernel_size=1, stride=1)

        self.conv2 = get_pyconv(inplans=(out_channels//4)*3, planes=out_channels, stride=1,
                                pyconv_kernels=[3, 5, 7], out_planes_div=[2, 2, 2], pyconv_groups=[1, 1, 1])
        self.bn2 = nn.BatchNorm2d((out_channels//2)*3)

        self.conv3 = nn.Conv2d(in_channels=(out_channels//2)*3, out_channels=out_channels, kernel_size=1, stride=1)
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.pool = nn.MaxPool2d(2)

    def forward(self, x23):
        x = self.conv1(x23)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = torch.relu(x)
        x = self.pool(x)
        return x


class Attention1(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super(Attention1, self).__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)
        proj_drop = 0.0
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.to_qk = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()
        self.proj = nn.Linear(32, 32)  # 32这个数得再改，不能写死
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x1, x2):
        kv = self.to_qk(x1).chunk(2, dim=-1)
        q = self.to_v(x2).chunk(1, dim=-1)
        k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), kv)
        q = torch.cat(list(map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), q)), dim=2)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        attn = self.dropout(attn)
        attn1 = torch.matmul(attn, v)
        attn2 = v - attn1
        attn3 = self.proj(attn2)
        attn3 = self.proj_drop(attn3)
        attn4 = attn3 + q
        out = rearrange(attn4, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super(Attention, self).__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.to_qk = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()
        self.proj = nn.Linear(32, 32)  # 32这个数得再改，不能写死

    def forward(self, x1, x2):
        kv = self.to_qk(x1).chunk(2, dim=-1)
        q = self.to_v(x2).chunk(1, dim=-1)
        k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), kv)
        q = torch.cat(list(map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), q)), dim=2)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super(FeedForward, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super(PreNorm, self).__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x1, x2=None, **kwargs):
        if x2 is not None:
            return self.fn(self.norm(x1), self.norm(x2), **kwargs)
        else:
            return self.fn(self.norm(x1), **kwargs)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super(Transformer, self).__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x1, x2):
        for attn, ff in self.layers:
            x = attn(x1, x2)
            x = ff(x) + x
        return x


class Transformer1(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super(Transformer1, self).__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention1(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x1, x2):
        for attn, ff in self.layers:
            x = attn(x1, x2)
            x = ff(x) + x
        return x


class add(nn.Module):
    def __init__(self, dim, dim_head, heads):
        super(add, self).__init__()
        inner_dim = dim_head * heads
        self.to_q = nn.Linear(dim, inner_dim)

    def forward(self, x1, x2):
        v1 = self.to_q(x1)
        v2 = self.to_q(x2)
#         out = torch.cat((v1, v2), dim=-1)
        out = v1 + v2

        return out


class PyionNet(nn.Module):
    def __init__(self, num_channels, patch_size, patch, dim, depth, heads, dim_head, mlp_dim, num_classes,
                 dropout):
        super(PyionNet, self).__init__()
        emb_dropout = 0.
        self.name = 'MYNet'
        self.num_classes = num_classes
        self.patch_to_embedding1 = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=1, p2=1),
        )
        self.patch_to_embedding2 = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=1, p2=1),
        )
        self.transformer1 = Transformer1(dim, depth[0], heads[0], dim_head, mlp_dim, dropout=dropout)
        self.transformer2 = Transformer(dim, depth[1], heads[0], dim_head, mlp_dim, dropout=dropout)
        self.transformer3 = Transformer(dim, depth[2], heads[0], dim_head, mlp_dim, dropout=dropout)
#         self.transformer4 = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout=dropout)
        self.add = add(dim, dim_head, heads[1])

        self.mlp_head0 = nn.Sequential(
            nn.LayerNorm(dim_head * heads[1]),
            nn.Linear(dim_head * heads[1], 1),
            nn.Softmax(dim=1)
        )

        self.mlp_head1 = nn.Sequential(
            nn.LayerNorm(dim_head * heads[1]),
            nn.Linear(dim_head * heads[1], num_classes)
        )
        self.pos_embedding1 = nn.Parameter(torch.randn(1, patch_size, dim))
        self.pos_embedding2 = nn.Parameter(torch.randn(1, patch_size, dim))
        self.hsi_encoder = HSI_Encoder(in_channels_3d=1, in_depth_3d=num_channels[0], out_channels_3d=16, # out_channels_3d=32
                                       out_channels_2d=dim)
        self.lidar_encoder = LiDAR_Encoder(in_channels=num_channels[1], out_channels=dim)
        self.dropout = nn.Dropout(emb_dropout)

    def forward(self, x1, x2):
        x1 = rearrange(x1, 'b c h w d -> b c d h w ')
        x2 = rearrange(x2, 'b c h w d -> b (c d) h w ')

        x1 = self.hsi_encoder(x1)
        x2 = self.lidar_encoder(x2)
        b, n, h, w = x1.shape
        x1_1 = self.patch_to_embedding1(x1)
        x2_1 = self.patch_to_embedding2(x2)

        x1_1 += self.pos_embedding1[:, :h * w]
        x1_1 = self.dropout(x1_1)
        x2_1 += self.pos_embedding2[:, :h * w]
        x2_1 = self.dropout(x2_1)

        x1_t = self.transformer1(x1_1, x2_1)
        x2_t = self.transformer2(x2_1, x1_t)
        x3_t = self.transformer3(x2_t, x1_1)
        x_ = self.add(x3_t, x1_1)

        xs = torch.squeeze(self.mlp_head0(x_))  # b-n （64，338）通道维度上的挤压操作
        x = torch.einsum('bn,bnd->bd', xs, x_)  # （64，64）  (64,338,64)

        out = self.mlp_head1(x)  # （64,15）

        return out

if __name__ == '__main__':
    model = PyionNet(num_channels=[63, 1],  # 深度可分离卷积中用到
                  patch_size=49,
                  patch=7,
                  dim=64,
                  depth=[2,1,1],
                  heads=[8,4],
                  dim_head=32,
                  mlp_dim=32,
                  num_classes=6,
                  dropout=0.1).to("cuda:0")

    input1 = torch.randn(128, 1, 7, 7, 63).to("cuda:0")
    input2 = torch.randn(128, 1, 7, 7, 1).to("cuda:0")
    out = model(input1, input2)
    print(out.size())