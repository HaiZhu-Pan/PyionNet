import torch.nn as nn
import torch
from einops import rearrange
from einops.layers.torch import Rearrange

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
#         out_3d = out_3d + x1  # 添加残差

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
#         x = x + x23 # 添加残差

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = torch.relu(x)
        x = self.pool(x)
        return x


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
#         out = self.proj(out)
#         out = out + q
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
#         self.heads = heads

    def forward(self, x1, x2):
        v1 = self.to_q(x1)
        v2 = self.to_q(x2)
#         out = torch.cat((v1, v2), dim=-1)
        out = v1 + v2

        return out


class Pyion(nn.Module):
    def __init__(self, num_patches, patch_size, patch, dim, depth, heads, dim_head, mlp_dim, num_classes,
                 dropout):  # x1(16, 144, 11, 11) x2(16, 1, 11, 11)
        super(Pyion, self).__init__()
        emb_dropout = 0.
        self.name = 'Pyion'
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
        self.hsi_encoder = HSI_Encoder(in_channels_3d=1, in_depth_3d=num_patches[0], out_channels_3d=16, # out_channels_3d=32
                                       out_channels_2d=dim)
        self.lidar_encoder = LiDAR_Encoder(in_channels=num_patches[1], out_channels=dim)
        self.dropout = nn.Dropout(emb_dropout)

    def forward(self, x1, x2):
        x1 = rearrange(x1, 'b c h w d -> b c d h w ')
        x2 = rearrange(x2, 'b c h w d -> b (c d) h w ')

        x1 = self.hsi_encoder(x1)
        x2 = self.lidar_encoder(x2)
        b, n, h, w = x1.shape
        x2_1 = self.patch_to_embedding1(x1)
        x1_1 = self.patch_to_embedding2(x2)

        x1_1 += self.pos_embedding1[:, :h * w]
        x1_1 = self.dropout(x1_1)
        x2_1 += self.pos_embedding2[:, :h * w]
        x2_1 = self.dropout(x2_1)

        x1_t = self.transformer1(x2_1, x1_1)
        x2_t = self.transformer2(x1_1, x1_t)
        x3_t = self.transformer3(x2_t, x2_1)
        x_ = self.add(x3_t, x2_1)

        xs = torch.squeeze(self.mlp_head0(x_))  # b-n （64，338）通道维度上的挤压操作
        x = torch.einsum('bn,bnd->bd', xs, x_)  # （64，64）  (64,338,64)

        out = self.mlp_head1(x)  # （64,15）

        return out

if __name__ == '__main__':
    model = Pyion(num_patches=[64, 1],  # 深度可分离卷积中用到
                  patch_size=81,
                  patch=9,
                  dim=64,
                  depth=[2,1,1],
                  heads=[8,4],
                  dim_head=32,
                  mlp_dim=32,
                  num_classes=6,
                  dropout=0.1).to("cuda:0")

    input1 = torch.randn(8, 1, 7, 7, 64).to("cuda:0")
    input2 = torch.randn(8, 1, 7, 7, 1).to("cuda:0")
    # input3 = torch.randn(1,63,243).to("cuda")
    out = model(input1, input2)
    print(out.size())
    # summary(model, [( 1,7, 7,63), (1,7, 7,1)],device='cuda')