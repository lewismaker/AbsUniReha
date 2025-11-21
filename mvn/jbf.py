import math

import torch
import torch.nn as nn

from .modules import Basic_Block
from .spatial_graph import SpatialGraph


def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)


class DeGCN(nn.Sequential):
    def __init__(self, block_args, A, k, eta):
        super(DeGCN, self).__init__()
        for i, [in_channels, out_channels, stride, residual, num_frame, num_joint, device,
                with_multi_scale] in enumerate(block_args):
            self.add_module(f'block-{i}_tcngcn', Basic_Block(in_channels,
                                                             out_channels,
                                                             A,
                                                             k,
                                                             eta,
                                                             stride=stride,
                                                             num_frame=num_frame,
                                                             num_joint=num_joint,
                                                             residual=residual,
                                                             device=device,
                                                             with_multi_scale=with_multi_scale))


class ElementwiseMul(nn.Module):
    def __init__(self):
        super(ElementwiseMul, self).__init__()

    def forward(self, xs):
        return torch.mul(xs[0], xs[1])


class DeGCNArc(nn.Module):
    def __init__(self, num_class=2, num_points=25, num_person=1, k=8, eta=4, num_stream=2, connectivity=None,
                 in_channels=3, drop_out=0, num_frame=30, device=None, is_ri=True):
        super(DeGCNArc, self).__init__()

        self.graph = SpatialGraph(num_joints=num_points, inward=connectivity)
        self.connectivity = connectivity
        self.is_ri = is_ri

        A = self.graph.A  # 3,25,25

        self.num_class = num_class
        self.num_points = num_points

        base_channel = 64
        base_frame = num_frame

        self.num_modal = 2
        # original
        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_points * self.num_modal)


        self.blockargs1 = [
            [in_channels, base_channel, 1, False, base_frame, num_points, device, True],
            [base_channel, base_channel * 2, 1, True, base_frame, num_points, device, True],
            [base_channel * 2, base_channel * 4, 1, True, base_frame, num_points, device, True],
        ]

        self.bn_mid = nn.BatchNorm2d(base_channel * 4)

        self.num_stream = 2
        self.streams1 = nn.ModuleList([DeGCN(self.blockargs1, A, k, eta) for _ in range(self.num_modal)])

        self.fc = nn.ModuleList([nn.Linear(base_channel * 4, num_class) for _ in range(self.num_stream - 1)])
        self.relu = nn.LeakyReLU(0.1)

        for fc in self.fc:
            nn.init.normal_(fc.weight, 0, math.sqrt(2. / num_class))
        bn_init(self.data_bn, 1)

        self.mul = ElementwiseMul()


    def forward(self, x):

        # joint
        x1 = x

        # bone
        x2 = torch.zeros_like(x)
        for v1, v2 in self.connectivity:
            x2[:, :, v1, :] = x[:, :, v1, :] - x[:, :, v2, :]

        if self.is_ri:
            x1 = torch.einsum('ijkm,ijml->ijkl', x1,
                              x1.permute(0, 1, 3, 2))
            x2 = torch.einsum('ijkm,ijml->ijkl', x2,
                              x2.permute(0, 1, 3, 2))

        # N, T, V, C -> N, C, T, V
        x1 = x1.permute(0, 3, 1, 2).contiguous()
        x2 = x2.permute(0, 3, 1, 2).contiguous()

        x = torch.cat([x1, x2], 1)

        N, C, T, V = x.size()
        x = x.permute(0, 3, 1, 2).contiguous().view(N, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, V, C, T).permute(0, 2, 3, 1).contiguous().view(N, C, T, V)

        xs = x.chunk(self.num_modal, 1)

        xs = [stream(x) for stream, x in zip(self.streams1, xs)]
        xs[0] = self.bn_mid(xs[0])
        xs[1] = self.bn_mid(xs[1])
        x = self.mul(xs)
        out = []
        for fc in self.fc:
            c_new = x.size(1)
            x = x.view(N, c_new, -1)
            x = x.mean(2)
            out.append(fc(x))

        return out
