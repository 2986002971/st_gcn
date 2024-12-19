import torch
import torch.nn as nn
import torch.nn.functional as F

from model_def.st_gcn import Graph, st_gcn_block


class LightClassifier(nn.Module):
    """轻量级动作分类网络"""

    def __init__(
        self,
        in_channels=2,  # 角度和置信度两个通道
        num_classes=15,  # 14个标准动作 + 1个其他类
        graph_cfg={"layout": "dual_coco", "strategy": "spatial", "max_hop": 2},
        edge_importance_weighting=False,
        **kwargs,
    ):
        super().__init__()

        # 加载图结构
        self.graph = Graph(**graph_cfg)
        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer("A", A)

        # 构建轻量级GCN网络
        spatial_kernel_size = A.size(0)
        temporal_kernel_size = 9
        kernel_size = (temporal_kernel_size, spatial_kernel_size)

        # 数据归一化层
        self.data_bn = nn.BatchNorm1d(in_channels * A.size(1))

        # GCN主干网络 - 使用较少的层和通道
        kwargs0 = {k: v for k, v in kwargs.items() if k != "dropout"}
        self.st_gcn_networks = nn.ModuleList(
            (
                st_gcn_block(
                    in_channels, 32, kernel_size, 1, residual=False, **kwargs0
                ),
                st_gcn_block(32, 64, kernel_size, 2, **kwargs),
                st_gcn_block(64, 128, kernel_size, 2, **kwargs),
            )
        )

        # 边的重要性权重
        if edge_importance_weighting:
            self.edge_importance = nn.ParameterList(
                [nn.Parameter(torch.ones(self.A.size())) for i in self.st_gcn_networks]
            )
        else:
            self.edge_importance = [1] * len(self.st_gcn_networks)

        # 分类头
        self.fcn = nn.Conv2d(128, num_classes, kernel_size=1)

    def forward(self, x):
        # 数据归一化
        N, C, T, V = x.size()
        x = x.view(N, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, C, T, V)

        # 前向传播
        for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
            x, _ = gcn(x, self.A * importance)

        # 全局池化
        x = F.avg_pool2d(x, x.size()[2:])

        # 分类预测
        x = self.fcn(x)
        return x.view(N, -1)  # [N, num_classes]
