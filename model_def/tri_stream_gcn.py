import torch
import torch.nn as nn
import torch.nn.functional as F

from model_def.st_gcn import Graph, st_gcn_block


class FeatureFusion(nn.Module):
    """特征融合模块，类似于TriStar的设计"""

    def __init__(self, in_channels, fused_in_channels=None, mlp_ratio=2, stride=1):
        super().__init__()

        self.fused_in_channels = fused_in_channels
        self.stride = stride

        # 升维层 - 输入通道数要匹配GCN输出
        self.f1_student = nn.Sequential(
            nn.Conv2d(in_channels, mlp_ratio * in_channels, (1, 1)),
            nn.BatchNorm2d(mlp_ratio * in_channels),
            nn.ReLU(),
        )

        self.f1_reference = nn.Sequential(
            nn.Conv2d(in_channels, mlp_ratio * in_channels, (1, 1)),
            nn.BatchNorm2d(mlp_ratio * in_channels),
            nn.ReLU(),
        )

        # 只有在指定了融合特征输入通道数时才创建处理层
        if self.fused_in_channels is not None:
            self.f2_fused = nn.Sequential(
                # 使用(kernel_size, 1)的卷积核，只在时间维度上进行下采样
                nn.Conv2d(
                    self.fused_in_channels,
                    mlp_ratio * in_channels,
                    kernel_size=(3, 1),  # 只在时间维度使用3x3卷积
                    stride=(stride, 1),  # 只在时间维度使用步长
                    padding=(1, 0),  # 相应的padding
                ),
                nn.BatchNorm2d(mlp_ratio * in_channels),
                nn.ReLU(),
            )

        # 降维层 - 输出通道数保持与输入相同
        self.g = nn.Sequential(
            nn.Conv2d(mlp_ratio * in_channels, in_channels, (1, 1)),
            nn.BatchNorm2d(in_channels),
        )

        self.act = nn.ReLU()

    def forward(self, student, reference, fused=None):
        # 保存输入用于残差连接
        student_res = student
        reference_res = reference

        # 1. 学生-参考序列融合(在高维空间)
        student_high = self.f1_student(student)
        reference_high = self.f1_reference(reference)
        sr_high = self.act(student_high) * reference_high

        # 如果没有融合特征输入，直接返回学生-参考融合结果
        if fused is None:
            out = self.g(sr_high)
            return out + student_res + reference_res

        # 2. 与已有的融合特征进行融合
        fused_high = self.f2_fused(fused)
        out = self.g(self.act(sr_high) * fused_high)

        # 3. 添加残差连接
        return out + student_res + reference_res


class TriStreamGCN(nn.Module):
    """三流图卷积网络"""

    def __init__(
        self,
        in_channels=2,  # 每个序列的输入通道数（角度+置信度）
        graph_cfg={"layout": "dual_coco", "strategy": "spatial", "max_hop": 2},
        edge_importance_weighting=True,
        **kwargs,
    ):
        super().__init__()

        # 加载图结构
        self.graph = Graph(**graph_cfg)
        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer("A", A)

        # 构建两个独立的GCN流
        self.student_stream = self._make_gcn_stream(in_channels, A.size(0))
        self.reference_stream = self._make_gcn_stream(in_channels, A.size(0))

        # 特征融合模块 - 从32通道开始融合，并指定步长
        self.fusion_modules = nn.ModuleList(
            [
                FeatureFusion(32),  # 第1次融合
                FeatureFusion(128, fused_in_channels=32, stride=2),  # 第2次融合，步长2
                FeatureFusion(512, fused_in_channels=128, stride=2),  # 第3次融合，步长2
            ]
        )

        # 边的重要性权重
        if edge_importance_weighting:
            self.edge_importance = nn.ParameterList(
                [
                    nn.Parameter(torch.ones(self.A.size()))
                    for i in range(len(self.student_stream))
                ]
            )
        else:
            self.edge_importance = [1] * len(self.student_stream)

        # 输出头
        self.fcn = nn.Conv2d(512, 1, kernel_size=1)

    def _make_gcn_stream(self, in_channels, spatial_kernel_size):
        """构建单个GCN流，通道数快速增长"""
        return nn.ModuleList(
            (
                st_gcn_block(in_channels, 8, (9, spatial_kernel_size), residual=False),
                st_gcn_block(8, 32, (9, spatial_kernel_size)),
                st_gcn_block(32, 128, (9, spatial_kernel_size), stride=2),
                st_gcn_block(128, 512, (9, spatial_kernel_size), stride=2),
            )
        )

    def forward(self, x):
        # 分离学生序列和参考序列
        N, C, T, V = x.size()
        student = x[:, :2]  # 学生序列的两个通道
        reference = x[:, 2:]  # 参考序列的两个通道

        fusion_features = []

        # 前向传播每个流
        for i, (student_gcn, reference_gcn) in enumerate(
            zip(self.student_stream, self.reference_stream)
        ):
            # GCN处理
            student, _ = student_gcn(student, self.A * self.edge_importance[i])
            reference, _ = reference_gcn(reference, self.A * self.edge_importance[i])

            # 从第2层(32通道)开始进行特征融合
            if i >= 1:
                fusion_idx = i - 1
                if i == 1:  # 第一次融合 (32通道)
                    fused = self.fusion_modules[fusion_idx](student, reference)
                else:  # 后续融合
                    fused = self.fusion_modules[fusion_idx](
                        student, reference, fusion_features[-1]
                    )
                fusion_features.append(fused)

        # 使用最后一层的融合特征
        final_fused = fusion_features[-1]  # 512通道的融合特征

        # 全局池化
        out = F.avg_pool2d(final_fused, final_fused.size()[2:])

        # 输出相似度分数
        out = self.fcn(out)
        return out.view(N)
