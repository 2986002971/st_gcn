import argparse
import os
import pickle
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from model_def.st_gcn import ST_GCN_18


def parse_args():
    parser = argparse.ArgumentParser(description="训练ST-GCN模型")
    parser.add_argument(
        "--work_dir",
        type=str,
        default="./model_def",
        help="工作目录",
    )
    parser.add_argument("--batch_size", type=int, default=16, help="批次大小")
    parser.add_argument("--resume_from", type=str, help="恢复训练的检查点文件")
    parser.add_argument(
        "--data_dir", type=str, default="./processed", help="数据集文件夹路径"
    )
    # 添加最大轮数参数
    parser.add_argument(
        "--max_epochs", type=int, default=500, help="每个阶段的最大训练轮数"
    )
    return parser.parse_args()


def auto_pading(data_numpy, size, random_pad=False):
    C, T, V, M = data_numpy.shape
    if T < size:
        begin = random.randint(0, size - T) if random_pad else 0
        data_numpy_paded = np.zeros((C, size, V, M), dtype=data_numpy.dtype)
        data_numpy_paded[:, begin : begin + T, :, :] = data_numpy
        return data_numpy_paded
    else:
        return data_numpy


def random_choose(data_numpy, size, auto_pad=True):
    # input: C,T,V,M
    C, T, V, M = data_numpy.shape
    if T == size:
        return data_numpy
    elif T < size:
        if auto_pad:
            return auto_pading(data_numpy, size, random_pad=True)
        else:
            return data_numpy
    else:
        begin = random.randint(0, T - size)
        return data_numpy[:, begin : begin + size, :, :]


def random_move(
    data_numpy,
    angle_candidate=[-10.0, -5.0, 0.0, 5.0, 10.0],
    scale_candidate=[0.9, 1.0, 1.1],
    transform_candidate=[-0.2, -0.1, 0.0, 0.1, 0.2],
    move_time_candidate=[1],
):
    # input: C,T,V,M
    C, T, V, M = data_numpy.shape
    move_time = random.choice(move_time_candidate)
    node = np.arange(0, T, T * 1.0 / move_time).round().astype(int)
    node = np.append(node, T)
    num_node = len(node)

    A = np.random.choice(angle_candidate, num_node)
    S = np.random.choice(scale_candidate, num_node)
    T_x = np.random.choice(transform_candidate, num_node)
    T_y = np.random.choice(transform_candidate, num_node)

    a = np.zeros(T)
    s = np.zeros(T)
    t_x = np.zeros(T)
    t_y = np.zeros(T)

    # linspace
    for i in range(num_node - 1):
        a[node[i] : node[i + 1]] = (
            np.linspace(A[i], A[i + 1], node[i + 1] - node[i]) * np.pi / 180
        )
        s[node[i] : node[i + 1]] = np.linspace(S[i], S[i + 1], node[i + 1] - node[i])
        t_x[node[i] : node[i + 1]] = np.linspace(
            T_x[i], T_x[i + 1], node[i + 1] - node[i]
        )
        t_y[node[i] : node[i + 1]] = np.linspace(
            T_y[i], T_y[i + 1], node[i + 1] - node[i]
        )

    theta = np.array([[np.cos(a) * s, -np.sin(a) * s], [np.sin(a) * s, np.cos(a) * s]])

    # perform transformation
    for i_frame in range(T):
        xy = data_numpy[0:2, i_frame, :, :]
        new_xy = np.dot(theta[:, :, i_frame], xy.reshape(2, -1))
        new_xy[0] += t_x[i_frame]
        new_xy[1] += t_y[i_frame]
        data_numpy[0:2, i_frame, :, :] = new_xy.reshape(2, V, M)

    return data_numpy


class SkeletonFeeder(Dataset):
    def __init__(
        self,
        data_path,
        label_path,
        standard_data_path="./processed_standard/train_data.npy",
        random_choose=False,
        random_move=False,
        window_size=-1,
    ):
        self.data_path = data_path
        self.label_path = label_path
        self.standard_data_path = standard_data_path
        self.random_choose = random_choose
        self.random_move = random_move
        self.window_size = window_size
        self.load_data()

    def load_data(self):
        # 加载原始数据
        with open(self.data_path, "rb") as f:
            self.data = np.load(f, allow_pickle=True)
        with open(self.label_path, "rb") as f:
            self.sample_name, self.label, self.accuracy = pickle.load(f)

        # 加载标准视频数据
        self.standard_data = np.load(self.standard_data_path, allow_pickle=True)
        print(f"标准视频数据形状: {self.standard_data.shape}")

        # 确保标签和准确度是正确的数据类型
        self.label = np.array(self.label, dtype=np.int64)
        self.accuracy = np.array(self.accuracy, dtype=np.float32)

    def __len__(self):
        return len(self.label)

    def get_phase2_data(self, index, pred_label):
        """获取第二阶段的数据（包含标准视频）"""
        data_numpy = self.data[index]
        if self.random_choose:
            data_numpy = random_choose(data_numpy, self.window_size, auto_pad=True)
        if self.random_move:
            data_numpy = random_move(data_numpy)

        # 获取对应预测标签的标准视频
        std_data = self.standard_data[pred_label]
        if self.random_choose:
            std_data = random_choose(std_data, self.window_size, auto_pad=True)
        if self.random_move:
            std_data = random_move(std_data)

        combined_data = np.concatenate([data_numpy, std_data], axis=0)
        return torch.FloatTensor(combined_data)

    def __getitem__(self, index):
        """获取第一阶段的数据（不包含标准视频）"""
        data_numpy = self.data[index]
        label = self.label[index]
        accuracy = self.accuracy[index]

        if self.random_choose:
            data_numpy = random_choose(data_numpy, self.window_size, auto_pad=True)
        if self.random_move:
            data_numpy = random_move(data_numpy)

        # Phase 1: 原始数据加零填充
        zero_channels = np.zeros_like(data_numpy)
        combined_data = np.concatenate([data_numpy, zero_channels], axis=0)

        return (
            torch.FloatTensor(combined_data),
            torch.LongTensor([label]).squeeze(),
            torch.FloatTensor([accuracy]).squeeze(),
            torch.LongTensor([index]),
        )


def get_dataloader(data_path, label_path, batch_size, **kwargs):
    dataset = SkeletonFeeder(data_path, label_path, **kwargs)
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True
    )


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 模型定义
    model = ST_GCN_18(
        in_channels=6,
        num_class=14,
        edge_importance_weighting=True,
        graph_cfg={"layout": "coco", "strategy": "spatial", "max_hop": 2},
    ).to(device)

    criterion_cls = nn.CrossEntropyLoss()
    criterion_quality = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_loader = get_dataloader(
        os.path.join(args.data_dir, "train_data.npy"),
        os.path.join(args.data_dir, "train_label.pkl"),
        args.batch_size,
        random_choose=True,
        random_move=True,
        window_size=600,
    )

    val_loader = get_dataloader(
        os.path.join(args.data_dir, "val_data.npy"),
        os.path.join(args.data_dir, "val_label.pkl"),
        args.batch_size,
    )

    best_quality_loss = float("inf")

    for epoch in range(args.max_epochs):
        model.train()
        total_cls_loss = 0
        total_quality_loss = 0
        correct_samples = 0
        total_samples = 0

        for batch_idx, (data, target, accuracy, indices) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            accuracy = accuracy.to(device)
            indices = indices.to(device)

            # Phase 1: 分类
            optimizer.zero_grad()  # 在每个批次开始时清除梯度
            cls_out, _ = model(data, output_quality=True)
            cls_loss = criterion_cls(cls_out, target)
            cls_loss.backward()  # 对所有样本进行反向传播
            total_cls_loss += cls_loss.item()

            pred = cls_out.argmax(dim=1)
            correct_mask = pred.eq(target)

            # Phase 2: 对分类正确的样本评估标准度
            if correct_mask.any():
                correct_indices = indices[correct_mask]
                correct_preds = pred[correct_mask]
                correct_accuracies = accuracy[correct_mask]

                phase2_data = []
                for idx, pred_label in zip(correct_indices, correct_preds):
                    phase2_data.append(
                        train_loader.dataset.get_phase2_data(idx, pred_label)
                    )
                phase2_data = torch.stack(phase2_data).to(device)

                # Phase 2前向传播
                _, phase2_quality = model(phase2_data, output_quality=True)

                # 计算标准度损失
                quality_loss = criterion_quality(
                    phase2_quality.squeeze(), correct_accuracies
                )
                quality_loss.backward()  # 对正确分类的样本进行标准度损失的反向传播
                total_quality_loss += quality_loss.item()

            # 统一进行优化器步骤
            optimizer.step()

            # 统计
            correct_samples += correct_mask.sum().item()
            total_samples += target.size(0)

            if batch_idx % 100 == 0:
                print(
                    f"Epoch {epoch} - Batch {batch_idx}: "
                    f"Classification Loss: {total_cls_loss/(batch_idx+1):.4f}, "
                    f"Quality Loss: {total_quality_loss/(batch_idx+1):.4f}, "
                    f"Accuracy: {correct_samples/total_samples:.4f}"
                )

        # 每5个epoch进行一次验证
        if epoch % 5 == 0:
            model.eval()
            val_cls_loss = 0
            val_quality_loss = 0
            correct = 0
            total = 0

            with torch.no_grad():
                for data, target, accuracy, _ in val_loader:
                    data, target = data.to(device), target.to(device)
                    accuracy = accuracy.to(device)

                    cls_out, quality_out = model(data, output_quality=True)
                    val_cls_loss += criterion_cls(cls_out, target).item()

                    pred = cls_out.argmax(dim=1)
                    correct_mask = pred.eq(target)

                    if correct_mask.any():
                        val_quality_loss += criterion_quality(
                            quality_out[correct_mask].squeeze(), accuracy[correct_mask]
                        ).item()

                    correct += correct_mask.sum().item()
                    total += target.size(0)

            val_accuracy = correct / total
            val_cls_loss /= len(val_loader)
            val_quality_loss /= len(val_loader)

            print(
                f"验证 - 分类损失: {val_cls_loss:.4f}, "
                f"质量损失: {val_quality_loss:.4f}, "
                f"准确率: {val_accuracy:.4f}"
            )

            # 保存最佳模型
            if val_quality_loss < best_quality_loss:
                best_quality_loss = val_quality_loss
                checkpoint_path = os.path.join(args.work_dir, "best_model.pth")
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "best_quality_loss": best_quality_loss,
                    },
                    checkpoint_path,
                )

    print("Training completed!")


if __name__ == "__main__":
    args = parse_args()
    train(args)
