import argparse
import os
import pickle
import random

import numpy as np
import torch
import torch.nn as nn
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
    parser.add_argument("--batch_size", type=int, default=1, help="批次大小")
    parser.add_argument("--resume_from", type=str, help="恢复训练的检查点文件")
    parser.add_argument(
        "--data_dir", type=str, default="./processed", help="数据集文件夹路径"
    )
    # 添加最大轮数参数
    parser.add_argument(
        "--max_epochs", type=int, default=2000, help="每个阶段的最大训练轮数"
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

        # 确保标签和准确度是正确的数据类型
        self.label = np.array(self.label, dtype=np.int64)
        self.accuracy = np.array(self.accuracy, dtype=np.float32)

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        data_numpy = self.data[index]
        label = self.label[index]
        accuracy = self.accuracy[index] * 100  # 将accuracy乘以100

        # 准备14个版本的数据，每个都与一个标准视频结合
        combined_data = []
        for std_idx in range(14):
            std_data = self.standard_data[std_idx]
            combined = np.concatenate([data_numpy, std_data], axis=0)

            if self.random_choose:
                combined = random_choose(combined, self.window_size)
            if self.random_move:
                combined = random_move(combined)

            combined_data.append(combined)

        combined_data = np.stack(combined_data)  # Shape: (14, C, T, V, M)

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


def evaluate(model, val_loader, criterion, device):
    model.eval()  # 确保模型处于评估模式
    total_loss = 0
    correct = 0
    total = 0

    # 添加统计信息
    class_correct = torch.zeros(15)  # 0-14类的正确数
    class_total = torch.zeros(15)  # 0-14类的总数

    with torch.no_grad():
        for data, target, accuracy, _ in val_loader:
            data, target = data.to(device), target.to(device)
            accuracy = accuracy.to(device)

            B, N, C, T, V, M = data.shape
            data = data.view(B * N, C, T, V, M)

            quality_out = model(data)
            quality_out = quality_out.view(B, N)

            # 修改目标矩阵创建逻辑
            target_matrix = torch.zeros_like(quality_out)
            for i in range(B):
                if target[i] < 14:
                    target_matrix[i, target[i]] = accuracy[i]

            # 计算损失
            loss = criterion(quality_out, target_matrix)

            # 对正确类别的损失赋予更高的权重
            weight_matrix = torch.ones_like(loss)
            for i in range(B):
                if target[i] < 14:
                    weight_matrix[i, target[i]] = 10

            weighted_loss = (loss * weight_matrix).mean()

            total_loss += weighted_loss.item()

            # 修改分类准确率计算逻辑
            pred = quality_out.max(dim=1)[1]
            for i in range(B):
                class_total[target[i]] += 1
                if target[i] == 14:  # 对于“其他”类
                    # 如果所有输出都小于某个阈值（比如0.3），则认为分类正确
                    if torch.all(quality_out[i] < 0.3):
                        correct += 1
                        class_correct[target[i]] += 1
                else:
                    if pred[i] == target[i]:
                        correct += 1
                        class_correct[target[i]] += 1
            total += B

    # 打印每个类别的准确率
    for i in range(15):
        if class_total[i] > 0:
            print(f"Class {i} Accuracy: {class_correct[i]/class_total[i]:.4f}")

    return total_loss / len(val_loader), correct / total


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 模型定义
    model = ST_GCN_18(
        in_channels=6,
        num_class=14,
        edge_importance_weighting=True,
        graph_cfg={"layout": "coco", "strategy": "spatial", "max_hop": 2},
    ).to(device)

    criterion = nn.MSELoss(reduction="none")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    train_loader = get_dataloader(
        os.path.join(args.data_dir, "train_data.npy"),
        os.path.join(args.data_dir, "train_label.pkl"),
        args.batch_size,
    )

    val_loader = get_dataloader(
        os.path.join(args.data_dir, "val_data.npy"),
        os.path.join(args.data_dir, "val_label.pkl"),
        args.batch_size,
    )

    # 添加从检查点加载模型的逻辑
    start_epoch = 0
    best_quality_loss = float("inf")

    if args.resume_from and os.path.exists(args.resume_from):
        print(f"正在从检查点加载模型: {args.resume_from}")
        checkpoint = torch.load(args.resume_from)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_quality_loss = checkpoint["best_quality_loss"]
        print(f"成功加载检查点，从第 {start_epoch} 轮继续训练")

    # 修改训练循环的起始位置
    for epoch in range(start_epoch, args.max_epochs):
        model.train()
        total_loss = 0
        correct_samples = 0
        total_samples = 0

        for batch_idx, (data, target, accuracy, indices) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            accuracy = accuracy.to(device)

            optimizer.zero_grad()

            B, N, C, T, V, M = data.shape
            data = data.view(B * N, C, T, V, M)

            quality_out = model(data)
            quality_out = quality_out.view(B, N)

            # 修改目标矩阵创建逻辑
            target_matrix = torch.zeros_like(quality_out)
            for i in range(B):
                if target[i] < 14:
                    target_matrix[i, target[i]] = accuracy[i]

            # 计算损失
            loss = criterion(quality_out, target_matrix)

            # 对正确类别的损失赋予更高的权重
            weight_matrix = torch.ones_like(loss)
            for i in range(B):
                if target[i] < 14:
                    weight_matrix[i, target[i]] = 10

            weighted_loss = (loss * weight_matrix).mean()

            weighted_loss.backward()
            optimizer.step()

            total_loss += weighted_loss.item()

            # 统计
            pred = quality_out.max(dim=1)[1]
            for i in range(B):
                if target[i] == 14:  # 对于“其他”类
                    # 如果所有输出都小于某个阈值（比如0.3），则认为分类正确
                    if torch.all(quality_out[i] < 0.3):
                        correct_samples += 1
                else:  # 对于0-13类
                    if pred[i] == target[i]:
                        correct_samples += 1
            total_samples += B

            if batch_idx % 100 == 0:
                print(
                    f"Epoch {epoch} - Batch {batch_idx}: Loss: {total_loss/(batch_idx+1):.4f}, Accuracy: {correct_samples/total_samples:.4f}"
                )

        # 每5个epoch进行一次验证
        if epoch % 5 == 0:
            val_loss, val_accuracy = evaluate(model, val_loader, criterion, device)
            print(
                f"Epoch {epoch} - Validation: Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.4f}"
            )

            # 保存最佳模型
            if val_loss < best_quality_loss:
                best_quality_loss = val_loss
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
                print(f"Saved best model with quality loss: {best_quality_loss:.4f}")

    print("Training completed!")


if __name__ == "__main__":
    args = parse_args()
    train(args)
