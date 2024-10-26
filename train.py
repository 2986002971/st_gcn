import argparse
import os
import pickle
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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
        print(f"标准视频数据形状: {self.standard_data.shape}")  # 应该是(14, C, T, V, M)

        # 确保标签和准确度是正确的数据类型
        self.label = np.array(self.label, dtype=np.int64)
        self.accuracy = np.array(self.accuracy, dtype=np.float32)

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        # 获取原始数据
        data_numpy = self.data[index]
        label = self.label[index]
        accuracy = self.accuracy[index]

        # 数据增强
        if self.random_choose:
            data_numpy = random_choose(data_numpy, self.window_size)
        if self.random_move:
            data_numpy = random_move(data_numpy)

        # 处理标准视频数据
        processed_standard_data = []
        for i in range(len(self.standard_data)):
            std_data = self.standard_data[i].copy()  # 复制以避免修改原始数据
            if self.random_choose:
                # 对标准视频使用相同的window_size
                std_data = random_choose(std_data, self.window_size, auto_pad=True)
            if self.random_move:
                std_data = random_move(std_data)
            processed_standard_data.append(std_data)

        # 合并所有标准视频数据
        # 原始数据: (C, T, V, M)
        # 标准数据: (14, C, T, V, M) -> 在通道维度上合并
        combined_data = np.concatenate(
            [data_numpy] + processed_standard_data,
            axis=0,
        )

        return (
            torch.FloatTensor(combined_data),
            torch.LongTensor([label]).squeeze(),
            torch.FloatTensor([accuracy]).squeeze(),
        )


def get_dataloader(data_path, label_path, batch_size, **kwargs):
    dataset = SkeletonFeeder(data_path, label_path, **kwargs)
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True
    )


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 数据加载
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

    # 模型定义
    num_class = 14  # 定义类别数
    graph_cfg = {"layout": "coco", "strategy": "spatial", "max_hop": 2}
    model = ST_GCN_18(
        in_channels=45,  # 修改为45 = 3*(1+14)，原始数据3通道 + 14个标准视频各3通道
        num_class=num_class,
        edge_importance_weighting=True,
        graph_cfg=graph_cfg,
    ).to(device)

    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=0.001,
    )

    # 恢复训练
    start_epoch = 0
    if args.resume_from:
        checkpoint = torch.load(args.resume_from)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"]
        print(f"Resuming from epoch {start_epoch}")

    # 训练循环
    total_epochs = 500
    for epoch in range(start_epoch, total_epochs):
        model.train()
        train_loss = 0
        for batch_idx, (data, target, accuracy) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            accuracy = accuracy.to(device)

            # 创建软标签
            background_value = (1 - accuracy) / (num_class - 1)
            soft_target = background_value.unsqueeze(1).expand(-1, num_class).clone()
            soft_target.scatter_(1, target.unsqueeze(1), accuracy.unsqueeze(1))

            # 前向传播
            logits = model(data)

            loss = criterion(logits, soft_target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            if batch_idx % 100 == 0:
                print(
                    f"Train Epoch: {epoch}/{total_epochs} [{batch_idx}/{len(train_loader)}] "
                    f"Loss: {loss.item():.6f}"
                )

        train_loss /= len(train_loader)
        print(f"Epoch {epoch} Average Training Loss: {train_loss:.6f}")

        # 每5个epoch进行一次验证和保存
        if epoch % 5 == 0:
            model.eval()
            val_loss = 0
            correct = 0
            total_accuracy_diff = 0
            with torch.no_grad():
                for data, target, accuracy in val_loader:
                    data, target = data.to(device), target.to(device)
                    accuracy = accuracy.to(device)

                    # 创建软标签
                    background_value = (1 - accuracy) / (num_class - 1)
                    soft_target = (
                        background_value.unsqueeze(1).expand(-1, num_class).clone()
                    )
                    soft_target.scatter_(1, target.unsqueeze(1), accuracy.unsqueeze(1))

                    # 前向传播
                    logits = model(data)

                    # 使用相同的损失函数计算验证损失
                    loss = criterion(logits, soft_target)
                    val_loss += loss.item()

                    # 计算分类准确率
                    pred = logits.argmax(dim=1)
                    correct += pred.eq(target).sum().item()

                    # 计算预测准确度与真实准确度的差异
                    pred_probs = F.softmax(logits, dim=1)
                    pred_accuracy = pred_probs.gather(1, target.unsqueeze(1))
                    total_accuracy_diff += (
                        torch.abs(pred_accuracy.squeeze() - accuracy).sum().item()
                    )

            val_loss /= len(val_loader)
            accuracy = correct / len(val_loader.dataset)
            mean_accuracy_diff = total_accuracy_diff / len(val_loader.dataset)

            print(
                f"Validation Loss: {val_loss:.4f}, "
                f"Classification Accuracy: {accuracy:.4f}, "
                f"Mean Accuracy Difference: {mean_accuracy_diff:.4f}"
            )

            # 保存检查点
            checkpoint_path = os.path.join(
                args.work_dir, f"checkpoint_epoch_{epoch}.pth"
            )
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": loss,
                },
                checkpoint_path,
            )

    print("训练完成")


if __name__ == "__main__":
    args = parse_args()
    train(args)
