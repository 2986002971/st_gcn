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
        "--max_epochs", type=int, default=300, help="每个阶段的最大训练轮数"
    )
    return parser.parse_args()


def auto_pading(data_numpy, size, random_pad=False):
    C, T, V = data_numpy.shape
    if T < size:
        begin = random.randint(0, size - T) if random_pad else 0
        data_numpy_paded = np.zeros((C, size, V), dtype=data_numpy.dtype)
        data_numpy_paded[:, begin : begin + T, :] = data_numpy
        return data_numpy_paded
    else:
        return data_numpy


def random_choose(data_numpy, size, auto_pad=True):
    # input: C,T,V
    C, T, V = data_numpy.shape
    if T == size:
        return data_numpy
    elif T < size:
        if auto_pad:
            return auto_pading(data_numpy, size, random_pad=True)
        else:
            return data_numpy
    else:
        begin = random.randint(0, T - size)
        return data_numpy[:, begin : begin + size, :]


def random_move(
    data_numpy,
    angle_candidate=[-10.0, -5.0, 0.0, 5.0, 10.0],
    scale_candidate=[0.9, 1.0, 1.1],
    transform_candidate=[-0.2, -0.1, 0.0, 0.1, 0.2],
    move_time_candidate=[1],
):
    # input: C,T,V
    C, T, V = data_numpy.shape
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
        xy = data_numpy[0:2, i_frame, :]
        new_xy = np.dot(theta[:, :, i_frame], xy.reshape(2, -1))
        new_xy[0] += t_x[i_frame]
        new_xy[1] += t_y[i_frame]
        data_numpy[0:2, i_frame, :] = new_xy.reshape(2, V)

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
            self.sample_name, self.label, self.accuracies = pickle.load(
                f
            )  # 改名以表明是多个准确度

        # 加载标准视频数据
        self.standard_data = np.load(self.standard_data_path, allow_pickle=True)

        # 确保标签和准确度是正确的数据类型
        self.label = np.array(self.label, dtype=np.int64)
        self.accuracies = np.array(self.accuracies, dtype=np.float32)  # [N, 14]
        assert (
            self.accuracies.shape[1] == 14
        ), f"每个样本应该有14个准确度分数，但得到了{self.accuracies.shape[1]}"

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        data_numpy = self.data[index]
        label = self.label[index]
        accuracies = self.accuracies[index]  # [14]

        # 准备14个版本的数据，每个都与一个标准视频结合
        combined_data = []
        for std_idx in range(14):
            std_data = self.standard_data[std_idx]
            combined = np.concatenate([data_numpy, std_data], axis=0)  # [4, T, V]

            if self.random_choose:
                combined = random_choose(combined, self.window_size)
            if self.random_move:
                combined = random_move(combined)

            combined_data.append(combined)

        combined_data = np.stack(combined_data)  # Shape: (14, 4, T, V)

        return (
            torch.FloatTensor(combined_data),  # [14, 4, T, V]
            torch.LongTensor([label]).squeeze(),
            torch.FloatTensor(accuracies),  # [14]
            torch.LongTensor([index]),
        )


def get_dataloader(data_path, label_path, batch_size, **kwargs):
    dataset = SkeletonFeeder(data_path, label_path, **kwargs)
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True
    )


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 模型定义 - 4个输入通道
    model = ST_GCN_18(
        in_channels=4,  # 2个通道(角度+置信度) × 2个序列(学生+参考)
        edge_importance_weighting=True,
        graph_cfg={"layout": "dual_coco", "strategy": "spatial", "max_hop": 2},
    ).to(device)

    criterion = nn.MSELoss(reduction="none")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

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

    # 加载检查点逻辑保持不变
    start_epoch = 0
    best_loss = float("inf")
    if args.resume_from and os.path.exists(args.resume_from):
        checkpoint = torch.load(args.resume_from)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_loss = checkpoint["best_loss"]
        print(f"成功加载检查点，从第 {start_epoch} 轮继续训练")

    # 修改训练循环
    for epoch in range(start_epoch, args.max_epochs):
        model.train()
        total_loss = 0

        for batch_idx, (data, _, accuracies, _) in enumerate(train_loader):
            data = data.to(device)
            accuracies = accuracies.to(device)  # [B, 14]

            optimizer.zero_grad()

            B, N, C, T, V = data.shape
            data = data.view(B * N, C, T, V)

            # 模型输出14个相似度分数
            similarity_scores = model(data)  # [B, 14]
            similarity_scores = similarity_scores.view(B, N)

            # 计算损失：直接使用14个准确度作为目标
            loss = criterion(similarity_scores, accuracies)  # [B, 14]
            loss = loss.mean(dim=0)  # [14] 只在批次维度上取平均
            individual_losses = loss.detach()  # 保存各个损失值用于监控
            loss = loss.sum()  # 转换为标量以进行反向传播
            loss.backward()

            # 可以打印各个标准序列的损失
            if batch_idx % 100 == 0:
                for i in range(14):
                    print(
                        f"Standard sequence {i} loss: {individual_losses[i].item():.4f}"
                    )

            optimizer.step()

            total_loss += loss.item()

            if batch_idx % 100 == 0:
                print(
                    f"Epoch {epoch} - Batch {batch_idx}: Loss: {total_loss/(batch_idx+1):.4f}"
                )

        # 每5个epoch进行一次验证
        if epoch % 5 == 0:
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for data, _, accuracies, _ in val_loader:
                    data = data.to(device)
                    accuracies = accuracies.to(device)

                    B, N, C, T, V = data.shape
                    data = data.view(B * N, C, T, V)

                    similarity_scores = model(data)
                    similarity_scores = similarity_scores.view(B, N)

                    loss = criterion(similarity_scores, accuracies)
                    val_loss += loss.mean().item()

            val_loss = val_loss / len(val_loader)
            print(f"Epoch {epoch} - Validation Loss: {val_loss:.4f}")

            # 保存最佳模型
            if val_loss < best_loss:
                best_loss = val_loss
                checkpoint_path = os.path.join(args.work_dir, "best_model.pth")
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "best_loss": best_loss,
                    },
                    checkpoint_path,
                )
                print(f"Saved best model with loss: {best_loss:.4f}")

    print("Training completed!")


if __name__ == "__main__":
    args = parse_args()
    train(args)
