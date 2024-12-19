import argparse
import os
import pickle

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from model_def.light_classifier import LightClassifier


class SkeletonClassifierDataset(Dataset):
    def __init__(self, data_path, label_path):
        # 加载数据
        self.data = np.load(data_path, allow_pickle=True)
        with open(label_path, "rb") as f:
            self.sample_name, self.label, _ = pickle.load(f)
        self.label = np.array(self.label, dtype=np.int64)

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        data = self.data[index]  # [2, T, V]
        label = self.label[index]
        return torch.FloatTensor(data), torch.LongTensor([label]).squeeze()


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 创建训练和验证数据加载器
    train_dataset = SkeletonClassifierDataset(
        os.path.join(args.data_dir, "train_data.npy"),
        os.path.join(args.data_dir, "train_label.pkl"),
    )
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4
    )

    val_dataset = SkeletonClassifierDataset(
        os.path.join(args.data_dir, "val_data.npy"),
        os.path.join(args.data_dir, "val_label.pkl"),
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4
    )

    # 创建模型
    model = LightClassifier(
        in_channels=2,
        num_classes=15,  # 14个标准动作 + 1个其他类
        edge_importance_weighting=True,
    ).to(device)

    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # 用于保存最佳模型
    best_val_acc = 0.0

    # 训练循环
    for epoch in range(args.max_epochs):
        # 训练阶段
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        for batch_idx, (data, label) in enumerate(train_loader):
            data, label = data.to(device), label.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, label)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = output.max(1)
            train_total += label.size(0)
            train_correct += predicted.eq(label).sum().item()

            if batch_idx % 50 == 0:
                print(
                    f"Epoch: {epoch}, Batch: {batch_idx}, "
                    f"Loss: {train_loss/(batch_idx+1):.4f}, "
                    f"Train Acc: {100.*train_correct/train_total:.2f}%"
                )

        # 验证阶段
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for data, label in val_loader:
                data, label = data.to(device), label.to(device)
                output = model(data)
                loss = criterion(output, label)

                val_loss += loss.item()
                _, predicted = output.max(1)
                val_total += label.size(0)
                val_correct += predicted.eq(label).sum().item()

        val_acc = 100.0 * val_correct / val_total
        print(
            f"Epoch: {epoch}, "
            f"Val Loss: {val_loss/len(val_loader):.4f}, "
            f"Val Acc: {val_acc:.2f}%"
        )

        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_path = os.path.join(args.work_dir, "best_model.pth")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_acc": best_val_acc,
                },
                best_model_path,
            )
            print(f"Saved best model with validation accuracy: {best_val_acc:.2f}%")

        # 定期保存检查点
        if (epoch + 1) % 5 == 0:
            checkpoint_path = os.path.join(args.work_dir, f"epoch_{epoch+1}.pth")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_acc": val_acc,
                },
                checkpoint_path,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--work_dir", default="./work_dir/classifier")
    parser.add_argument("--data_dir", default="./processed")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--max_epochs", type=int, default=50)

    args = parser.parse_args()
    os.makedirs(args.work_dir, exist_ok=True)
    train(args)
