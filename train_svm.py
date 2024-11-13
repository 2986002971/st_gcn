import argparse
import os
import pickle

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR
from torch.utils.data import DataLoader

from model_def.st_gcn import ST_GCN_18
from train import SkeletonFeeder  # 复用原来的数据加载器


def parse_args():
    parser = argparse.ArgumentParser(description="训练SVM质量评估模型")
    parser.add_argument(
        "--stgcn_model",
        type=str,
        default="./model_def/best_model.pth",
        help="训练好的ST-GCN模型路径",
    )
    parser.add_argument(
        "--data_dir", type=str, default="./processed", help="数据集文件夹路径"
    )
    parser.add_argument(
        "--output_dir", type=str, default="./model_def", help="SVM模型保存路径"
    )
    return parser.parse_args()


def collect_features(model, data_loader, device):
    """使用ST-GCN模型提取特征"""
    features = []
    targets = []
    scores = []

    model.eval()
    with torch.no_grad():
        for data, target, accuracy, _ in data_loader:
            data = data.to(device)

            B, N, C, T, V, M = data.shape
            data = data.view(B * N, C, T, V, M)
            stgcn_out = model(data)
            stgcn_out = stgcn_out.view(B, N)

            features.append(stgcn_out.cpu().numpy())
            targets.append(target.numpy())
            scores.append(accuracy.numpy() / 100.0)  # 转换回0-1范围

    return (np.vstack(features), np.concatenate(targets), np.concatenate(scores))


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载ST-GCN模型
    model = ST_GCN_18(
        in_channels=6,
        num_class=14,
        edge_importance_weighting=True,
        graph_cfg={"layout": "coco", "strategy": "spatial", "max_hop": 2},
    ).to(device)

    checkpoint = torch.load(args.stgcn_model, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # 准备数据加载器
    train_loader = DataLoader(
        SkeletonFeeder(
            os.path.join(args.data_dir, "train_data.npy"),
            os.path.join(args.data_dir, "train_label.pkl"),
        ),
        batch_size=1,
        shuffle=False,
        num_workers=16,
        pin_memory=True,
    )

    # 收集特征
    print("收集训练特征...")
    train_features, train_targets, train_scores = collect_features(
        model, train_loader, device
    )

    # 训练SVM
    print("训练SVM分类器和回归器...")

    # 特征标准化
    feature_scaler = StandardScaler()
    scaled_features = feature_scaler.fit_transform(train_features)

    # 训练分类器
    classifier = SVC(kernel="rbf", probability=True)
    classifier.fit(scaled_features, train_targets)

    # 训练回归器
    regressor = SVR(kernel="rbf")
    regressor.fit(scaled_features, train_scores)

    # 保存模型
    os.makedirs(args.output_dir, exist_ok=True)
    svm_path = os.path.join(args.output_dir, "quality_predictor.pkl")

    with open(svm_path, "wb") as f:
        pickle.dump(
            {
                "feature_scaler": feature_scaler,
                "classifier": classifier,
                "regressor": regressor,
            },
            f,
        )

    print(f"SVM模型已保存到: {svm_path}")

    # 简单评估
    train_pred_classes = classifier.predict(scaled_features)
    train_pred_scores = regressor.predict(scaled_features)

    class_acc = (train_pred_classes == train_targets).mean()
    score_mse = ((train_pred_scores - train_scores) ** 2).mean()

    print(f"训练集分类准确率: {class_acc:.4f}")
    print(f"训练集分数MSE: {score_mse:.4f}")


if __name__ == "__main__":
    main()
