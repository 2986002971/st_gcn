import argparse
import csv
import os
import pickle
from pathlib import Path

import numpy as np
import torch

from model_def.st_gcn import ST_GCN_18
from utils_data_process.estimate_pose import PoseEstimator
from utils_data_process.kinetics_gendata import KineticsDataProcessor


class ActionPredictor:
    def __init__(
        self,
        model_path: str = "./model_def/best_model.pth",
        svm_path: str = "./model_def/quality_predictor.pkl",
        output_path: str = "./submit.csv",
        temp_dir: str = "./temp",
        standard_data_path: str = "./processed_standard/train_data.npy",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.model_path = model_path
        self.svm_path = svm_path
        self.output_path = output_path
        self.temp_dir = Path(temp_dir)
        self.device = device
        self.standard_data_path = standard_data_path

        # 创建临时目录
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.json_dir = self.temp_dir / "json"
        self.json_dir.mkdir(exist_ok=True)

        # 初始化模型
        self._init_model()

        # 初始化姿态估计器
        self.pose_estimator = PoseEstimator(output_dir=self.json_dir)

        # 加载标准视频数据
        self.load_standard_data()

    def _init_model(self):
        """初始化ST-GCN模型和SVM模型"""
        # 初始化ST-GCN
        num_class = 14
        graph_cfg = {"layout": "coco", "strategy": "spatial", "max_hop": 2}
        self.model = ST_GCN_18(
            in_channels=6,
            num_class=num_class,
            edge_importance_weighting=True,
            graph_cfg=graph_cfg,
        ).to(self.device)

        # 加载ST-GCN权重
        checkpoint = torch.load(self.model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        # 加载SVM模型
        with open(self.svm_path, "rb") as f:
            self.quality_predictor = pickle.load(f)

    def load_standard_data(self):
        # 加载标准视频数据
        self.standard_data = np.load(self.standard_data_path, allow_pickle=True)

    def _process_video(self, video_path: str) -> np.ndarray:
        """处理单个视频文件"""
        # 姿态估计
        json_data = self.pose_estimator.process_video(video_path)

        # 转换为模型输入格式
        processor = KineticsDataProcessor(self.temp_dir)
        data = processor.json_to_data(json_data)

        # 确保数据维度正确 (C, T, V, M)
        if len(data.shape) == 3:
            data = np.expand_dims(data, axis=-1)

        return data

    def predict(self, video_paths: list) -> None:
        """对多个视频进行预测并保存结果"""
        results = []

        with torch.no_grad():
            for video_path in video_paths:
                print(f"Processing {video_path}...")

                data = self._process_video(video_path)

                # 准备14个版本的数据
                combined_data = []
                for std_idx in range(14):
                    std_data = self.standard_data[std_idx]
                    combined = np.concatenate([data, std_data], axis=0)
                    combined_data.append(combined)

                combined_data = np.stack(combined_data)
                combined_data = torch.FloatTensor(combined_data).to(self.device)

                # ST-GCN推理
                stgcn_out = self.model(combined_data)
                stgcn_features = stgcn_out.cpu().numpy()

                # 计算平均特征并确保是2D数组
                mean_features = stgcn_features.reshape(1, -1)

                # 使用SVM进行预测
                # 标准化特征
                scaled_features = self.quality_predictor["feature_scaler"].transform(
                    mean_features
                )

                # 预测类别和质量分数
                pred_class = self.quality_predictor["classifier"].predict(
                    scaled_features
                )[0]
                pred_accuracy = self.quality_predictor["regressor"].predict(
                    scaled_features
                )[0]

                # 确保质量分数在0-1范围内
                pred_accuracy = np.clip(pred_accuracy, 0, 1)

                # 保存结果
                results.append(
                    [Path(video_path).name, int(pred_class), float(pred_accuracy)]
                )

        # 写入结果
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        with open(self.output_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["video_name", "predicted_class", "accuracy"])
            writer.writerows(results)

        print(f"预测结果已保存到: {self.output_path}")


def parse_args():
    script_dir = Path(__file__).parent

    parser = argparse.ArgumentParser(description="动作识别推理脚本")
    parser.add_argument(
        "--model_path",
        type=str,
        default=str(script_dir / "model_def/best_model.pth"),
        help="ST-GCN模型权重文件路径",
    )
    parser.add_argument(
        "--svm_path",
        type=str,
        default=str(script_dir / "model_def/quality_predictor.pkl"),
        help="SVM模型文件路径",
    )
    parser.add_argument(
        "--video_dir", type=str, default="/home/service/video", help="视频文件夹路径"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=str("./submit.csv"),
        help="输出CSV文件路径",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    video_dir = Path(args.video_dir)
    video_paths = list(video_dir.glob("*.mp4"))

    if not video_paths:
        print(f"在 {args.video_dir} 中未找到视频文件")
        exit(1)

    predictor = ActionPredictor(
        model_path=args.model_path,
        svm_path=args.svm_path,
        output_path=args.output_path,
        temp_dir="./temp",
        standard_data_path="./processed_standard/train_data.npy",
    )
    predictor.predict(video_paths)
