import argparse
import csv
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
        output_path: str = "./15802228557_submit.csv",
        temp_dir: str = "./temp",
        standard_data_path: str = "./processed_standard/train_data.npy",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.model_path = model_path
        self.output_path = output_path
        self.temp_dir = Path(temp_dir)
        self.device = device
        self.standard_data_path = standard_data_path  # 保存标准视频数据路径

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
        """初始化ST-GCN模型"""
        num_class = 14
        graph_cfg = {"layout": "coco", "strategy": "spatial", "max_hop": 2}
        self.model = ST_GCN_18(
            in_channels=6,  # 修改为6，以匹配训练时的输入通道数
            num_class=num_class,
            edge_importance_weighting=True,
            graph_cfg=graph_cfg,
        ).to(self.device)

        # 加载模型权重
        checkpoint = torch.load(self.model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

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

        # 添加零填充通道以匹配训练时的输入
        zero_channels = np.zeros_like(data)
        data = np.concatenate([data, zero_channels], axis=0)

        return data

    def predict(self, video_paths: list) -> None:
        """对多个视频进行预测并保存结果"""
        results = []

        with torch.no_grad():
            for video_path in video_paths:
                print(f"Processing {video_path}...")

                # Phase 1: 分类
                data = self._process_video(video_path)
                data = torch.FloatTensor(data).unsqueeze(0).to(self.device)
                cls_out, _ = self.model(data, output_quality=True)

                # 获取预测类别
                pred_class = torch.argmax(cls_out, dim=1).item()

                # Phase 2: 标准度评估
                std_data = self._get_standard_video(pred_class)

                # 确保标准视频数据的形状与原始数据相匹配
                if std_data.shape[1] > data.shape[2]:
                    std_data = std_data[:, : data.shape[2], :, :]
                elif std_data.shape[1] < data.shape[2]:
                    pad_width = (
                        (0, 0),
                        (0, data.shape[2] - std_data.shape[1]),
                        (0, 0),
                        (0, 0),
                    )
                    std_data = np.pad(std_data, pad_width, mode="constant")

                # 只使用原始数据的前3个通道（去掉零填充通道）和标准数据
                original_data = data.cpu().numpy().squeeze(0)[:3]  # 只取前3个通道
                combined_data = np.concatenate([original_data, std_data], axis=0)
                combined_data = (
                    torch.FloatTensor(combined_data).unsqueeze(0).to(self.device)
                )
                # 进行标准度评估
                _, quality_out = self.model(combined_data, output_quality=True)
                pred_accuracy = quality_out.squeeze().item()

                # 保存结果
                results.append([Path(video_path).name, pred_class, pred_accuracy])

        # 保存为CSV文件
        with open(self.output_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["video_name", "predicted_class", "accuracy"])
            writer.writerows(results)

        print(f"预测结果已保存到: {self.output_path}")

    def _get_standard_video(self, pred_class):
        # 获取对应预测类别的标准视频数据
        std_data = self.standard_data[pred_class]
        return std_data


def parse_args():
    # 获取脚本所在目录
    script_dir = Path(__file__).parent

    parser = argparse.ArgumentParser(description="动作识别推理脚本")
    parser.add_argument(
        "--model_path",
        type=str,
        default=str(script_dir / "model_def/best_model.pth"),
        help="模型权重文件路径",
    )
    parser.add_argument(
        "--video_dir", type=str, default="/home/service/video", help="视频文件夹路径"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=str(script_dir / "15802228557_submit.csv"),
        help="输出CSV文件路径",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # 获取所有视频文件
    video_dir = Path(args.video_dir)
    video_paths = list(video_dir.glob("*.mp4"))

    if not video_paths:
        print(f"在 {args.video_dir} 中未找到视频文件")
        exit(1)

    # 创建预测器并运行预测
    predictor = ActionPredictor(
        model_path=args.model_path,
        output_path=args.output_path,
        temp_dir="./temp",
        standard_data_path="./processed_standard/train_data.npy",  # 添加标准视频数据路径
    )
    predictor.predict(video_paths)
