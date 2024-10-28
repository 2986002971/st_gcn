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

        return data

    def predict(self, video_paths: list) -> None:
        """对多个视频进行预测并保存结果"""
        results = []

        with torch.no_grad():
            for video_path in video_paths:
                print(f"Processing {video_path}...")

                data = self._process_video(video_path)

                # 准备14个版本的数据，每个都与一个标准视频结合
                combined_data = []
                for std_idx in range(14):
                    std_data = self.standard_data[std_idx]
                    combined = np.concatenate([data, std_data], axis=0)
                    combined_data.append(combined)

                combined_data = np.stack(combined_data)  # Shape: (14, C, T, V, M)
                combined_data = torch.FloatTensor(combined_data).to(self.device)

                # 进行14次推理
                quality_out = self.model(combined_data)

                # 选择最大标准度及其对应的类别
                max_quality, pred_class = quality_out.max(dim=0)
                pred_class = pred_class.item()

                # 检查是否属于“其他”类（所有输出都接近0）
                threshold = 0.3  # 阈值
                if torch.all(quality_out < threshold):
                    pred_class = 14  # “其他”类，标准度为0
                    pred_accuracy = 0.0
                else:
                    pred_accuracy = max_quality.item()

                # 保存结果
                results.append([Path(video_path).name, pred_class, pred_accuracy])

        # 保存为CSV文件
        with open(self.output_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["video_name", "predicted_class", "accuracy"])
            writer.writerows(results)

        print(f"预测结果已保存到: {self.output_path}")


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
        standard_data_path="./processed_standard/train_data.npy",
    )
    predictor.predict(video_paths)
