import argparse
import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from model_def.st_gcn import ST_GCN_18
from utils_data_process.estimate_pose import PoseEstimator
from utils_data_process.kinetics_gendata import KineticsDataProcessor


class ActionPredictor:
    def __init__(
        self,
        model_path: str,
        output_path: str,
        temp_dir: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.model_path = model_path
        self.output_path = output_path
        self.temp_dir = Path(temp_dir)
        self.device = device

        # 创建临时目录
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.json_dir = self.temp_dir / "json"
        self.json_dir.mkdir(exist_ok=True)

        # 初始化模型
        self._init_model()

        # 初始化姿态估计器
        self.pose_estimator = PoseEstimator(output_dir=self.json_dir)

    def _init_model(self):
        """初始化ST-GCN模型"""
        num_class = 14
        graph_cfg = {"layout": "coco", "strategy": "spatial", "max_hop": 2}
        self.model = ST_GCN_18(
            in_channels=3,
            num_class=num_class,
            edge_importance_weighting=True,
            graph_cfg=graph_cfg,
        ).to(self.device)

        # 加载模型权重
        checkpoint = torch.load(self.model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

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

                # 处理视频
                data = self._process_video(video_path)

                # 转换为tensor
                data = torch.FloatTensor(data).unsqueeze(0).to(self.device)

                # 模型推理
                logits = self.model(data)

                # 获取预测类别和置信度
                probs = F.softmax(logits, dim=1)
                pred_class = torch.argmax(logits, dim=1).item()
                pred_accuracy = probs[0, pred_class].item()

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
        default=str(script_dir / "model_def/model.pth"),
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
        model_path=args.model_path, output_path=args.output_path
    )
    predictor.predict(video_paths)
