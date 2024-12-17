import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from dual_coco import dual_coco_edges  # 导入边的定义
from ultralytics import YOLO


class PoseEstimator:
    def __init__(
        self,
        model_path: Optional[str] = None,
        dataset_path: str = "./dataset",
        output_dir: str = "./processed/json",
    ):
        """
        姿态估计器初始化

        Args:
            model_path: 模型路径, 默认为None时会在当前目录查找模型文件
            dataset_path: 数据集路径，默认为 "./dataset"
            output_dir: 输出目录，默认为 "./processed/json"
        """
        self.dataset_path = Path(dataset_path)
        self.output_dir = Path(output_dir)

        if model_path is None:
            model_path = Path(__file__).parent / "yolov8s-pose.pt"
        self.model = YOLO(str(model_path))

    @staticmethod
    def _input_reader(video_path: str) -> Tuple[cv2.VideoCapture, float, float, float]:
        """
        读取视频文件

        Args:
            video_path: 视频文件路径

        Returns:
            Tuple[cv2.VideoCapture, float, float, float]: 视频对象、帧率、总帧数、时长
        """
        cap = cv2.VideoCapture(str(video_path))
        rate = cap.get(5)  # 帧速率
        frame_number = cap.get(7)  # 视频文件的帧数
        duration = frame_number / rate  # 帧速率/视频总帧数 是时间，除以60之后单位是分钟
        return cap, rate, frame_number, duration

    @staticmethod
    def _calculate_edge_angle(point1: np.ndarray, point2: np.ndarray) -> float:
        """
        计算从point1指向point2的有向角度

        Args:
            point1: 起始点坐标 [x, y]
            point2: 终止点坐标 [x, y]

        Returns:
            float: 角度值（度数，范围-180到180）
        """
        dx = point2[0] - point1[0]
        dy = point2[1] - point1[1]
        angle = np.degrees(np.arctan2(dy, dx))
        return float(angle)

    def _process_single_frame(self, img) -> Tuple[List[float], List[float]]:
        """
        处理单帧图像，计算边的角度和置信度

        Args:
            img: 输入图像

        Returns:
            Tuple[List[float], List[float]]: 边的角度和置信度
        """
        result = self.model(img, conf=0.5)[0]
        keypoints = result.keypoints.xyn  # 归一化坐标 (0-1)
        keypoints_conf = result.keypoints.conf

        if keypoints_conf is None:
            # 如果没有检测到关键点，返回全0
            return [0.0] * len(dual_coco_edges), [0.0] * len(dual_coco_edges)

        # 获取关键点坐标和置信度
        coords = keypoints[0].cpu().numpy()  # [17, 2]
        conf = keypoints_conf.cpu().numpy()[0]  # [17]

        # 计算每条边的角度和置信度
        angles = []
        edge_conf = []
        for start_idx, end_idx in dual_coco_edges:
            # 注意：这里的索引已经是0-based
            start_point = coords[start_idx - 1]  # 转换为0-based索引
            end_point = coords[end_idx - 1]

            angle = self._calculate_edge_angle(start_point, end_point)
            # 边的置信度取两个端点置信度的最小值
            edge_confidence = min(conf[start_idx - 1], conf[end_idx - 1])

            angles.append(float(round(angle, 6)))
            edge_conf.append(float(round(edge_confidence, 6)))

        return angles, edge_conf

    def process_video(
        self,
        video_path: str,
        class_name: Optional[str] = None,
        label_index: Optional[int] = None,
    ) -> Dict:
        """
        处理单个视频文件

        Args:
            video_path: 视频文件路径
            class_name: 类别名称
            label_index: 类别索引

        Returns:
            Dict: 处理结果
        """
        # 从文件名中提取14个评分
        video_name = Path(video_path).stem
        if "_" in video_name:
            # 格式: "01_0.603_0.472_0.622..." -> [0.603, 0.472, 0.622, ...]
            parts = video_name.split("_")
            accuracies = [float(score) for score in parts[1:]]
            # 确保有14个评分，如果不足则补0
            accuracies.extend([0.0] * (14 - len(accuracies)))
        else:
            accuracies = [0.0] * 14

        cap, rate, frame_number, duration = self._input_reader(video_path)
        frame_index = 0
        jsdata = []

        while cap.isOpened():
            rec, img = cap.read()
            if not rec:
                break
            frame_index += 1

            angles, conf = self._process_single_frame(img)

            # 只有当至少有一个有效的角度时才添加数据
            frame_data = {
                "frame_index": frame_index,
                "skeleton": [{"angles": angles, "score": conf}] if any(conf) else [],
            }
            jsdata.append(frame_data)

        cap.release()
        return {
            "data": jsdata,
            "label": class_name,
            "label_index": label_index,
            "accuracies": accuracies,  # 现在是一个列表，包含14个评分
        }

    def process_dataset(self) -> None:
        """处理整个数据集"""
        # 确保输出目录存在
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 创建类别索引映射
        class_to_index = {}
        index = 0
        for class_folder in sorted(
            p.name
            for p in self.dataset_path.iterdir()
            if p.is_dir() and not p.name.startswith(".")
        ):
            class_to_index[class_folder] = index
            index += 1

        # 处理每个类别下的视频
        for class_folder, label_index in class_to_index.items():
            class_path = self.dataset_path / class_folder
            class_output_dir = self.output_dir / class_folder
            class_output_dir.mkdir(exist_ok=True)

            for video_file in class_path.glob("*.mp4"):
                # 处理视频
                result = self.process_video(
                    video_file, class_name=class_folder, label_index=label_index
                )

                # 保存结果
                output_path = class_output_dir / f"{video_file.stem}.json"
                with open(output_path, "w") as f:
                    json.dump(result, f, indent=None, separators=(",", ":"))

        print("数据处理完成")


if __name__ == "__main__":
    estimator = PoseEstimator()
    estimator.process_dataset()
