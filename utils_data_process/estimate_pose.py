import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import torch
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

    def _process_single_frame(self, img) -> Tuple[List[float], List[float]]:
        """
        处理单帧图像

        Args:
            img: 输入图像

        Returns:
            Tuple[List[float], List[float]]: 关键点坐标和置信度
        """
        result = self.model(img, conf=0.5)[0]
        keypoints_conf = result.keypoints.conf
        keypoints_coords = result.keypoints.xyn

        if keypoints_conf is None:
            keypoints_conf = torch.zeros(17).unsqueeze(0)
            keypoints_coords = torch.zeros(34).unsqueeze(0).unsqueeze(0)

        coords = keypoints_coords[0].cpu().numpy()
        coords_flat = coords.flatten()
        conf = keypoints_conf.cpu().numpy()[0]

        point = [float(round(i, 6)) for i in coords_flat]
        conf = [float(round(i, 6)) for i in conf]
        return point, conf

    @staticmethod
    def _normalize_coordinates(
        points: List[float], ref_nose: Tuple[float, float], ref_shoulder_width: float
    ) -> List[float]:
        """
        归一化坐标

        Args:
            points: 原始坐标点列表
            ref_nose: 参考鼻子位置
            ref_shoulder_width: 参考肩宽

        Returns:
            List[float]: 归一化后的坐标点列表
        """
        normalized_points = []
        for i in range(0, len(points), 2):
            x, y = points[i : i + 2]
            norm_x = (x - ref_nose[0]) / ref_shoulder_width
            norm_y = (y - ref_nose[1]) / ref_shoulder_width
            normalized_points.extend([round(norm_x, 3), round(norm_y, 3)])
        return normalized_points

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
        # 从文件名中提取准确度
        video_name = Path(video_path).stem
        accuracy = float(video_name.split("_")[1]) if "_" in video_name else 0.0

        cap, rate, frame_number, duration = self._input_reader(video_path)
        frame_index = 0
        jsdata = []
        ref_nose = None
        ref_shoulder_width = None

        while cap.isOpened():
            rec, img = cap.read()
            if not rec:
                break
            frame_index += 1

            point, conf = self._process_single_frame(img)

            # 设置参考帧数据
            if sum(point) != 0 and ref_nose is None:
                ref_nose = (point[0], point[1])
                left_shoulder = (point[10], point[11])
                right_shoulder = (point[12], point[13])
                ref_shoulder_width = (
                    (left_shoulder[0] - right_shoulder[0]) ** 2
                    + (left_shoulder[1] - right_shoulder[1]) ** 2
                ) ** 0.5

            # 坐标归一化
            if (
                sum(point) != 0
                and ref_nose is not None
                and ref_shoulder_width is not None
            ):
                point = self._normalize_coordinates(point, ref_nose, ref_shoulder_width)

            pose_data = {"pose": point, "score": conf}
            frame_data = {
                "frame_index": frame_index,
                "skeleton": [pose_data] if sum(point) != 0 else [],
            }
            jsdata.append(frame_data)

        cap.release()
        return {
            "data": jsdata,
            "label": class_name,
            "label_index": label_index,
            "accuracy": accuracy,  # 添加准确度信息
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
