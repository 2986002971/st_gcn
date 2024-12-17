import json
import os
import pickle
from pathlib import Path
from typing import Tuple

import numpy as np
from numpy.lib.format import open_memmap
from torch.utils.data import Dataset
from tqdm import tqdm


class KineticsDataset(Dataset):
    """Kinetics骨架数据集处理类"""

    def __init__(
        self,
        data_path: str,
        label_path: str,
        max_frames: int = 2000,
        ignore_empty: bool = True,
        debug: bool = False,
    ):
        """
        初始化数据集

        Args:
            data_path: 数据路径
            label_path: 标签文件路径
            max_frames: 最大帧数, 默认2000
            ignore_empty: 是否忽略空样本, 默认True
            debug: 是否开启调试模式, 默认False
        """
        self.data_path = Path(data_path)
        self.label_path = Path(label_path)
        self.max_frames = max_frames
        self.ignore_empty = ignore_empty
        self.debug = debug

        # 数据维度
        self.channels = 2  # 角度值和置信度
        self.num_edges = 14  # 边的数量

        self._load_data()

    def _load_data(self) -> None:
        """加载数据和标签"""
        # 加载文件列表并排序
        self.sample_names = sorted(
            [f for f in os.listdir(self.data_path) if f.endswith(".json")]
        )
        if self.debug:
            self.sample_names = self.sample_names[:2]

        # 加载标签
        with open(self.label_path) as f:
            label_info = json.load(f)

        # 处理标签
        # 使用完整文件名（不带.json）作为键
        sample_ids = [name.rsplit(".json", 1)[0] for name in self.sample_names]

        self.labels = np.array([label_info[id]["label_index"] for id in sample_ids])
        self.accuracies = np.array(
            [label_info[id]["accuracies"] for id in sample_ids]
        )  # 现在是14个评分
        has_skeleton = np.array([label_info[id]["has_skeleton"] for id in sample_ids])

        # 过滤空样本
        if self.ignore_empty:
            valid_indices = has_skeleton
            self.sample_names = [
                s for h, s in zip(has_skeleton, self.sample_names) if h
            ]
            self.labels = self.labels[valid_indices]
            self.accuracies = self.accuracies[valid_indices]

        self.num_samples = len(self.sample_names)

    def __len__(self) -> int:
        """返回数据集大小"""
        return self.num_samples

    def __getitem__(
        self, index: int
    ) -> Tuple[np.ndarray, int, np.ndarray]:  # 修改返回类型
        """获取单个样本数据"""
        # 加载数据
        sample_path = self.data_path / self.sample_names[index]
        with open(sample_path) as f:
            video_info = json.load(f)

        # 初始化数据数组 [C=2, T, V=14]
        data = np.zeros((self.channels, self.max_frames, self.num_edges))

        # 填充数据
        for frame_info in video_info["data"]:
            frame_idx = frame_info["frame_index"]
            # 添加截断逻辑
            if frame_idx >= self.max_frames:
                continue  # 跳过超出最大帧数的帧

            if frame_info["skeleton"]:  # 如果有骨架数据
                skeleton = frame_info["skeleton"][0]  # 只取第一个人的数据
                data[0, frame_idx, :] = skeleton["angles"]  # 角度数据
                data[1, frame_idx, :] = skeleton["score"]  # 置信度数据

        # 验证标签
        label = int(video_info["label_index"])
        accuracies = self.accuracies[index]  # 获取14个准确度值
        assert self.labels[index] == label

        return data, label, accuracies


class KineticsDataProcessor:
    """Kinetics数据处理器"""

    def __init__(
        self,
        data_root: str = "./processed",
        max_frames: int = 2000,
    ):
        """
        初始化数据处理器

        Args:
            data_root: 数据根目录, 默认 "./processed"
            max_frames: 最大帧数, 默认2000
        """
        self.data_root = Path(data_root)
        self.max_frames = max_frames

    def process(self) -> None:
        """处理训练集和验证集数据"""
        for split in ["train", "val"]:
            self._process_split(split)

    def _process_split(self, split: str) -> None:
        """处理单个数据集分割"""
        # 设置路径
        data_path = self.data_root / split
        label_path = self.data_root / f"{split}_label.json"
        data_out_path = self.data_root / f"{split}_data.npy"
        label_out_path = self.data_root / f"{split}_label.pkl"

        # 检查目录是否存在且非空
        if not data_path.exists() or not any(data_path.iterdir()):
            print(f"跳过 {split} 数据处理：目录为空或不存在")
            return

        # 创建数据集
        dataset = KineticsDataset(
            data_path=data_path,
            label_path=label_path,
            max_frames=self.max_frames,
        )

        # 如果数据集为空，直接返回
        if len(dataset) == 0:
            print(f"跳过 {split} 数据处理：没有有效样本")
            return

        # 创建内存映射文件
        fp = open_memmap(
            str(data_out_path),
            dtype="float32",
            mode="w+",
            shape=(len(dataset), 2, self.max_frames, 14),  # [N, C=2, T, V=14]
        )

        # 处理数据
        sample_labels = []
        sample_accuracies = []  # 添加准确度列表
        for i in tqdm(range(len(dataset)), desc=f"Processing {split} data"):
            data, label, accuracy = dataset[i]  # 获取数据、标签和准确度
            fp[i] = data
            sample_labels.append(label)
            sample_accuracies.append(accuracy)  # 保存准确度

        # 保存标签和准确度
        with open(label_out_path, "wb") as f:
            pickle.dump(
                (dataset.sample_names, sample_labels, sample_accuracies), f
            )  # 添加准确度

        print(f"{split} 数据处理完成")

    def json_to_data(self, json_data: dict) -> np.ndarray:
        """
        将JSON数据转换为模型输入格式的数据

        Args:
            json_data: 包含骨架数据的字典

        Returns:
            np.ndarray: 形状为(C=2, T, V=14)的数据数组
        """
        # 初始化数据数组
        data = np.zeros((2, self.max_frames, 14))  # [C=2, T, V=14]

        # 填充数据
        for frame_info in json_data["data"]:
            frame_idx = frame_info["frame_index"]
            # 添加截断逻辑
            if frame_idx >= self.max_frames:
                continue  # 跳过超出最大帧数的帧

            if frame_info["skeleton"]:  # 如果有骨架数据
                skeleton = frame_info["skeleton"][0]  # 只取第一个人的数据
                data[0, frame_idx, :] = skeleton["angles"]  # 角度数据
                data[1, frame_idx, :] = skeleton["score"]  # 置信度数据

        return data


if __name__ == "__main__":
    processor = KineticsDataProcessor()
    processor.process()
