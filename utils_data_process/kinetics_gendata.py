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
        num_person_in: int = 5,
        num_person_out: int = 1,
        max_frames: int = 2000,
        ignore_empty: bool = True,
        debug: bool = False,
    ):
        """
        初始化数据集

        Args:
            data_path: 数据路径
            label_path: 标签文件路径
            num_person_in: 输入的人数, 默认5
            num_person_out: 输出的人数, 默认1
            max_frames: 最大帧数, 默认2000
            ignore_empty: 是否忽略空样本, 默认True
            debug: 是否开启调试模式, 默认False
        """
        self.data_path = Path(data_path)
        self.label_path = Path(label_path)
        self.num_person_in = num_person_in
        self.num_person_out = num_person_out
        self.max_frames = max_frames
        self.ignore_empty = ignore_empty
        self.debug = debug

        # 数据维度
        self.channels = 3  # x, y, score
        self.num_joints = 17  # 关键点数量

        self._load_data()

    def _load_data(self) -> None:
        """加载数据和标签"""
        # 加载文件列表
        self.sample_names = [
            f for f in os.listdir(self.data_path) if f.endswith(".json")
        ]
        if self.debug:
            self.sample_names = self.sample_names[:2]

        # 加载标签
        with open(self.label_path) as f:
            label_info = json.load(f)

        # 处理标签
        # 使用完整文件名（不带.json）作为键
        sample_ids = [name.rsplit(".json", 1)[0] for name in self.sample_names]

        self.labels = np.array([label_info[id]["label_index"] for id in sample_ids])
        self.accuracies = np.array([label_info[id]["accuracy"] for id in sample_ids])
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

    def __getitem__(self, index: int) -> Tuple[np.ndarray, int, float]:  # 修改返回类型
        """获取单个样本数据"""
        # 加载数据
        sample_path = self.data_path / self.sample_names[index]
        with open(sample_path) as f:
            video_info = json.load(f)

        # 初始化数据数组
        data = np.zeros(
            (self.channels, self.max_frames, self.num_joints, self.num_person_in)
        )

        # 填充数据
        for frame_info in video_info["data"]:
            frame_idx = frame_info["frame_index"]
            for m, skeleton_info in enumerate(frame_info["skeleton"]):
                if m >= self.num_person_in:
                    break
                pose = skeleton_info["pose"]
                score = skeleton_info["score"]
                data[0, frame_idx, :, m] = pose[0::2]  # x坐标
                data[1, frame_idx, :, m] = pose[1::2]  # y坐标
                data[2, frame_idx, :, m] = score  # 置信度

        # 数据归一化
        data[0:2] = data[0:2] - 0.5
        data[0][data[2] == 0] = 0
        data[1][data[2] == 0] = 0

        # 验证标签
        label = int(video_info["label_index"])
        accuracy = self.accuracies[index]  # 获取准确度
        assert self.labels[index] == label

        # 按置信度排序
        sort_idx = (-data[2, :, :, :].sum(axis=1)).argsort(axis=1)
        for t, s in enumerate(sort_idx):
            data[:, t, :, :] = data[:, t, :, s].transpose((1, 2, 0))

        # 选择指定数量的人
        data = data[:, :, :, : self.num_person_out]

        return data, label, accuracy  # 返回数据、标签和准确度


class KineticsDataProcessor:
    """Kinetics数据处理器"""

    def __init__(
        self,
        data_root: str = "./processed",
        num_person_in: int = 2,
        num_person_out: int = 1,
        max_frames: int = 2000,
    ):
        """
        初始化数据处理器

        Args:
            data_root: 数据根目录, 默认 "./processed"
            num_person_in: 输入人数, 默认2
            num_person_out: 输出人数, 默认1
            max_frames: 最大帧数, 默认2000
        """
        self.data_root = Path(data_root)
        self.num_person_in = num_person_in
        self.num_person_out = num_person_out
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

        # 创建数据集
        dataset = KineticsDataset(
            data_path=data_path,
            label_path=label_path,
            num_person_in=self.num_person_in,
            num_person_out=self.num_person_out,
            max_frames=self.max_frames,
        )

        # 创建内存映射文件
        fp = open_memmap(
            str(data_out_path),
            dtype="float32",
            mode="w+",
            shape=(len(dataset), 3, self.max_frames, 17, self.num_person_out),
        )

        # 处理数据
        sample_labels = []
        sample_accuracies = []  # 添加准确度列表
        for i in tqdm(range(len(dataset)), desc=f"Processing {split} data"):
            data, label, accuracy = dataset[i]  # 获取数据、标签和准确度
            fp[i, :, : data.shape[1], :, :] = data
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
            np.ndarray: 形状为(C, T, V, M)的数据数组
        """
        # 初始化数据数组
        data = np.zeros(
            (
                3,
                self.max_frames,
                17,
                self.num_person_in,
            )  # channels, frames, joints, persons
        )

        # 填充数据
        for frame_info in json_data["data"]:
            frame_idx = frame_info["frame_index"]
            for m, skeleton_info in enumerate(frame_info["skeleton"]):
                if m >= self.num_person_in:
                    break
                pose = skeleton_info["pose"]
                score = skeleton_info["score"]
                data[0, frame_idx, :, m] = pose[0::2]  # x坐标
                data[1, frame_idx, :, m] = pose[1::2]  # y坐标
                data[2, frame_idx, :, m] = score  # 置信度

        # 数据归一化
        data[0:2] = data[0:2] - 0.5
        data[0][data[2] == 0] = 0
        data[1][data[2] == 0] = 0

        # 按置信度排序
        sort_idx = (-data[2, :, :, :].sum(axis=1)).argsort(axis=1)
        for t, s in enumerate(sort_idx):
            data[:, t, :, :] = data[:, t, :, s].transpose((1, 2, 0))

        # 选择指定数量的人
        data = data[:, :, :, : self.num_person_out]

        return data


if __name__ == "__main__":
    processor = KineticsDataProcessor()
    processor.process()
