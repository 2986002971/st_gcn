import os
import random
import shutil
from typing import List

# 默认标签列表
DEFAULT_LABELS = [
    "00",
    "01",
    "02",
    "03",
    "04",
    "05",
    "06",
    "07",
    "10",
    "11",
    "12",
    "13",
    "14",
    "15",
    "20",
]


class DatasetSplitter:
    def __init__(
        self,
        data_root: str = "./processed/json",
        train_folder: str = "./processed/train",
        test_folder: str = "./processed/val",
        labels: List[str] = DEFAULT_LABELS,
        split_ratio: float = 0.8,
    ):
        """
        数据集分割器初始化

        Args:
            data_root: 原始数据根目录，默认为 "./processed/json"
            train_folder: 训练集输出目录，默认为 "./processed/train"
            test_folder: 测试集输出目录，默认为 "./processed/val"
            labels: 标签列表，默认为 DEFAULT_LABELS
            split_ratio: 训练集比例, 默认0.8
        """
        self.data_root = data_root
        self.train_folder = train_folder
        self.test_folder = test_folder
        self.labels = labels
        self.split_ratio = split_ratio

    def _split_and_copy_json_files(
        self, data_folder: str, train_folder: str, test_folder: str, label: str
    ) -> None:
        """单个标签数据的分割和复制"""
        json_files = [f for f in os.listdir(data_folder) if f.endswith(".json")]
        random.shuffle(json_files)

        num_train_files = int(len(json_files) * self.split_ratio)
        train_files = json_files[:num_train_files]
        test_files = json_files[num_train_files:]

        os.makedirs(train_folder, exist_ok=True)
        os.makedirs(test_folder, exist_ok=True)

        # 复制训练集文件
        for file in train_files:
            source_path = os.path.join(data_folder, file)
            dest_path = os.path.join(train_folder, f"{label}_{file}")
            shutil.copy(source_path, dest_path)

        # 复制测试集文件
        for file in test_files:
            source_path = os.path.join(data_folder, file)
            dest_path = os.path.join(test_folder, f"{label}_{file}")
            shutil.copy(source_path, dest_path)

    def split(self) -> None:
        """执行数据集分割"""
        for label in self.labels:
            folder_name = next(
                (f for f in os.listdir(self.data_root) if f.startswith(label)), None
            )
            if folder_name:
                data_folder = os.path.join(self.data_root, folder_name)
                self._split_and_copy_json_files(
                    data_folder, self.train_folder, self.test_folder, label
                )
            else:
                print(f"警告: 没有找到以 {label} 开头的文件夹，已跳过。")


if __name__ == "__main__":
    splitter = DatasetSplitter()
    splitter.split()
