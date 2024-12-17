from pathlib import Path
from typing import Optional

from estimate_pose import PoseEstimator
from kinetics_gendata import KineticsDataProcessor
from label_generator import LabelGenerator
from split import DatasetSplitter


class StandardDataProcessor:
    """标准视频数据处理流水线"""

    def __init__(
        self,
        dataset_path: str = "./standard",  # 标准视频目录
        processed_path: str = "./processed_standard",  # 标准视频处理结果目录
        model_path: Optional[str] = None,
    ):
        self.dataset_path = Path(dataset_path)
        self.processed_path = Path(processed_path)
        self.model_path = model_path

        # 创建必要的目录
        self.json_path = self.processed_path / "json"
        self.train_path = self.processed_path / "train"
        self.val_path = self.processed_path / "val"

        for path in [
            self.processed_path,
            self.json_path,
            self.train_path,
            self.val_path,
        ]:
            path.mkdir(parents=True, exist_ok=True)

    def process(self) -> None:
        """执行完整的标准视频处理流程"""
        print("=== 开始标准视频处理流水线 ===")

        # 1. 视频处理：生成骨架JSON文件
        print("\n1. 处理标准视频生成骨架数据...")
        pose_estimator = PoseEstimator(
            model_path=self.model_path,
            dataset_path=self.dataset_path,
            output_dir=self.json_path,
        )
        pose_estimator.process_dataset()

        # 2. 数据集分割：全部作为训练集
        print("\n2. 分配数据集...")
        splitter = DatasetSplitter(
            data_root=self.json_path,
            train_folder=self.train_path,
            test_folder=self.val_path,
            split_ratio=1.0,  # 关键修改：将所有数据分配到训练集
        )
        splitter.split()

        # 3. 生成标签文件
        print("\n3. 生成标签文件...")
        label_generator = LabelGenerator(
            output_path=self.processed_path,
            train_folder=self.train_path,
            val_folder=self.val_path,
        )
        label_generator.generate()

        # 4. 生成最终的NPY文件
        print("\n4. 生成NPY标准文件...")
        data_processor = KineticsDataProcessor(
            data_root=self.processed_path,
        )
        data_processor.process()

        print("\n=== 标准视频处理流水线完成 ===")
        print(f"处理后的文件保存在: {self.processed_path}")


if __name__ == "__main__":
    processor = StandardDataProcessor()
    processor.process()
