import json
import os
from typing import Dict


class LabelGenerator:
    def __init__(
        self,
        output_path: str = "./processed",
        train_folder: str = "./processed/train",
        val_folder: str = "./processed/val",
    ):
        """
        标签生成器初始化

        Args:
            output_path: 输出目录路径，默认为 "./processed"
            train_folder: 训练数据目录，默认为 "./processed/train"
            val_folder: 验证数据目录，默认为 "./processed/val"
        """
        self.output_path = output_path
        self.train_folder = train_folder
        self.val_folder = val_folder
        self.label_index_mapping = self._get_label_index_mapping()

    def _get_label_index_mapping(self) -> Dict[str, int]:
        """
        获取标签到索引的映射

        Returns:
            Dict[str, int]: 标签到索引的映射字典
        """
        class_to_index = {}
        index = 0

        # 修改为在 json 目录下查找类别文件夹
        json_path = os.path.join(self.output_path, "json")

        for class_folder in sorted(os.listdir(json_path)):
            if os.path.isdir(
                os.path.join(json_path, class_folder)
            ) and not class_folder.startswith("."):
                class_to_index[class_folder[:2]] = index
                print(f"找到类别: {class_folder[:2]} -> {index}")  # 添加调试信息
                index += 1

        if not class_to_index:
            print(f"警告: 在 {json_path} 目录下没有找到任何类别文件夹")

        return class_to_index

    def _process_json_file(self, file_path: str) -> Dict:
        """
        处理单个JSON文件

        Args:
            file_path: JSON文件路径

        Returns:
            Dict: 处理后的数据条目
        """
        with open(file_path, "r") as json_file:
            json_data = json.load(json_file)

            # 检查骨架数据是否存在
            has_skeleton = any(
                any(
                    skeleton["pose"] or skeleton["score"]
                    for skeleton in frame["skeleton"]
                )
                for frame in json_data["data"]
            )

            # 从文件名中提取标签和准确度
            filename = os.path.basename(file_path)
            # 使用正确的分割方式解析文件名
            parts = filename.split("_")
            label = parts[0]  # 获取类别标签
            accuracy = float(parts[2].split(".json")[0])  # 正确提取标准度

            return {
                "has_skeleton": has_skeleton,
                "label": label,
                "label_index": self.label_index_mapping.get(label, -1),
                "accuracy": accuracy,
            }

    def _combine_json_files(self, folder_path: str, output_filename: str) -> None:
        """
        合并指定文件夹中的所有JSON文件

        Args:
            folder_path: 输入文件夹路径
            output_filename: 输出文件名
        """
        data = {}

        for filename in os.listdir(folder_path):
            if filename.endswith(".json"):
                json_file_path = os.path.join(folder_path, filename)
                entry = self._process_json_file(json_file_path)
                file_key = filename.rsplit(".json", 1)[0]
                data[file_key] = entry

        output_path = os.path.join(self.output_path, output_filename)
        with open(output_path, "w") as output_json_file:
            json.dump(data, output_json_file, indent=4)

    def _sort_and_save_label_file(self, input_file: str) -> None:
        """
        读取标签文件，按标签索引排序后重新保存

        Args:
            input_file: 输入文件路径
        """
        # 读取原始json文件
        with open(input_file, "r") as f:
            data = json.load(f)

        # 将数据转换为列表并按label_index排序
        sorted_items = sorted(data.items(), key=lambda x: x[1]["label_index"])

        # 转换回有序字典
        sorted_data = {k: v for k, v in sorted_items}

        # 保存排序后的数据
        with open(input_file, "w") as f:
            json.dump(sorted_data, f, indent=4)

    def generate(self, sort_by_label: bool = False) -> None:
        """
        生成训练集和验证集的标签文件

        Args:
            sort_by_label: 是否按标签索引排序，默认为False
        """
        self._combine_json_files(self.train_folder, "train_label.json")
        self._combine_json_files(self.val_folder, "val_label.json")

        if sort_by_label:
            train_label_path = os.path.join(self.output_path, "train_label.json")
            val_label_path = os.path.join(self.output_path, "val_label.json")
            self._sort_and_save_label_file(train_label_path)
            self._sort_and_save_label_file(val_label_path)
            print("标签文件已按标签索引排序")

        print(f"标签生成完成！文件保存在: {self.output_path}")


if __name__ == "__main__":
    generator = LabelGenerator()
    generator.generate()
