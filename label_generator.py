import json
import os


def get_label_index_mapping():
    root_path = "."  # 假设我们在 data 目录下
    output_path = os.path.join(root_path, "output")

    class_to_index = {}
    index = 0

    for class_folder in sorted(os.listdir(output_path)):
        if os.path.isdir(
            os.path.join(output_path, class_folder)
        ) and not class_folder.startswith("."):
            class_to_index[class_folder[:2]] = index  # 使用文件夹名称的前两位数字作为键
            index += 1

    return class_to_index


def combine_js(file_path, out_path, label_index_mapping):
    data = {}

    for filename in os.listdir(file_path):
        if filename.endswith(".json"):
            json_file_path = os.path.join(file_path, filename)

            with open(json_file_path, "r") as json_file:
                json_data = json.load(json_file)
                # 检查空数据
                has_skeleton = any(
                    any(
                        skeleton["pose"] or skeleton["score"]
                        for skeleton in frame["skeleton"]
                    )
                    for frame in json_data["data"]
                )

                # 从文件名中提取标签
                label = filename.split("_")[0]

                entry = {
                    "has_skeleton": has_skeleton,
                    "label": label,
                    "label_index": label_index_mapping.get(
                        label, -1
                    ),  # 如果找不到映射，默认为-1
                }

                data[filename.split(".")[0]] = entry

    with open(out_path, "w") as output_json_file:
        json.dump(data, output_json_file, indent=4)


# 获取标签索引映射
label_index_mapping = get_label_index_mapping()

# 调用函数，传入文件夹路径和输出文件名
combine_js("train", "train_label.json", label_index_mapping)
combine_js("val", "val_label.json", label_index_mapping)
