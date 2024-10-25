import os
import json
import random
import shutil

def split_and_copy_json_files(data_folder, train_folder, test_folder, label, split_ratio=0.8):
    json_files = [f for f in os.listdir(data_folder) if f.endswith(".json")]
    random.shuffle(json_files)

    # 计算训练集文件数量
    num_train_files = int(len(json_files) * split_ratio)
    train_files = json_files[:num_train_files]
    test_files = json_files[num_train_files:]

    # 如果不存在,创建训练和测试文件夹
    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(test_folder, exist_ok=True)

    # 复制选定的JSON文件到训练文件夹
    for file in train_files:
        source_path = os.path.join(data_folder, file)
        dest_path = os.path.join(train_folder, f"{label}_{file}")
        shutil.copy(source_path, dest_path)

    # 复制选定的JSON文件到测试文件夹
    for file in test_files:
        source_path = os.path.join(data_folder, file)
        dest_path = os.path.join(test_folder, f"{label}_{file}")
        shutil.copy(source_path, dest_path)

# 设置标签列表
labels = ['00', '01', '02', '03', '04', '05', '06', '07', '10', '11', '12', '13', '14', '15']

# 设置文件夹路径和分割比例
data_root = '.'
train_folder = "../train"
test_folder = "../val"
split_ratio = 0.8  # 80% 用于训练, 20% 用于测试

for label in labels:
    folder_name = next((f for f in os.listdir(data_root) if f.startswith(label)), None)
    if folder_name:
        data_folder = os.path.join(data_root, folder_name)
        split_and_copy_json_files(data_folder, train_folder, test_folder, label, split_ratio)
    else:
        print(f"警告: 没有找到以 {label} 开头的文件夹,已跳过。")