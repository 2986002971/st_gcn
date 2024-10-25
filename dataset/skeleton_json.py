import json
import os

import cv2
import numpy as np
import torch
from ultralytics import YOLO


def InputReader(path):
    cap = cv2.VideoCapture("{}".format(path))  # 视频流读取
    rate = cap.get(5)  # 帧速率
    FrameNumber = cap.get(7)  # 视频文件的帧数
    duration = FrameNumber / rate  # 帧速率/视频总帧数 是时间，除以60之后单位是分钟
    return cap, rate, FrameNumber, duration


def single_video_output(img, model):
    result = model(img, imgsz=320, conf=0.5)[0]
    a = result.keypoints.conf
    b = result.keypoints.xyn

    # 如果预测结果为空，则补0；a补n*1，b补n*2，n为关节点数目
    if a is None:
        a = torch.zeros(17).unsqueeze(0)
        b = torch.zeros(34).unsqueeze(0).unsqueeze(0)  # 需要  扩维度
    # 输出 关键点
    np1 = b[0].cpu().numpy()
    np2 = np.around(np1, 3).flatten()

    # 输出 置信度
    conf1 = a.cpu().numpy()
    conf2 = np.around(conf1, 3)[0]

    point = [round(i, 3) for i in np2]
    conf = [round(i, 3) for i in conf2]
    return point, conf


class single_video_json_output:
    def __init__(self):
        self.model = YOLO("yolov8n-pose.pt")  # 确保这个路径是正确的
        self.capture = ""
        self.class_name = ""
        self.label_index = -1

    def _inference(self, name):
        cap, rate, FrameNumber, duration = InputReader(self.capture)

        # json存储
        frame_index = 0
        jsdata = []

        while cap.isOpened():
            rec, img = cap.read()
            if not rec:
                break
            frame_index += 1

            pose_data = {"pose": "", "score": ""}
            frame_data = {"frame_index": 0, "skeleton": []}

            point, conf = single_video_output(img, self.model)
            point = [round(float(x), 3) for x in point]  # 数据保留小数点位数
            conf = [round(float(x), 3) for x in conf]

            # 字典内部存储 此处需要修改 因为有多个人的视频
            pose_data = {"pose": point, "score": conf}
            frame_data = {
                "frame_index": frame_index,
                "skeleton": [pose_data],
            }  # 注意加个括号
            if sum(point) == 0:
                frame_data = {"frame_index": frame_index, "skeleton": []}
            jsdata.append(frame_data)

            del img
            # 按q结束
            if cv2.waitKey(1) == ord("q"):
                break

        output_data = {
            "data": jsdata,
            "label": self.class_name,
            "label_index": self.label_index,
        }

        # 修改文件路径生成逻辑
        file_path = os.path.join(
            "..", "output", self.class_name, name.split(".")[0] + ".json"
        )

        # 确保目录存在
        directory = os.path.dirname(file_path)
        os.makedirs(directory, exist_ok=True)
        with open(file_path, "w") as json_file:
            json.dump(output_data, json_file, indent=None, separators=(",", ":"))


# 运行 所有类别
if __name__ == "__main__":
    root_path = "."  # 假设脚本在dataset文件夹内运行

    # 创建一个字典来映射文件夹名称到索引
    class_to_index = {}
    index = 0

    for class_folder in sorted(os.listdir(root_path)):
        if os.path.isdir(
            os.path.join(root_path, class_folder)
        ) and not class_folder.startswith("."):
            class_to_index[class_folder] = index
            index += 1

    for class_folder, label_index in class_to_index.items():
        class_path = os.path.join(root_path, class_folder)
        a = single_video_json_output()
        a.class_name = class_folder
        a.label_index = label_index

        for video_name in os.listdir(class_path):
            if video_name.endswith(".mp4"):
                a.capture = os.path.join(class_path, video_name)
                a._inference(video_name)

    print("处理完成")
