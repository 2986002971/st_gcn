"""
COCO关键点定义和对偶图边的定义

关键点索引：
1 - 鼻子
2 - 左眼
3 - 右眼
4 - 左耳
5 - 右耳
6 - 左肩
7 - 右肩
8 - 左肘
9 - 右肘
10 - 左手
11 - 右手
12 - 左胯
13 - 右胯
14 - 左膝
15 - 右膝
16 - 左脚
17 - 右脚
"""

# 定义对偶图的边（注意：这里使用1-based索引，与COCO保持一致）
dual_coco_edges = [
    # 肩部连接
    (6, 7),  # 肩线
    (12, 13),  # 胯线
    (6, 12),  # 左躯干
    (7, 13),  # 右躯干
    (0, 6),  # 左颈部
    (0, 7),  # 右颈部
    # 手臂连接
    (6, 8),  # 左上臂
    (7, 9),  # 右上臂
    (8, 10),  # 左前臂
    (9, 11),  # 右前臂
    # 腿部连接
    (12, 14),  # 左大腿
    (13, 15),  # 右大腿
    (14, 16),  # 左小腿
    (15, 17),  # 右小腿
]

# 定义对偶图中的连接关系（基于点线对偶）
neighbor_link = [
    # 鼻子
    (4, 5),  # 鼻子
    # 肩线
    (0, 2),  # 左肩
    (0, 3),  # 右肩
    (0, 4),  # 左肩
    (0, 5),  # 右肩
    (0, 6),  # 左肩
    (0, 8),  # 右肩
    (2, 4),  # 左肩
    (2, 6),  # 左肩
    (3, 5),  # 右肩
    (3, 8),  # 右肩
    (4, 6),  # 左肩
    (5, 8),  # 右肩
    # 肘
    (6, 7),  # 左肘
    (8, 9),  # 右肘
    # 胯
    (1, 2),  # 左胯
    (1, 3),  # 右胯
    (1, 10),  # 左胯
    (1, 12),  # 右胯
    (2, 10),  # 左胯
    (3, 12),  # 右胯
    # 膝
    (10, 11),  # 左膝
    (12, 13),  # 右膝
]
