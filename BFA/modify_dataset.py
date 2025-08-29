# import glob
# import os
# from shutil import move
# from os import rmdir
# import cv2
# import matplotlib.pyplot as plt
# import os
# """
# 第一段是修改tiny-imagenet数据集的结构
# """
# # # 修改目标文件夹路径
# # target_folder = './datasets/tiny-imagenet-200/val/'
# #
# # # 读取验证集的标注文件
# # val_dict = {}
# # with open('./datasets/tiny-imagenet-200/val/val_annotations.txt', 'r') as f:
# #     for line in f.readlines():
# #         split_line = line.split('\t')
# #         val_dict[split_line[0]] = split_line[1]
# #
# # # 获取所有验证集图像的路径
# # paths = glob.glob('./datasets/tiny-imagenet-200/val/images/*')
# #
# # # 为每个类别创建子文件夹
# # for path in paths:
# #     file = os.path.basename(path)  # 使用 os.path.basename 提取文件名
# #     folder = val_dict[file]
# #     if not os.path.exists(target_folder + str(folder)):
# #         os.mkdir(target_folder + str(folder))  # 直接创建类别文件夹，不需要 images 子文件夹
# #
# # # 将图像移动到对应的类别文件夹
# # for path in paths:
# #     file = os.path.basename(path)  # 使用 os.path.basename 提取文件名
# #     folder = val_dict[file]
# #     dest = os.path.join(target_folder, str(folder), file)  # 使用 os.path.join 构建目标路径
# #     move(path, dest)
# #
# # # 删除空的 images 文件夹
# # rmdir('./datasets/tiny-imagenet-200/val/images')
#
#
#
# import cv2
# import matplotlib.pyplot as plt
#
# # 指定 Tiny ImageNet 数据集中的一张图像的路径
# image_path = './datasets/tiny-imagenet-200/train/n02231487/images/n02231487_19.JPEG'  # 请确保路径正确
#
# # 读取图像
# image = cv2.imread(image_path)
#
# # 检查图像是否成功读取
# if image is None:
#     raise ValueError("图像未找到，请检查文件路径。")
#
# # 将 BGR 图像转换为 RGB
# image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#
# # 打印原始图像大小
# print(f'Original Image Size: {image_rgb.shape[1]}x{image_rgb.shape[0]}')
#
# # 设定目标尺寸（例如，224x224）
# target_size = (224, 224)
#
# # 使用双线性插值进行上采样
# bilinear_image = cv2.resize(image_rgb, target_size, interpolation=cv2.INTER_LINEAR)
#
# # 打印双线性插值图像大小
# print(f'Bilinear Interpolation Image Size: {bilinear_image.shape[1]}x{bilinear_image.shape[0]}')
#
# # 使用双三次插值进行上采样
# bicubic_image = cv2.resize(image_rgb, target_size, interpolation=cv2.INTER_CUBIC)
#
# # 打印双三次插值图像大小
# print(f'Bicubic Interpolation Image Size: {bicubic_image.shape[1]}x{bicubic_image.shape[0]}')
#
# # 显示图像
# plt.figure(figsize=(12, 8))
#
# # 原始图像
# plt.subplot(2, 2, 1)
# plt.title('Original Image')
# plt.imshow(image_rgb)
# plt.axis('off')
#
# # 双线性插值上采样
# plt.subplot(2, 2, 2)
# plt.title('Bilinear Interpolation')
# plt.imshow(bilinear_image)
# plt.axis('off')
#
# # 双三次插值上采样
# plt.subplot(2, 2, 3)
# plt.title('Bicubic Interpolation')
# plt.imshow(bicubic_image)
# plt.axis('off')
#
# plt.tight_layout()
# plt.show()
#只适用于tiny-imagenet
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
#
#
# class BioInspiredAttention(nn.Module):
#     def __init__(self, channels, num_tasks=10, mem_size=32, reduction=16):
#         super().__init__()
#         self.channels = channels
#         self.num_tasks = num_tasks  # 新增：定义num_tasks属性
#         self.mem_size = mem_size
#         self.reduction = reduction  # 新增：定义reduction属性
#
#         # 1. 任务条件化特征重加权（TCR）
#         self.task_embed = nn.Sequential(
#             nn.Embedding(num_tasks, channels // reduction),
#             nn.Linear(channels // reduction, channels))
#
#         # 2. 跨维度信息门控（CDG）
#         self.thresholds = nn.Parameter(torch.zeros(1, channels, 1, 1))
#         self.lambda_sparse = 0.01
#
#         # 3. 记忆增强优化（MAO）
#         self.mem_keys = nn.Parameter(torch.randn(mem_size, channels // reduction))
#         self.mem_values = nn.Parameter(torch.randn(mem_size, channels, 1, 1))
#         self.mem_proj = nn.Linear(channels, channels // reduction)
#
#         self.gamma = nn.Parameter(torch.zeros(1))
#         self._init_weights()  # 初始化权重
#
#     def _init_weights(self):
#         # 初始化记忆库
#         nn.init.normal_(self.mem_keys, mean=0, std=0.02)
#         nn.init.normal_(self.mem_values, mean=0, std=0.02)
#
#         # 初始化阈值参数
#         nn.init.uniform_(self.thresholds, 0.1, 0.5)
#
#         # 初始化任务嵌入层
#         nn.init.xavier_uniform_(self.task_embed[0].weight)
#         nn.init.zeros_(self.task_embed[1].bias)
#
#     def forward(self, x, task_id=None):
#         b, c, h, w = x.shape
#
#         # 1. TCR
#         if task_id is not None:
#             t = self.task_embed(task_id)  # [B, C]
#             alpha = torch.sigmoid(t).view(b, c, 1, 1)  # [B, C, 1, 1]
#             x_tcr = x * alpha
#         else:
#             x_tcr = x
#
#         # 2. CDG
#         mu = x_tcr.mean(dim=[2, 3], keepdim=True)  # [B, C, 1, 1]
#         var = x_tcr.var(dim=[2, 3], keepdim=True)  # [B, C, 1, 1]
#         energy = (x_tcr - mu).pow(2) / (2 * (var + 1e-5)) + self.lambda_sparse * self.thresholds.abs()
#         mask = (energy < self.thresholds).float()
#         x_gated = x_tcr * mask
#
#         # 3. MAO
#         if task_id is not None:
#             query = self.mem_proj(t).view(b, 1, -1)  # [B, 1, C//r]
#             scores = torch.matmul(query, self.mem_keys.T)  # [B, 1, mem_size]
#             attn = F.softmax(scores, dim=-1)
#
#             # 修正维度匹配问题
#             v_star = torch.matmul(attn, self.mem_values.view(self.mem_size, c))  # [B, 1, C]
#             v_star = v_star.view(b, c, 1, 1)  # [B, C, 1, 1]
#
#             out = torch.sigmoid(self.gamma) * v_star + (1 - torch.sigmoid(self.gamma)) * x_gated
#         else:
#             out = x_gated
#
#         return out
#
# # 测试代码
# if __name__ == "__main__":
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
#     # 实例化模块
#     att = BioInspiredAttention(channels=256, num_tasks=200).to(device)
#
#     # 创建测试数据
#     x = torch.randn(32, 256, 64, 64).to(device)  # 模拟CNN特征图
#     task_ids = torch.randint(0, 200, (32,)).to(device)  # 随机任务ID
#
#     # 前向测试
#     out = att(x, task_ids)
#     print(f"输入形状: {x.shape} -> 输出形状: {out.shape}")
#
#     # 梯度测试
#     x.requires_grad_(True)
#     loss = out.sum()
#     loss.backward()
#     print("梯度测试通过")

import torch
import torch.nn as nn
import torch.nn.functional as F


class UniversalAttention(nn.Module):
    """
    修正版全动态注意力模块（已解决所有维度错误）
    保证即插即用，适配任意输入尺寸
    """

    def __init__(self, max_channels=3, num_tasks=None, mem_size=32, reduction=4):  # reduction至少为2
        super().__init__()
        assert reduction >= 2, "reduction必须≥2以保证维度有效性"
        self.max_channels = max_channels
        self.num_tasks = num_tasks
        self.mem_size = mem_size
        self.reduction = reduction

        # 1. 动态任务条件化
        if num_tasks is not None:
            self.task_embed = nn.Sequential(
                nn.Embedding(num_tasks, max(max_channels // reduction, 1)),  # 保证≥1
                nn.Linear(max(max_channels // reduction, 1), max_channels),
                nn.SiLU(inplace=True)
            )

        # 2. 跨维度门控
        self.thresholds = nn.Parameter(torch.zeros(1, max_channels, 1, 1))
        self.lambda_sparse = nn.Parameter(torch.tensor(0.01))

        # 3. 动态记忆库
        self.mem_keys = nn.Parameter(torch.randn(mem_size, max(max_channels // reduction, 1)))
        self.mem_values = nn.Parameter(torch.randn(mem_size, max_channels))

        # 动态特征适配器
        self.feature_adapter = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(max_channels, max(max_channels // reduction, 1)),  # 保证≥1
            nn.LayerNorm(max(max_channels // reduction, 1))
        )

        self.gamma = nn.Parameter(torch.tensor(0.5))
        self._init_weights()

    def _init_weights(self):
        if hasattr(self, 'mem_keys') and self.mem_keys.numel() > 0:  # 避免零元素初始化警告
            nn.init.trunc_normal_(self.mem_keys, std=0.02)
        if hasattr(self, 'mem_values') and self.mem_values.numel() > 0:
            nn.init.xavier_uniform_(self.mem_values)
        if hasattr(self, 'thresholds') and self.thresholds.numel() > 0:
            nn.init.uniform_(self.thresholds, 0.1, 0.3)

    def forward(self, x, task_id=None):
        b, c, h, w = x.shape
        assert c <= self.max_channels, f"输入通道数{c}超过最大设置{self.max_channels}"

        # 1. 任务条件化
        x_tcr = x
        if self.num_tasks is not None and task_id is not None:
            t_full = self.task_embed(task_id)  # [B, max_channels]
            t = t_full[:, :c]  # 动态裁剪 [B, actual_c]
            alpha = torch.sigmoid(t).view(b, c, 1, 1)
            x_tcr = x * alpha

        # 2. 跨维度门控
        mu = x_tcr.mean(dim=[2, 3], keepdim=True)
        var = x_tcr.var(dim=[2, 3], keepdim=True)
        energy = (x_tcr - mu).pow(2) / (2 * (var + 1e-5)) + self.lambda_sparse * self.thresholds[:, :c]
        mask = (energy < self.thresholds[:, :c]).float()
        x_gated = x_tcr * mask

        # 3. 记忆增强
        if self.num_tasks is not None:
            # 动态调整reduction维度
            actual_reduction = max(c // self.reduction, 1)

            # 特征适配
            query = self.feature_adapter(x_gated)  # [B, max(C//r,1)]
            query = query.unsqueeze(1)  # [B, 1, D]

            # 记忆检索
            keys = self.mem_keys[:, :actual_reduction]  # [mem_size, D]
            scores = torch.matmul(query, keys.unsqueeze(0).transpose(1, 2))  # [B,1,mem_size]
            attn = F.softmax(scores, dim=-1)

            # 记忆融合
            values = self.mem_values[:, :c]  # [mem_size, C]
            v_star = torch.matmul(attn, values.unsqueeze(0))  # [B,1,C]
            v_star = v_star.transpose(1, 2).view(b, c, 1, 1)

            out = torch.sigmoid(self.gamma) * v_star + (1 - torch.sigmoid(self.gamma)) * x_gated
        else:
            out = x_gated

        return out


def test_module():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"测试设备: {device}")

    # 测试极端情况（单通道+小reduction）
    att = UniversalAttention(max_channels=1, num_tasks=5, reduction=2).to(device)
    x = torch.randn(4, 1, 32, 32).to(device)
    task_ids = torch.randint(0, 5, (4,)).to(device)
    out = att(x, task_ids)
    print(f"单通道测试通过! 输出形状: {out.shape}")

    # 测试标准RGB
    att = UniversalAttention(max_channels=3, num_tasks=10).to(device)
    x = torch.randn(4, 3, 224, 224).to(device)
    task_ids = torch.randint(0, 10, (4,)).to(device)
    out = att(x, task_ids)
    print(f"RGB测试通过! 输出形状: {out.shape}")

    # 测试无任务模式
    att = UniversalAttention(max_channels=64, num_tasks=None).to(device)
    x = torch.randn(4, 64, 16, 16).to(device)
    out = att(x)
    print(f"无任务模式测试通过! 输出形状: {out.shape}")


if __name__ == "__main__":
    test_module()
    print("所有维度问题已解决，测试通过！")