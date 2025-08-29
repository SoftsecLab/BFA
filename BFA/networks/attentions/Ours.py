import torch
import torch.nn as nn
import torch.nn.init as init
from thop import profile  # 用于计算 FLOPs 和参数数量

# Ours 直接串联
class Ours(nn.Module):
    def __init__(self, kernel_size=3, e_lambda=1e-4):
        super(Ours, self).__init__()
        # 通道部分
        self.gap = nn.AdaptiveAvgPool2d(1)  # 全局平均池化层
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2)  # 1D 卷积
        self.sigmoid = nn.Sigmoid()  # Sigmoid 激活函数

        # 空间部分
        self.e_lambda = e_lambda  # 平滑项
        self.act = nn.Sigmoid()  # Sigmoid 激活函数

    def forward(self, x):
        # 通道注意力
        y = self.gap(x)  # 全局平均池化，得到 [bs, c, 1, 1]，将输入特征图的每个通道压缩为1*1的标量
        y = y.squeeze(-1).permute(0, 2, 1)  # 转换为 [bs, 1, c]
        y = self.conv(y)  # 1D 卷积，得到 [bs, 1, c]
        y = self.sigmoid(y)  # Sigmoid 激活，得到通道注意力权重
        y = y.permute(0, 2, 1).unsqueeze(-1)  # 转换为 [bs, c, 1, 1]
        x = x * y.expand_as(x)  # 通道注意力加权

        # 空间注意力
        b, c, h, w = x.size()  # 获取输入尺寸
        n = w * h - 1  # 特征图元素数量减一
        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)  # 计算 (x - μ)^2
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)) + 0.5
        x = x * self.act(y)  # 空间注意力加权

        return x


# Our 并行+自适应结构
class ParallelECASimAM(nn.Module):
    """
    并行ECA与SimAM注意力模块
    特点：
    1. 并行计算通道(ECA)和空间(SimAM)注意力，避免串行干扰
    2. 自适应融合权重，无需手动调参
    3. 即插即用，支持4D输入张量 [B, C, H, W]

    参数:
        channels (int): 输入通道数
        kernel_size (int): ECA的1D卷积核大小，默认为3
        e_lambda (float): SimAM能量函数的平滑系数，默认为1e-4
    """

    def __init__(self, channels, kernel_size=3, e_lambda=1e-4):
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.e_lambda = e_lambda

        # ---- ECA分支（通道注意力）----
        self.gap = nn.AdaptiveAvgPool2d(1)  # 全局平均池化
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size,
                              padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

        # ---- SimAM分支（空间注意力）----
        self.spatial_act = nn.Sigmoid()

        # ---- 自适应融合权重 ----
        # 初始化为0.5（平衡两个分支）
        self.fusion_weight = nn.Parameter(torch.tensor([0.5]))

    def forward(self, x):
        """ 前向传播
        输入: x [B, C, H, W]
        输出: out [B, C, H, W]
        """
        b, c, h, w = x.shape

        # ===== 1. ECA分支（通道注意力）=====
        # 全局平均池化 -> [B,C,1,1]
        eca = self.gap(x)
        # 调整维度 -> [B,1,C]
        eca = eca.squeeze(-1).permute(0, 2, 1)
        # 1D卷积捕捉跨通道交互 -> [B,1,C]
        eca = self.conv(eca)
        # 激活 -> [B,C,1,1]
        eca_weights = self.sigmoid(eca).permute(0, 2, 1).unsqueeze(-1)
        # 通道注意力加权
        x_eca = x * eca_weights.expand_as(x)

        # ===== 2. SimAM分支（空间注意力）=====
        n = h * w - 1  # 归一化因子
        # 计算均值μ和方差σ^2
        mean = x.mean(dim=[2, 3], keepdim=True)
        var = (x - mean).pow(2).sum(dim=[2, 3], keepdim=True) / n
        # 能量计算 -> [B,C,H,W]
        energy = (x - mean).pow(2) / (4 * (var + self.e_lambda)) + 0.5
        # 空间注意力加权
        x_simam = x * self.spatial_act(energy)

        # ===== 3. 自适应融合 =====
        # 动态权重（通过Sigmoid约束到0~1）
        alpha = torch.sigmoid(self.fusion_weight)
        # 加权融合
        out = alpha * x_simam + (1 - alpha) * x_eca

        return out


class OptimizedParallelECASimAM(nn.Module):
    """即插即用优化版ECA+SimAM模块
    特点：
    1. 完全兼容原SimAM项目结构
    2. 动态核大小ECA + 增强SimAM
    3. 自动维度处理，无运行时错误
    """

    def __init__(self, channels, gamma=2, b=1, e_lambda=1e-4):
        super().__init__()
        self.channels = channels
        self.e_lambda = e_lambda

        # ---- 动态核ECA ----
        k_size = self._get_kernel_size(channels, gamma, b)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size,
                              padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

        # ---- 融合模块 ----
        self.fusion = nn.Sequential(
            nn.Linear(channels, max(4, channels // 4), bias=False),
            nn.ReLU(),
            nn.Linear(max(4, channels // 4), channels, bias=False),
            nn.Sigmoid()
        )

    def _get_kernel_size(self, channels, gamma, b):
        k_size = int(abs((math.log(channels, 2) + b) / gamma))
        return k_size if k_size % 2 else k_size + 1

    def forward(self, x):
        b, c, h, w = x.size()

        # ===== 1. ECA分支 =====
        # 正确处理4D->3D转换
        y = self.avg_pool(x)  # [b,c,1,1]
        y = y.view(b, c, 1)  # [b,c,1]
        y = y.transpose(1, 2)  # [b,1,c]
        y = self.conv(y)  # [b,1,c]
        y = self.sigmoid(y)  # [b,1,c]
        y = y.transpose(1, 2).view(b, c, 1, 1)  # [b,c,1,1]
        x_eca = x * y.expand_as(x)

        # ===== 2. SimAM分支 =====
        n = w * h - 1
        x_mean = x.mean(dim=[2, 3], keepdim=True)
        x_var = (x - x_mean).pow(2).sum(dim=[2, 3], keepdim=True) / n
        energy = (x - x_mean).pow(2) / (4 * (x_var + self.e_lambda)) + 0.5
        x_simam = x * torch.sigmoid(energy)

        # ===== 3. 智能融合 =====
        gap = F.avg_pool2d(x_eca + x_simam, (h, w)).view(b, c)
        alpha = self.fusion(gap).view(b, c, 1, 1)
        return alpha * x_simam + (1 - alpha) * x_eca

    @staticmethod
    def get_module_name():
        """保持与原项目兼容的接口"""
        return "se"

## 生物启发代码
class se_module(nn.Module):
    # reduction=16:这个一定要加否则会报错
    def __init__(self, max_channels=3, num_tasks=None, mem_size=32, reduction=4):
        super().__init__()
        # reduction至少为2
        # 确保reduction是整数且≥2
        self.reduction = max(int(reduction), 2)  # 强制转换为整数
        self.num_tasks = num_tasks
        # 所有维度计算使用整除运算符 //
        self.max_channels = int(max_channels)
        self.mem_size = int(mem_size)

        # 计算特征维度（确保≥1且为整数）
        self.feature_dim = max(self.max_channels // self.reduction, 1)

        # 1. 动态任务条件化
        if num_tasks is not None:
            self.task_embed = nn.Sequential(
                nn.Embedding(num_tasks, self.feature_dim),
                nn.Linear(self.feature_dim, self.max_channels),
                nn.SiLU(inplace=True)
            )

        # 2. 跨维度门控
        self.thresholds = nn.Parameter(torch.zeros(1, self.max_channels, 1, 1))
        self.lambda_sparse = nn.Parameter(torch.tensor(0.01))

        # 3. 动态记忆库（确保所有维度是整数）
        self.mem_keys = nn.Parameter(torch.randn(self.mem_size, self.feature_dim))
        self.mem_values = nn.Parameter(torch.randn(self.mem_size, self.max_channels))

        # 动态特征适配器
        self.feature_adapter = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(self.max_channels, self.feature_dim),
            nn.LayerNorm(self.feature_dim)
        )

        self.gamma = nn.Parameter(torch.tensor(0.5))
        self._init_weights()

    def _init_weights(self):
        if hasattr(self, 'mem_keys'):
            nn.init.trunc_normal_(self.mem_keys, std=0.02)
        if hasattr(self, 'mem_values'):
            nn.init.xavier_uniform_(self.mem_values)
        if hasattr(self, 'thresholds'):
            nn.init.uniform_(self.thresholds, 0.1, 0.3)

    @staticmethod
    def get_module_name():
        return "se"

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
