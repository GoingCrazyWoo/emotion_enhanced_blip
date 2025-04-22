import math
import torch
import torch.nn.functional as F
from torch import nn

from .rotary import apply_rotary_emb  # 使用相对位置编码模块

# 定义RMSNorm类，用于归一化
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, elementwise_affine=True, memory_efficient=False):
        """
        初始化RMSNorm类
        :param dim: 归一化的维度
        :param eps: 避免除零错误的小常数
        :param elementwise_affine: 是否使用逐元素的仿射变换（即是否有可学习的权重）
        :param memory_efficient: 是否使用内存高效模式（这里没有用到）
        """
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            # 如果需要仿射变换，则初始化权重为1
            self.weight = nn.Parameter(torch.ones(dim))
        else:
            self.register_parameter('weight', None)

    def _norm(self, x):
        """
        进行RMS归一化
        :param x: 输入张量
        :return: 归一化后的张量
        """
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        """
        前向传播
        :param x: 输入张量
        :return: 归一化后的张量
        """
        output = self._norm(x.float()).type_as(x)  # 先进行RMS归一化，再转换回输入张量类型
        if self.weight is not None:
            # 如果有仿射变换，则对归一化后的张量进行权重缩放
            output = output * self.weight
        return output

    def extra_repr(self) -> str:
        """
        打印该层的额外信息
        :return: RMSNorm层的额外描述
        """
        return f'dim={self.dim}, eps={self.eps}, elementwise_affine={self.elementwise_affine}'

# 定义repeat_kv函数，用于重复键值对
def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    重复键值对（k, v）以适应多头注意力
    :param x: 输入张量，形状为[batch_size, num_heads, seq_len, head_dim]
    :param n_rep: 重复次数
    :return: 重复后的张量
    """
    bs, n_kv_heads, slen, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, None, :, :]  # 增加一个维度，使得每个键值对在该维度上可以重复
        .expand(bs, n_kv_heads, n_rep, slen, head_dim)  # 扩展至重复n_rep次
        .reshape(bs, n_kv_heads * n_rep, slen, head_dim)  # 展平重复维度，得到新的形状
    )

# 定义lambda初始化函数，用于调整注意力权重的衰减因子
def lambda_init_fn(depth):
    """
    根据当前层数计算lambda衰减因子
    :param depth: 当前层数
    :return: 衰减因子的值
    """
    return 0.8 - 0.6 * math.exp(-0.3 * depth)

# 定义多头差分注意力（MultiheadDiffAttn）层
class MultiheadDiffAttn(nn.Module):
    def __init__(
        self,
        embed_dim,
        depth,  # 当前层的索引
        num_heads,
        num_kv_heads=None,  # 键值头数，默认为None
    ):
        """
        初始化多头差分注意力层
        :param embed_dim: 输入的嵌入维度
        :param depth: 当前层的索引
        :param num_heads: 注意力头的数量
        :param num_kv_heads: 键值头的数量（如果使用GQA，则需要指定）
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads  # 设置注意力头数量
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads  # 设置键值头数量
        self.n_rep = self.num_heads // self.num_kv_heads  # 计算重复次数

        # 每个头的维度
        self.head_dim = embed_dim // num_heads // 2
        self.scaling = self.head_dim ** -0.5  # 缩放因子

        # 定义查询、键、值的线性变换层
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim // self.n_rep, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim // self.n_rep, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        # 根据当前层的深度初始化lambda值
        self.lambda_init = lambda_init_fn(depth)
        # 定义lambda参数，用于差分注意力的加权
        self.lambda_q1 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0, std=0.1))
        self.lambda_k1 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0, std=0.1))
        self.lambda_q2 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0, std=0.1))
        self.lambda_k2 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0, std=0.1))

        # 使用RMSNorm进行子层归一化
        self.subln = RMSNorm(2 * self.head_dim, eps=1e-5, elementwise_affine=True)

    def forward(
        self,
        x,
        rel_pos,
        attn_mask=None,
    ):
        """
        前向传播
        :param x: 输入特征张量，形状为[batch_size, seq_len, embed_dim]
        :param rel_pos: 相对位置编码，元组形式(cos, sin)
        :param attn_mask: 注意力掩码，默认None
        :return: 计算后的注意力输出
        """
        bsz, tgt_len, embed_dim = x.size()  # 获取输入的形状
        src_len = tgt_len  # 源序列的长度

        # 计算查询、键、值
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # 将查询、键、值变换为多头形式
        q = q.view(bsz, tgt_len, 2 * self.num_heads, self.head_dim)
        k = k.view(bsz, src_len, 2 * self.num_kv_heads, self.head_dim)
        v = v.view(bsz, src_len, self.num_kv_heads, 2 * self.head_dim)

        # 确保相对位置编码的cos和sin与q具有相同的数据类型
        cos, sin = rel_pos
        if cos.dtype != q.dtype:
            cos = cos.to(q.dtype)
            sin = sin.to(q.dtype)
        rel_pos_casted = (cos, sin)

        # 应用旋转位置编码
        q = apply_rotary_emb(q, *rel_pos_casted, interleaved=True)
        k = apply_rotary_emb(k, *rel_pos_casted, interleaved=True)

        # 计算注意力权重
        offset = src_len - tgt_len  # 计算偏移量
        q = q.transpose(1, 2)  # 转置
        k = repeat_kv(k.transpose(1, 2), self.n_rep)  # 重复k
        v = repeat_kv(v.transpose(1, 2), self.n_rep)  # 重复v
        q *= self.scaling  # 缩放查询
        attn_weights = torch.matmul(q, k.transpose(-1, -2))  # 计算注意力权重

        # 如果没有提供掩码，则创建一个上三角掩码
        if attn_mask is None:
            attn_mask = torch.triu(
                torch.zeros([tgt_len, src_len])
                .float()
                .fill_(float("-inf"))
                .type_as(attn_weights),
                1 + offset,
            )
        attn_weights = torch.nan_to_num(attn_weights)  # 将NaN值替换为零
        attn_weights += attn_mask  # 加入掩码
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).type_as(attn_weights)  # 归一化

        # 计算lambda系数
        lambda_1 = torch.exp(torch.sum(self.lambda_q1 * self.lambda_k1, dim=-1).float()).type_as(q)
        lambda_2 = torch.exp(torch.sum(self.lambda_q2 * self.lambda_k2, dim=-1).float()).type_as(q)
        lambda_full = lambda_1 - lambda_2 + self.lambda_init  # 总的lambda系数
        attn_weights = attn_weights.view(bsz, self.num_heads, 2, tgt_len, src_len)
        attn_weights = attn_weights[:, :, 0] - lambda_full * attn_weights[:, :, 1]  # 计算加权的注意力权重

        # 计算最终的注意力输出
        attn = torch.matmul(attn_weights, v)
        attn = self.subln(attn)  # 归一化
        attn = attn * (1 - self.lambda_init)  # 调整
        attn = attn.transpose(1, 2).reshape(bsz, tgt_len, self.num_heads * 2 * self.head_dim)

        # 投影输出
        attn = self.out_proj(attn)
        return attn
