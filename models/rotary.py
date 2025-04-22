# Copyright (c) 2023, Tri Dao.

from typing import Optional, Union

import torch


def apply_rotary(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    seqlen_offsets: Union[int, torch.Tensor] = 0,
    cu_seqlens: Optional[torch.Tensor] = None,
    max_seqlen: Optional[int] = None,
    interleaved=False,
    inplace=False,
    conjugate=False,
) -> torch.Tensor:
    """
    Arguments:
        x: (batch, seqlen, nheads, headdim) if cu_seqlens is None
            else (total_seqlen, nheads, headdim).
        cos: (seqlen_ro, rotary_dim / 2)
        sin: (seqlen_ro, rotary_dim / 2)
        seqlen_offsets: integer or integer tensor of size (batch,)
        cu_seqlens: (batch + 1,) or None
        max_seqlen: int
    Returns:
        y: (batch, seqlen, nheads, headdim)
    """
    is_varlen = cu_seqlens is not None
    if not is_varlen:
        batch, seqlen, nheads, headdim = x.shape
    else:
        assert max_seqlen is not None, "If cu_seqlens is passed in, then max_seqlen must be passed"
        total_seqlen, nheads, headdim = x.shape
        batch_p_1 = cu_seqlens.shape[0]
        batch = batch_p_1 - 1
        seqlen = max_seqlen
    seqlen_ro, rotary_dim = cos.shape
    assert sin.shape == cos.shape
    rotary_dim *= 2
    assert rotary_dim <= headdim, "rotary_dim must be <= headdim"
    assert seqlen_ro >= seqlen, "seqlen_ro must be >= seqlen"

    assert (
        cos.dtype == sin.dtype
    ), f"cos and sin must have the same dtype, got {cos.dtype} and {sin.dtype}"
    assert (
        x.dtype == cos.dtype
    ), f"Input and cos/sin must have the same dtype, got {x.dtype} and {cos.dtype}"

    cos, sin = cos.contiguous(), sin.contiguous()
    if isinstance(seqlen_offsets, torch.Tensor):
        assert seqlen_offsets.shape == (batch,)
        assert seqlen_offsets.dtype in [torch.int32, torch.int64]
        seqlen_offsets = seqlen_offsets.contiguous()
    else:
        assert seqlen_offsets + seqlen <= seqlen_ro

    # 创建输出张量
    output = torch.empty_like(x) if not inplace else x
    if rotary_dim < headdim and not inplace:
        output[..., rotary_dim:].copy_(x[..., rotary_dim:])
    
    # 基于 cu_seqlens 处理变长序列的情况
    if is_varlen:
        # 变长序列的处理逻辑
        # 因为这种情况比较复杂，我们这里提供一个基本实现
        if not interleaved:
            # 处理未交错的情况（前半部分和后半部分分开旋转）
            for i in range(batch):
                start_idx = cu_seqlens[i].item()
                end_idx = cu_seqlens[i + 1].item()
                seq_len_i = end_idx - start_idx
                offset_i = seqlen_offsets[i].item() if isinstance(seqlen_offsets, torch.Tensor) else seqlen_offsets
                
                # 获取当前批次的输入
                xi = x[start_idx:end_idx]  # [seq_len_i, nheads, headdim]
                
                # 应用旋转
                for j in range(seq_len_i):
                    j_offset = j + offset_i
                    if j_offset >= seqlen_ro:
                        continue
                    
                    # 旋转前半部分
                    x_j = xi[j, :, :rotary_dim//2]
                    y_j = xi[j, :, rotary_dim//2:rotary_dim]
                    
                    cos_j = cos[j_offset].unsqueeze(0)  # [1, rotary_dim//2]
                    sin_j = sin[j_offset].unsqueeze(0)  # [1, rotary_dim//2]
                    
                    if conjugate:
                        sin_j = -sin_j
                    
                    # 应用旋转
                    output[start_idx + j, :, :rotary_dim//2] = x_j * cos_j - y_j * sin_j
                    output[start_idx + j, :, rotary_dim//2:rotary_dim] = x_j * sin_j + y_j * cos_j
        else:
            # 交错旋转的情况（GPT-J风格）
            for i in range(batch):
                start_idx = cu_seqlens[i].item()
                end_idx = cu_seqlens[i + 1].item()
                seq_len_i = end_idx - start_idx
                offset_i = seqlen_offsets[i].item() if isinstance(seqlen_offsets, torch.Tensor) else seqlen_offsets
                
                # 获取当前批次的输入
                xi = x[start_idx:end_idx]  # [seq_len_i, nheads, headdim]
                
                # 应用旋转
                for j in range(seq_len_i):
                    j_offset = j + offset_i
                    if j_offset >= seqlen_ro:
                        continue
                    
                    # 旋转交错维度
                    # 交错模式下，我们对偶数和奇数位置分别应用旋转
                    for d in range(0, rotary_dim, 2):
                        if d + 1 >= headdim:
                            break
                        
                        # 获取当前位置的值
                        x0 = xi[j, :, d]  # [nheads]
                        x1 = xi[j, :, d+1]  # [nheads]
                        
                        # 获取对应的cos、sin值
                        cos_idx = j_offset
                        cos_d = cos[cos_idx, d//2]  # 标量
                        sin_d = sin[cos_idx, d//2]  # 标量
                        
                        if conjugate:
                            sin_d = -sin_d
                        
                        # 应用旋转
                        output[start_idx + j, :, d] = x0 * cos_d - x1 * sin_d
                        output[start_idx + j, :, d+1] = x0 * sin_d + x1 * cos_d
    else:
        # 处理固定长度序列的情况
        # 偏移量处理
        if isinstance(seqlen_offsets, torch.Tensor):
            # 每个批次有不同的偏移量
            for i in range(batch):
                offset_i = seqlen_offsets[i].item()
                
                # 获取有效的序列长度（考虑偏移量）
                valid_seqlen = min(seqlen, seqlen_ro - offset_i)
                
                if not interleaved:
                    # 非交错模式（GPT-NeoX风格）
                    # 前半部分
                    x1 = x[i, :valid_seqlen, :, :rotary_dim//2]  # [valid_seqlen, nheads, rotary_dim//2]
                    # 后半部分
                    x2 = x[i, :valid_seqlen, :, rotary_dim//2:rotary_dim]  # [valid_seqlen, nheads, rotary_dim//2]
                    
                    # 提取对应的cos和sin值
                    cos_i = cos[offset_i:offset_i+valid_seqlen]  # [valid_seqlen, rotary_dim//2]
                    sin_i = sin[offset_i:offset_i+valid_seqlen]  # [valid_seqlen, rotary_dim//2]
                    
                    if conjugate:
                        sin_i = -sin_i
                        
                    # 扩展维度以便广播
                    cos_i = cos_i.unsqueeze(1)  # [valid_seqlen, 1, rotary_dim//2]
                    sin_i = sin_i.unsqueeze(1)  # [valid_seqlen, 1, rotary_dim//2]
                    
                    # 应用旋转变换
                    output[i, :valid_seqlen, :, :rotary_dim//2] = x1 * cos_i - x2 * sin_i
                    output[i, :valid_seqlen, :, rotary_dim//2:rotary_dim] = x1 * sin_i + x2 * cos_i
                else:
                    # 交错模式（GPT-J风格）
                    for j in range(valid_seqlen):
                        for d in range(0, rotary_dim, 2):
                            if d + 1 >= headdim:
                                break
                            
                            # 交错旋转对偶数和相邻的奇数位置一起旋转
                            x0 = x[i, j, :, d]  # [nheads]
                            x1 = x[i, j, :, d+1]  # [nheads]
                            
                            cos_d = cos[j + offset_i, d//2]  # 标量
                            sin_d = sin[j + offset_i, d//2]  # 标量
                            
                            if conjugate:
                                sin_d = -sin_d
                            
                            # 应用旋转
                            output[i, j, :, d] = x0 * cos_d - x1 * sin_d
                            output[i, j, :, d+1] = x0 * sin_d + x1 * cos_d
        else:
            # 所有批次使用相同的偏移量
            offset = seqlen_offsets
            valid_seqlen = min(seqlen, seqlen_ro - offset)
            
            if not interleaved:
                # 非交错模式（GPT-NeoX风格）
                # 前半部分
                x1 = x[:, :valid_seqlen, :, :rotary_dim//2]  # [batch, valid_seqlen, nheads, rotary_dim//2]
                # 后半部分
                x2 = x[:, :valid_seqlen, :, rotary_dim//2:rotary_dim]  # [batch, valid_seqlen, nheads, rotary_dim//2]
                
                # 提取对应的cos和sin值
                cos_seq = cos[offset:offset+valid_seqlen]  # [valid_seqlen, rotary_dim//2]
                sin_seq = sin[offset:offset+valid_seqlen]  # [valid_seqlen, rotary_dim//2]
                
                if conjugate:
                    sin_seq = -sin_seq
                
                # 扩展维度以便广播
                cos_seq = cos_seq.unsqueeze(0).unsqueeze(2)  # [1, valid_seqlen, 1, rotary_dim//2]
                sin_seq = sin_seq.unsqueeze(0).unsqueeze(2)  # [1, valid_seqlen, 1, rotary_dim//2]
                
                # 应用旋转变换
                output[:, :valid_seqlen, :, :rotary_dim//2] = x1 * cos_seq - x2 * sin_seq
                output[:, :valid_seqlen, :, rotary_dim//2:rotary_dim] = x1 * sin_seq + x2 * cos_seq
            else:
                # 批量处理交错模式
                for j in range(valid_seqlen):
                    for d in range(0, rotary_dim, 2):
                        if d + 1 >= headdim:
                            break
                        
                        # 交错旋转对偶数和相邻的奇数位置一起旋转
                        x0 = x[:, j, :, d]  # [batch, nheads]
                        x1 = x[:, j, :, d+1]  # [batch, nheads]
                        
                        cos_d = cos[j + offset, d//2]  # 标量
                        sin_d = sin[j + offset, d//2]  # 标量
                        
                        if conjugate:
                            sin_d = -sin_d
                        
                        # 应用旋转
                        output[:, j, :, d] = x0 * cos_d - x1 * sin_d
                        output[:, j, :, d+1] = x0 * sin_d + x1 * cos_d
    
    return output


class ApplyRotaryEmb(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x,
        cos,
        sin,
        interleaved=False,
        inplace=False,
        seqlen_offsets: Union[int, torch.Tensor] = 0,
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seqlen: Optional[int] = None,
    ):
        out = apply_rotary(
            x,
            cos,
            sin,
            seqlen_offsets=seqlen_offsets,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            interleaved=interleaved,
            inplace=inplace,
        )
        ctx.save_for_backward(cos, sin)
        ctx.interleaved = interleaved
        ctx.inplace = inplace
        ctx.seqlen_offsets = seqlen_offsets
        ctx.cu_seqlens = cu_seqlens
        ctx.max_seqlen = max_seqlen
        return out if not inplace else x

    @staticmethod
    def backward(
        ctx,
        dout,
    ):
        cos, sin = ctx.saved_tensors
        # 反向传播使用共轭操作
        dx = apply_rotary(
            dout,
            cos,
            sin,
            interleaved=ctx.interleaved,
            conjugate=True,
            seqlen_offsets=ctx.seqlen_offsets,
            cu_seqlens=ctx.cu_seqlens,
            max_seqlen=ctx.max_seqlen,
        )
        return dx, None, None, None, None, None, None, None


def apply_rotary_emb(
    x,
    cos,
    sin,
    interleaved=False,
    inplace=False,
    seqlen_offsets: Union[int, torch.Tensor] = 0,
    cu_seqlens: Optional[torch.Tensor] = None,
    max_seqlen: Optional[int] = None,
):
    """
    Arguments:
        x: (batch_size, seqlen, nheads, headdim) if cu_seqlens is None
            else (total_seqlen, nheads, headdim)
        cos, sin: (seqlen_rotary, rotary_dim / 2)
        interleaved: if True, rotate pairs of even and odd dimensions (GPT-J style) instead
            of 1st half and 2nd half (GPT-NeoX style).
        inplace: if True, apply rotary embedding in-place.
        seqlen_offsets: (batch_size,) or int. Each sequence in x is shifted by this amount.
            Most commonly used in inference when we have KV cache.
        cu_seqlens: (batch + 1,) or None
        max_seqlen: int
    Return:
        out: (batch_size, seqlen, nheads, headdim) if cu_seqlens is None
            else (total_seqlen, nheads, headdim)
    rotary_dim must be <= headdim
    Apply rotary embedding to the first rotary_dim of x.
    """
    return ApplyRotaryEmb.apply(
        x, cos, sin, interleaved, inplace, seqlen_offsets, cu_seqlens, max_seqlen
    )


# 添加一个RotaryEmbedding类，作为模块使用，用于计算和应用旋转嵌入
class RotaryEmbedding(torch.nn.Module):
    """
    旋转位置编码模块，基于RoPE (Rotary Position Embedding)
    
    该类适配torch.autograd.Function为标准的nn.Module，便于在模型中使用
    """
    
    def __init__(self, dim: int, seq_len: int, base: int = 10000):
        """
        初始化旋转位置编码模块
        
        参数:
            dim: 隐藏维度，必须是偶数
            seq_len: 序列长度
            base: 位置编码的基数，默认为10000
        """
        super().__init__()
        
        # 确保dim是偶数
        assert dim % 2 == 0, f"维度 {dim} 必须是偶数"
        
        # 旋转位置编码需要的cos和sin表
        # cos_cached和sin_cached形状: [seq_len, dim/2]
        theta = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        seq_idx = torch.arange(seq_len).float()
        # 计算旋转角度
        idx_theta = torch.einsum('i,j->ij', seq_idx, theta)
        
        # 缓存cos和sin值，便于重复使用
        self.cos_cached = torch.cos(idx_theta)
        self.sin_cached = torch.sin(idx_theta)
        
        # 配置选项
        self.dim = dim
        self.seq_len = seq_len
        self.interleaved = True  # 使用GPT-J风格的交错旋转
        
    def forward(self, x: torch.Tensor):
        """
        前向传播，计算并返回旋转位置编码
        
        参数:
            x: 输入张量，通常形状为 [batch_size, seq_len, hidden_dim]
        
        返回:
            cos: 余弦值张量，形状为 [seq_len, dim/2]
            sin: 正弦值张量，形状为 [seq_len, dim/2]
        """
        # 确保cos和sin在正确的设备上且数据类型与输入匹配
        device = x.device
        dtype = x.dtype
        
        # 强制转换 cos_cached 和 sin_cached 到与输入相同的设备和数据类型
        self.cos_cached = self.cos_cached.to(device=device, dtype=dtype)
        self.sin_cached = self.sin_cached.to(device=device, dtype=dtype)
            
        # 返回正余弦值供注意力层使用
        return self.cos_cached, self.sin_cached
    
    def apply_rotary(self, q: torch.Tensor, k: torch.Tensor):
        """
        应用旋转位置编码到查询和键
        
        参数:
            q: 查询张量，形状为 [batch_size, seq_len, num_heads, head_dim]
            k: 键张量，形状为 [batch_size, seq_len, num_heads, head_dim]
        
        返回:
            q_rot: 应用旋转后的查询
            k_rot: 应用旋转后的键
        """
        # 获取或计算cos和sin
        cos, sin = self.forward(q)
        
        # 确保cos和sin的数据类型与q和k匹配
        # 这样做可以避免在fp16训练时出现数据类型不匹配的问题
        if cos.dtype != q.dtype:
            cos = cos.to(dtype=q.dtype)
            sin = sin.to(dtype=q.dtype)
        
        # 应用旋转位置编码
        q_rot = apply_rotary_emb(
            q, cos, sin, interleaved=self.interleaved
        )
        
        k_rot = apply_rotary_emb(
            k, cos, sin, interleaved=self.interleaved
        )
        
        return q_rot, k_rot