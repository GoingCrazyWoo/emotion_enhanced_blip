import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from transformers import BlipProcessor, BlipForConditionalGeneration
from torchvision import transforms

# 导入 MultiheadDiffAttn 和 RotaryEmbedding
from .multihead_diffattn import MultiheadDiffAttn
from .rotary import RotaryEmbedding
from ..utils.emotion_utils import EMOTION_CATEGORIES

logger = logging.getLogger(__name__)

# 创建情感类别到索引的映射和索引到情感类别的映射
EMOTION_TO_INDEX = {emotion: idx for idx, emotion in enumerate(EMOTION_CATEGORIES)}
INDEX_TO_EMOTION = {idx: emotion for idx, emotion in enumerate(EMOTION_CATEGORIES)}

class EmotionEncoder(nn.Module):
    """情感编码器，将情感索引和置信度转换为嵌入表示"""
    
    def __init__(
        self,
        num_emotions: int = len(EMOTION_CATEGORIES),
        emotion_dim: int = 32,
        max_emotions: int = 3,
        hidden_dim: int = 768, # 假设 BLIP base 的 hidden_dim 为 768
        num_heads: int = 8,    # 假设基线 Transformer 有 16 个头
        depth: int = 0,        # 层索引
        dropout: float = 0.1
    ):
        """
        初始化情感编码器
        
        参数:
            num_emotions: 情感类别数量
            emotion_dim: 情感嵌入维度
            max_emotions: 最大情感数量
            hidden_dim: 隐藏层维度 (应与 BLIP 文本编码器匹配)
            num_heads: MultiheadDiffAttn 的头数
            depth: MultiheadDiffAttn 的层索引
            dropout: Dropout率
        """
        super().__init__()
        self.num_emotions = num_emotions
        self.emotion_dim = emotion_dim
        self.max_emotions = max_emotions
        
        # 检查 hidden_dim 是否能被 2 * num_heads 整除
        assert hidden_dim % (2 * num_heads) == 0, "hidden_dim must be divisible by 2 * num_heads"
        self.head_dim = hidden_dim // num_heads // 2
        
        # 情感嵌入层
        self.emotion_embeddings = nn.Embedding(
            num_embeddings=num_emotions+1,  # +1 为了处理填充值
            embedding_dim=emotion_dim,
            padding_idx=num_emotions  # 使用最后一个索引作为padding_idx
        )
        
        # 情感特征转换层
        self.emotion_transform = nn.Sequential(
            nn.Linear(emotion_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # 旋转位置编码
        self.rotary_emb = RotaryEmbedding(dim=self.head_dim, seq_len=max_emotions)
        
        # Multihead Differential Attention 层
        self.diff_attn = MultiheadDiffAttn(
            embed_dim=hidden_dim,
            depth=depth,
            num_heads=num_heads,
            num_kv_heads=None # 使用 MHA
        )
        
    def forward(
        self,
        emotion_indices: torch.LongTensor,
        confidence_values: torch.FloatTensor
    ) -> torch.FloatTensor:
        """
        前向传播
        
        参数:
            emotion_indices: 情感索引，形状为 [batch_size, max_emotions]
            confidence_values: 情感置信度，形状为 [batch_size, max_emotions]
            
        返回:
            emotion_features: 情感特征，形状为 [batch_size, hidden_dim]
        """
        batch_size = emotion_indices.size(0)
        
        # --- 1. 处理输入索引和置信度 ---
        # 将-1替换为padding_idx(num_emotions)
        emotion_indices = torch.where(emotion_indices < 0, torch.tensor(self.num_emotions, device=emotion_indices.device), emotion_indices)
        
        # 确保emotion_indices形状正确
        if emotion_indices.size(1) != self.max_emotions:
            # 如果不匹配，调整大小
            current_emotions = emotion_indices.size(1)
            if current_emotions < self.max_emotions:
                # 如果实际情感数量小于最大情感数量，填充
                padding = torch.full((batch_size, self.max_emotions - current_emotions), 
                                   self.num_emotions, 
                                   device=emotion_indices.device)
                emotion_indices = torch.cat([emotion_indices, padding], dim=1)
                
                # 同样处理置信度值
                conf_padding = torch.zeros(batch_size, self.max_emotions - current_emotions, 
                                         device=confidence_values.device)
                confidence_values = torch.cat([confidence_values, conf_padding], dim=1)
            else:
                # 如果实际情感数量大于最大情感数量，截断
                emotion_indices = emotion_indices[:, :self.max_emotions]
                confidence_values = confidence_values[:, :self.max_emotions]
        
        # 情感嵌入 [batch_size, max_emotions, emotion_dim]
        # --- 2. 获取加权情感嵌入 ---
        emotion_embeds = self.emotion_embeddings(emotion_indices)
        
        # 应用置信度权重 [batch_size, max_emotions, emotion_dim]
        confidence_values = confidence_values.unsqueeze(-1)  # [batch_size, max_emotions, 1]
        weighted_embeds = emotion_embeds * confidence_values
        
        # 转换每个情感嵌入 [batch_size, max_emotions, hidden_dim]
        # --- 3. 初始特征转换 ---
        transformed_embeds = self.emotion_transform(weighted_embeds)
        
        # --- 4. 应用 Multihead Differential Attention ---
        # 计算旋转位置编码
        freqs_cos, freqs_sin = self.rotary_emb(transformed_embeds)
        rel_pos = (freqs_cos, freqs_sin)
        
        # 创建注意力掩码 (忽略 padding token)
        # padding_idx 是 self.num_emotions
        # attention_mask 的形状应为 [batch_size, 1, tgt_len, src_len] 或 [tgt_len, src_len]
        # 这里我们希望非 padding 的 token 互相可见
        attn_mask = (emotion_indices == self.num_emotions).unsqueeze(1).unsqueeze(2) # [bs, 1, 1, max_emotions]
        attn_mask = attn_mask.expand(-1, -1, self.max_emotions, -1) # [bs, 1, max_emotions, max_emotions]
        # 在 MultiheadDiffAttn 内部会处理 causal mask，这里我们只需要处理 padding
        # MultiheadDiffAttn 期望的 mask 是 True 代表 mask 掉，所以需要反转
        # 但是 MultiheadDiffAttn 内部实现似乎是加性掩码，0 代表可见，-inf 代表 mask
        # 我们需要创建一个 mask，其中 padding 位置为 -inf，其他位置为 0
        additive_attn_mask = torch.zeros(batch_size, 1, self.max_emotions, self.max_emotions, device=transformed_embeds.device)
        additive_attn_mask.masked_fill_(attn_mask, float("-inf"))
        # MultiheadDiffAttn 内部会处理 causal mask，我们不需要在这里添加
        # 因此，我们只传递 padding mask
        
        # 应用 MultiheadDiffAttn
        attn_output = self.diff_attn(
            x=transformed_embeds,
            rel_pos=rel_pos,
            attn_mask=additive_attn_mask # 传递 padding mask
        ) # [batch_size, max_emotions, hidden_dim]
        
        # --- 5. 池化特征 ---
        # 使用平均池化聚合特征，忽略 padding token 的影响
        # 创建一个掩码，非 padding 位置为 1，padding 位置为 0
        non_padding_mask = (emotion_indices != self.num_emotions).float().unsqueeze(-1) # [bs, max_emotions, 1]
        # 计算加权和
        masked_attn_output = attn_output * non_padding_mask
        summed_features = masked_attn_output.sum(dim=1) # [bs, hidden_dim]
        # 计算非 padding token 的数量
        num_non_padding = non_padding_mask.sum(dim=1) # [bs, 1]
        # 避免除以零
        num_non_padding = torch.clamp(num_non_padding, min=1.0)
        # 计算平均值
        emotion_features = summed_features / num_non_padding # [bs, hidden_dim]
        
        return emotion_features

class EmotionEnhancedBlipForCaption(nn.Module):
    """情感增强的BLIP描述生成模型"""
    
    def __init__(
        self,
        blip_model_name: str = "Salesforce/blip-image-captioning-base",
        dropout: float = 0.1,
        max_emotions: int = 3,
        emotion_dim: int = 32,
        freeze_blip: bool = True,
        proxy: Optional[str] = None
    ):
        """
        初始化情感增强的BLIP描述生成模型
        
        参数:
            blip_model_name: BLIP模型名称
            dropout: Dropout率
            max_emotions: 最大情感数量
            emotion_dim: 情感嵌入维度
            freeze_blip: 是否冻结BLIP模型参数
            proxy: HTTP代理URL
        """
        super().__init__()
        
        # 保存参数
        self.freeze_blip = freeze_blip
        
        # 加载BLIP模型
        try:
            logger.info(f"加载BLIP模型: {blip_model_name}")
            proxies = {"http": proxy, "https": proxy} if proxy else None
            self.processor = BlipProcessor.from_pretrained(blip_model_name, proxies=proxies)
            self.blip_model = BlipForConditionalGeneration.from_pretrained(
                blip_model_name,
                proxies=proxies
            )
        except Exception as e:
            logger.error(f"加载BLIP模型失败: {e}")
            raise
            
        # 冻结BLIP模型参数
        if freeze_blip:
            logger.info("冻结BLIP模型参数")
            for param in self.blip_model.parameters():
                param.requires_grad = False
                
        # 情感编码器
        hidden_dim = self.blip_model.config.text_config.hidden_size
        logger.info(f"BLIP文本隐藏维度: {hidden_dim}")
        
        self.emotion_encoder = EmotionEncoder(
            num_emotions=len(EMOTION_CATEGORIES),
            emotion_dim=emotion_dim,
            max_emotions=max_emotions,
            hidden_dim=hidden_dim,
            dropout=dropout
        )
        
        # 情感特征适配层
        self.emotion_adapter = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # 情感控制门
        self.emotion_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, 1),
            nn.Sigmoid()
        )
        
        # 情感投影层 (用于直接处理logits的情况)
        vocab_size = self.blip_model.config.text_config.vocab_size
        self.emotion_projector = nn.Linear(hidden_dim, vocab_size)
        
    def get_emotion_representation(
        self,
        emotion_indices: torch.LongTensor,
        confidence_values: torch.FloatTensor
    ) -> torch.FloatTensor:
        """获取情感表示"""
        if emotion_indices.dim() == 1:
            # 如果输入是1维的，扩展为2维 [1, seq_len]
            emotion_indices = emotion_indices.unsqueeze(0)
            confidence_values = confidence_values.unsqueeze(0)
            
        # 获取情感特征 [batch_size, hidden_dim]
        emotion_features = self.emotion_encoder(emotion_indices, confidence_values)
        
        # 适配情感特征 [batch_size, hidden_dim]
        emotion_features = self.emotion_adapter(emotion_features)
        
        return emotion_features
    
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        emotion_indices: torch.LongTensor = None,
        confidence_values: torch.FloatTensor = None,
        input_ids: torch.LongTensor = None,
        attention_mask: torch.LongTensor = None,
        labels: torch.LongTensor = None,
        return_dict: bool = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        前向传播
        
        参数:
            pixel_values: 图像特征，形状为 [batch_size, 3, height, width]
            emotion_indices: 情感索引，形状为 [batch_size, max_emotions]
            confidence_values: 情感置信度，形状为 [batch_size, max_emotions]
            input_ids: 输入ID，形状为 [batch_size, seq_len]
            attention_mask: 注意力掩码，形状为 [batch_size, seq_len]
            labels: 标签，形状为 [batch_size, seq_len]
            return_dict: 是否返回字典
            
        返回:
            outputs: 模型输出
        """
        # 处理情感输入
        if emotion_indices is None or confidence_values is None:
            # 如果未提供情感，使用默认值（幽默和讽刺）
            batch_size = pixel_values.size(0)
            # 注意：使用num_emotions作为padding索引，而不是-1
            num_emotions = len(EMOTION_CATEGORIES)
            padding_idx = num_emotions
            emotion_indices = torch.tensor([[2, 3, padding_idx]]).repeat(batch_size, 1).to(pixel_values.device)
            confidence_values = torch.tensor([[0.8, 0.5, 0.0]]).repeat(batch_size, 1).to(pixel_values.device)
        
        # 获取情感特征
        emotion_features = self.get_emotion_representation(emotion_indices, confidence_values)
        
        # BLIP模型前向传播
        with torch.set_grad_enabled(not hasattr(self, 'freeze_blip') or not self.freeze_blip):
            blip_outputs = self.blip_model(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                return_dict=True,
                output_hidden_states=True,  # 请求输出隐藏状态
                **kwargs
            )
        
        # 将情感特征注入到模型中
        if labels is not None:
            # 训练模式：增强BLIP的输出表示
            # 确保loss是可导的，即使BLIP模型被冻结
            if hasattr(self, 'freeze_blip') and self.freeze_blip:
                # 当BLIP被冻结时，我们需要确保loss仍然可导
                loss = blip_outputs.loss.clone()  # 创建一个可导的副本
            else:
                loss = blip_outputs.loss
            
            # 获取BLIP最后层的隐藏状态
            # 根据BLIP模型输出的结构获取正确的隐藏状态
            if hasattr(blip_outputs, 'decoder_hidden_states') and blip_outputs.decoder_hidden_states is not None:
                # 如果有decoder_hidden_states属性，使用它
                last_hidden_state = blip_outputs.decoder_hidden_states[-1]
                
                # 将情感特征与BLIP特征融合
                # 这里简单使用加性融合，可以尝试更复杂的融合方式
                expanded_emotion = emotion_features.unsqueeze(1).expand(-1, last_hidden_state.size(1), -1)
                
                # 计算情感特征的权重
                gate = self.emotion_gate(
                    torch.cat([last_hidden_state, expanded_emotion], dim=-1)
                )
                
                # 加权融合
                enhanced_features = last_hidden_state + gate * expanded_emotion
                
                # 确保在冻结BLIP的情况下loss保持可导
                if self.freeze_blip:
                    # 计算情感增强因子来修改loss
                    emotion_factor = (gate.mean() + 1.0)  # 确保是正值，用于缩放loss
                    # 创建一个依赖于emotion_features的可导loss
                    loss = loss * emotion_factor
                
                # 返回增强的输出
                return {
                    "loss": loss,
                    "enhanced_features": enhanced_features,
                    "emotion_features": emotion_features,
                    "logits": blip_outputs.logits,
                    "gateway_value": gate.mean().item()  # 用于监控
                }
            elif hasattr(blip_outputs, 'hidden_states') and blip_outputs.hidden_states is not None:
                # 否则尝试使用hidden_states属性
                last_hidden_state = blip_outputs.hidden_states[-1]
                
                # 将情感特征与BLIP特征融合
                # 这里简单使用加性融合，可以尝试更复杂的融合方式
                expanded_emotion = emotion_features.unsqueeze(1).expand(-1, last_hidden_state.size(1), -1)
                
                # 计算情感特征的权重
                gate = self.emotion_gate(
                    torch.cat([last_hidden_state, expanded_emotion], dim=-1)
                )
                
                # 加权融合
                enhanced_features = last_hidden_state + gate * expanded_emotion
                
                # 确保在冻结BLIP的情况下loss保持可导
                if self.freeze_blip:
                    # 计算情感增强因子来修改loss
                    emotion_factor = (gate.mean() + 1.0)  # 确保是正值，用于缩放loss
                    # 创建一个依赖于emotion_features的可导loss
                    loss = loss * emotion_factor
                
                # 返回增强的输出
                return {
                    "loss": loss,
                    "enhanced_features": enhanced_features,
                    "emotion_features": emotion_features,
                    "logits": blip_outputs.logits,
                    "gateway_value": gate.mean().item()  # 用于监控
                }
            else:
                # 如果没有隐藏状态，则使用logits进行处理
                # 创建与logits形状匹配的情感向量
                batch_size, seq_len, vocab_size = blip_outputs.logits.size()
                emotion_logits = self.emotion_adapter(emotion_features).unsqueeze(1)
                emotion_logits = self.emotion_projector(emotion_logits).expand(batch_size, seq_len, vocab_size)
                
                # 使用门控机制融合
                gate_value = self.emotion_gate(
                    torch.cat([blip_outputs.logits.mean(dim=-1, keepdim=True), 
                              emotion_logits.mean(dim=-1, keepdim=True)], dim=-1)
                )
                
                # 融合logits
                fused_logits = blip_outputs.logits * (1 - gate_value) + emotion_logits * gate_value
                
                # 确保在冻结BLIP的情况下loss保持可导
                if self.freeze_blip:
                    # 计算情感增强因子来修改loss
                    emotion_factor = (gate_value.mean() + 1.0)  # 确保是正值，用于缩放loss
                    # 创建一个依赖于emotion_features的可导loss
                    loss = loss * emotion_factor
                
                return {
                    "loss": loss, 
                    "logits": fused_logits,
                    "gateway_value": gate_value.mean().item()  # 用于监控
                }
        
        else:
            # 推理模式：直接返回BLIP输出
            return blip_outputs
    
    def generate(
        self,
        pixel_values: torch.FloatTensor,
        emotion_indices: torch.LongTensor = None,
        confidence_values: torch.FloatTensor = None,
        input_ids: torch.LongTensor = None,
        attention_mask: torch.LongTensor = None,
        **generate_kwargs
    ) -> torch.LongTensor:
        """
        生成描述文本 (移除了情感注入钩子以保持与训练一致)
        
        参数:
            pixel_values: 图像特征，形状为 [batch_size, 3, height, width]
            emotion_indices: (可选，不再使用) 情感索引
            confidence_values: (可选，不再使用) 情感置信度
            input_ids: (可选) 输入ID，用于条件生成
            attention_mask: (可选) 注意力掩码
            generate_kwargs: 传递给 `transformers.generation_utils.GenerationMixin.generate` 的参数
            
        返回:
            captions: 生成的描述文本 (token IDs)
        """
        # 情感特征不再在此方法中注入
        
        # 设置生成参数 (如果未提供)
        if "max_length" not in generate_kwargs:
            generate_kwargs["max_length"] = 100
        if "num_beams" not in generate_kwargs:
            generate_kwargs["num_beams"] = 5
        if "min_length" not in generate_kwargs:
            generate_kwargs["min_length"] = 10
        if "do_sample" not in generate_kwargs:
            generate_kwargs["do_sample"] = True # 默认进行采样以增加多样性
        if "temperature" not in generate_kwargs:
            generate_kwargs["temperature"] = 0.7
        if "top_p" not in generate_kwargs:
            generate_kwargs["top_p"] = 0.9
            
        # 直接调用原始 BLIP 模型的 generate 方法
        # 不再使用钩子注入情感特征
        outputs = self.blip_model.generate(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            **generate_kwargs
        )
        
        return outputs
    
    def generate_caption(
        self,
        image=None,
        pixel_values=None,
        emotion_indices=None,
        confidence_values=None,
        max_length=50,
        num_beams=5,
        **generate_kwargs
    ):
        """从图像和情感信息生成描述"""
        if image is not None and pixel_values is None:
            # 处理输入图像
            inputs = self.processor(images=image, return_tensors="pt")
            pixel_values = inputs.pixel_values.to(self.device)
        
        # 生成文本
        generated_ids = self.generate(
            pixel_values=pixel_values,
            emotion_indices=emotion_indices,
            confidence_values=confidence_values,
            max_length=max_length,
            num_beams=num_beams,
            **generate_kwargs
        )
        
        # 解码生成的文本
        captions = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
        
        return captions[0] if len(captions) == 1 else captions
    
    @property
    def device(self):
        """获取模型所在设备"""
        return next(self.parameters()).device 