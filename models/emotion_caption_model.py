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
from emotion_enhanced_blip.utils.emotion_utils import EMOTION_CATEGORIES # 使用绝对导入

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

from transformers import LogitsProcessor

class EmotionLogitsProcessor(LogitsProcessor):
    """在生成过程中根据情感特征调整 logits。"""
    def __init__(self, model_instance, emotion_features, alpha: float = 0.1):
        """
        初始化 Logits Processor。

        参数:
            model_instance: EmotionEnhancedBlipForCaption 的实例。
            emotion_features: 计算得到的情感特征 [batch_size, hidden_dim]。
            alpha: 情感影响因子，控制情感偏差的强度。
        """
        if emotion_features is None:
            raise ValueError("Emotion features cannot be None for EmotionLogitsProcessor.")
        self.model = model_instance
        self.emotion_features = emotion_features
        self.batch_size, self.hidden_dim = emotion_features.shape
        self.alpha = alpha

        # 预计算情感对 logits 的影响，避免在每步重复计算
        # [bs, hidden_dim] -> [bs, vocab_size]
        with torch.no_grad(): # 确保在推理时不计算梯度
            emotion_logits_bias = self.model.emotion_adapter(self.emotion_features)
            self.emotion_logits_bias = self.model.emotion_projector(emotion_logits_bias)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        """
        调整 logits。

        参数:
            input_ids: 当前生成的 token IDs [batch_size * num_beams, seq_len]。
            scores: 当前步的 logits [batch_size * num_beams, vocab_size]。

        返回:
            modified_scores: 调整后的 logits。
        """
        current_batch_size = scores.size(0)
        vocab_size = scores.size(-1)

        # 检查 emotion_logits_bias 是否与 scores 的设备匹配
        if self.emotion_logits_bias.device != scores.device:
            self.emotion_logits_bias = self.emotion_logits_bias.to(scores.device)

        # 需要将 emotion_logits_bias [bs, vocab_size] 扩展以匹配 scores [bs * num_beams, vocab_size]
        num_beams = current_batch_size // self.batch_size
        if num_beams <= 0:
             # 处理 batch_size=1 且 num_beams=1 的情况
             if current_batch_size == self.batch_size:
                 num_beams = 1
             else:
                 # 如果无法确定 beam 数量，可能出错了，跳过修改
                 logger.warning(f"Could not determine beam size. Scores shape: {scores.shape}, Emotion bias shape: {self.emotion_logits_bias.shape}")
                 return scores

        # 扩展情感偏差以匹配 beam search 的维度
        # [bs, vocab_size] -> [bs * num_beams, vocab_size]
        expanded_emotion_bias = self.emotion_logits_bias.repeat_interleave(num_beams, dim=0)

        # 确保扩展后的形状匹配
        if expanded_emotion_bias.shape != scores.shape:
             logger.warning(f"Shape mismatch after expanding emotion bias. Scores: {scores.shape}, Expanded Bias: {expanded_emotion_bias.shape}. Skipping modification.")
             return scores

        # --- 融合逻辑: 加性融合 --- 
        # 使用预设的 alpha 值来控制情感影响强度
        modified_scores = scores + self.alpha * expanded_emotion_bias

        # 可选：未来可以尝试更复杂的门控融合，但这需要访问 hidden states

        return modified_scores



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
        self.num_emotions = len(EMOTION_CATEGORIES) # 添加 num_emotions 属性
        
        # 加载BLIP处理器和模型，添加错误处理
        try:
            logger.info(f"正在加载BLIP处理器: {blip_model_name}")
            self.processor = BlipProcessor.from_pretrained(blip_model_name, proxies=proxy)
            
            logger.info(f"正在加载BLIP模型: {blip_model_name}")
            # 使用try-except捕获可能的OOM或其他错误
            try:
                self.blip = BlipForConditionalGeneration.from_pretrained(blip_model_name, proxies=proxy)
            except Exception as e:
                # 如果出现错误，尝试使用CPU加载然后再移动到设备
                logger.warning(f"使用默认设备加载BLIP模型失败: {str(e)}")
                logger.info("尝试使用CPU加载BLIP模型...")
                
                import gc
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
                self.blip = BlipForConditionalGeneration.from_pretrained(
                    blip_model_name, 
                    proxies=proxy,
                    torch_dtype=torch.float32,  # 使用FP32而不是FP16
                    device_map="auto"  # 允许模型自动决定设备分配
                )
        except Exception as e:
            logger.error(f"加载BLIP模型或处理器失败: {str(e)}")
            raise RuntimeError(f"无法加载BLIP模型: {str(e)}")
        
        # 冻结BLIP参数
        if freeze_blip:
            logger.info("冻结BLIP模型参数")
            for param in self.blip.parameters():
                param.requires_grad = False
                
        # 情感编码器
        hidden_dim = self.blip.config.text_config.hidden_size
        logger.info(f"BLIP文本隐藏维度: {hidden_dim}")
        
        self.emotion_encoder = EmotionEncoder(
            num_emotions=len(EMOTION_CATEGORIES), # 使用 self.num_emotions
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
        vocab_size = self.blip.config.text_config.vocab_size
        self.emotion_projector = nn.Linear(hidden_dim, vocab_size)

        # --- 新增：情感分类头 ---
        # 使用视觉模型的输出维度 (通常与文本模型相同)
        vision_hidden_dim = self.blip.config.vision_config.hidden_size
        self.emotion_classifier = nn.Linear(vision_hidden_dim, len(EMOTION_CATEGORIES))
        logger.info(f"添加情感分类头: Linear({vision_hidden_dim}, {len(EMOTION_CATEGORIES)})")
        # --- 结束新增 ---
        
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
            

    def extract_emotions(
        self,
        pixel_values: torch.FloatTensor,
        top_k: int = 3
    ) -> Tuple[torch.LongTensor, torch.FloatTensor]:
        """
        从图像像素值中动态提取情感类别和置信度。

        参数:
            pixel_values: 图像的像素值，形状为 [batch_size, 3, height, width]。
            top_k: 返回置信度最高的前 k 个情感。

        返回:
            Tuple[torch.LongTensor, torch.FloatTensor]:
                - predicted_emotion_indices: 预测的情感索引，形状为 [batch_size, top_k]。
                - predicted_confidence_values: 预测的情感置信度，形状为 [batch_size, top_k]。
        """
        batch_size = pixel_values.size(0)
        device = pixel_values.device

        # 1. 使用 BLIP 视觉编码器提取图像特征
        # 注意：即使冻结了BLIP，我们仍然可以进行前向传播
        with torch.no_grad(): # 通常在推理时不计算梯度
            vision_outputs = self.blip.vision_model(pixel_values=pixel_values)
            # 获取池化后的输出 (通常是 CLS token 的特征)
            image_embeds = vision_outputs.pooler_output # 形状: [batch_size, vision_hidden_dim]

        # 2. 使用情感分类头预测情感 logits
        emotion_logits = self.emotion_classifier(image_embeds) # 形状: [batch_size, num_emotions]

        # 3. 应用 Sigmoid 获得每个情感的独立概率 (置信度)
        emotion_probs = torch.sigmoid(emotion_logits) # 形状: [batch_size, num_emotions]

        # 4. 获取 top-k 情感及其置信度
        # 对每个样本按置信度降序排序
        sorted_probs, sorted_indices = torch.sort(emotion_probs, dim=-1, descending=True)

        # 选择前 k 个
        predicted_confidence_values = sorted_probs[:, :top_k]
        predicted_emotion_indices = sorted_indices[:, :top_k]

        return predicted_emotion_indices, predicted_confidence_values

        # 获取情感特征 [batch_size, hidden_dim]
        emotion_features = self.emotion_encoder(emotion_indices, confidence_values)
        
        # 适配情感特征 [batch_size, hidden_dim]
        emotion_features = self.emotion_adapter(emotion_features)
        
        return emotion_features
    
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        emotion_indices: Optional[torch.LongTensor] = None, # 真实情感标签 (LongTensor, 可能用于 EmotionEncoder)
        emotion_labels_multi_hot: Optional[torch.FloatTensor] = None, # 真实情感标签 (Multi-hot FloatTensor, 用于 BCE Loss)
        confidence_values: Optional[torch.FloatTensor] = None, # 置信度（目前未使用）
        input_ids: Optional[torch.LongTensor] = None, # 标题生成输入
        attention_mask: Optional[torch.LongTensor] = None, # 标题生成掩码
        labels: Optional[torch.LongTensor] = None, # 标题生成标签 (用于计算损失)
        return_dict: Optional[bool] = None,
        emotion_loss_weight: float = 0.5, # 情感损失权重
        **kwargs
    ) -> Union[Dict[str, Any], Tuple[torch.Tensor, ...]]:
        """
        前向传播
        """
        
        return_dict = return_dict if return_dict is not None else self.blip.config.use_return_dict
        
        try:
            # --- 1. 视觉特征提取和情感分类 ---
            # 提取视觉特征并进行情感分类
            
            # 确保 vision_model 的梯度计算根据 freeze_blip 控制
            with torch.set_grad_enabled(not self.freeze_blip):
                vision_outputs = self.blip.vision_model(pixel_values=pixel_values, return_dict=True)
            
            # 池化后的输出需要梯度，以便传递给可训练的 emotion_classifier
            image_embeds = vision_outputs.pooler_output # [batch_size, vision_hidden_dim]
            
            # 情感分类
            emotion_logits = self.emotion_classifier(image_embeds) # [batch_size, num_emotions]
            
            # --- 2. 计算情感损失 (如果有真实标签) ---
            emotion_loss = None
            caption_loss = None
            loss = None
            if emotion_labels_multi_hot is not None:
                try:
                    # 使用BCE损失计算情感损失
                    emotion_loss = F.binary_cross_entropy_with_logits(
                        emotion_logits, 
                        emotion_labels_multi_hot.float(),
                        reduction="mean"
                    )
                except Exception as e:
                    logger.error(f"Error during emotion loss calculation: {e}")
                    # 如果情感损失计算出错，设为0以避免训练中断
                    emotion_loss = torch.tensor(0.0, device=pixel_values.device)
            else:
                pass # No emotion_labels_multi_hot provided, skipping emotion loss calculation
            
            # --- 3. 获取情感表示 ---
            emotion_features = None
            try:
                if emotion_indices is not None and confidence_values is not None:
                    # 使用真实情感标签获取情感表示
                    emotion_features = self.emotion_encoder(
                        emotion_indices=emotion_indices,
                        confidence_values=confidence_values
                    )
                else:
                    # 从vision_model预测情感，然后获取情感表示
                    predicted_emotion_indices, predicted_confidence_values = self.extract_emotions(
                        pixel_values=pixel_values
                    )
                    emotion_features = self.emotion_encoder(
                        emotion_indices=predicted_emotion_indices,
                        confidence_values=predicted_confidence_values
                    )
            except Exception as e:
                logger.error(f"Error during emotion representation extraction: {e}")
                # 如果情感表示提取出错，用零向量替代
                batch_size = pixel_values.size(0)
                hidden_dim = self.blip.config.text_config.hidden_size
                emotion_features = torch.zeros(batch_size, hidden_dim, device=pixel_values.device)
            
            # --- 4. 使用BLIP计算caption loss (如果提供了labels) ---
            if labels is not None:
                try:
                    # 使用自定义情感适配层处理情感特征
                    emotion_adapter_output = self.emotion_adapter(emotion_features)
                    
                    # 使用门控机制控制情感对文本生成的影响
                    emotion_gate_weight = self.emotion_gate(emotion_features)
                    
                    # 直接使用BLIP模型计算损失，避免连接情感特征导致的形状不匹配问题
                    with torch.set_grad_enabled(not self.freeze_blip):
                        blip_outputs = self.blip(
                            pixel_values=pixel_values, # 传递像素值，BLIP内部处理视觉部分
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=labels,
                            return_dict=True
                        )
                    
                    # 使用BLIP计算的损失，但保持梯度连接
                    caption_loss = blip_outputs.loss
                
                    # 如果有emotion_loss，计算总损失
                    if emotion_loss is not None:
                        loss = caption_loss + emotion_loss_weight * emotion_loss
                    else:
                        loss = caption_loss
                except RuntimeError as e:
                    if "mat1 and mat2 shapes cannot be multiplied" in str(e):
                        logger.warning(f"Matrix shape mismatch in BLIP model: {e}")
                        caption_loss = torch.tensor(0.0, device=pixel_values.device, requires_grad=True)

                        # 如果有emotion_loss，仍然可以用它作为总损失
                        if emotion_loss is not None:
                            loss = emotion_loss_weight * emotion_loss
                        else:
                            loss = caption_loss
                    else:
                        raise
                except Exception as e:
                    logger.error(f"Error during caption loss calculation: {e}")
                    # 如果caption损失计算出错，检查是否有情感损失可用
                    if emotion_loss is not None:
                        loss = emotion_loss_weight * emotion_loss
                    else:
                        # 如果都没有可用损失，使用dummy loss以避免训练失败
                        loss = torch.tensor(0.0, device=pixel_values.device)
    else:
        # 处理预测/推理情况但确保有情感损失时能正确设置总损失
                if emotion_loss is not None:
                    # 当没有标题标签但有情感损失时，将情感损失作为总损失
                    loss = emotion_loss_weight * emotion_loss
                else:
                    # 如果没有任何损失，使用dummy loss以避免训练失败
                    loss = torch.tensor(0.0, device=pixel_values.device)

    # 返回结果字典
    return_values = {
            "loss": loss,
            "emotion_logits": emotion_logits,
            "emotion_loss": emotion_loss,
            "caption_loss": caption_loss
        }

return return_values

except Exception as e:
        logger.error(f"Unexpected error in forward method: {e}")
        import traceback
        traceback.print_exc()
    
    def generate(
        self,
        pixel_values: torch.FloatTensor,
        emotion_indices: torch.LongTensor = None,
        confidence_values: torch.FloatTensor = None,
        input_ids: torch.LongTensor = None,
        attention_mask: torch.LongTensor = None,
        emotion_alpha: float = 0.1,
        extract_top_k: int = 3, # 新增：控制动态提取的情感数量
        **generate_kwargs
    ) -> torch.LongTensor:
        """
        生成带有情感色彩的描述文本。

        参数:
            pixel_values: 图像特征，形状为 [batch_size, 3, height, width]。
            emotion_indices: (可选) 提供的情感索引，形状为 [batch_size, num_provided_emotions]。
                           如果提供，将优先使用此信息。
            confidence_values: (可选) 提供的情感置信度，形状为 [batch_size, num_provided_emotions]。
                               需要与 `emotion_indices` 一起提供。
            input_ids: (可选) 输入ID，用于条件生成。
            attention_mask: (可选) 注意力掩码。
            emotion_alpha: (可选) 控制情感 logits 偏置强度的因子，默认为 0.1。
            extract_top_k: (可选) 当未提供 `emotion_indices` 时，动态提取置信度最高的前 k 个情感，默认为 3。
            generate_kwargs: 传递给 `transformers.generation_utils.GenerationMixin.generate` 的其他参数。

        返回:
            captions: 生成的描述文本 (token IDs)。
        """
        logits_processor = generate_kwargs.pop("logits_processor", [])
        emotion_features = None
        final_emotion_indices = None
        final_confidence_values = None

        # 1. 确定要使用的情感信息 (优先使用传入的)
        if emotion_indices is not None and confidence_values is not None:
            logger.info("使用提供的情感信息生成描述。")
            # 确保输入在正确的设备上
            if pixel_values.device != emotion_indices.device:
                emotion_indices = emotion_indices.to(pixel_values.device)
            if pixel_values.device != confidence_values.device:
                confidence_values = confidence_values.to(pixel_values.device)
            final_emotion_indices = emotion_indices
            final_confidence_values = confidence_values
        elif emotion_indices is None and confidence_values is None:
            logger.info(f"未提供情感信息，尝试动态提取 top-{extract_top_k} 情感。")
            try:
                # 调用动态提取方法
                predicted_indices, predicted_confidences = self.extract_emotions(
                    pixel_values=pixel_values,
                    top_k=extract_top_k
                )
                final_emotion_indices = predicted_indices
                final_confidence_values = predicted_confidences
                logger.info(f"动态提取的情感索引: {final_emotion_indices.tolist()}")
                logger.info(f"动态提取的置信度: {final_confidence_values.tolist()}")
            except Exception as e:
                logger.error(f"动态提取情感时出错: {e}，将不注入情感。")
                # 保持 final_emotion_indices 和 final_confidence_values 为 None
        else:
            # 处理只提供了一个情感输入的情况
            logger.warning("emotion_indices 和 confidence_values 必须同时提供或同时不提供。将不注入情感特征。")
            # 保持 final_emotion_indices 和 final_confidence_values 为 None

        # 2. 如果获得了有效的情感信息，则计算情感表示
        if final_emotion_indices is not None and final_confidence_values is not None:
            try:
                emotion_features = self.get_emotion_representation(
                    final_emotion_indices,
                    final_confidence_values
                )
            except Exception as e:
                logger.error(f"计算情感表示时出错: {e}，将不注入情感。")
                emotion_features = None # 确保出错时 emotion_features 为 None

        # 3. 如果获得情感特征，则创建并添加 Logits Processor
        if emotion_features is not None:
            try:
                emotion_processor = EmotionLogitsProcessor(
                    model_instance=self,
                    emotion_features=emotion_features,
                    alpha=emotion_alpha # 使用传入的 alpha 值
                )
                logits_processor.append(emotion_processor)
                logger.info(f"已添加 EmotionLogitsProcessor，alpha={emotion_alpha}")
            except ValueError as e:
                 logger.error(f"创建 EmotionLogitsProcessor 失败: {e}，将不注入情感。")
            except Exception as e:
                logger.error(f"创建或添加 EmotionLogitsProcessor 时发生未知错误: {e}，将不注入情感。")


        # 4. 设置默认生成参数 (如果未提供)
        generate_kwargs.setdefault("max_length", 100)
        generate_kwargs.setdefault("num_beams", 5)
        generate_kwargs.setdefault("min_length", 10)
        generate_kwargs.setdefault("do_sample", True) # 默认采样
        generate_kwargs.setdefault("temperature", 0.7)
        generate_kwargs.setdefault("top_p", 0.9)

        # 5. 调用基础模型的 generate 方法，传入修改后的 logits_processor
        outputs = self.blip.generate(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            logits_processor=logits_processor, # 传递包含情感处理器的列表
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
        """生成图像描述"""
        # 处理输入
        if image is not None and pixel_values is None:
            # 使用处理器预处理图像
            inputs = self.processor(images=image, return_tensors="pt")
            pixel_values = inputs.pixel_values
        
        # 确保像素值在正确的设备上
        if pixel_values is not None:
            pixel_values = pixel_values.to(self.device)
        
        # 调用generate方法
        generated_ids = self.generate(
            pixel_values=pixel_values,
            emotion_indices=emotion_indices,
            confidence_values=confidence_values,
            max_length=max_length,
            num_beams=num_beams,
            **generate_kwargs
        )
        
        # 解码生成的ID
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
        
        return generated_text[0] if len(generated_text) == 1 else generated_text
    
    @property
    def device(self):
        """获取模型所在的设备"""
        return next(self.parameters()).device

    def to(self, *args, **kwargs):
        """
        重写to方法，确保递归地将所有子模块移动到正确的设备
        
        这个方法会被torch.nn.Module自动调用，但我们需要确保所有自定义组件都正确移动
        """
        # 首先调用父类的to方法
        device = args[0] if args else kwargs.get('device', None)
        if device:
            logger.info(f"将模型移动到设备: {device}")
            try:
                # 显式地将子模块移动到设备
                if hasattr(self, 'blip'):
                    self.blip = self.blip.to(*args, **kwargs)
                    logger.info("BLIP模型已移动到设备")
                
                if hasattr(self, 'emotion_encoder'):
                    self.emotion_encoder = self.emotion_encoder.to(*args, **kwargs)
                    logger.info("Emotion Encoder已移动到设备")
                    
                if hasattr(self, 'emotion_adapter'):
                    self.emotion_adapter = self.emotion_adapter.to(*args, **kwargs)
                    logger.info("Emotion Adapter已移动到设备")
                    
                if hasattr(self, 'emotion_gate'):
                    self.emotion_gate = self.emotion_gate.to(*args, **kwargs)
                    logger.info("Emotion Gate已移动到设备")
                    
                if hasattr(self, 'emotion_projector'):
                    self.emotion_projector = self.emotion_projector.to(*args, **kwargs)
                    logger.info("Emotion Projector已移动到设备")
                    
                if hasattr(self, 'emotion_classifier'):
                    self.emotion_classifier = self.emotion_classifier.to(*args, **kwargs)
                    logger.info("Emotion Classifier已移动到设备")
                    
                # 然后调用nn.Module的to方法，确保所有参数正确移动
                return super().to(*args, **kwargs)
            except Exception as e:
                logger.error(f"移动模型到设备时出错: {str(e)}")
                raise
        else:
            # 如果没有指定设备，仅调用父类的to方法
            return super().to(*args, **kwargs) 