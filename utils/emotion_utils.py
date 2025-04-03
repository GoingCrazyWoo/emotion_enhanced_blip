#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import logging
import os
#from emotion_enhanced_blip.models.emotion_caption_model import EMOTION_CATEGORIES, EMOTION_TO_INDEX, INDEX_TO_EMOTION
import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, Union

logger = logging.getLogger(__name__)

# 定义情感类别
EMOTION_CATEGORIES = [
    "happiness",  # 快乐
    "sadness",    # 悲伤
    "humor",      # 幽默
    "satire",     # 讽刺
    "confusion",  # 困惑
    "surprise",   # 惊讶
    "embarrassment",  # 尴尬
    "warmth"      # 温馨
]

# 情感类别的中文名称，用于日志输出
EMOTION_CATEGORIES_ZH = [
    "快乐",
    "悲伤",
    "幽默",
    "讽刺",
    "困惑",
    "惊讶", 
    "尴尬",
    "温馨"
]

# 创建映射字典
EMOTION_TO_INDEX = {emotion: idx for idx, emotion in enumerate(EMOTION_CATEGORIES)}
INDEX_TO_EMOTION = {idx: emotion for idx, emotion in enumerate(EMOTION_CATEGORIES)}

def get_emotion_mapping() -> Dict[str, int]:
    """获取情感类别名称到索引的映射"""
    return EMOTION_TO_INDEX

def get_emotion_name(index: int) -> str:
    """根据索引获取情感名称"""
    return INDEX_TO_EMOTION.get(index, "unknown")

def get_emotion_name_zh(index: int) -> str:
    """根据索引获取情感中文名称"""
    if 0 <= index < len(EMOTION_CATEGORIES_ZH):
        return EMOTION_CATEGORIES_ZH[index]
    return "未知"

def load_annotations(annotation_path: str) -> List[Dict[str, Any]]:
    """加载情感标注数据"""
    with open(annotation_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def create_emotion_embedding(emotion_indices: List[int], 
                           confidence_values: List[float], 
                           max_emotions: int = 3,
                           embedding_dim: int = 16) -> torch.Tensor:
    """
    将情感索引和置信度转换为嵌入张量
    
    参数:
        emotion_indices: 情感类别索引列表 [0-7]
        confidence_values: 对应的置信度列表 [0-1]
        max_emotions: 最大情感数量
        embedding_dim: 每个情感的嵌入维度
        
    返回:
        torch.Tensor: 形状为 [max_emotions, embedding_dim]
    """
    # 使用类别长度作为填充索引
    padding_idx = len(EMOTION_CATEGORIES)
    
    # 补齐到最大情感数量
    if len(emotion_indices) < max_emotions:
        padding_count = max_emotions - len(emotion_indices)
        emotion_indices = emotion_indices + [padding_idx] * padding_count
        confidence_values = confidence_values + [0.0] * padding_count
    
    # 截断超出的部分
    emotion_indices = emotion_indices[:max_emotions]
    confidence_values = confidence_values[:max_emotions]
    
    # 为每个情感创建一个独热编码
    emotion_embeddings = []
    for idx, conf in zip(emotion_indices, confidence_values):
        if idx == padding_idx:  # 填充情感
            embedding = torch.zeros(embedding_dim)
        else:
            # 创建情感嵌入：前8位是独热编码，后8位可以是额外信息
            embedding = torch.zeros(embedding_dim)
            embedding[idx] = conf  # 用置信度作为权重
        
        emotion_embeddings.append(embedding)
    
    return torch.stack(emotion_embeddings)

def format_emotions_for_display(emotion_indices: List[int], 
                               confidence_values: List[float]) -> str:
    """格式化情感和置信度为可读字符串"""
    emotions_str = []
    for idx, conf in zip(emotion_indices, confidence_values):
        emotion_name = get_emotion_name(idx)
        emotion_name_zh = get_emotion_name_zh(idx)
        emotions_str.append(f"{emotion_name}({emotion_name_zh}): {conf:.2f}")
    
    return ", ".join(emotions_str)

def save_descriptions_to_json(descriptions: List[Dict[str, Any]], 
                             output_path: str) -> None:
    """保存生成的描述到JSON文件"""
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(descriptions, f, ensure_ascii=False, indent=2)
        
    print(f"描述已保存到 {output_path}")

def extract_emotions_from_annotation(annotation: Dict[str, Any]) -> Tuple[List[int], List[float]]:
    """从标注数据中提取情感索引和置信度"""
    emotion_indices = []
    confidence_values = []
    
    # 处理新格式（直接包含emotions_with_indices和confidences字段）
    if "emotions_with_indices" in annotation and "confidences" in annotation:
        for emotion_info in annotation["emotions_with_indices"]:
            emotion_indices.append(emotion_info["emotion_index"])
        confidence_values = annotation["confidences"]
    
    # 处理旧格式
    elif "emotions" in annotation:
        emotion_mapping = get_emotion_mapping()
        for emotion in annotation["emotions"]:
            if emotion["name"] in emotion_mapping:
                emotion_indices.append(emotion_mapping[emotion["name"]])
                confidence_values.append(emotion["confidence"])
    
    return emotion_indices, confidence_values

def emotion_indices_to_names(emotion_indices: List[int]) -> List[str]:
    """将情感索引列表转换为情感名称列表"""
    return [get_emotion_name(idx) for idx in emotion_indices]

def load_emotion_mapping(mapping_file="annotations/emotion_mapping.json"):
    """
    加载情感映射文件
    
    Args:
        mapping_file: 情感映射文件路径
    
    Returns:
        emotion_to_index: 情感到索引的映射
        index_to_emotion: 索引到情感的映射
    """
    try:
        if os.path.exists(mapping_file):
            with open(mapping_file, "r", encoding="utf-8") as f:
                mapping = json.load(f)
                
            # 将字符串键转换为整数键
            index_to_emotion = {int(k): v for k, v in mapping["index_to_emotion"].items()}
            emotion_to_index = mapping["emotion_to_index"]
            
            return emotion_to_index, index_to_emotion
        else:
            logger.warning(f"找不到情感映射文件: {mapping_file}，使用默认映射")
            return EMOTION_TO_INDEX, INDEX_TO_EMOTION
    except Exception as e:
        logger.error(f"加载情感映射文件时出错: {str(e)}，使用默认映射")
        return EMOTION_TO_INDEX, INDEX_TO_EMOTION

def emotion_indices_to_emotions(emotion_indices, index_to_emotion=None):
    """
    将情感索引转换为情感名称
    
    Args:
        emotion_indices: 情感索引列表
        index_to_emotion: 索引到情感名称的映射
    
    Returns:
        emotions: 情感名称列表
    """
    if index_to_emotion is None:
        _, index_to_emotion = load_emotion_mapping()
    
    return [index_to_emotion.get(idx, f"未知情感({idx})") for idx in emotion_indices]

def update_emotion_annotation(annotation, emotions, confidence_values):
    """
    更新情感标注
    
    Args:
        annotation: 原始标注
        emotions: 情感名称列表
        confidence_values: 置信度值列表
    
    Returns:
        updated_annotation: 更新后的标注
    """
    emotion_to_index, _ = load_emotion_mapping()
    
    # 创建emotions_with_indices字段
    emotions_with_indices = []
    for i, emotion in enumerate(emotions):
        if emotion in emotion_to_index:
            emotions_with_indices.append({
                "emotion_index": emotion_to_index[emotion],
                "emotion_name": emotion
            })
    
    # 更新标注
    updated_annotation = annotation.copy()
    updated_annotation["emotions"] = emotions
    updated_annotation["confidence"] = confidence_values
    updated_annotation["emotions_with_indices"] = emotions_with_indices
    
    return updated_annotation

def analyze_emotion_annotations(annotations_file):
    """
    分析情感标注数据
    
    Args:
        annotations_file: 标注文件路径
    
    Returns:
        stats: 统计信息
    """
    try:
        with open(annotations_file, "r", encoding="utf-8") as f:
            annotations = json.load(f)
        
        emotion_count = {}
        total_annotations = len(annotations)
        emotions_per_sample = []
        
        for ann in annotations:
            if "emotions" in ann and isinstance(ann["emotions"], list):
                emotions = ann["emotions"]
                emotions_per_sample.append(len(emotions))
                
                for emotion in emotions:
                    emotion_count[emotion] = emotion_count.get(emotion, 0) + 1
        
        # 计算平均每个样本的情感数量
        avg_emotions = sum(emotions_per_sample) / len(emotions_per_sample) if emotions_per_sample else 0
        
        # 计算每种情感的百分比
        emotion_percentage = {emotion: count / total_annotations * 100 
                             for emotion, count in emotion_count.items()}
        
        # 排序
        sorted_emotions = sorted(emotion_percentage.items(), key=lambda x: x[1], reverse=True)
        
        return {
            "total_annotations": total_annotations,
            "avg_emotions_per_sample": avg_emotions,
            "emotion_count": emotion_count,
            "emotion_percentage": emotion_percentage,
            "sorted_emotions": sorted_emotions
        }
    
    except Exception as e:
        logger.error(f"分析情感标注数据时出错: {str(e)}")
        return None 