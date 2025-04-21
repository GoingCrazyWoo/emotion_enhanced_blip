#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
优化的New Yorker数据集类，使用预处理后的标注文件
以避免运行时的处理开销和错误。
"""

import os
import json
import random
from PIL import Image
import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from transformers import BlipProcessor
import logging
from typing import Dict, List, Optional, Tuple, Any, Union
import io
import base64
import requests
from pathlib import Path
import numpy as np

logger = logging.getLogger(__name__)

class NewYorkerCaptionDataset(Dataset):
    """优化的New Yorker漫画描述数据集，使用预处理后的标注"""
    
    def __init__(
        self,
        split: str = "train",
        preprocessed_annotations_path: Optional[str] = "../annotations/preprocessed_annotations_with_titles.json",
        processor: Optional[BlipProcessor] = None,
        blip_model_name: str = "Salesforce/blip-image-captioning-large",
        max_target_length: int = 128,
        max_emotions: int = 3,
        limit_samples: Optional[int] = None,
        proxy: Optional[str] = None,
        dataset_cache_dir: Optional[str] = None,
        local_dataset_path: Optional[str] = None
    ):
        """
        初始化优化的New Yorker漫画描述数据集
        
        参数:
            split: 数据集分割
            preprocessed_annotations_path: 预处理后的标注文件路径（JSON格式，键为instance_id）
            processor: BLIP预处理器，不提供则自动创建
            blip_model_name: BLIP模型名称
            max_target_length: 最大目标长度
            max_emotions: 每张图像的最大情感数量
            limit_samples: 限制样本数量
            proxy: HTTP代理URL
            dataset_cache_dir: 数据集缓存目录
            local_dataset_path: 本地数据集路径，如果提供，将优先从本地加载
        """
        super().__init__()
        self.split = split
        self.processor = processor
        self.max_target_length = max_target_length
        self.max_emotions = max_emotions
        self.proxy = proxy
        self.limit_samples = limit_samples
        
        # 设置代理
        if proxy:
            logger.info(f"使用代理: {proxy}")
            self.proxies = {
                "http": proxy,
                "https": proxy
            }
        else:
            self.proxies = None
            
        # 加载BLIP处理器
        if self.processor is None:
            from transformers import BlipProcessor
            try:
                logger.info(f"加载BLIP处理器: {blip_model_name}")
                proxies = self.proxies if self.proxies else None
                
                # 使用代理下载模型
                self.processor = BlipProcessor.from_pretrained(
                    blip_model_name,
                    proxies=proxies
                )
            except Exception as e:
                logger.error(f"加载BLIP处理器失败: {e}")
                raise
            
        # 加载数据集
        try:
            # 首先检查是否存在本地数据集
            if local_dataset_path and os.path.exists(local_dataset_path):
                logger.info(f"从本地路径加载New Yorker数据集: {local_dataset_path}")
                try:
                    # 尝试从本地加载数据集
                    # from datasets import load_dataset # 删除此处的局部导入，因为它导致 UnboundLocalError，应使用文件顶部的全局导入
                    
                    # 尝试作为HuggingFace datasets格式加载
                    try:
                        complete_dataset = load_dataset(local_dataset_path)
                        logger.info(f"从HF数据集目录加载: {local_dataset_path}")
                    except Exception as e:
                        logger.warning(f"从HF数据集目录加载失败: {e}")
                        
                        # 尝试作为缓存目录加载
                        try:
                            complete_dataset = load_dataset(
                                "jmhessel/newyorker_caption_contest", 
                                "explanation",
                                split=split,
                                cache_dir=os.path.dirname(local_dataset_path)
                            )
                            logger.info(f"从缓存目录加载: {os.path.dirname(local_dataset_path)}")
                        except Exception as e:
                            logger.error(f"所有本地加载方法都失败: {e}")
                            raise
                    
                    # 处理加载到的数据集
                    if hasattr(complete_dataset, 'get') and split in complete_dataset:
                        self.dataset = complete_dataset[split]
                        logger.info(f"成功从本地加载数据集分割 '{split}'")
                    elif isinstance(complete_dataset, dict) and split in complete_dataset:
                        self.dataset = complete_dataset[split]
                        logger.info(f"成功从本地加载数据集分割 '{split}'")
                    else:
                        # 假设整个数据集是单个分割
                        self.dataset = complete_dataset
                        logger.info("从本地加载了完整数据集")
                        
                except Exception as e:
                    logger.warning(f"从本地加载数据集失败: {e}，将尝试在线下载")
                    local_dataset_path = None
            
            # 如果本地加载失败或未提供本地路径，则从在线下载
            if not local_dataset_path or not hasattr(self, 'dataset'):
                logger.info(f"从在线加载New Yorker数据集 ({split})")
                self.dataset = load_dataset(
                    "jmhessel/newyorker_caption_contest", 
                    "explanation",
                    split=split,
                    cache_dir=dataset_cache_dir,
                    proxies=self.proxies
                )
                
                # 如果提供了本地路径但不存在，则保存下载的数据集
                if local_dataset_path and not os.path.exists(local_dataset_path):
                    try:
                        # 创建目录（如果不存在）
                        os.makedirs(os.path.dirname(local_dataset_path), exist_ok=True)
                        
                        logger.info(f"保存数据集到本地: {local_dataset_path}")
                        # 保存数据集到本地
                        self.dataset.save_to_disk(local_dataset_path)
                    except Exception as e:
                        logger.error(f"保存数据集到本地失败: {e}")
                
            # 打印一个样本示例
            if len(self.dataset) > 0:
                # 获取第一个样本的ID，如果不存在则生成一个默认值
                sample = self.dataset[0]
                instance_id = sample.get("instance_id", f"unknown_{0}")
                logger.info(f"样本示例：instance_id={instance_id}")
                
        except Exception as e:
            logger.error(f"加载数据集失败: {e}")
            raise
            
        # 限制样本数量
        if self.limit_samples is not None:
            logger.info(f"限制样本数量: {self.limit_samples}")
            self.dataset = self.dataset.select(range(min(self.limit_samples, len(self.dataset))))
            
        logger.info(f"数据集大小: {len(self.dataset)}")
        
        # 加载预处理后的标注
        self.annotations = {}
        if preprocessed_annotations_path and os.path.exists(preprocessed_annotations_path):
            logger.info(f"加载预处理标注: {preprocessed_annotations_path}")
            try:
                with open(preprocessed_annotations_path, 'r', encoding='utf-8') as f:
                    self.annotations = json.load(f)
                
                logger.info(f"成功加载 {len(self.annotations)} 条预处理标注")
                
                # 检查标注格式
                if len(self.annotations) > 0:
                    sample_id = next(iter(self.annotations))
                    sample_annotation = self.annotations[sample_id]
                    logger.info(f"标注样本示例：{sample_annotation.keys()}")
                    
                    # 验证标注是否包含必要字段
                    required_fields = ["emotion_indices", "confidence_values"]
                    missing_fields = [field for field in required_fields if field not in sample_annotation]
                    
                    if missing_fields:
                        logger.warning(f"标注缺少必要字段: {missing_fields}")
            except Exception as e:
                logger.error(f"加载预处理标注失败: {e}")
                self.annotations = {}
        else:
            logger.warning(f"未找到预处理标注文件或未提供路径: {preprocessed_annotations_path}")
    
    def __len__(self) -> int:
        """返回数据集长度"""
        return len(self.dataset)
    
    def get_image(self, idx: int) -> Image.Image:
        """获取图像"""
        try:
            # 尝试从数据集获取图像
            sample = self.dataset[idx]
            image = sample["image"]
            
            # 确保图像是RGB模式
            if image.mode != "RGB":
                image = image.convert("RGB")
                
            return image
        except Exception as e:
            logger.error(f"获取图像失败 (idx={idx}): {e}")
            # 返回空白图像
            return Image.new('RGB', (384, 384), color='white')
    
    def get_emotions(self, instance_id: str) -> Tuple[List[int], List[float]]:
        """获取情感标注，直接从预处理标注中读取"""
        if instance_id in self.annotations:
            annotation = self.annotations[instance_id]
            
            # 从预处理标注中直接获取情感索引和置信度
            emotion_indices = annotation.get("emotion_indices", [])
            confidence_values = annotation.get("confidence_values", [])
            
            # 确保列表非空
            if not emotion_indices:
                return [], []
                
            return emotion_indices, confidence_values
        return [], []
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """获取数据集样本"""
        try:
            sample = self.dataset[idx]
            instance_id = sample.get("instance_id", f"unknown_{idx}")
            # print(f"[DEBUG] Processing sample idx: {idx}, ID: {instance_id}")
            image = self.get_image(idx)
            emotion_indices, confidence_values = self.get_emotions(instance_id)

            # 获取真实标题和描述
            ground_truth_title = None
            reference_description = None # 使用更明确的变量名

            # 优先从预处理标注中获取标题和描述
            if instance_id in self.annotations:
                annotation = self.annotations[instance_id]
                ground_truth_title = annotation.get("title") # 尝试从标注获取标题
                reference_description = annotation.get("explanation") # 尝试从标注获取描述

            # 如果标注中没有，尝试从原始数据集中获取
            if ground_truth_title is None and "title" in sample and sample["title"]:
                ground_truth_title = sample["title"]
            if reference_description is None and "explanation" in sample and sample["explanation"]:
                reference_description = sample["explanation"]

            # 如果两者都没有，设置默认值
            if ground_truth_title is None:
                ground_truth_title = "" # 设为空字符串而不是None，方便后续处理
                logger.warning(f"样本 {instance_id} 缺少标题。")
            if reference_description is None:
                 reference_description = "" # 设为空字符串

            # 处理图像
            try:
                pixel_values = self.processor(
                    images=image,
                    return_tensors="pt"
                ).pixel_values.squeeze()
            except Exception as e:
                logger.error(f"处理图像失败 (idx={idx}): {e}")
                pixel_values = torch.zeros((3, 384, 384))

            result = {
                "id": instance_id,
                "pixel_values": pixel_values,
                "emotion_indices": emotion_indices,
                "confidence_values": confidence_values,
                "ground_truth_title": ground_truth_title, # 添加真实标题
                "reference_description": reference_description, # 添加参考描述
            }

            # 处理目标文本（用于训练，评估时可能不需要）
            # 决定使用标题还是描述作为训练目标，这里我们假设评估时不需要labels
            target_text = ground_truth_title # 使用标题作为训练目标
            if target_text:
                try:
                    target_encoding = self.processor(
                        text=target_text,
                        padding="max_length",
                        truncation=True,
                        max_length=self.max_target_length,
                        return_tensors="pt"
                    )
                    # 确保 labels 是一个张量
                    labels = target_encoding.input_ids.squeeze()
                    if labels.dim() == 0: # 如果squeeze后变成0维标量
                        labels = labels.unsqueeze(0) # 恢复为1维张量
                    result["labels"] = labels
                except Exception as e:
                    logger.error(f"处理文本失败 (idx={idx}): {e}")
                    # 如果处理失败，添加一个空的或填充的标签以避免后续错误
                    result["labels"] = torch.full((self.max_target_length,), fill_value=-100, dtype=torch.long)
            else: # 如果没有目标文本，也需要提供一个labels字段给collate_fn
                # 添加空的标签值，确保所有样本都有labels字段
                result["labels"] = torch.full((self.max_target_length,), fill_value=-100, dtype=torch.long)
                
            return result
        except Exception as e:
            logger.error(f"处理样本失败 (idx={idx}): {str(e)}")
            print(f"[DEBUG] Error in __getitem__ (idx={idx}): {e}")
            return {
                "id": f"error_{idx}",
                "pixel_values": torch.zeros((3, 384, 384)),
                "emotion_indices": [],
                "confidence_values": [],
                "ground_truth_title": "", # 确保返回空字符串
                "reference_description": "", # 确保返回空字符串
                "labels": torch.full((self.max_target_length,), fill_value=-100, dtype=torch.long) # 添加空标签
            }


def optimized_collate_fn(batch):
    """为优化数据集定制的数据批次整理函数"""
    batch = [item for item in batch if item is not None and "pixel_values" in item]

    if not batch:
        return {
            "pixel_values": torch.zeros((1, 3, 384, 384)),
            "labels": torch.zeros((1, 10), dtype=torch.long), # 保留以防训练时需要
            "emotion_indices": torch.zeros((1, 1), dtype=torch.long),
            "confidence_values": torch.zeros((1, 1), dtype=torch.float),
            "ids": ["empty_batch"],
            "ground_truth_titles": [""], # 修改为 titles
            "reference_descriptions": [[]] # 修改为 descriptions
        }

    pixel_values = torch.stack([item["pixel_values"] for item in batch])

    max_emotions = max(len(item.get("emotion_indices", [])) for item in batch)
    if max_emotions == 0: max_emotions = 1
    emotion_indices = torch.full((len(batch), max_emotions), fill_value=-1, dtype=torch.long)
    confidence_values = torch.zeros((len(batch), max_emotions), dtype=torch.float)

    # 保留标签处理逻辑，以防训练时需要
    has_labels = all("labels" in item for item in batch)
    labels = None
    attention_mask = None
    if has_labels:
        max_label_len = max(item["labels"].size(0) for item in batch if "labels" in item)
        labels = torch.full((len(batch), max_label_len), fill_value=-100, dtype=torch.long)
        attention_mask = torch.zeros((len(batch), max_label_len), dtype=torch.long)

    ids = []
    ground_truth_titles_list = [] # 用于收集真实标题
    reference_descriptions_list = [] # 用于收集参考描述

    for i, item in enumerate(batch):
        if "emotion_indices" in item and len(item["emotion_indices"]) > 0:
            emotion_len = min(len(item["emotion_indices"]), max_emotions)
            emotion_indices[i, :emotion_len] = torch.tensor(item["emotion_indices"][:emotion_len], dtype=torch.long)
            confidence_values[i, :emotion_len] = torch.tensor(item["confidence_values"][:emotion_len], dtype=torch.float)

        if has_labels and "labels" in item:
            seq_len = item["labels"].size(0)
            labels[i, :seq_len] = item["labels"]
            # 创建注意力掩码 - 非填充位置设为1
            attention_mask[i, :seq_len] = (item["labels"] != -100).long()

        ids.append(item.get("id", f"unknown_{i}"))
        ground_truth_titles_list.append(item.get("ground_truth_title", "")) # 收集真实标题
        reference_descriptions_list.append(item.get("reference_description", "")) # 收集参考描述

    result = {
        "pixel_values": pixel_values,
        "emotion_indices": emotion_indices,
        "confidence_values": confidence_values,
        "ids": ids,
        "ground_truth_titles": ground_truth_titles_list, # 返回真实标题列表
        "reference_descriptions": reference_descriptions_list # 返回参考描述列表
    }

    if has_labels:
        result["labels"] = labels
        result["attention_mask"] = attention_mask

    return result