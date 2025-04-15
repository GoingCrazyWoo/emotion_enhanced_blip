#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
标注预处理脚本，用于将原始标注文件转换为标准格式，
以便直接被数据集加载使用，避免在运行时进行处理。

处理后的标注文件将包含以下标准字段：
- emotion_indices: 情感索引列表
- confidence_values: 置信度列表
- instance_id: 图像ID或实例ID，与dataset中的ID对应

使用方法:
python src/scripts/preprocess_annotations.py --input annotations/results_0_to_2339(1).json --output annotations/preprocessed_annotations.json
"""

import os
import sys
import json
import logging
import argparse
from tqdm import tqdm
from typing import Dict, List, Any
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from emotion_enhanced_blip.utils.emotion_utils import (
    extract_emotions_from_annotation,
    get_emotion_name,
    format_emotions_for_display
)

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def standardize_annotation(annotation: Dict[str, Any]) -> Dict[str, Any]:
    """
    将标注数据标准化为统一格式
    
    参数:
        annotation: 原始标注数据
    
    返回:
        标准化后的标注数据
    """
    try:
        # 直接从annotation中获取情感信息，不使用extract_emotions_from_annotation
        emotion_indices = []
        confidence_values = []
        
        # 处理emotions_with_indices字段
        if "emotions_with_indices" in annotation and isinstance(annotation["emotions_with_indices"], list):
            for emotion_info in annotation["emotions_with_indices"]:
                if isinstance(emotion_info, dict) and "emotion_index" in emotion_info:
                    emotion_indices.append(emotion_info["emotion_index"])
        
        # 处理confidence或confidences字段
        if "confidences" in annotation and isinstance(annotation["confidences"], list):
            confidence_values = annotation["confidences"]
        elif "confidence" in annotation:
            confidence = annotation["confidence"]
            if isinstance(confidence, list):
                confidence_values = confidence
            elif isinstance(confidence, (int, float)):
                confidence_values = [confidence]
        
        # 确保confidence_values和emotion_indices长度匹配
        while len(confidence_values) < len(emotion_indices):
            confidence_values.append(0.5)  # 默认中等置信度
        confidence_values = confidence_values[:len(emotion_indices)]  # 截断多余的置信度
        
        # 如果emotions_with_indices处理失败，尝试处理emotions字段
        if not emotion_indices and "emotions" in annotation:
            emotions = annotation["emotions"]
            
            # 如果emotions是字符串，尝试解析它
            if isinstance(emotions, str):
                try:
                    # 尝试作为JSON解析
                    import json
                    parsed_emotions = json.loads(emotions)
                    if isinstance(parsed_emotions, list):
                        emotions = parsed_emotions
                except:
                    # 如果解析失败，尝试按逗号分割
                    emotions = [e.strip() for e in emotions.split(",")]
            
            # 处理不同格式的emotions字段
            if isinstance(emotions, list):
                from emotion_enhanced_blip.utils.emotion_utils import get_emotion_mapping
                emotion_mapping = get_emotion_mapping()
                
                for emotion in emotions:
                    if isinstance(emotion, dict) and "name" in emotion:
                        emotion_name = emotion["name"]
                        if emotion_name in emotion_mapping:
                            emotion_indices.append(emotion_mapping[emotion_name])
                            if "confidence" in emotion and isinstance(emotion["confidence"], (int, float)):
                                confidence_values.append(emotion["confidence"])
                            else:
                                confidence_values.append(0.5)  # 默认中等置信度
                    elif isinstance(emotion, str) and emotion in emotion_mapping:
                        emotion_indices.append(emotion_mapping[emotion])
                        confidence_values.append(0.5)  # 默认中等置信度
        
        # 提取或生成ID
        instance_id = None
        for id_field in ["original_id", "image_id", "id", "instance_id"]:
            if id_field in annotation and annotation[id_field]:
                instance_id = annotation[id_field]
                break
        
        if not instance_id:
            raise ValueError(f"无法提取有效ID: {list(annotation.keys())}")
        
        # 创建标准化后的标注
        standardized = {
            "instance_id": instance_id,
            "emotion_indices": emotion_indices,
            "confidence_values": confidence_values
        }
        
        # 可选: 保留原始字段以供参考
        if "explanation" in annotation:
            standardized["explanation"] = annotation["explanation"]
        if "label" in annotation:
            standardized["label"] = annotation["label"]
        
        # 添加情感名称，方便查看
        emotion_names = [get_emotion_name(idx) for idx in emotion_indices]
        standardized["emotion_names"] = emotion_names
        
        return standardized
    except Exception as e:
        logger.error(f"标准化标注时出错 (ID={annotation.get('original_id', annotation.get('image_id', 'unknown'))}): {e}")
        raise

def preprocess_annotations(input_path: str, output_path: str, check_dataset: bool = False) -> None:
    """
    预处理情感标注文件
    
    参数:
        input_path: 输入文件路径
        output_path: 输出文件路径
        check_dataset: 是否检查与数据集的匹配情况
    """
    logger.info(f"开始处理标注文件: {input_path}")
    
    # 加载原始标注
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            annotations = json.load(f)
        
        logger.info(f"成功加载 {len(annotations)} 条标注记录")
        
        # 检查标注格式
        if len(annotations) > 0:
            if isinstance(annotations, list):
                logger.info(f"原始标注格式（列表）: {list(annotations[0].keys()) if isinstance(annotations[0], dict) else '非字典类型'}")
            elif isinstance(annotations, dict):
                first_key = next(iter(annotations))
                logger.info(f"原始标注格式（字典）: 键示例='{first_key}'，值类型={type(annotations[first_key]).__name__}")
                if isinstance(annotations[first_key], dict):
                    logger.info(f"值结构: {list(annotations[first_key].keys())}")
    except Exception as e:
        logger.error(f"加载标注文件失败: {e}")
        return
    
    # 标准化标注
    standardized_annotations = {}
    errors = 0
    success = 0
    
    # 如果annotations是字典，将其转换为列表处理
    annotations_list = []
    if isinstance(annotations, dict):
        for key, value in annotations.items():
            if isinstance(value, dict):
                # 确保value包含ID字段
                value["id"] = key  # 使用键作为ID
                annotations_list.append(value)
            else:
                logger.warning(f"跳过非字典值: 键={key}, 值类型={type(value).__name__}")
    else:
        annotations_list = annotations
    
    for annotation in tqdm(annotations_list, desc="处理标注"):
        try:
            if not isinstance(annotation, dict):
                logger.warning(f"跳过非字典标注: {type(annotation).__name__}")
                errors += 1
                continue
                
            standardized = standardize_annotation(annotation)
            instance_id = standardized["instance_id"]
            standardized_annotations[instance_id] = standardized
            success += 1
        except Exception as e:
            errors += 1
            if errors <= 10:  # 只显示前10个错误
                logger.error(f"处理标注时出错: {e}")
    
    logger.info(f"完成标准化处理, 共 {success} 条有效标注, {errors} 条处理失败")
    
    # 检查与数据集的匹配情况
    if check_dataset:
        try:
            from datasets import load_dataset
            logger.info("加载数据集以检查ID匹配情况...")
            
            # 加载数据集
            dataset = load_dataset("jmhessel/newyorker_caption_contest", "explanation", split="train")
            
            # 计算匹配数量
            matches = 0
            for item in tqdm(dataset, desc="检查匹配"):
                if item["instance_id"] in standardized_annotations:
                    matches += 1
            
            logger.info(f"与数据集匹配情况: {matches}/{len(dataset)} ({matches/len(dataset)*100:.2f}%)")
        except Exception as e:
            logger.error(f"检查数据集匹配时出错: {e}")
    
    # 保存处理后的标注
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(standardized_annotations, f, ensure_ascii=False, indent=2)
    
    logger.info(f"标准化后的标注已保存到: {output_path}")
    
    # 输出样例
    if len(standardized_annotations) > 0:
        sample_id = next(iter(standardized_annotations))
        sample = standardized_annotations[sample_id]
        logger.info(f"标准化后的标注示例:")
        logger.info(f"ID: {sample['instance_id']}")
        logger.info(f"情感索引: {sample['emotion_indices']}")
        logger.info(f"置信度: {sample['confidence_values']}")
        logger.info(f"情感名称: {sample['emotion_names']}")
        
        emotions_display = format_emotions_for_display(
            sample['emotion_indices'], 
            sample['confidence_values']
        )
        logger.info(f"情感显示: {emotions_display}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="情感标注预处理脚本")
    parser.add_argument("--input", type=str, required=True, help="输入标注文件路径")
    parser.add_argument("--output", type=str, required=True, help="输出标注文件路径")
    parser.add_argument("--check", action="store_true", help="检查与数据集的匹配情况")
    
    args = parser.parse_args()
    
    # 处理标注
    preprocess_annotations(args.input, args.output, args.check)

if __name__ == "__main__":
    main() 