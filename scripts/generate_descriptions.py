#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import json
import logging
import torch
from PIL import Image
from tqdm import tqdm
import time
from pathlib import Path
from transformers import BlipProcessor
from datetime import datetime

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from emotion_enhanced_blip.models.emotion_caption_model import EmotionEnhancedBlipForCaption
from emotion_enhanced_blip.data.newyorker_dataset import NewYorkerCaptionDataset
from emotion_enhanced_blip.utils.emotion_utils import (
    format_emotions_for_display, 
    save_descriptions_to_json, 
    emotion_indices_to_names
)

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def generate_descriptions(args):
    """生成描述文本"""
    # 设置设备
    device = torch.device(args.device)
    logger.info(f"使用设备: {device}")
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载模型
    logger.info(f"使用BLIP模型: {args.blip_model}")
    model = EmotionEnhancedBlipForCaption(
        blip_model_name=args.blip_model,
        proxy=args.proxy
    )
    
    # 加载模型权重
    if args.model_path and os.path.exists(args.model_path):
        logger.info(f"加载模型权重: {args.model_path}")
        model.load_state_dict(torch.load(args.model_path, map_location=device))
    else:
        logger.info("使用未微调的模型")
    
    model = model.to(device)
    model.eval()
    
    # 加载数据集
    logger.info(f"加载数据集 (split={args.split})")
    dataset = NewYorkerCaptionDataset(
        split=args.split,
        annotations_path=args.annotations_path,
        processor=None,  # 使用模型中的处理器
        blip_model_name=args.blip_model,
        limit_samples=args.max_samples,
        proxy=args.proxy,
        dataset_cache_dir=args.cache_dir
    )
    
    logger.info(f"数据集大小: {len(dataset)}")
    
    # 准备输出结果
    results = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(args.output_dir, f"descriptions_{args.split}_{timestamp}.json")
    
    # 开始生成描述
    start_time = time.time()
    
    with torch.no_grad():
        for idx in tqdm(range(len(dataset)), desc="生成描述"):
            try:
                # 获取样本
                sample = dataset[idx]
                sample_id = sample["id"]
                
                # 获取图像和情感
                pixel_values = sample["pixel_values"].unsqueeze(0).to(device)
                emotion_indices = sample["emotion_indices"]
                confidence_values = sample["confidence_values"]
                
                # 如果没有情感标注，使用默认值
                if not emotion_indices:
                    emotion_indices = [2]  # 默认使用幽默
                    confidence_values = [0.8]
                
                # 转换为tensor
                emotion_indices_tensor = torch.tensor([emotion_indices], dtype=torch.long).to(device)
                confidence_values_tensor = torch.tensor([confidence_values], dtype=torch.float).to(device)
                
                # 生成描述
                output_ids = model.generate(
                    pixel_values=pixel_values,
                    # emotion_indices 和 confidence_values 不再传递给 generate
                    max_length=args.max_length,
                    num_beams=args.num_beams,
                    min_length=args.min_length,
                    do_sample=args.do_sample,
                    temperature=args.temperature,
                    top_p=args.top_p
                )
                
                # 解码生成的描述
                caption = dataset.processor.decode(output_ids[0], skip_special_tokens=True)
                
                # 获取参考描述
                reference_captions = sample.get("reference_captions", [])
                
                # 记录结果
                result = {
                    "id": sample_id,
                    "generated_description": caption,
                    "reference_captions": reference_captions,
                    "emotion_indices": emotion_indices,
                    "emotion_names": emotion_indices_to_names(emotion_indices),
                    "confidence_values": confidence_values
                }
                
                results.append(result)
                
                # 周期性保存结果
                if args.save_interval > 0 and (idx + 1) % args.save_interval == 0:
                    save_descriptions_to_json(results, output_path)
                    logger.info(f"已保存 {len(results)} 个结果到 {output_path}")
            
            except Exception as e:
                logger.error(f"处理样本 {idx} 时出错: {e}")
                continue
    
    # 最终保存结果
    save_descriptions_to_json(results, output_path)
    
    # 统计信息
    elapsed_time = time.time() - start_time
    avg_time_per_sample = elapsed_time / len(results) if results else 0
    
    logger.info(f"完成! 生成了 {len(results)} 个描述")
    logger.info(f"总耗时: {elapsed_time:.2f}秒, 平均每样本: {avg_time_per_sample:.2f}秒")
    logger.info(f"结果已保存到: {output_path}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="生成带有情感特征的描述文本")
    
    # 数据参数
    parser.add_argument("--split", type=str, default="train", help="数据集划分(train, validation, test)")
    parser.add_argument("--annotations_path", type=str, default="annotations/results_0_to_2339(1).json", help="情感标注文件路径")
    parser.add_argument("--cache_dir", type=str, default=None, help="数据集缓存目录")
    parser.add_argument("--output_dir", type=str, default="output/descriptions", help="输出目录")
    parser.add_argument("--max_samples", type=int, default=None, help="最大样本数")
    parser.add_argument("--save_interval", type=int, default=50, help="保存结果的间隔样本数")
    
    # 模型参数
    parser.add_argument("--model_path", type=str, default=None, help="模型权重路径")
    parser.add_argument("--blip_model", type=str, default="Salesforce/blip-image-captioning-large", help="BLIP模型名称")
    
    # 生成参数
    parser.add_argument("--max_length", type=int, default=100, help="生成的最大长度")
    parser.add_argument("--min_length", type=int, default=10, help="生成的最小长度")
    parser.add_argument("--num_beams", type=int, default=5, help="束搜索数量")
    parser.add_argument("--do_sample", action="store_true", help="是否进行采样")
    parser.add_argument("--temperature", type=float, default=0.7, help="采样温度")
    parser.add_argument("--top_p", type=float, default=0.9, help="top-p采样")
    
    # 其他参数
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="设备")
    parser.add_argument("--proxy", type=str, default="http://127.0.0.1:7890", help="HTTP代理URL")
    
    args = parser.parse_args()
    
    # 生成描述
    generate_descriptions(args)

if __name__ == "__main__":
    main() 