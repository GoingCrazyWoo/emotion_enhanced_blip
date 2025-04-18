#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
测试脚本：验证情感增强BLIP模型的generate函数是否正常工作
使用validation数据集进行测试
"""

import os
import sys
import torch
from PIL import Image
import argparse
from tqdm import tqdm

# 添加项目根目录到系统路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# 导入模型和必要的工具
from models.emotion_caption_model import EmotionEnhancedBlipForCaption
from utils.emotion_utils import EMOTION_CATEGORIES, EMOTION_CATEGORIES_ZH
from data.newyorker_dataset import NewYorkerCaptionDataset, optimized_collate_fn

def main():
    # 命令行参数解析
    parser = argparse.ArgumentParser(description="测试情感增强BLIP模型的generate函数")
    parser.add_argument("--model_path", type=str, default="output/caption_model/best_model.pth", help="模型权重路径")
    parser.add_argument("--annotations_path", type=str, 
                       default="annotations/preprocessed_annotations_0_to_129_validation_with_titles.json", 
                       help="验证集标注文件路径")
    parser.add_argument("--blip_model_name", type=str, default="Salesforce/blip-image-captioning-base", 
                        help="基础BLIP模型名称")
    parser.add_argument("--batch_size", type=int, default=1, help="批次大小")
    parser.add_argument("--num_samples", type=int, default=5, help="测试样本数量")
    parser.add_argument("--device", type=str, default=None, 
                        help="设备 (默认: 自动检测)")
    parser.add_argument("--proxy", type=str, default=None, 
                        help="HTTP代理URL (例如 'http://localhost:7890')")
    args = parser.parse_args()
    
    # 设备检测
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"使用设备: {device}")
    
    # 加载数据集
    print(f"加载验证数据集: {args.annotations_path}")
    try:
        dataset = NewYorkerCaptionDataset(
            split="validation",
            preprocessed_annotations_path=args.annotations_path,
            blip_model_name=args.blip_model_name,
            max_target_length=100,
            proxy=args.proxy
        )
        
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=optimized_collate_fn
        )
        
        print(f"成功加载数据集，共 {len(dataset)} 个样本")
    except Exception as e:
        print(f"加载数据集时出错: {e}")
        sys.exit(1)
    
    # 加载模型
    print(f"正在加载模型: {args.model_path}")
    model = EmotionEnhancedBlipForCaption(
        blip_model_name=args.blip_model_name,
        freeze_blip=False,  # 测试时不需要冻结
        proxy=args.proxy
    )
    
    try:
        state_dict = torch.load(args.model_path, map_location=device)
        # 处理可能的 DataParallel 或 DDP 包装
        if 'module.' in list(state_dict.keys())[0]:
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict, strict=False)
        print(f"模型权重从 {args.model_path} 加载成功")
    except Exception as e:
        print(f"加载模型权重时出错: {e}")
        sys.exit(1)
    
    model.to(device)
    model.eval()
    processor = dataset.processor
    
    # 执行测试
    print("\n开始测试生成功能...")
    
    with torch.no_grad():
        # 限制测试样本数量
        sample_count = 0
        
        for batch in tqdm(dataloader, desc="生成测试中"):
            if sample_count >= args.num_samples:
                break
                
            # 获取批次数据
            pixel_values = batch["pixel_values"].to(device)
            emotion_indices = batch["emotion_indices"].to(device)
            confidence_values = batch["confidence_values"].to(device)
            instance_ids = batch["instance_ids"]
            
            # 原始标注
            labels = batch["labels"].to(device)
            decoded_labels = processor.batch_decode(labels, skip_special_tokens=True)
            
            # 标题 (如果存在)
            titles = batch.get("titles", ["无标题"] * len(instance_ids))
            
            for i in range(len(instance_ids)):
                sample_count += 1
                if sample_count > args.num_samples:
                    break
                    
                instance_id = instance_ids[i]
                print(f"\n\n样本 {sample_count}: {instance_id}")
                print(f"标题: {titles[i]}")
                print(f"原始描述: {decoded_labels[i]}")
                
                # 获取情感信息
                sample_emotion_indices = emotion_indices[i:i+1]
                sample_confidence_values = confidence_values[i:i+1]
                sample_pixel_values = pixel_values[i:i+1]
                
                # 显示情感类别
                if torch.any(sample_emotion_indices >= 0):
                    valid_indices = [idx.item() for idx in sample_emotion_indices[0] if idx >= 0]
                    emotion_names = [EMOTION_CATEGORIES[idx] for idx in valid_indices]
                    emotion_names_zh = [EMOTION_CATEGORIES_ZH[idx] for idx in valid_indices]
                    valid_confidences = [sample_confidence_values[0][j].item() for j, idx in enumerate(sample_emotion_indices[0]) if idx >= 0]
                    
                    print(f"情感: {', '.join(emotion_names)} ({', '.join(emotion_names_zh)})")
                    emotion_with_conf = [f"{emotion}({emotion_zh}, {conf:.2f})" 
                                       for emotion, emotion_zh, conf in zip(emotion_names, emotion_names_zh, valid_confidences)]
                    print(f"情感及置信度: {', '.join(emotion_with_conf)}")
                
                # 1. 使用原始情感生成
                try:
                    print("\n1. 原始情感生成:")
                    generated_ids = model.generate(
                        pixel_values=sample_pixel_values,
                        emotion_indices=sample_emotion_indices,
                        confidence_values=sample_confidence_values,
                        max_length=50,
                        num_beams=5
                    )
                    captions = processor.batch_decode(generated_ids, skip_special_tokens=True)
                    print(f"生成描述: {captions[0]}")
                except Exception as e:
                    print(f"生成出错: {e}")
                
                # 2. 使用不同情感组合测试（幽默+讽刺）
                try:
                    print("\n2. 幽默+讽刺生成:")
                    humor_satire_emotions = torch.tensor([[2, 3, -1]]).to(device)  # humor(2), satire(3)
                    humor_satire_confidence = torch.tensor([[0.8, 0.5, 0.0]]).to(device)
                    
                    generated_ids = model.generate(
                        pixel_values=sample_pixel_values,
                        emotion_indices=humor_satire_emotions,
                        confidence_values=humor_satire_confidence,
                        max_length=50,
                        num_beams=5
                    )
                    captions = processor.batch_decode(generated_ids, skip_special_tokens=True)
                    print(f"生成描述: {captions[0]}")
                except Exception as e:
                    print(f"生成出错: {e}")
                
                # 3. 使用不同情感组合测试（快乐+温馨）
                try:
                    print("\n3. 快乐+温馨生成:")
                    happy_warm_emotions = torch.tensor([[0, 7, -1]]).to(device)  # happiness(0), warmth(7)
                    happy_warm_confidence = torch.tensor([[0.9, 0.7, 0.0]]).to(device)
                    
                    generated_ids = model.generate(
                        pixel_values=sample_pixel_values,
                        emotion_indices=happy_warm_emotions,
                        confidence_values=happy_warm_confidence,
                        max_length=50,
                        num_beams=5
                    )
                    captions = processor.batch_decode(generated_ids, skip_special_tokens=True)
                    print(f"生成描述: {captions[0]}")
                except Exception as e:
                    print(f"生成出错: {e}")
    
    print("\n测试完成!")

if __name__ == "__main__":
    main() 