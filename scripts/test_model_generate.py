#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
测试脚本：使用训练好的情感增强BLIP模型为验证集中的所有图像生成标题
并将结果保存到项目目录中的JSON文件，使用模型自动提取情感
"""

import os
import sys
import torch
import json
from PIL import Image
import argparse
from tqdm import tqdm
from datetime import datetime

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from emotion_enhanced_blip.models.emotion_caption_model import EmotionEnhancedBlipForCaption
from emotion_enhanced_blip.data.newyorker_dataset import NewYorkerCaptionDataset
from emotion_enhanced_blip.utils.emotion_utils import EMOTION_CATEGORIES_ZH, EMOTION_CATEGORIES


def optimized_collate_fn(batch):
    """优化的批次收集函数"""
    batch_dict = {}

    # 提取所有批次的项目
    if "pixel_values" in batch[0]:
        batch_dict["pixel_values"] = torch.stack([item["pixel_values"] for item in batch])

    # 处理标签
    if "labels" in batch[0]:
        batch_dict["labels"] = torch.stack([item["labels"] for item in batch])

    # 收集其他非张量数据 (如ID, 标题等)
    for key in batch[0].keys():
        if key not in batch_dict:
            batch_dict[key] = [item[key] for item in batch]

    return batch_dict


def main():
    # 命令行参数解析
    parser = argparse.ArgumentParser(description="使用情感增强BLIP模型为验证集生成标题并保存结果")
    parser.add_argument("--model_path", type=str, default="output/caption_model/best_model.pth", help="模型权重路径")
    parser.add_argument("--annotations_path", type=str,
                        default="annotations/preprocessed_annotations_0_to_129_validation_with_titles.json",
                        help="验证集标注文件路径")
    parser.add_argument("--blip_model_name", type=str, default="Salesforce/blip-image-captioning-base",
                        help="基础BLIP模型名称")
    parser.add_argument("--batch_size", type=int, default=4, help="批次大小")
    parser.add_argument("--output_dir", type=str, default="output/generated_captions",
                        help="输出目录路径")
    parser.add_argument("--device", type=str, default=None,
                        help="设备 (默认: 自动检测)")
    parser.add_argument("--proxy", type=str, default=None,
                        help="HTTP代理URL (例如 'http://localhost:7890')")
    parser.add_argument("--emotion_alpha", type=float, default=0.1,
                        help="情感影响因子，控制情感对生成的影响强度 (默认: 0.1)")
    parser.add_argument("--num_beams", type=int, default=5,
                        help="束搜索的束数 (默认: 5)")
    parser.add_argument("--max_length", type=int, default=50,
                        help="生成标题的最大长度 (默认: 50)")
    parser.add_argument("--extract_top_k", type=int, default=3,
                        help="自动提取情感时，提取置信度最高的前k个情感 (默认: 3)")
    args = parser.parse_args()

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

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

    # 用于收集结果的列表
    results = []

    # 执行生成
    print("\n开始为验证集生成标题...")

    with torch.no_grad():
        batch_index = 0
        for batch in tqdm(dataloader, desc="处理批次"):
            # 获取批次数据
            pixel_values = batch["pixel_values"].to(device)

            # 获取样本ID
            if "id" in batch:
                instance_ids = batch["id"]
            elif "ids" in batch:
                instance_ids = batch["ids"]
            else:
                # 使用批次索引和样本在批次中的位置生成唯一ID
                instance_ids = [f"sample_{batch_index * args.batch_size + i}" for i in range(len(pixel_values))]

            # 原始标注
            labels = batch["labels"].to(device)
            decoded_labels = processor.batch_decode(labels, skip_special_tokens=True)

            # 标题 (如果存在)
            if "ground_truth_titles" in batch:
                titles = batch["ground_truth_titles"]
            elif "titles" in batch:
                titles = batch["titles"]
            else:
                titles = ["无标题"] * len(instance_ids)

            for i in range(len(instance_ids)):
                instance_id = instance_ids[i]
                original_caption = decoded_labels[i]
                title = titles[i] if i < len(titles) else "无标题"

                # 获取当前样本的像素值
                sample_pixel_values = pixel_values[i:i + 1]

                # 用于存储当前样本生成结果
                sample_result = {
                    "id": instance_id,
                    "title": title,
                    "original_caption": original_caption
                }

                # 提取情感并生成标题
                try:
                    # 先提取情感
                    extracted_indices, extracted_confidences = model.extract_emotions(
                        pixel_values=sample_pixel_values,
                        top_k=args.extract_top_k
                    )

                    # 记录提取的情感信息
                    if torch.any(extracted_indices >= 0):
                        valid_indices = [idx.item() for idx in extracted_indices[0] if idx >= 0]
                        emotion_names = [EMOTION_CATEGORIES[idx] for idx in valid_indices]
                        emotion_names_zh = [EMOTION_CATEGORIES_ZH[idx] for idx in valid_indices]
                        valid_confidences = [extracted_confidences[0][j].item() for j, idx in
                                             enumerate(extracted_indices[0]) if idx >= 0]

                        # 保存到结果
                        sample_result["extracted_emotion_info"] = {
                            "emotions": emotion_names,
                            "emotions_zh": emotion_names_zh,
                            "confidences": valid_confidences
                        }

                    # 使用自动提取的情感生成标题
                    generated_ids = model.generate(
                        pixel_values=sample_pixel_values,
                        emotion_indices=None,  # 传递None让模型内部自动提取情感
                        confidence_values=None,
                        max_length=args.max_length,
                        num_beams=args.num_beams,
                        emotion_alpha=args.emotion_alpha,
                        extract_top_k=args.extract_top_k
                    )

                    captions = processor.batch_decode(generated_ids, skip_special_tokens=True)
                    sample_result["generated_caption"] = captions[0]

                except Exception as e:
                    print(f"为样本 {instance_id} 生成标题时出错: {e}")
                    sample_result["error"] = str(e)

                # 将当前样本的结果添加到总结果列表
                results.append(sample_result)

            # 更新批次索引
            batch_index += 1

    # 生成输出文件名（包含时间戳和参数信息）
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"validation_captions_auto_extract_alpha{args.emotion_alpha}_{timestamp}.json"
    output_path = os.path.join(args.output_dir, output_filename)

    # 保存结果到JSON文件
    print(f"\n保存生成结果到: {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # 保存生成参数信息
    params_info = {
        "timestamp": timestamp,
        "model_path": args.model_path,
        "blip_model_name": args.blip_model_name,
        "generation_mode": "auto_extract",
        "emotion_alpha": args.emotion_alpha,
        "num_beams": args.num_beams,
        "max_length": args.max_length,
        "extract_top_k": args.extract_top_k,
        "total_samples": len(results)
    }

    params_output_path = os.path.join(args.output_dir, f"params_{timestamp}.json")
    with open(params_output_path, 'w', encoding='utf-8') as f:
        json.dump(params_info, f, ensure_ascii=False, indent=2)

    print(f"\n处理完成! 共生成 {len(results)} 个样本的标题")
    print(f"结果已保存至: {output_path}")
    print(f"参数信息已保存至: {params_output_path}")


if __name__ == "__main__":
    main()