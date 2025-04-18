#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import json
import logging
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup, BlipProcessor, BlipForConditionalGeneration
from torch.cuda.amp import autocast, GradScaler

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def train(args):
    """
    简化版训练脚本 - 只使用基础BLIP模型
    """
    # 设置设备
    device = torch.device("cpu") if args.force_cpu else torch.device(args.device)
    logger.info(f"使用设备: {device}")
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 直接加载原始BLIP模型
    logger.info(f"加载BLIP模型: {args.blip_model}")
    try:
        processor = BlipProcessor.from_pretrained(args.blip_model)
        model = BlipForConditionalGeneration.from_pretrained(args.blip_model)
        logger.info("BLIP模型加载成功")
    except Exception as e:
        logger.error(f"加载BLIP模型失败: {e}")
        sys.exit(1)
    
    # 移动模型到设备
    try:
        model.to(device)
        logger.info(f"模型已移动到设备: {device}")
    except Exception as e:
        logger.error(f"移动模型到设备失败: {e}")
        # 尝试CPU
        device = torch.device("cpu")
        model.to(device)
        logger.info("回退到CPU设备")
    
    # 验证模型能够进行一次前向传播
    try:
        # 创建一个随机输入
        pixel_values = torch.randn(1, 3, 384, 384).to(device)
        input_ids = torch.randint(0, 30522, (1, 10)).to(device)
        
        # 执行一次前向传播
        with torch.no_grad():
            outputs = model(pixel_values=pixel_values, input_ids=input_ids)
        
        logger.info("前向传播测试成功!")
        logger.info(f"输出形状: {outputs.logits.shape}")
    except Exception as e:
        logger.error(f"前向传播测试失败: {e}")
        sys.exit(1)
    
    logger.info("测试完成，模型可以正常工作")

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="简化版BLIP模型训练测试")
    
    # 模型参数
    parser.add_argument("--blip_model", type=str, default="Salesforce/blip-image-captioning-base", help="BLIP模型名称")
    parser.add_argument("--output_dir", type=str, default="output/test", help="输出目录")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--force_cpu", action="store_true", help="强制使用CPU")
    
    args = parser.parse_args()
    
    # 开始测试
    train(args)

if __name__ == "__main__":
    main() 