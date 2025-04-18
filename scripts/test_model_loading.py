#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import torch
import logging
from transformers import BlipProcessor, BlipForConditionalGeneration

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_blip_model(args):
    """测试BLIP模型加载和前向传播"""
    logger.info("===== 测试BLIP模型 =====")
    
    device = torch.device("cpu") if args.force_cpu else torch.device(args.device)
    logger.info(f"使用设备: {device}")
    
    try:
        # 步骤1: 加载处理器
        logger.info("加载BLIP处理器...")
        processor = BlipProcessor.from_pretrained(args.blip_model)
        logger.info("处理器加载成功")
        
        # 步骤2: 加载模型
        logger.info("加载BLIP模型...")
        model = BlipForConditionalGeneration.from_pretrained(args.blip_model)
        logger.info("模型加载成功")
        
        # 步骤3: 移动模型到设备
        logger.info(f"将模型移动到设备: {device}")
        try:
            model.to(device)
            logger.info("模型成功移动到设备")
        except Exception as e:
            logger.error(f"移动模型到设备失败: {e}")
            if device.type == "cuda":
                logger.info("尝试使用CPU...")
                device = torch.device("cpu")
                model.to(device)
                logger.info("模型已移动到CPU")
        
        # 步骤4: 测试前向传播
        logger.info("测试前向传播...")
        # 创建随机输入
        pixel_values = torch.randn(1, 3, 384, 384).to(device)
        input_ids = torch.randint(0, 30522, (1, 10)).to(device)
        
        # 执行前向传播
        with torch.no_grad():
            outputs = model(pixel_values=pixel_values, input_ids=input_ids)
        
        logger.info(f"前向传播成功，输出形状: {outputs.logits.shape}")
        logger.info("BLIP模型测试通过!")
        
        return True, model, processor
    
    except Exception as e:
        logger.error(f"BLIP模型测试失败: {e}")
        return False, None, None

def test_emotion_components(args):
    """测试情感组件加载"""
    logger.info("===== 测试情感组件 =====")
    
    from emotion_enhanced_blip.models.emotion_caption_model import (
        EmotionEncoder, 
        EmotionLogitsProcessor, 
        EmotionEnhancedBlipForCaption
    )
    
    device = torch.device("cpu") if args.force_cpu else torch.device(args.device)
    logger.info(f"使用设备: {device}")
    
    try:
        # 步骤1: 创建情感编码器
        logger.info("创建EmotionEncoder...")
        emotion_encoder = EmotionEncoder(
            num_emotions=7,  # 假设有7种情感类别
            emotion_dim=32,
            max_emotions=3,
            hidden_dim=768  # BLIP隐藏维度
        )
        logger.info("EmotionEncoder创建成功")
        
        # 步骤2: 移动情感编码器到设备
        logger.info(f"将EmotionEncoder移动到设备: {device}")
        emotion_encoder.to(device)
        logger.info("EmotionEncoder成功移动到设备")
        
        # 步骤3: 测试EmotionEncoder前向传播
        logger.info("测试EmotionEncoder前向传播...")
        # 创建随机输入
        emotion_indices = torch.randint(0, 7, (1, 3)).to(device)
        confidence_values = torch.rand(1, 3).to(device)
        
        # 执行前向传播
        with torch.no_grad():
            emotion_features = emotion_encoder(
                emotion_indices=emotion_indices,
                confidence_values=confidence_values
            )
        
        logger.info(f"EmotionEncoder前向传播成功，输出形状: {emotion_features.shape}")
        logger.info("情感组件测试通过!")
        
        return True, emotion_encoder
    
    except Exception as e:
        logger.error(f"情感组件测试失败: {e}")
        return False, None

def test_full_model(args):
    """测试完整模型加载和前向传播"""
    logger.info("===== 测试完整模型 =====")
    
    from emotion_enhanced_blip.models.emotion_caption_model import EmotionEnhancedBlipForCaption
    
    device = torch.device("cpu") if args.force_cpu else torch.device(args.device)
    logger.info(f"使用设备: {device}")
    
    try:
        # 步骤1: 创建完整模型
        logger.info("创建EmotionEnhancedBlipForCaption...")
        model = EmotionEnhancedBlipForCaption(
            blip_model_name=args.blip_model,
            freeze_blip=True,
            proxy=None  # 禁用代理
        )
        logger.info("EmotionEnhancedBlipForCaption创建成功")
        
        # 步骤2: 移动模型到设备
        logger.info("清理内存...")
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        logger.info(f"将模型移动到设备: {device}")
        try:
            # 逐个组件移动
            if hasattr(model, 'processor'):
                logger.info("移动processor...")
            
            if hasattr(model, 'blip'):
                logger.info("移动blip...")
                model.blip = model.blip.to(device)
            
            if hasattr(model, 'emotion_encoder'):
                logger.info("移动emotion_encoder...")
                model.emotion_encoder = model.emotion_encoder.to(device)
            
            if hasattr(model, 'emotion_adapter'):
                logger.info("移动emotion_adapter...")
                model.emotion_adapter = model.emotion_adapter.to(device)
            
            if hasattr(model, 'emotion_gate'):
                logger.info("移动emotion_gate...")
                model.emotion_gate = model.emotion_gate.to(device)
            
            if hasattr(model, 'emotion_projector'):
                logger.info("移动emotion_projector...")
                model.emotion_projector = model.emotion_projector.to(device)
            
            if hasattr(model, 'emotion_classifier'):
                logger.info("移动emotion_classifier...")
                model.emotion_classifier = model.emotion_classifier.to(device)
                
            logger.info("模型组件移动完成")
        except Exception as e:
            logger.error(f"移动模型组件失败: {e}")
            if device.type == "cuda":
                logger.info("尝试使用CPU...")
                device = torch.device("cpu") 
                # 重复上面的移动过程...
        
        # 步骤3: 测试前向传播
        logger.info("测试前向传播...")
        # 创建随机输入
        pixel_values = torch.randn(1, 3, 384, 384).to(device)
        emotion_indices = torch.randint(0, 7, (1, 3)).to(device)
        confidence_values = torch.rand(1, 3).to(device)
        
        # 执行前向传播
        with torch.no_grad():
            outputs = model(
                pixel_values=pixel_values,
                emotion_indices=emotion_indices,
                confidence_values=confidence_values
            )
        
        logger.info("前向传播成功!")
        logger.info("完整模型测试通过!")
        
        return True, model
    
    except Exception as e:
        logger.error(f"完整模型测试失败: {e}")
        return False, None

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="测试模型加载和前向传播")
    
    # 模型参数
    parser.add_argument("--blip_model", type=str, default="Salesforce/blip-image-captioning-base", help="BLIP模型名称")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--force_cpu", action="store_true", help="强制使用CPU")
    parser.add_argument("--test_type", type=str, choices=["blip", "emotion", "full", "all"], default="all", help="测试类型")
    
    args = parser.parse_args()
    
    # 开始测试
    if args.test_type == "blip" or args.test_type == "all":
        success, _, _ = test_blip_model(args)
        if not success and args.test_type == "all":
            logger.error("BLIP模型测试失败，终止后续测试")
            return
    
    if args.test_type == "emotion" or args.test_type == "all":
        success, _ = test_emotion_components(args)
        if not success and args.test_type == "all":
            logger.error("情感组件测试失败，终止后续测试")
            return
    
    if args.test_type == "full" or args.test_type == "all":
        success, _ = test_full_model(args)
        if not success:
            logger.error("完整模型测试失败")
        else:
            logger.info("所有测试通过!")
    
if __name__ == "__main__":
    main() 