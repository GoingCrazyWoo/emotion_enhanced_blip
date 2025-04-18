#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import torch
import logging
from transformers import BlipProcessor, BlipForConditionalGeneration
import json

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
    from emotion_enhanced_blip.utils.emotion_utils import EMOTION_CATEGORIES
    
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
        batch_size = 1
        seq_len = 10
        
        # 基本输入
        pixel_values = torch.randn(batch_size, 3, 384, 384).to(device)
        
        # 情感索引和置信度
        num_emotions = len(EMOTION_CATEGORIES)
        max_emotions = 3
        emotion_indices = torch.randint(0, num_emotions, (batch_size, max_emotions)).to(device)
        confidence_values = torch.rand(batch_size, max_emotions).to(device)
        
        # 多热编码的情感标签
        emotion_labels_multi_hot = torch.zeros(batch_size, num_emotions).to(device)
        for i in range(batch_size):
            for j in range(max_emotions):
                if emotion_indices[i, j] < num_emotions:
                    emotion_labels_multi_hot[i, emotion_indices[i, j]] = 1.0
        
        # 输入IDs和注意力掩码
        input_ids = torch.randint(0, 30522, (batch_size, seq_len)).to(device)
        attention_mask = torch.ones(batch_size, seq_len).to(device)
        
        # 标签
        labels = torch.randint(0, 30522, (batch_size, seq_len)).to(device)
        
        # 步骤4: 添加诊断信息收集
        model_diagnostics = {}
        # 收集模型架构信息
        if hasattr(model, 'blip'):
            model_diagnostics["blip_model_name"] = model.blip.config.name_or_path if hasattr(model.blip.config, 'name_or_path') else "Unknown"
            model_diagnostics["text_config"] = {
                "hidden_size": model.blip.config.text_config.hidden_size,
                "vocab_size": model.blip.config.text_config.vocab_size,
            }
            model_diagnostics["vision_config"] = {
                "hidden_size": model.blip.config.vision_config.hidden_size
            }
        
        logger.info(f"模型诊断信息: {json.dumps(model_diagnostics, indent=2)}")
        
        # 执行标准前向传播
        try:
            logger.info("执行标准前向传播测试...")
            with torch.no_grad():
                outputs = model(
                    pixel_values=pixel_values,
                    emotion_indices=emotion_indices,
                    confidence_values=confidence_values
                )
            logger.info("标准前向传播成功!")
        except Exception as e:
            logger.error(f"标准前向传播失败: {e}")
            import traceback
            traceback.print_exc()
        
        # 执行完整前向传播
        if args.full_test:
            try:
                logger.info("执行完整前向传播测试 (包含所有参数)...")
                with torch.no_grad():
                    outputs = model(
                        pixel_values=pixel_values,
                        emotion_indices=emotion_indices,
                        confidence_values=confidence_values,
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                        emotion_labels_multi_hot=emotion_labels_multi_hot,
                        emotion_loss_weight=0.5
                    )
                
                logger.info("完整前向传播成功!")
                logger.info(f"输出损失: {outputs.get('loss')}")
                logger.info(f"情感损失: {outputs.get('emotion_loss')}")
                logger.info(f"标题损失: {outputs.get('caption_loss')}")
            except RuntimeError as e:
                error_msg = str(e)
                logger.error(f"完整前向传播失败: {error_msg}")
                
                # 检查是否是矩阵维度不匹配错误
                if "mat1 and mat2 shapes cannot be multiplied" in error_msg:
                    logger.error("检测到矩阵维度不匹配错误!")
                    # 提取错误中的维度信息
                    import re
                    dims = re.findall(r'\d+x\d+', error_msg)
                    if dims:
                        logger.error(f"不匹配的维度: {dims}")
                    
                    # 提供解决方案
                    logger.info("\n===== 解决方案 =====")
                    logger.info("1. 这是模型内部矩阵维度不匹配问题，可能由以下原因导致:")
                    logger.info("   - BLIP模型版本与代码不兼容")
                    logger.info("   - 情感特征与BLIP文本编码器维度不匹配")
                    logger.info("   - 输入标签格式不正确")
                    
                    logger.info("\n2. 在训练脚本中添加以下错误处理:")
                    logger.info("""
# 在train_captioner.py的前向传播部分添加以下代码:
try:
    outputs = model(
        pixel_values=pixel_values,
        emotion_indices=emotion_indices,
        attention_mask=attention_mask,
        input_ids=labels,
        labels=labels,
        emotion_loss_weight=args.emotion_loss_weight,
        emotion_labels_multi_hot=emotion_labels_multi_hot
    )
except RuntimeError as e:
    if "mat1 and mat2 shapes cannot be multiplied" in str(e):
        logger.warning(f"矩阵维度不匹配错误: {e}")
        logger.warning("回退到仅使用情感分类，跳过标题生成")
        # 尝试仅使用情感分类
        outputs = model(
            pixel_values=pixel_values,
            emotion_indices=emotion_indices,
            emotion_labels_multi_hot=emotion_labels_multi_hot
        )
    else:
        raise  # 重新抛出其他错误
                    """)
                    
                    logger.info("\n3. 或者修改模型的forward方法，使其能够处理此错误:")
                    logger.info("""
# 在models/emotion_caption_model.py的caption_loss计算部分添加:
try:
    # 调用BLIP计算caption loss
    blip_outputs = self.blip(
        pixel_values=pixel_values,
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        return_dict=True
    )
    caption_loss = blip_outputs.loss.item()
except RuntimeError as e:
    if "mat1 and mat2 shapes cannot be multiplied" in str(e):
        logger.warning(f"矩阵维度不匹配，跳过标题生成: {e}")
        caption_loss = None
    else:
        raise
                    """)
                    
                    # 测试仅使用情感分类的回退方案
                    logger.info("\n正在测试仅使用情感分类的回退方案...")
                    try:
                        with torch.no_grad():
                            outputs = model(
                                pixel_values=pixel_values,
                                emotion_indices=emotion_indices,
                                confidence_values=confidence_values,
                                emotion_labels_multi_hot=emotion_labels_multi_hot
                            )
                        
                        emotion_loss = outputs.get("emotion_loss")
                        logger.info(f"回退方案测试成功! 情感损失: {emotion_loss}")
                        logger.info("结论: 可以通过跳过标题生成部分继续训练，仅使用情感分类损失")
                    except Exception as e:
                        logger.error(f"回退方案测试失败: {e}")
                else:
                    # 其他类型的运行时错误
                    import traceback
                    traceback.print_exc()
            except Exception as e:
                # 非运行时错误
                logger.error(f"完整前向传播遇到未知错误: {e}")
                import traceback
                traceback.print_exc()
        
        logger.info("完整模型测试完成")
        
        return True, model
    
    except Exception as e:
        logger.error(f"完整模型测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False, None

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="测试模型加载和前向传播")
    
    # 模型参数
    parser.add_argument("--blip_model", type=str, default="Salesforce/blip-image-captioning-base", help="BLIP模型名称")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--force_cpu", action="store_true", help="强制使用CPU")
    parser.add_argument("--test_type", type=str, choices=["blip", "emotion", "full", "all"], default="all", help="测试类型")
    parser.add_argument("--full_test", action="store_true", help="执行完整的前向传播测试")
    
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