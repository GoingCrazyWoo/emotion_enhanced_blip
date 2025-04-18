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
from transformers import get_linear_schedule_with_warmup, BlipProcessor
from torch.cuda.amp import autocast, GradScaler  # 导入 AMP 相关组件
from huggingface_hub import snapshot_download

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from emotion_enhanced_blip.models.emotion_caption_model import EmotionEnhancedBlipForCaption
from emotion_enhanced_blip.data.newyorker_dataset import NewYorkerCaptionDataset
from emotion_enhanced_blip.utils.emotion_utils import EMOTION_CATEGORIES

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def collate_fn(batch):
    """数据批次整理函数"""
    # 过滤掉空样本
    batch = [item for item in batch if item is not None and "pixel_values" in item and "labels" in item]
    
    # 如果批次为空，返回默认批次
    if not batch:
        return {
            "pixel_values": torch.zeros((1, 3, 384, 384)),
            "labels": torch.zeros((1, 10), dtype=torch.long),
            "emotion_indices": torch.zeros((1, 1), dtype=torch.long),
            "confidence_values": torch.zeros((1, 1), dtype=torch.float),
            "attention_mask": torch.zeros((1, 10), dtype=torch.long)  # 添加默认注意力掩码
        }
    
    # 提取所有批次的项目
    pixel_values = torch.stack([item["pixel_values"] for item in batch])
    
    # 获取标签的最大长度
    max_label_len = max(item["labels"].size(0) for item in batch if "labels" in item)
    
    # 准备批处理张量
    labels = torch.full((len(batch), max_label_len), fill_value=-100, dtype=torch.long)
    attention_mask = torch.zeros((len(batch), max_label_len), dtype=torch.long)  # 注意力掩码
    
    # 情感标签处理
    max_emotions = max(len(item["emotion_indices"]) for item in batch)
    if max_emotions == 0:
        max_emotions = 1  # 确保至少有一个情感
    
    # 使用正确的填充索引
    padding_idx = len(EMOTION_CATEGORIES)
    emotion_indices = torch.full((len(batch), max_emotions), fill_value=padding_idx, dtype=torch.long)
    confidence_values = torch.zeros((len(batch), max_emotions), dtype=torch.float)
    
    ids = []
    
    # 填充或截断批次中的每个项目
    for i, item in enumerate(batch):
        # 处理标签
        if "labels" in item:
            seq_len = item["labels"].size(0)
            labels[i, :seq_len] = item["labels"]
            # 设置注意力掩码：非填充位置（非-100）为1，填充位置为0
            attention_mask[i, :seq_len] = (item["labels"] != -100).long()
        
        # 处理情感标签
        if "emotion_indices" in item and len(item["emotion_indices"]) > 0:
            emotion_len = min(len(item["emotion_indices"]), max_emotions)
            emotion_indices[i, :emotion_len] = torch.tensor(item["emotion_indices"][:emotion_len], dtype=torch.long)
            confidence_values[i, :emotion_len] = torch.tensor(item["confidence_values"][:emotion_len], dtype=torch.float)
        
        # 收集ID
        if "id" in item:
            ids.append(item["id"])
    
    return {
        "pixel_values": pixel_values,
        "labels": labels,
        "attention_mask": attention_mask,  # 添加注意力掩码
        "emotion_indices": emotion_indices,
        "confidence_values": confidence_values,
        "ids": ids
    }

def calculate_trainable_params(model):
    """计算可训练参数数量"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info(f"模型总参数量: {total_params:,}")
    logger.info(f"可训练参数量: {trainable_params:,} ({trainable_params/total_params*100:.2f}%)")
    
    return total_params, trainable_params

def train(args):
    """
    训练模型
    
    Args:
        args: 命令行参数
    """
    # 设置设备
    device = torch.device(args.device)
    logger.info(f"使用设备: {device}")
    
    # 检查是否可以使用FP16：在CPU上禁用FP16
    if device.type != "cuda":
        args.fp16 = False
        logger.warning("警告: 在CPU设备上无法使用半精度训练。已自动禁用FP16。")
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 初始化模型
    logger.info(f"使用BLIP模型: {args.blip_model}")
    model = EmotionEnhancedBlipForCaption(
        blip_model_name=args.blip_model,
        freeze_blip=args.freeze_blip,
        proxy=args.proxy
    )
    
    # 加载模型参数
    if args.load_model_path:
        if os.path.exists(args.load_model_path):
            logger.info(f"从本地加载模型参数: {args.load_model_path}")
            state_dict = torch.load(args.load_model_path, map_location=device)
            model.load_state_dict(state_dict, strict=False)
        else:
            logger.error(f"错误: 本地模型参数文件不存在: {args.load_model_path}")
            sys.exit(1) # 退出程序
    else:
        logger.info("未指定本地模型路径，从Hugging Face下载模型...")
        # 检查并下载模型快照
        snapshot_dir = "../snapshots"
        snapshot_file = "blip_state_dict.pth"
        snapshot_path = os.path.join(snapshot_dir, snapshot_file)
        
        if not os.path.exists(snapshot_path):
            logger.info(f"下载模型快照到 {snapshot_dir}...")
            try:
                snapshot_download(
                    repo_id="Opps/blip_base_newyorker",
                    local_dir=snapshot_dir,
                    local_dir_use_symlinks=False,
                    allow_patterns=snapshot_file,
                    proxy=args.proxy # 使用代理进行下载
                )
                logger.info("模型快照下载完成。")
            except Exception as e:
                logger.error(f"下载模型快照失败: {e}")
                sys.exit(1) # 下载失败则退出
        else:
            logger.info(f"找到现有模型快照: {snapshot_path}")
            
        state_dict = torch.load(snapshot_path, map_location=device)
        model.load_state_dict(state_dict, strict=False)
        logger.info("模型参数加载完成。")
        
    model.to(device)
    
    # 计算可训练参数
    calculate_trainable_params(model)
    
    # 创建数据加载器
    logger.info("创建数据集...")
    train_dataset = NewYorkerCaptionDataset(
        split="train",
        preprocessed_annotations_path=args.annotations_path,
        blip_model_name=args.blip_model,
        max_target_length=args.max_length,
        limit_samples=args.num_samples,
        proxy=args.proxy,
        dataset_cache_dir=args.cache_dir
    )
    
    val_dataset = NewYorkerCaptionDataset(
        split="validation",
        preprocessed_annotations_path=args.annotations_path,
        processor=train_dataset.processor,  # 重用处理器
        blip_model_name=args.blip_model,
        max_target_length=args.max_length,
        limit_samples=args.num_samples,
        proxy=args.proxy,
        dataset_cache_dir=args.cache_dir
    )
    
    logger.info(f"训练集: {len(train_dataset)} 样本")
    logger.info(f"验证集: {len(val_dataset)} 样本")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # 设置优化器和学习率调度器
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * 0.1),
        num_training_steps=total_steps
    )
    
    # 初始化梯度缩放器用于混合精度训练
    scaler = GradScaler(enabled=args.fp16)
    if args.fp16:
        logger.info("启用半精度(FP16)训练")
    
    # 记录训练配置
    config = {
        "blip_model": args.blip_model,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "weight_decay": args.weight_decay,
        "epochs": args.epochs,
        "max_length": args.max_length,
        "freeze_blip": args.freeze_blip,
        "emotion_loss_weight": args.emotion_loss_weight, # 添加情感损失权重
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "annotations_path": args.annotations_path,
        "fp16": args.fp16  # 添加半精度设置到配置
    }

    with open(os.path.join(args.output_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    
    # 训练循环
    best_val_loss = float("inf")
    training_history = []
    
    for epoch in range(args.epochs):
        logger.info(f"开始训练第 {epoch+1}/{args.epochs} 轮")
        
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_caption_loss = 0.0
        train_emotion_loss = 0.0
        train_batches = 0

        train_progress = tqdm(train_loader, desc=f"训练轮次 {epoch+1}")
        for batch in train_progress:
            # 将数据移到设备
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)
            emotion_indices = batch["emotion_indices"].to(device)
            # confidence_values = batch["confidence_values"].to(device) # 当前未使用
            attention_mask = batch["attention_mask"].to(device)  # 使用batch中的attention_mask

            # 清除梯度
            optimizer.zero_grad()

            # 使用混合精度训练
            with autocast(enabled=args.fp16):
                # 前向传播
                outputs = model(
                    pixel_values=pixel_values,
                    emotion_indices=emotion_indices, # 传递真实情感标签用于计算损失
                    # confidence_values=confidence_values, # 当前未使用
                    attention_mask=attention_mask,  # 传入注意力掩码
                    input_ids=labels, # input_ids 用于生成
                    labels=labels,    # labels 用于计算 caption loss
                    emotion_loss_weight=args.emotion_loss_weight # 传递损失权重
                )

                # 获取总损失和子损失 (如果存在)
                loss = outputs.get("loss") # 总损失
                caption_loss = outputs.get("caption_loss")
                emotion_loss = outputs.get("emotion_loss")

            # 只有在存在可训练的总损失时才进行反向传播
            if loss is not None and loss.requires_grad:
                # 使用梯度缩放器进行反向传播
                if args.fp16:
                    scaler.scale(loss).backward()
                    # 梯度裁剪
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    # 更新参数
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    # 常规反向传播
                    loss.backward()
                    # 梯度裁剪
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    # 更新参数
                    optimizer.step()

                scheduler.step()

                # 更新训练损失统计 (仅当loss有效时)
                train_loss += loss.item()
                if caption_loss is not None:
                    train_caption_loss += caption_loss # caption_loss 已经是 item()
                if emotion_loss is not None:
                    train_emotion_loss += emotion_loss # emotion_loss 已经是 item()
                train_batches += 1

                # 更新进度条显示子损失
                log_dict = {"loss": f"{loss.item():.4f}"}
                if caption_loss is not None:
                    log_dict["cap_loss"] = f"{caption_loss:.4f}"
                if emotion_loss is not None:
                    log_dict["emo_loss"] = f"{emotion_loss:.4f}"
                train_progress.set_postfix(log_dict)
            else:
                 # 如果 loss 为 None (例如冻结BLIP且没有情感标签)，跳过优化步骤
                 logger.debug("跳过优化步骤，因为没有可训练的损失。")


            # 定期保存检查点
            if args.save_steps > 0 and train_batches > 0 and (train_batches % args.save_steps == 0):
                checkpoint_path = os.path.join(args.output_dir, f"checkpoint_e{epoch+1}_b{train_batches}.pth")
                torch.save(model.state_dict(), checkpoint_path)
                logger.info(f"保存检查点到 {checkpoint_path}")
        
        # 计算平均训练损失 (确保 train_batches > 0)
        avg_train_loss = train_loss / train_batches if train_batches > 0 else 0.0
        avg_train_caption_loss = train_caption_loss / train_batches if train_batches > 0 else 0.0
        avg_train_emotion_loss = train_emotion_loss / train_batches if train_batches > 0 else 0.0
        logger.info(f"平均训练损失: {avg_train_loss:.4f} (标题: {avg_train_caption_loss:.4f}, 情感: {avg_train_emotion_loss:.4f})")

        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_caption_loss = 0.0
        val_emotion_loss = 0.0
        val_batches = 0

        with torch.no_grad():
            val_progress = tqdm(val_loader, desc=f"验证轮次 {epoch+1}")
            for batch in val_progress:
                # 将数据移到设备
                pixel_values = batch["pixel_values"].to(device)
                labels = batch["labels"].to(device)
                emotion_indices = batch["emotion_indices"].to(device)
                # confidence_values = batch["confidence_values"].to(device) # 当前未使用
                attention_mask = batch["attention_mask"].to(device)  # 使用batch中的attention_mask

                # 验证时也使用混合精度，但不需要梯度
                with autocast(enabled=args.fp16):
                    # 前向传播
                    outputs = model(
                        pixel_values=pixel_values,
                        emotion_indices=emotion_indices,
                        # confidence_values=confidence_values,
                        attention_mask=attention_mask,  # 传入注意力掩码
                        labels=labels,
                        input_ids=labels, # 验证时也需要计算损失
                        emotion_loss_weight=args.emotion_loss_weight # 传递权重
                    )

                    loss = outputs.get("loss")
                    caption_loss = outputs.get("caption_loss")
                    emotion_loss = outputs.get("emotion_loss")

                # 更新验证损失 (仅当 loss 有效时)
                if loss is not None:
                    val_loss += loss.item()
                    if caption_loss is not None:
                        val_caption_loss += caption_loss # item()
                    if emotion_loss is not None:
                        val_emotion_loss += emotion_loss # item()
                    val_batches += 1

                    # 更新进度条
                    log_dict = {"loss": f"{loss.item():.4f}"}
                    if caption_loss is not None:
                        log_dict["cap_loss"] = f"{caption_loss:.4f}"
                    if emotion_loss is not None:
                        log_dict["emo_loss"] = f"{emotion_loss:.4f}"
                    val_progress.set_postfix(log_dict)

        # 计算平均验证损失 (确保 val_batches > 0)
        avg_val_loss = val_loss / val_batches if val_batches > 0 else 0.0
        avg_val_caption_loss = val_caption_loss / val_batches if val_batches > 0 else 0.0
        avg_val_emotion_loss = val_emotion_loss / val_batches if val_batches > 0 else 0.0
        logger.info(f"平均验证损失: {avg_val_loss:.4f} (标题: {avg_val_caption_loss:.4f}, 情感: {avg_val_emotion_loss:.4f})")

        # 记录训练历史
        epoch_history = {
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "train_caption_loss": avg_train_caption_loss,
            "train_emotion_loss": avg_train_emotion_loss,
            "val_loss": avg_val_loss,
            "val_caption_loss": avg_val_caption_loss,
            "val_emotion_loss": avg_val_emotion_loss,
            "learning_rate": scheduler.get_last_lr()[0]
        }
        training_history.append(epoch_history)

        # 保存训练历史
        with open(os.path.join(args.output_dir, "training_history.json"), "w", encoding="utf-8") as f:
            json.dump(training_history, f, ensure_ascii=False, indent=2)
        
        # 保存模型
        model_path = os.path.join(args.output_dir, f"model_epoch_{epoch+1}.pth")
        torch.save(model.state_dict(), model_path)
        logger.info(f"模型已保存到 {model_path}")
        
        # 保存最佳模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_path = os.path.join(args.output_dir, "best_model.pth")
            torch.save(model.state_dict(), best_model_path)
            logger.info(f"最佳模型已保存到 {best_model_path}，验证损失: {best_val_loss:.4f}")
    
    logger.info("训练完成！")
    logger.info(f"最佳验证损失: {best_val_loss:.4f}")

def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="训练情感增强的BLIP描述生成模型")
    
    # 数据参数
    parser.add_argument("--annotations_path", type=str, default=r"../annotations/preprocessed_annotations_with_titles.json", help="包含标题的情感标注文件路径")
    parser.add_argument("--cache_dir", type=str, default=None, help="数据集缓存目录")
    parser.add_argument("--output_dir", type=str, default="output/caption_model", help="模型输出目录")
    parser.add_argument("--num_samples", type=int, default=None, help="用于调试的最大样本数")
    
    # 模型参数
    parser.add_argument("--load_model_path", type=str, default=None, help="从本地加载模型参数的路径 (.pth 文件)")
    parser.add_argument("--blip_model", type=str, default="Salesforce/blip-image-captioning-base", help="BLIP模型名称")
    parser.add_argument("--max_length", type=int, default=100, help="文本最大长度")
    parser.add_argument("--freeze_blip", default=True, action="store_true", help="是否冻结BLIP基础模型")
    parser.add_argument("--no-freeze-blip", action="store_true", help="是否不冻结BLIP基础模型")
    
    # 训练参数
    parser.add_argument("--batch_size", type=int, default=2, help="批次大小")
    parser.add_argument("--lr", type=float, default=2e-5, help="学习率")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="权重衰减")
    parser.add_argument("--epochs", type=int, default=3, help="训练轮次")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--num_workers", type=int, default=4, help="数据加载器的工作线程数")
    parser.add_argument("--save_steps", type=int, default=0, help="多少步保存一次检查点，0表示不保存")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--fp16", action="store_true", default=False, help="是否使用半精度(FP16)训练")
    parser.add_argument("--emotion_loss_weight", type=float, default=0.5, help="情感分类损失的权重") # 新增参数

    # 其他参数
    parser.add_argument("--proxy", type=str, default="http://127.0.0.1:7890", help="HTTP代理URL")
    parser.add_argument("--no-proxy", action="store_true", help="是否禁用代理")
    
    args = parser.parse_args()
    
    # 如果指定了禁用代理，则设置代理为None
    if args.no_proxy:
        args.proxy = None
        logger.info("已禁用代理")
    
    # 如果指定了不冻结BLIP，则关闭冻结
    if getattr(args, 'no_freeze_blip', False):
        args.freeze_blip = False
        logger.info("已禁用BLIP模型冻结，所有参数将参与训练")
    
    # 检查CUDA是否可用，如果不可用则强制使用CPU
    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"
        logger.warning("警告: CUDA不可用，将使用CPU进行训练。")
    
    # 检查是否可以使用FP16
    if args.fp16:
        if not torch.cuda.is_available():
            logger.warning("警告: GPU不可用，无法使用半精度训练。已自动禁用FP16。")
            args.fp16 = False
        elif args.device != "cuda":
            logger.warning("警告: 选择了非CUDA设备，无法使用半精度训练。已自动禁用FP16。")
            args.fp16 = False
    
    # 开始训练
    train(args)

if __name__ == "__main__":
    main() 