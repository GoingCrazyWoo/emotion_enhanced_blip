import os
import torch
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from tqdm import tqdm
from emotion_enhanced_blip.models.emotion_caption_model import EmotionEnhancedBlipForCaption
# 导入正确的 collate_fn 和数据集类
from emotion_enhanced_blip.data.newyorker_dataset import NewYorkerCaptionDataset, optimized_collate_fn
from transformers import BlipProcessor
import evaluate  # huggingface evaluate库，支持BLEU/ROUGE/CIDEr等
from emotion_enhanced_blip.utils.emotion_utils import EMOTION_CATEGORIES  # 如有需要
import argparse # 添加 argparse 用于命令行参数

def load_model(model_path, device, blip_model_name, proxy=None):
    """加载训练好的模型"""
    model = EmotionEnhancedBlipForCaption(
        blip_model_name=blip_model_name,
        freeze_blip=False, # 评估时通常不需要冻结
        proxy=proxy
    )
    try:
        state_dict = torch.load(model_path, map_location=device)
        # 处理可能的 DataParallel 或 DDP 包装
        if 'module.' in list(state_dict.keys())[0]:
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict, strict=False) # 使用 strict=False 允许部分加载
        print(f"模型权重从 {model_path} 加载成功。")
    except FileNotFoundError:
        print(f"错误：找不到模型权重文件 {model_path}。请检查路径。")
        sys.exit(1)
    except Exception as e:
        print(f"加载模型权重时出错: {e}")
        sys.exit(1)

    model.to(device)
    model.eval()
    return model

def evaluate_model(model, dataloader, device, processor, max_samples=None):
    """评估模型性能"""
    metric_bleu = evaluate.load("bleu")
    metric_rouge = evaluate.load("rouge")
    # 你也可以加载cider、meteor等

    all_preds = []
    all_labels = []
    sample_count = 0

    for batch in tqdm(dataloader, desc="评估中"):
        pixel_values = batch["pixel_values"].to(device)
        emotion_indices = batch["emotion_indices"].to(device)
        confidence_values = batch["confidence_values"].to(device)
        # 真实标签（用于比较）
        labels_ids = batch["labels"].to(device) # 确保标签也在设备上

        # 生成文本 (使用我们修改后的 model.generate)
        with torch.no_grad():
            generated_ids = model.generate(
                pixel_values=pixel_values,
                emotion_indices=emotion_indices,
                confidence_values=confidence_values,
                max_length=50 # 可以通过参数调整
                # 其他 generate_kwargs 如 num_beams 等可以在这里传递
            )
        # 解码
        preds = processor.batch_decode(generated_ids, skip_special_tokens=True)
        labels = processor.batch_decode(labels_ids, skip_special_tokens=True) # 解码真实标签

        all_preds.extend(preds)
        all_labels.extend(labels)

        sample_count += len(preds)
        if max_samples is not None and sample_count >= max_samples:
            print(f"\n达到最大评估样本数 {max_samples}，停止评估。")
            break

    if not all_preds or not all_labels:
        print("没有生成任何预测或标签，无法计算指标。")
        return

    # 计算指标
    # Hugging Face evaluate 的 compute 需要 references 是 list of lists
    references = [[l] for l in all_labels]
    try:
        bleu = metric_bleu.compute(predictions=all_preds, references=references)
        rouge = metric_rouge.compute(predictions=all_preds, references=all_labels) # rouge 不需要 list of lists
        print("\n--- 评估结果 ---")
        print("BLEU:", bleu)
        print("ROUGE:", rouge)
        print("----------------\n")
    except Exception as e:
        print(f"计算评估指标时出错: {e}")


    # 打印部分样例 (确保索引不越界)
    num_samples_to_print = min(5, len(all_labels))
    print("--- 部分生成样例 ---")
    for i in range(num_samples_to_print):
        print(f"样例 {i+1}:")
        print(f"  真实: {all_labels[i]}")
        print(f"  生成: {all_preds[i]}")
        print("-" * 20)
    print("--------------------\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="评估 Emotion Enhanced BLIP 模型")
    parser.add_argument("--model_path", type=str, default="snapshots/best_model.pth", help="训练好的模型权重路径")
    parser.add_argument("--annotations_path", type=str, default="annotations/preprocessed_annotations_with_titles.json", help="预处理后的标注文件路径")
    parser.add_argument("--blip_model_name", type=str, default="Salesforce/blip-image-captioning-base", help="基础 BLIP 模型名称")
    parser.add_argument("--batch_size", type=int, default=16, help="评估时的批次大小")
    parser.add_argument("--max_samples", type=int, default=None, help="最多评估多少个样本 (默认评估全部)")
    parser.add_argument("--proxy", type=str, default=None, help="下载模型时使用的代理 (例如 'http://localhost:7890')")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")

    # 确保路径相对于项目根目录是正确的
    # 如果脚本在 scripts/ 目录下运行，需要调整路径
    # 假设脚本从项目根目录运行 (例如 python scripts/evaluate_captioner.py)
    model_path = args.model_path
    annotations_path = args.annotations_path

    # 检查文件是否存在
    if not os.path.exists(model_path):
        print(f"错误: 模型文件未找到于 {model_path}")
        sys.exit(1)
    if not os.path.exists(annotations_path):
        print(f"错误: 标注文件未找到于 {annotations_path}")
        sys.exit(1)


    # 加载数据集
    print("加载数据集中...")
    try:
        dataset = NewYorkerCaptionDataset(
            split="validation", # 通常在验证集上评估
            preprocessed_annotations_path=annotations_path,
            blip_model_name=args.blip_model_name,
            max_target_length=100, # 这个长度主要用于训练，评估时可以忽略
            proxy=args.proxy
        )
    except Exception as e:
        print(f"加载数据集时出错: {e}")
        sys.exit(1)

    # 使用导入的 optimized_collate_fn
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False, # 评估时不需要打乱
        collate_fn=optimized_collate_fn # 使用正确的 collate 函数
    )
    print("数据集加载完毕。")

    # 加载模型
    print("加载模型中...")
    model = load_model(model_path, device, args.blip_model_name, args.proxy)
    processor = dataset.processor # 从数据集中获取 processor
    print("模型加载完毕。")

    # 评估
    print("开始评估...")
    evaluate_model(model, dataloader, device, processor, max_samples=args.max_samples)
    print("评估完成。")
