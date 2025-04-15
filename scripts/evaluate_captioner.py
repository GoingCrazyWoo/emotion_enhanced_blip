import os
import torch
from tqdm import tqdm
from models.emotion_caption_model import EmotionEnhancedBlipForCaption
from data.newyorker_dataset import NewYorkerCaptionDataset
from transformers import BlipProcessor
import evaluate  # huggingface evaluate库，支持BLEU/ROUGE/CIDEr等
from utils.emotion_utils import EMOTION_CATEGORIES  # 如有需要

def load_model(model_path, device, blip_model_name, proxy=None):
    model = EmotionEnhancedBlipForCaption(
        blip_model_name=blip_model_name,
        freeze_blip=False,
        proxy=proxy
    )
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    return model

def evaluate_model(model, dataloader, device, processor):
    metric_bleu = evaluate.load("bleu")
    metric_rouge = evaluate.load("rouge")
    # 你也可以加载cider、meteor等

    all_preds = []
    all_labels = []

    for batch in tqdm(dataloader, desc="评估中"):
        pixel_values = batch["pixel_values"].to(device)
        emotion_indices = batch["emotion_indices"].to(device)
        confidence_values = batch["confidence_values"].to(device)
        # 生成文本
        with torch.no_grad():
            generated_ids = model.generate(
                pixel_values=pixel_values,
                emotion_indices=emotion_indices,
                confidence_values=confidence_values,
                max_length=50
            )
        # 解码
        preds = processor.batch_decode(generated_ids, skip_special_tokens=True)
        labels = processor.batch_decode(batch["labels"], skip_special_tokens=True)
        all_preds.extend(preds)
        all_labels.extend(labels)

    # 计算指标
    bleu = metric_bleu.compute(predictions=all_preds, references=[[l] for l in all_labels])
    rouge = metric_rouge.compute(predictions=all_preds, references=all_labels)
    print("BLEU:", bleu)
    print("ROUGE:", rouge)
    # 可扩展更多指标

    # 打印部分样例
    for i in range(5):
        print(f"真实: {all_labels[i]}")
        print(f"生成: {all_preds[i]}")
        print("-" * 30)

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_path = "output/caption_model/best_model.pth"
    blip_model_name = "Salesforce/blip-image-captioning-base"
    annotations_path = "../annotations/preprocessed_annotations_with_titles.json"

    # 加载数据集
    dataset = NewYorkerCaptionDataset(
        split="validation",
        preprocessed_annotations_path=annotations_path,
        blip_model_name=blip_model_name,
        max_target_length=100
    )
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=False, collate_fn=collate_fn)

    # 加载模型
    model = load_model(model_path, device, blip_model_name)
    processor = dataset.processor

    # 评估
    evaluate_model(model, dataloader, device, processor)
