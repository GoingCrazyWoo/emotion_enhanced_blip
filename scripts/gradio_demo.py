#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Gradio 演示脚本：上传图片，使用情感增强BLIP模型生成标题
"""

import os
import sys
import torch
from PIL import Image
import gradio as gr

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from emotion_enhanced_blip.models.emotion_caption_model import EmotionEnhancedBlipForCaption
from emotion_enhanced_blip.utils.emotion_utils import EMOTION_CATEGORIES_ZH

# 默认参数
DEFAULT_MODEL_PATH = "../output/caption_model/best_model.pth"
DEFAULT_BLIP_MODEL_NAME = "Salesforce/blip-image-captioning-base"
DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_model(model_path=DEFAULT_MODEL_PATH, blip_model_name=DEFAULT_BLIP_MODEL_NAME, device=DEFAULT_DEVICE):
    # 加载模型
    model = EmotionEnhancedBlipForCaption(
        blip_model_name=blip_model_name,
        freeze_blip=False,
        proxy=None
    )
    # 加载权重
    state_dict = torch.load(model_path, map_location=device)
    if 'module.' in list(state_dict.keys())[0]:
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    return model

# 只加载一次模型
model = load_model()

def predict(image):
    """
    image: PIL.Image
    emotion_type: str, 选择的情感类型
    """
    device = DEFAULT_DEVICE
    emotion_indices = None
    confidence_values = None

    # 图像预处理
    inputs = model.processor(images=image, return_tensors="pt")
    pixel_values = inputs.pixel_values.to(device)

    with torch.no_grad():
        generated_ids = model.generate(
            pixel_values=pixel_values,
            emotion_indices=emotion_indices,
            confidence_values=confidence_values,
            max_length=50,
            num_beams=5
        )
        captions = model.processor.batch_decode(generated_ids, skip_special_tokens=True)
        return captions[0]

# Gradio 界面
demo = gr.Interface(
    fn=predict,
    inputs=[
        gr.Image(type="pil", label="上传图片"),
    ],
    outputs=gr.Textbox(label="生成标题"),
    title="情感增强BLIP图像标题生成演示",
)

if __name__ == "__main__":
    demo.launch()