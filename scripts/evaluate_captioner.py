import os
import torch
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from tqdm import tqdm
from emotion_enhanced_blip.models.emotion_caption_model import EmotionEnhancedBlipForCaption
# 导入正确的 collate_fn 和数据集类
from emotion_enhanced_blip.data.newyorker_dataset import NewYorkerCaptionDataset, optimized_collate_fn
from transformers import BlipProcessor
# import evaluate  # 不再主要使用 evaluate，改用 pycocoevalcap
from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap
import json
import tempfile
from emotion_enhanced_blip.utils.emotion_utils import EMOTION_CATEGORIES  # 如有需要
import argparse # 添加 argparse 用于命令行参数

# 确保已安装 pycocotools 和 pycocoevalcap
# pip install pycocotools pycocoevalcap
# 对于 Windows 用户，安装 pycocotools 可能需要 C++ Build Tools:
# https://visualstudio.microsoft.com/visual-cpp-build-tools/
# 或者尝试: pip install pycocotools-windows

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

def evaluate_model(model, dataloader, device, processor, max_samples=None, output_path=None):
    """使用 pycocoevalcap 评估模型性能 (BLEU, ROUGE, CIDEr, METEOR, SPICE) 并进行情感评估"""
    all_preds = []
    all_image_ids = [] # 存储图像ID以匹配预测和标签
    all_reference_captions = [] # 存储所有参考描述
    all_emotion_labels = [] # 存储真实情感标签
    sample_count = 0

    print("开始生成预测...")
    for batch in tqdm(dataloader, desc="生成描述"):
        # 假设 dataloader 返回的 batch 包含 'ids', 'reference_captions', 和 'emotion_indices'
        if "ids" not in batch or "reference_captions" not in batch or "emotion_indices" not in batch:
            print("错误：Dataloader 未提供 'ids', 'reference_captions' 或 'emotion_indices'。无法进行评估。")
            print("请修改 NewYorkerCaptionDataset 或 optimized_collate_fn 以返回这些字段。")
            return # 或者引发错误 sys.exit(1)

        pixel_values = batch["pixel_values"].to(device)
        emotion_indices = batch["emotion_indices"].to(device)
        confidence_values = batch["confidence_values"].to(device)
        image_ids = batch["ids"] # 获取图像 ID
        reference_captions_batch = batch["reference_captions"] # 获取参考描述列表
        emotion_labels_batch = batch["emotion_indices"] # 获取真实情感标签 (索引形式)

        # 生成文本
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

        all_preds.extend(preds)
        all_image_ids.extend(image_ids) # 收集图像 ID
        all_reference_captions.extend(reference_captions_batch) # 收集所有参考描述
        all_emotion_labels.extend(emotion_labels_batch.cpu().tolist()) # 收集真实情感标签 (转为列表)


        sample_count += len(preds)
        if max_samples is not None and sample_count >= max_samples:
            print(f"\n达到最大评估样本数 {max_samples}，停止评估。")
            break

    if not all_preds or not all_image_ids or not all_reference_captions or not all_emotion_labels:
        print("没有生成任何预测、图像ID、参考描述或情感标签，无法计算指标。")
        return
    if len(all_preds) != len(all_image_ids) or len(all_preds) != len(all_reference_captions) or len(all_preds) != len(all_emotion_labels):
        print(f"错误：预测({len(all_preds)})、图像ID({len(all_image_ids)})、参考描述({len(all_reference_captions)})和情感标签({len(all_emotion_labels)})的数量不匹配。")
        return

    print(f"\n共生成 {len(all_preds)} 个描述，准备使用 pycocoevalcap 进行评估...")

    # 准备 pycocoevalcap 需要的格式
    # gts (Ground Truth Sets): {img_id: [{'caption': ref1}, {'caption': ref2}, ...]}
    # res (Results): [{'image_id': img_id, 'caption': pred}]
    gts_dict = {}
    res_list = []
    annotation_id_counter = 0
    img_id_map = {} # 创建从原始 ID 到临时整数 ID 的映射
    temp_int_id_counter = 0 # 临时整数 ID 计数器

    for i, img_id_raw in enumerate(all_image_ids):
        # 确保 image_id 是 JSON 兼容的类型 (int or str)
        if isinstance(img_id_raw, torch.Tensor):
            img_id = img_id_raw.item() # 假设是单个元素的 Tensor
        else:
            img_id = img_id_raw
        # 确保是 int 或 str
        img_id = str(img_id) if not isinstance(img_id, (int, str)) else img_id

        # 获取或创建临时整数 ID
        if img_id not in img_id_map:
            img_id_map[img_id] = temp_int_id_counter
            temp_int_id_counter += 1
        img_id_int = img_id_map[img_id] # 使用映射后的整数 ID

        pred = all_preds[i]
        reference_captions = all_reference_captions[i] # 获取当前样本的所有参考描述

        # 添加到 res
        res_list.append({'image_id': img_id_int, 'caption': pred})

        # 添加到 gts (处理一个图像可能有多个参考的情况)
        if img_id_int not in gts_dict:
            gts_dict[img_id_int] = []

        # 为每个参考描述创建一个标注条目
        for ref_caption in reference_captions:
             gts_dict[img_id_int].append({
                'image_id': img_id_int,
                'id': annotation_id_counter, # 唯一的注释 ID
                'caption': ref_caption
            })
             annotation_id_counter += 1


    # 转换 gts_dict 为 COCO API 需要的格式
    coco_gts_format = {
        'annotations': [ann for anns in gts_dict.values() for ann in anns],
        'images': [{'id': img_id} for img_id in gts_dict.keys()],
        'type': 'captions', # 指定类型
        'info': {},        # 可选信息
        'licenses': []     # 可选信息
    }


    # 使用临时文件进行评估
    gts_path = None
    res_path = None
    pycoco_eval_results = {}
    try:
        # 创建临时文件保存 gts 和 res
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json', encoding='utf-8') as tmp_gts_file:
            json.dump(coco_gts_format, tmp_gts_file, ensure_ascii=False)
            gts_path = tmp_gts_file.name

        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json', encoding='utf-8') as tmp_res_file:
            json.dump(res_list, tmp_res_file, ensure_ascii=False)
            res_path = tmp_res_file.name

        # 初始化 COCO API
        coco = COCO(gts_path)
        cocoRes = coco.loadRes(res_path)

        # 创建评估器
        cocoEval = COCOEvalCap(coco, cocoRes)

        # 仅评估我们有结果的图像
        cocoEval.params['image_id'] = cocoRes.getImgIds()

        # 执行评估
        cocoEval.evaluate()

        print("\n--- 评估结果 (pycocoevalcap) ---")
        for metric, score in cocoEval.eval.items():
            # 将指标名称转换为大写并左对齐
            print(f'{metric.upper():<7}: {score:.4f}')
            pycoco_eval_results[metric] = score # 存储结果
        print("---------------------------------\n")

    except ImportError:
         print("\n错误：无法导入 pycocotools 或 pycocoevalcap。")
         print("请安装：pip install pycocotools pycocoevalcap")
         print("对于 Windows，可能需要 C++ Build Tools 或 pip install pycocotools-windows")
    except FileNotFoundError as e:
        print(f"\n错误：找不到临时文件或依赖项: {e}")
    except Exception as e:
        print(f"\n使用 pycocoevalcap 计算评估指标时出错: {e}")
        print("请检查数据格式和库安装情况。")
    finally:
        # 清理临时文件
        if gts_path and os.path.exists(gts_path):
            try:
                os.remove(gts_path)
            except Exception as e_clean:
                print(f"警告：无法删除临时 gts 文件 {gts_path}: {e_clean}")
        if res_path and os.path.exists(res_path):
             try:
                os.remove(res_path)
             except Exception as e_clean:
                print(f"警告：无法删除临时 res 文件 {res_path}: {e_clean}")


    # --- 情感评估 ---
    print("--- 情感评估 ---")
    # 实现简单的情感匹配逻辑
    # 遍历每个样本，检查生成的描述是否包含真实情感标签对应的关键词
    emotion_match_count = 0
    total_samples_with_emotion = 0

    for i, pred in enumerate(all_preds):
        # 获取真实情感标签 (索引形式)
        true_emotion_indices = [idx for idx in all_emotion_labels[i] if idx != -1] # 过滤掉填充值 -1

        if true_emotion_indices:
            total_samples_with_emotion += 1
            # 将真实情感索引转换为情感名称
            true_emotion_names = [EMOTION_CATEGORIES[idx] for idx in true_emotion_indices if idx < len(EMOTION_CATEGORIES)]

            # 检查生成的描述是否包含任何真实情感关键词
            # 简单匹配：检查情感名称（英文）是否在生成描述中
            matched = False
            for emotion_name in true_emotion_names:
                if emotion_name.lower() in pred.lower():
                    matched = True
                    break
            if matched:
                emotion_match_count += 1

    emotion_accuracy = (emotion_match_count / total_samples_with_emotion) if total_samples_with_emotion > 0 else 0
    print(f"情感匹配准确率: {emotion_accuracy:.4f} ({emotion_match_count}/{total_samples_with_emotion})")
    print("----------------\n")

    # 打印部分样例 (确保索引不越界)
    num_samples_to_print = min(5, len(all_preds))
    print("--- 部分生成样例 ---")
    for i in range(num_samples_to_print):
        print(f"样例 {i+1}:")
        # 获取真实参考描述和情感标签
        true_captions = all_reference_captions[i]
        true_emotion_indices = [idx for idx in all_emotion_labels[i] if idx != -1]
        true_emotion_names = [EMOTION_CATEGORIES[idx] for idx in true_emotion_indices if idx < len(EMOTION_CATEGORIES)]

        print(f"  真实描述: {true_captions}")
        print(f"  真实情感: {', '.join(true_emotion_names)}")
        print(f"  生成描述: {all_preds[i]}")
        print("-" * 20)
    print("--------------------\n")

    # 保存评估结果
    if output_path:
        evaluation_results = {
            "pycoco_metrics": pycoco_eval_results,
            "emotion_accuracy": emotion_accuracy,
            "emotion_match_count": emotion_match_count,
            "total_samples_with_emotion": total_samples_with_emotion,
            "generated_captions": all_preds,
            "image_ids": all_image_ids,
            "reference_captions": all_reference_captions,
            "emotion_labels_indices": all_emotion_labels # 保存原始索引
        }
        try:
            output_dir = os.path.dirname(output_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(evaluation_results, f, ensure_ascii=False, indent=4)
            print(f"评估结果已保存到 {output_path}")
        except Exception as e:
            print(f"保存评估结果到 {output_path} 时出错: {e}")


def main():
    parser = argparse.ArgumentParser(description="评估 Emotion Enhanced BLIP 模型")
    parser.add_argument("--model_path", type=str, default="scripts/output/caption_model/best_model.pth", help="训练好的模型权重路径")
    parser.add_argument("--annotations_path", type=str, default="annotations/preprocessed_annotations_0_to_130_validation_with_titles.json", help="预处理后的标注文件路径")
    parser.add_argument("--blip_model_name", type=str, default="Salesforce/blip-image-captioning-base", help="基础 BLIP 模型名称")
    parser.add_argument("--batch_size", type=int, default=16, help="评估时的批次大小")
    parser.add_argument("--max_samples", type=int, default=None, help="最多评估多少个样本 (默认评估全部)")
    parser.add_argument("--proxy", type=str, default=None, help="下载模型时使用的代理 (例如 'http://localhost:7890')")
    parser.add_argument("--output_path", type=str, default="scripts/output/evaluation_results.json", help="评估结果保存路径") # 添加输出路径参数
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")

    # 确保路径相对于项目根目录是正确的
    # 如果脚本在 scripts/ 目录下运行，需要调整路径
    # 假设脚本从项目根目录运行 (例如 python scripts/evaluate_captioner.py)
    model_path = args.model_path
    annotations_path = args.annotations_path
    output_path = args.output_path # 获取输出路径

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
    # 修改调用以传递 output_path
    evaluate_model(model, dataloader, device, processor, max_samples=args.max_samples, output_path=output_path)
    print("评估完成。")


if __name__ == "__main__":
    main() # 调用 main 函数
