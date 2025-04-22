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

def calculate_exact_match(predictions: list[str], references: list[str]) -> float:
    """计算精确匹配率"""
    if len(predictions) != len(references):
        print("警告：预测和参考的数量不匹配，无法计算精确匹配率。")
        return 0.0
    if not predictions:
        return 0.0 # 或根据需要处理空列表的情况

    match_count = 0
    for pred, ref in zip(predictions, references):
        # 简单的大小写不敏感比较
        if pred.strip().lower() == ref.strip().lower():
            match_count += 1
    return match_count / len(predictions)

def evaluate_model(model, dataloader, device, processor, max_samples=None, output_path=None, max_title_length=20): # 添加 max_title_length 参数
    """评估模型生成标题的性能，使用 BLEU, ROUGE 和精确匹配"""
    all_preds = [] # 存储生成的标题
    all_image_ids = [] # 存储图像ID
    all_ground_truth_titles = [] # 存储真实标题
    all_emotion_labels = [] # 存储真实情感标签 (保留用于情感评估)
    sample_count = 0

    print("开始生成预测标题...")
    for batch in tqdm(dataloader, desc="生成标题"):
        # 检查 Dataloader 是否提供了必要的字段
        required_keys = ["ids", "ground_truth_titles", "emotion_indices", "pixel_values", "confidence_values"]
        if not all(key in batch for key in required_keys):
            missing_keys = [key for key in required_keys if key not in batch]
            print(f"错误：Dataloader 未提供必需的键: {missing_keys}。请检查 data/newyorker_dataset.py。")
            return # 或者引发错误 sys.exit(1)

        pixel_values = batch["pixel_values"].to(device)
        emotion_indices = batch["emotion_indices"].to(device)
        confidence_values = batch["confidence_values"].to(device)
        image_ids = batch["ids"] # 获取图像 ID
        ground_truth_titles_batch = batch["ground_truth_titles"] # 获取真实标题列表
        emotion_labels_batch = batch["emotion_indices"] # 获取真实情感标签 (索引形式)

        # 生成标题
        with torch.no_grad():
            # 调整生成参数以生成标题
            generated_ids = model.generate(
                pixel_values=pixel_values,
                # emotion_indices=emotion_indices, # 移除情感输入，强制动态提取
                # confidence_values=confidence_values, # 移除置信度输入
                max_length=max_title_length, # 使用参数控制最大标题长度
                num_beams=4, # 可以尝试使用 beam search
                early_stopping=True # 提前停止生成
                # 可以添加其他适合短文本生成的参数
            )
        # 解码
        preds = processor.batch_decode(generated_ids, skip_special_tokens=True)

        all_preds.extend(preds)
        all_image_ids.extend(image_ids) # 收集图像 ID
        all_ground_truth_titles.extend(ground_truth_titles_batch) # 收集真实标题
        all_emotion_labels.extend(emotion_labels_batch.cpu().tolist()) # 收集真实情感标签 (转为列表)


        sample_count += len(preds)
        if max_samples is not None and sample_count >= max_samples:
            print(f"\n达到最大评估样本数 {max_samples}，停止评估。")
            break

    if not all_preds or not all_image_ids or not all_ground_truth_titles or not all_emotion_labels:
        print("没有生成任何预测、图像ID、真实标题或情感标签，无法计算指标。")
        return
    # 检查长度是否一致
    if not (len(all_preds) == len(all_image_ids) == len(all_ground_truth_titles) == len(all_emotion_labels)):
        print(f"错误：预测({len(all_preds)})、图像ID({len(all_image_ids)})、真实标题({len(all_ground_truth_titles)})和情感标签({len(all_emotion_labels)})的数量不匹配。")
        return

    print(f"\n共生成 {len(all_preds)} 个标题，准备进行评估...")

    # --- 标题评估 (pycocoevalcap for BLEU, ROUGE) ---
    # pycocoevalcap 设计用于评估描述，但 BLEU 和 ROUGE 对于标题也有一定参考价值。
    # 注意：CIDEr, METEOR, SPICE 可能不适用于短标题评估。
    print("--- 标题评估 (BLEU, ROUGE using pycocoevalcap) ---")
    # 准备 pycocoevalcap 需要的格式
    # gts (Ground Truth Sets): {img_id: [{'caption': gt_title}]}
    # res (Results): [{'image_id': img_id, 'caption': pred_title}]
    gts_dict = {}
    res_list = []
    annotation_id_counter = 0
    img_id_map = {} # 创建从原始 ID 到临时整数 ID 的映射
    temp_int_id_counter = 0 # 临时整数 ID 计数器

    for i, img_id_raw in enumerate(all_image_ids):
        if isinstance(img_id_raw, torch.Tensor): img_id = img_id_raw.item()
        else: img_id = img_id_raw
        img_id = str(img_id) if not isinstance(img_id, (int, str)) else img_id

        if img_id not in img_id_map:
            img_id_map[img_id] = temp_int_id_counter
            temp_int_id_counter += 1
        img_id_int = img_id_map[img_id]

        pred_title = all_preds[i]
        gt_title = all_ground_truth_titles[i] # 获取当前样本的真实标题

        # 添加到 res
        res_list.append({'image_id': img_id_int, 'caption': pred_title})

        # 添加到 gts (每个图像只有一个真实标题)
        if img_id_int not in gts_dict:
            gts_dict[img_id_int] = []

        # 为真实标题创建一个标注条目
        # 注意：即使只有一个参考，pycocoevalcap 也需要列表格式
        gts_dict[img_id_int].append({
            'image_id': img_id_int,
            'id': annotation_id_counter, # 唯一的注释 ID
            'caption': gt_title # 使用真实标题
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
        print("初始化COCO API完成")
        # 创建评估器
        cocoEval = COCOEvalCap(coco, cocoRes)
        print("创建评估器完成")
        # 仅评估我们有结果的图像
        cocoEval.params['image_id'] = cocoRes.getImgIds()

        # 执行评估
        print("开始评估")
        cocoEval.evaluate()

        print("\n--- 评估结果 (BLEU, ROUGE using pycocoevalcap) ---")
        for metric, score in cocoEval.eval.items():
            print(f'{metric.upper():<7}: {score:.4f}')
            pycoco_eval_results[metric] = score
        print("--------------------------------------------------\n")

    except ImportError:
         print("\n错误：无法导入 pycocotools 或 pycocoevalcap。")
         # ... [保留错误处理和清理逻辑] ...
    except FileNotFoundError as e:
        print(f"\n错误：找不到临时文件或依赖项: {e}")
    except Exception as e:
        print(f"\n使用 pycocoevalcap 计算 BLEU/ROUGE 时出错: {e}")
    finally:
        if gts_path and os.path.exists(gts_path):
            try: os.remove(gts_path)
            except Exception as e_clean: print(f"警告：无法删除临时 gts 文件 {gts_path}: {e_clean}")
        if res_path and os.path.exists(res_path):
             try: os.remove(res_path)
             except Exception as e_clean: print(f"警告：无法删除临时 res 文件 {res_path}: {e_clean}")

    # --- 标题评估 (精确匹配) ---
    print("--- 标题评估 (精确匹配) ---")
    exact_match_score = calculate_exact_match(all_preds, all_ground_truth_titles)
    print(f"精确匹配率 (Exact Match): {exact_match_score:.4f}")
    print("---------------------------\n")

    # --- 情感评估 (基于生成的标题) ---
    # 注意：由于动态情感提取的结果目前不易直接获取用于评估，
    # 此部分已被注释掉。
    # print("--- 情感评估 (基于生成的标题) ---")
    # emotion_match_count = 0
    # total_samples_with_emotion = 0
    # for i, pred_title in enumerate(all_preds): # 使用生成的标题进行评估
    #     true_emotion_indices = [idx for idx in all_emotion_labels[i] if idx != -1]
    #     if true_emotion_indices:
    #         total_samples_with_emotion += 1
    #         true_emotion_names = [EMOTION_CATEGORIES[idx] for idx in true_emotion_indices if idx < len(EMOTION_CATEGORIES)]
    #         matched = False
    #         for emotion_name in true_emotion_names:
    #             # 检查情感名称是否在 *生成的标题* 中
    #             if emotion_name.lower() in pred_title.lower():
    #                 matched = True
    #                 break
    #         if matched:
    #             emotion_match_count += 1
    #
    # emotion_accuracy = (emotion_match_count / total_samples_with_emotion) if total_samples_with_emotion > 0 else 0
    # print(f"情感匹配准确率 (基于标题): {emotion_accuracy:.4f} ({emotion_match_count}/{total_samples_with_emotion})")
    # print("---------------------------------\n")

    # 打印部分样例
    num_samples_to_print = min(5, len(all_preds))
    print("--- 部分生成样例 ---")
    for i in range(num_samples_to_print):
        print(f"样例 {i+1}:")
        true_title = all_ground_truth_titles[i]
        true_emotion_indices = [idx for idx in all_emotion_labels[i] if idx != -1]
        true_emotion_names = [EMOTION_CATEGORIES[idx] for idx in true_emotion_indices if idx < len(EMOTION_CATEGORIES)]

        print(f"  真实标题: {true_title}") # 显示真实标题
        print(f"  真实情感: {', '.join(true_emotion_names)}")
        print(f"  生成标题: {all_preds[i]}") # 显示生成标题
        print("-" * 20)
    print("--------------------\n")

    # 保存评估结果
    if output_path:
        evaluation_results = {
            "title_metrics": { # 将指标分组
                 **pycoco_eval_results, # BLEU, ROUGE etc. from pycocoevalcap
                "ExactMatch": exact_match_score # 添加精确匹配
            },
            # "emotion_accuracy_on_title": emotion_accuracy, # 已注释掉
            # "emotion_match_count_on_title": emotion_match_count, # 已注释掉
            # "total_samples_with_emotion": total_samples_with_emotion, # 已注释掉
            "generated_titles": all_preds, # 更新字段名称
            "image_ids": all_image_ids,
            "ground_truth_titles": all_ground_truth_titles, # 更新字段名称
            "emotion_labels_indices": all_emotion_labels # 保留原始标签以供参考
        }
        try:
            output_dir = os.path.dirname(output_path)
            if output_dir and not os.path.exists(output_dir): os.makedirs(output_dir)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(evaluation_results, f, ensure_ascii=False, indent=4)
            print(f"评估结果已保存到 {output_path}")
        except Exception as e:
            print(f"保存评估结果到 {output_path} 时出错: {e}")


def main():
    parser = argparse.ArgumentParser(description="评估 Emotion Enhanced BLIP 模型生成标题的能力") # 更新描述
    parser.add_argument("--model_path", type=str, default="output/caption_model/best_model.pth", help="训练好的模型权重路径")
    parser.add_argument("--annotations_path", type=str, default="annotations/preprocessed_annotations_0_to_129_validation_with_titles.json", help="包含真实标题的预处理标注文件路径") # 更新帮助文本
    parser.add_argument("--blip_model_name", type=str, default="Salesforce/blip-image-captioning-base", help="基础 BLIP 模型名称")
    parser.add_argument("--batch_size", type=int, default=16, help="评估时的批次大小")
    parser.add_argument("--max_samples", type=int, default=None, help="最多评估多少个样本 (默认评估全部)")
    parser.add_argument("--max_title_length", type=int, default=20, help="生成标题的最大长度") # 添加新参数
    parser.add_argument("--proxy", type=str, default=None, help="下载模型时使用的代理 (例如 'http://localhost:7890')")
    parser.add_argument("--output_path", type=str, default="scripts/output/title_evaluation_results.json", help="标题评估结果保存路径") # 更新默认输出路径和帮助文本
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")

    model_path = args.model_path
    annotations_path = args.annotations_path
    output_path = args.output_path

    if not os.path.exists(model_path):
        print(f"错误: 模型文件未找到于 {model_path}")
        sys.exit(1)
    if not os.path.exists(annotations_path):
        print(f"错误: 标注文件未找到于 {annotations_path}")
        sys.exit(1)


    # 加载数据集
    print("加载数据集中...")
    try:
        # 确保传递正确的 blip_model_name 和 proxy
        dataset = NewYorkerCaptionDataset(
            split="validation", # 或 "test"
            preprocessed_annotations_path=annotations_path,
            blip_model_name=args.blip_model_name,
            # max_target_length 在评估标题时可能不直接使用，但保留以兼容类定义
            max_target_length=args.max_title_length * 2, # 可以设置一个稍大的值
            proxy=args.proxy
        )
    except Exception as e:
        print(f"加载数据集时出错: {e}")
        sys.exit(1)

    # 使用导入的 optimized_collate_fn
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=optimized_collate_fn
    )
    print("数据集加载完毕。")

    # 加载模型
    print("加载模型中...")
    model = load_model(model_path, device, args.blip_model_name, args.proxy)
    # 确保 processor 从 dataset 正确获取
    if hasattr(dataset, 'processor') and dataset.processor:
         processor = dataset.processor
    else:
        print("错误：无法从数据集中获取 processor。")
        sys.exit(1)
    print("模型加载完毕。")

    # 评估
    print("开始评估标题生成...")
    # 传递 max_title_length 参数
    evaluate_model(model, dataloader, device, processor,
                   max_samples=args.max_samples,
                   output_path=output_path,
                   max_title_length=args.max_title_length)
    print("评估完成。")


if __name__ == "__main__":
    main()
