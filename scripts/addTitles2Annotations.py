import json
import argparse
import logging
import os
from typing import Dict, List, Any

# --- 基本日志设置 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def add_titles(
        annotations_path: str,
        titles_path: str,
        mapping_path: str,  # 假设这个文件将 image_id 映射到 instance_id
        output_path: str
) -> None:
    """
    使用映射文件将标题从标题文件添加到注释文件。

    参数：
        annotations_path: 预处理注释 JSON 文件的路径。
        titles_path: 包含标题和 image_id 的 JSON 文件路径。
        mapping_path: 映射 image_id 到 instance_id 的 JSON 文件路径。
        output_path: 保存更新后注释 JSON 文件的路径。
    """
    logger.info(f"正在加载注释文件: {annotations_path}")
    try:
        with open(annotations_path, 'r', encoding='utf-8') as f:
            annotations_data: Dict[str, Dict[str, Any]] = json.load(f)
    except FileNotFoundError:
        logger.error(f"找不到注释文件: {annotations_path}")
        return
    except json.JSONDecodeError:
        logger.error(f"解码注释文件 JSON 时出错: {annotations_path}")
        return

    logger.info(f"正在加载标题文件: {titles_path}")
    try:
        with open(titles_path, 'r', encoding='utf-8') as f:
            titles_data: List[Dict[str, Any]] = json.load(f)
    except FileNotFoundError:
        logger.error(f"找不到标题文件: {titles_path}")
        return
    except json.JSONDecodeError:
        logger.error(f"解码标题文件 JSON 时出错: {titles_path}")
        return

    logger.info(f"正在加载映射文件: {mapping_path}")
    try:
        # *** 假设 mapping_path JSON 文件是一个字典列表，形式如 [{'image_id': 0, 'instance_id': 'hex_string'}, ...] ***
        with open(mapping_path, 'r', encoding='utf-8') as f:
            mapping_data: List[Dict[str, Any]] = json.load(f)
    except FileNotFoundError:
        logger.error(f"找不到映射文件: {mapping_path}，无法将标题映射到注释。")
        return
    except json.JSONDecodeError:
        logger.error(f"解码映射文件 JSON 时出错: {mapping_path}")
        return
    except Exception as e:
        logger.error(f"加载映射文件时发生意外错误 {mapping_path}: {e}")
        return

    # --- 准备查找表 ---
    try:
        # 创建标题查找字典：{image_id: title}
        image_id_to_title: Dict[int, str] = {
            item['image_id']: item['generated_title']
            for item in titles_data
            if 'image_id' in item and 'generated_title' in item
        }
        logger.info(f"创建了包含 {len(image_id_to_title)} 条目的标题查找表。")

        # 使用映射文件创建从原始 ID 到 image_id 的映射
        # 假设映射数据是一个字典列表，形式如 [{'image_id': 0, 'original_id': 'hex_string'}, ...]
        instance_id_to_image_id: Dict[str, int] = {}
        duplicates = 0
        processed_mapping_items = 0
        missing_keys_count = 0
        for item in mapping_data:
            # 使用 'original_id' 作为实例 ID，'image_id' 作为图像 ID
            if 'original_id' in item and 'image_id' in item:
                processed_mapping_items += 1
                instance_id = item['original_id']  # <-- 使用 'original_id'
                image_id = item['image_id']
                if instance_id in instance_id_to_image_id:
                    # 如果 original_id 多次出现，记录日志
                    # logger.warning(f"在映射文件中发现重复的 original_id '{instance_id}'，保留首次出现的 image_id ({instance_id_to_image_id[instance_id]}).")
                    duplicates += 1
                else:
                    try:
                        instance_id_to_image_id[instance_id] = int(image_id)
                    except (ValueError, TypeError):
                        logger.warning(f"无法将 image_id '{image_id}' 转换为 int 类型，原始 ID '{instance_id}' 被跳过。")
                        continue  # 如果 image_id 不是有效的整数表示，则跳过此条目
            else:
                # logger.warning(f"映射文件中缺少 'original_id' 或 'image_id'，已跳过：{item}")
                missing_keys_count += 1

        logger.info(f"处理了 {processed_mapping_items} 条映射文件中的条目。")
        if missing_keys_count > 0:
            logger.warning(f"由于缺少 'original_id' 或 'image_id'，跳过了 {missing_keys_count} 条映射文件条目。")
        if duplicates > 0:
            logger.warning(f"在映射文件中发现了 {duplicates} 个重复的 original_id。")
        logger.info(f"创建了 {len(instance_id_to_image_id)} 个唯一的 original_id 到 image_id 的映射。")  # 日志消息更新
        if not instance_id_to_image_id:
            logger.warning("original_id 到 image_id 的映射为空，请检查映射文件的格式和内容。")


    except KeyError as e:
        # 保持通用的错误处理
        logger.error(f"标题或映射数据中缺少预期的键 '{e}'，请检查文件格式。")
        return
    except Exception as e:
        logger.error(f"创建查找表时发生错误: {e}")
        return

    # --- 将标题添加到注释 ---
    updated_count = 0
    missing_mapping_count = 0
    missing_title_count = 0
    for instance_id, annotation in annotations_data.items():
        if instance_id in instance_id_to_image_id:
            image_id = instance_id_to_image_id[instance_id]
            if image_id in image_id_to_title:
                annotation['title'] = image_id_to_title[image_id]
                updated_count += 1
            else:
                logger.warning(f"没有找到 image_id {image_id} (instance_id: {instance_id}) 对应的标题")
                missing_title_count += 1
        else:
            logger.warning(f"没有找到 instance_id/original_id ({instance_id}) 对应的 image_id 映射")  # 日志消息更新
            missing_mapping_count += 1

    logger.info(f"已为 {updated_count} 条注释添加标题。")
    if missing_mapping_count > 0:
        logger.warning(f"无法为 {missing_mapping_count} 个 instance_id/original_id 找到映射。")  # 日志消息更新
    if missing_title_count > 0:
        logger.warning(f"无法为 {missing_title_count} 个已映射的 image_id 找到标题。")

    # --- 保存更新后的注释 ---
    logger.info(f"正在保存更新后的注释到: {output_path}")
    try:
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(annotations_data, f, indent=2, ensure_ascii=False)
        logger.info("注释已成功更新并保存。")
    except IOError as e:
        logger.error(f"保存更新后的注释到 {output_path} 时失败: {e}")
    except Exception as e:
        logger.error(f"保存文件时发生了意外错误: {e}")


def main():
    parser = argparse.ArgumentParser(description="将标题添加到预处理注释中。")
    parser.add_argument(
        "--annotations_path",
        type=str,
        default="../annotations/preprocessed_annotations.json",
        help="输入预处理注释 JSON 文件的路径。"
    )
    parser.add_argument(
        "--titles_path",
        type=str,
        default="../annotations/titles_0_to_2339_title.json",
        help="包含标题和 image_id 的 JSON 文件路径。"
    )
    parser.add_argument(
        "--mapping_path",
        type=str,
        default="../annotations/results_0_to_2339.json",  # 假设这个文件提供了映射
        help="映射 image_id 到 instance_id 的 JSON 文件路径。"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="../annotations/preprocessed_annotations_with_titles.json",  # 默认保存为一个新文件
        help="保存更新后注释 JSON 文件的路径。"
    )
    args = parser.parse_args()

    add_titles(args.annotations_path, args.titles_path, args.mapping_path, args.output_path)


if __name__ == "__main__":
    main()
