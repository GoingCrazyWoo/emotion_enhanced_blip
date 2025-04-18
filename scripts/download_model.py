import os
from huggingface_hub import hf_hub_download

# --- 用户需要修改的参数 ---
# 您的 Hugging Face 仓库 ID (格式: "用户名/仓库名")
REPO_ID = "Opps/blip_base_newyorker" # 请替换为您的实际仓库 ID

# 要下载的文件在仓库中的路径和文件名
# 例如，如果您想下载仓库根目录下的 "best_model.pth"，可以设置为 "best_model.pth"
# 如果您想下载仓库的 "models" 目录下的 "latest_model.pth"，可以设置为 "models/latest_model.pth"
REPO_FILE_PATH = "best_model.pth" # 仓库中的文件名

# 下载到本地的目标路径和文件名 (相对于项目根目录)
# 例如，如果您想下载到项目根目录并保持原文件名，可以设置为 REPO_FILE_PATH
# 如果您想下载到本地的 "downloaded_models" 目录下并重命名为 "my_model.pth"，可以设置为 "downloaded_models/my_model.pth"
LOCAL_FILE_PATH = "downloaded_models/best_model.pth" # 下载到本地的文件路径

# --- 脚本逻辑 ---

def download_model(repo_id: str, repo_file_path: str, local_file_path: str):
    """
    从 Hugging Face 仓库下载模型文件

    Args:
        repo_id: 仓库 ID (格式: "用户名/仓库名")
        repo_file_path: 要下载的文件在仓库中的路径和文件名
        local_file_path: 下载到本地的目标路径和文件名
    """
    print(f"正在从仓库 {repo_id} 下载文件 {repo_file_path} 到 {local_file_path}...")
    try:
        # 确保本地目标目录存在
        local_dir = os.path.dirname(local_file_path)
        if local_dir and not os.path.exists(local_dir):
            os.makedirs(local_dir, exist_ok=True)
            print(f"已创建本地目录: {local_dir}")

        hf_hub_download(
            repo_id=repo_id,
            filename=repo_file_path,
            local_dir=os.dirpath(local_file_path), # 指定下载到哪个本地目录
            local_files_only=False, # 如果文件不存在则下载
            cache_dir=None # 不使用缓存，直接下载到指定目录
        )
        print("文件下载成功！")
        print(f"文件已保存到本地：{local_file_path}")
    except Exception as e:
        print(f"文件下载失败: {e}")

if __name__ == "__main__":
    download_model(REPO_ID, REPO_FILE_PATH, LOCAL_FILE_PATH)