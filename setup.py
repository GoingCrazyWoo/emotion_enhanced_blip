from setuptools import setup, find_packages

setup(
    name='emotion_enhanced_blip',
    version='0.1.0',
    packages=find_packages(exclude=["scripts*", "annotations*", "snapshots*", ".history*"]), # 自动查找包，排除非代码目录
    # 可以根据需要添加其他元数据，例如：
    # author='Your Name',
    # description='Emotion Enhanced BLIP Model',
    # install_requires=[ # 列出项目依赖项
    #     'torch',
    #     'transformers',
    #     'datasets',
    #     'Pillow',
    #     'tqdm',
    #     'numpy',
    #     'pycocotools', # 或 pycocotools-windows
    #     'pycocoevalcap',
    #     'huggingface_hub',
    #     'requests',
    # ],
    python_requires='>=3.8', # 指定Python版本要求
)