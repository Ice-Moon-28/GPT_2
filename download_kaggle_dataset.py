import kagglehub
import shutil
import os

# 设置目标路径
target_path = '/root/autodl-tmp/data/openwebtext'

# 下载数据集
openwebtext_path = kagglehub.dataset_download('zhanglinghua011228/openwebtext')

# 检查下载路径是否有效
if not os.path.exists(openwebtext_path):
    raise FileNotFoundError(f"Downloaded dataset not found at {openwebtext_path}")
