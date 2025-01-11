import kagglehub
import shutil
import os

os.environ['TMPDIR'] = '/root/autodl-tmp/data/openwebtext'
# 设置目标路径

# 下载数据集
openwebtext_path = kagglehub.dataset_download('zhanglinghua011228/openwebtext')

# 检查下载路径是否有效
if not os.path.exists(openwebtext_path):
    raise FileNotFoundError(f"Downloaded dataset not found at {openwebtext_path}")
