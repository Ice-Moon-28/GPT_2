import kagglehub
import shutil

# 下载数据集
openwebtext_path = kagglehub.dataset_download('zhanglinghua011228/openwebtext')

# 将数据集移动到指定路径
target_path = 'data/openwebtext'
shutil.move(openwebtext_path, target_path)
print(f"Data moved to {target_path}")