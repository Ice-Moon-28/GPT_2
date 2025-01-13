from datasets import load_dataset

# 加载数据集
dataset = load_dataset("/icemoon28/openwebtext",  cache_dir="/root/autodl-tmp/cache")

# 查看数据集信息
print(dataset)