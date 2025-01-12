import os
import argparse
from tqdm import tqdm
import numpy as np
import tiktoken
from datasets import load_dataset  # huggingface datasets

# number of workers in .map() call
num_proc = 8
num_proc_load_dataset = num_proc

enc = tiktoken.get_encoding("gpt2")

def ensure_dir(directory):
    """
    Ensure that a directory exists.
    If not, create it.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)

def main(output_dir):
    """
    主函数，处理数据并保存到指定文件夹。

    :param output_dir: 保存处理后文件的目标文件夹
    """
    ensure_dir(output_dir)

    # 加载数据集
    dataset = load_dataset("openwebtext", num_proc=num_proc_load_dataset, trust_remote_code=True)

    # 创建验证集
    split_dataset = dataset["train"].train_test_split(test_size=0.0005, seed=2357, shuffle=True)
    split_dataset['val'] = split_dataset.pop('test')  # 重命名 test 为 val

    # 定义编码函数
    def process(example):
        ids = enc.encode_ordinary(example['text'])  # 普通编码
        ids.append(enc.eot_token)  # 添加 End-Of-Text (EOT) token
        return {'ids': ids, 'len': len(ids)}

    # 令牌化数据集
    tokenized = split_dataset.map(
        process,
        remove_columns=['text'],
        desc="tokenizing the splits",
        num_proc=num_proc,
    )


    # 将每个数据集的所有 ID 连接成一个大文件用于训练
    for split, dset in tokenized.items():
        arr_len = np.sum(dset['len'], dtype=np.uint64)
        bin_path = os.path.join(output_dir, f'{split}.bin')
        dtype = np.uint16  # 数据类型，适合 GPT-2 的 token 值范围
        arr = np.memmap(bin_path, dtype=dtype, mode='w+', shape=(arr_len,))
        total_batches = 1024

        idx = 0
        for batch_idx in tqdm(range(total_batches), desc=f'writing {bin_path}'):
            # 分批处理
            batch = dset.shard(num_shards=total_batches, index=batch_idx, contiguous=True).with_format('numpy')
            arr_batch = np.concatenate(batch['ids'])
            arr[idx: idx + len(arr_batch)] = arr_batch
            idx += len(arr_batch)
        arr.flush()
        print(f"Saved {split} split to {bin_path}")

    # 提示完成
    print(f"All processed data saved in: {os.path.abspath(output_dir)}")

if __name__ == '__main__':
    # 定义命令行参数
    parser = argparse.ArgumentParser(description="Process and save the openwebtext dataset.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="processed_data",
        help="Directory to save processed files. Defaults to 'processed_data'."
    )

    args = parser.parse_args()
    main(args.output_dir)