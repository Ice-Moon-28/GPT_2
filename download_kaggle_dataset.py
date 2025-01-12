import zipfile
import os
import argparse
from tqdm import tqdm  # 用于显示进度条

def unzip_with_progress(zip_file, dest_dir):
    """
    解压文件并显示进度条

    :param zip_file: 要解压的 .zip 文件路径
    :param dest_dir: 解压目标目录
    """
    # 检查目标目录是否存在，不存在则创建
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    # 打开压缩文件
    with zipfile.ZipFile(zip_file, 'r') as zf:
        file_list = zf.namelist()  # 获取压缩文件中的所有文件名
        total_files = len(file_list)  # 总文件数

        print(f"开始解压文件：{zip_file}")
        print(f"目标目录：{dest_dir}")
        print(f"总文件数：{total_files}")

        # 使用 tqdm 显示进度
        with tqdm(total=total_files, unit="file") as progress_bar:
            for file in file_list:
                # 解压当前文件
                zf.extract(file, dest_dir)
                progress_bar.update(1)  # 每解压一个文件，更新进度
        print("解压完成！")

def main():
    # 定义命令行参数
    parser = argparse.ArgumentParser(description="解压 ZIP 文件并显示进度条")
    parser.add_argument("zip_file", help="需要解压的 ZIP 文件路径")
    parser.add_argument("dest_dir", help="解压目标目录")

    # 解析命令行参数
    args = parser.parse_args()

    # 调用解压函数
    unzip_with_progress(args.zip_file, args.dest_dir)

if __name__ == "__main__":
    main()