from huggingface_hub import snapshot_download

# 下载数据集到指定文件夹
snapshot_download(
    repo_id="Skylion007/openwebtext",
    cache_dir="/root/autodl-tmp/cache"
)

 poetry run huggingface-cli download --repo-type dataset --resume-download openwebtext  --local-dir /root/autodl-tmp/cache 