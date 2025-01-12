sudo apt update
sudo apt install pipx
pipx ensurepath
pipx install poetry


mkdir -p /root/autodl-tmp/cache
export HF_ENDPOINT=https://hf-mirror.com
# pip install kaggle
# kaggle datasets download -d zhanglinghua011228/openwebtext  -p /root/autodl-tmp/cache
# python download_kaggle_dataset.py  /root/autodl-tmp/cache/openwebtext.zip  /root/autodl-tmp/cache

pip install datasets==2.19.1 wandb kagglehub
pip install tiktoken
python3.12 data/openwebtext/prepare.py --output-dir /root/autodl-tmp/cache