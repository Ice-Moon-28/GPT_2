mkdir -p /root/autodl-tmp/cache
ln -s /root/autodl-tmp/cache /root/.cache
pip install kaggle
kaggle datasets download -d zhanglinghua011228/openwebtext  -p /root/autodl-tmp/cache

pip install datasets wandb kagglehub
python download_kaggle_dataset.py