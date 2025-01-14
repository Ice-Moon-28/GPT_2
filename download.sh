sudo python3 -m pip install pipx
pipx ensurepath
pipx install poetry
poetry install 

export HF_ENDPOINT=https://hf-mirror.com
poetry run huggingface-cli login
poetry run huggingface-cli download icemoon28/openwebtext --local-dir /root/autodl-tmp/cache/ --repo-type dataset

poetry run python train_pytorch.py config/train_gpt2.py 