from transformers import GPT2LMHeadModel, GPT2Config

model_args = GPT2Config(
    vocab_size=50257,  # 根据数据集大小调整
    n_positions=1024,  # 输入序列的最大长度
    n_ctx=1024,  # 上下文窗口大小
    n_embd=768,  # 嵌入层大小
    n_layer=12,  # Transformer 层数
    n_head=12,  # 自注意力头数
    dropout=0.1,  # Dropout 比例
)

print(model_args['vocab_size'])