from transformers import GPT2LMHeadModel, GPT2Config


def get_model(config):
    return GPT2LMHeadModel(config)

if __name__ == '__main__':

    model = get_model(GPT2Config.from_pretrained('gpt2'))

    print(model)