from transformers import AutoTokenizer

squad_v2 = True
model_checkpoint = "distilbert-base-uncased"
batch_size = 16
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_auth_token=True)
