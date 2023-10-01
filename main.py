from transformers import *
import json

model_name = 'allenai/scibert_scivocab_cased'
tokenizer_name = 'allenai/scibert_scivocab_cased'

tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
model_name = AutoTokenizer.from_pretrained(model_name)
