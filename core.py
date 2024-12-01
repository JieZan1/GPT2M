import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel
import torch
from transformers import TrainingArguments, Trainer

# https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge/data
# load dataset and filter for one label
ds = load_dataset("google-research-datasets/go_emotions", "simplified")
ds_with_one_label = ds.filter(lambda example: len(example['labels']) == 1)

train_full = ds["train"]
test_full = ds["test"]
validation_full = ds["validation"]

# one label 
train_one_label = ds_with_one_label["train"]
test_one_label = ds_with_one_label["test"]
validation_one_label = ds_with_one_label["validation"]

# alias, for convenience
train = train_one_label
test = test_one_label
validation = validation_one_label

tokenizer = AutoTokenizer.from_pretrained("/opt/models/bert-base-uncased") 
model = AutoModel.from_pretrained("/opt/models/bert-base-uncased")

model = model.to('cuda')

class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if self.labels:
            item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.encodings["input_ids"])
    


X_train_tokenized = tokenizer(train_one_label["text"], padding=True, truncation=True, max_length=512)
X_val_tokenized = tokenizer(validation_one_label["text"], padding=True, truncation=True, max_length=512)
    
train_dataset = Dataset(X_train_tokenized, train_one_label["labels"])
val_dataset = Dataset(X_val_tokenized, validation_one_label["labels"])

print(X_train_tokenized.keys())