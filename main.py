from datasets import load_dataset
from tqdm import tqdm
from transformers import BertTokenizerFast
from itertools import chain
import pandas as pd
from transformers import BertConfig, AutoTokenizer, BertForSequenceClassification
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data_pre_train = load_dataset("wikipedia", "20220301.en", split="train[:1%]")

# create a python generator to dynamically load the data
def batch_iterator(batch_size=3000):
    for i in tqdm(range(0, len(data_pre_train), batch_size)):
        yield data_pre_train[i : i + batch_size]["text"]

# create a tokenizer from existing one to re-use special tokens
tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

bert_tokenizer = tokenizer.train_new_from_iterator(text_iterator=batch_iterator(), vocab_size=32_000)

bert_tokenizer.save_pretrained("tokenizer")

from transformers import AutoTokenizer
import multiprocessing

# load tokenizer
tokenizer = AutoTokenizer.from_pretrained("tokenizer")
num_proc = multiprocessing.cpu_count()
print(f"The max length for the tokenizer is: {tokenizer.model_max_length}")

def group_texts(examples):
    tokenized_inputs = tokenizer(
       examples["text"], return_special_tokens_mask=True, truncation=True, max_length=tokenizer.model_max_length
    )
    return tokenized_inputs

# preprocess dataset
tokenized_datasets = data_pre_train.map(group_texts, batched=True, remove_columns=["text"], num_proc=num_proc)
tokenized_datasets.features

# Main data processing function that will concatenate all texts from our dataset and generate chunks of
# max_seq_length.
def group_texts(examples):
    # Concatenate all texts.
    concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
    # customize this part to your needs.
    if total_length >= tokenizer.model_max_length:
        total_length = (total_length // tokenizer.model_max_length) * tokenizer.model_max_length
    # Split by chunks of max_len.
    result = {
        k: [t[i : i + tokenizer.model_max_length] for i in range(0, total_length, tokenizer.model_max_length)]
        for k, t in concatenated_examples.items()
    }
    return result

tokenized_datasets = tokenized_datasets.map(group_texts, batched=True, num_proc=num_proc)
# shuffle dataset
tokenized_datasets = tokenized_datasets.shuffle(seed=34)

print(f"the dataset contains in total {len(tokenized_datasets)*tokenizer.model_max_length} tokens")

import torch
from torch.utils.data import DataLoader
from transformers import (
    BertConfig,
    BertForMaskedLM,
    DataCollatorForLanguageModeling,
    BertTokenizerFast,
)
from tqdm.notebook import tqdm

# Thiết bị (GPU nếu có)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=True,
    mlm_probability=0.15,
)

# Tạo mô hình
config = BertConfig(
    vocab_size=tokenizer.vocab_size,  
    hidden_size=256,                 
    num_hidden_layers=6,             
    num_attention_heads=8,           
    intermediate_size=1024,          
    max_position_embeddings=512,      
    type_vocab_size=2,                
    pad_token_id=tokenizer.pad_token_id,
)
model = BertForMaskedLM(config).to(device)

# DataLoader
train_dataloader = DataLoader(
    tokenized_datasets, shuffle=True, batch_size=32, collate_fn=data_collator
)

# Optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

# Train loop
model.train()
epochs = 4
for epoch in range(epochs):
    loop = tqdm(train_dataloader, desc=f"Epoch {epoch+1}")
    for batch in loop:
        # Chuyển dữ liệu lên GPU nếu cần
        batch = {k: v.to(device) for k, v in batch.items()}

        # Tính loss và backward
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        # Cập nhật progress bar
        loop.set_postfix(loss=loss.item())

print("✅ Training completed!")

save_path = "./pre_training"
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)

model_pre_path = "pre_training"

tokenizer = AutoTokenizer.from_pretrained(model_pre_path)
config = BertConfig.from_pretrained(model_pre_path, num_labels=2)
model = BertForSequenceClassification.from_pretrained(model_pre_path, config=config)

data = pd.read_csv("data/sentiment_analysis")

texts = data['content'].tolist()
labels = data['label'].tolist()  

labels = [l-1 for l in labels]

# Tách train-validation
train_texts, val_texts, train_labels, val_labels = train_test_split(
    texts, labels, test_size=0.1, random_state=42
)

# 3. Dataset class
class MyDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# 4. Tạo DataLoader
train_dataset = MyDataset(train_texts, train_labels, tokenizer)
val_dataset = MyDataset(val_texts, val_labels, tokenizer)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

# 5. Training setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

optimizer = Adam(model.parameters(), lr=2e-5)
epochs = 4

from tqdm import tqdm

# 6. Training loop
for epoch in range(epochs):
    model.train()
    total_loss = 0

    for batch in tqdm(train_loader):
        optimizer.zero_grad()

        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()

        loss.backward()
        optimizer.step()

    avg_train_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{epochs}, Train loss: {avg_train_loss:.4f}")

 # Validation
    model.eval()
    val_preds = []
    val_true = []

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1)

            val_preds.extend(preds.cpu().numpy())
            val_true.extend(labels.cpu().numpy())

    acc = accuracy_score(val_true, val_preds)
    print(f"Validation Accuracy: {acc:.4f}")

output_dir = "./model_sentiment_analysis"

# Lưu model
model.save_pretrained(output_dir)

# Lưu tokenizer
tokenizer.save_pretrained(output_dir)

