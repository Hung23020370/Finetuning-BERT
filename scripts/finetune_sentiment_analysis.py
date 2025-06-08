# -------------------- FILE 2: finetune.py --------------------
import pandas as pd
from transformers import AutoTokenizer, BertConfig, BertForSequenceClassification
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tqdm import tqdm

# Load model & tokenizer
model_pre_path = "../models/pre_training"
tokenizer = AutoTokenizer.from_pretrained(model_pre_path)
config = BertConfig.from_pretrained(model_pre_path, num_labels=2)
model = BertForSequenceClassification.from_pretrained(model_pre_path, config=config)

# Load and process dataset
data = pd.read_csv("data/sentiment_analysis/train_sentiment_analysis.csv")
texts = data['content'].tolist()
labels = [l - 1 for l in data['label'].tolist()]  # Convert labels to 0-based

train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=0.1, random_state=42)

class MyDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer.encode_plus(
            self.texts[idx],
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }

# DataLoaders
train_dataset = MyDataset(train_texts, train_labels, tokenizer)
val_dataset = MyDataset(val_texts, val_labels, tokenizer)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

# Training setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
optimizer = Adam(model.parameters(), lr=2e-5)

# Training loop
for epoch in range(4):
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
    print(f"Epoch {epoch+1}/4, Train Loss: {total_loss / len(train_loader):.4f}")

    # Validation
    model.eval()
    val_preds, val_true = [], []
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=1)
            val_preds.extend(preds.cpu().numpy())
            val_true.extend(labels.cpu().numpy())
    acc = accuracy_score(val_true, val_preds)
    print(f"Validation Accuracy: {acc:.4f}")

# Save fine-tuned model
output_dir = "../models/sentiment_analysis"
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
