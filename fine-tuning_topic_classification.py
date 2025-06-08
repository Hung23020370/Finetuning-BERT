import pandas as pd
import torch
from transformers import BertConfig, AutoTokenizer, BertForSequenceClassification
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tqdm import tqdm

# Đường dẫn đến mô hình pre-trained
model_pre_path = "models/pre_training"

# Load tokenizer, config, và mô hình BERT với num_labels=10
tokenizer = AutoTokenizer.from_pretrained(model_pre_path)
config = BertConfig.from_pretrained(model_pre_path, num_labels=10)
model = BertForSequenceClassification.from_pretrained(model_pre_path, config=config)

# Đọc dữ liệu
train_df = pd.read_csv(
    "data/yahoo-email-classification/train.csv",
    header=None,
    names=["label", "title", "content", "answer"]
)
test_df = pd.read_csv(
    "/data/yahoo-email-classification/test.csv",
    header=None,
    names=["label", "title", "content", "answer"]
)


# Lấy mẫu nhỏ để huấn luyện thử
train_df = train_df.sample(frac=0.5, random_state=42).reset_index(drop=True)
test_df  = test_df.sample(frac=0.5, random_state=42).reset_index(drop=True)

# Tạo cột text gộp
train_df["text"] = train_df.apply(lambda row: f"{row['title']} {row['content']} {row['answer']}", axis=1)
test_df["text"]  = test_df.apply(lambda row: f"{row['title']} {row['content']} {row['answer']}", axis=1)

# Chỉnh lại nhãn từ 0 đến 9
train_df["label"] -= 1
test_df["label"] -= 1

# Tách train/val
texts = train_df["text"].tolist()
labels = train_df["label"].tolist()
train_texts, val_texts, train_labels, val_labels = train_test_split(
    texts, labels, test_size=0.1, random_state=42
)

# Dataset class
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

# DataLoader
train_dataset = MyDataset(train_texts, train_labels, tokenizer)
val_dataset = MyDataset(val_texts, val_labels, tokenizer)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

# Setup training
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
optimizer = Adam(model.parameters(), lr=2e-5)
epochs = 3

# Train loop
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

# Lưu mô hình và tokenizer
output_dir = "models/topic_classification"
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)