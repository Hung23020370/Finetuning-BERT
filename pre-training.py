from datasets import load_dataset
from tqdm import tqdm
from transformers import BertTokenizerFast, BertConfig, BertForMaskedLM, DataCollatorForLanguageModeling
from torch.utils.data import DataLoader
import torch
from itertools import chain
import multiprocessing

# Load raw dataset
data_pre_train = load_dataset("wikipedia", "20220301.en", split="train[:1%]")

def batch_iterator(batch_size=3000):
    for i in tqdm(range(0, len(data_pre_train), batch_size)):
        yield data_pre_train[i : i + batch_size]["text"]

# Train new tokenizer
tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
bert_tokenizer = tokenizer.train_new_from_iterator(text_iterator=batch_iterator(), vocab_size=32_000)
bert_tokenizer.save_pretrained("tokenizer")

# Tokenizer & preprocessing
tokenizer = BertTokenizerFast.from_pretrained("tokenizer")
num_proc = multiprocessing.cpu_count()

# Pre-tokenize
def group_texts_init(examples):
    tokenized_inputs = tokenizer(
        examples["text"], return_special_tokens_mask=True, truncation=True, max_length=tokenizer.model_max_length
    )
    return tokenized_inputs

tokenized_datasets = data_pre_train.map(group_texts_init, batched=True, remove_columns=["text"], num_proc=num_proc)

# Grouping into chunks
def group_texts(examples):
    concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    if total_length >= tokenizer.model_max_length:
        total_length = (total_length // tokenizer.model_max_length) * tokenizer.model_max_length
    result = {
        k: [t[i : i + tokenizer.model_max_length] for i in range(0, total_length, tokenizer.model_max_length)]
        for k, t in concatenated_examples.items()
    }
    return result

tokenized_datasets = tokenized_datasets.map(group_texts, batched=True, num_proc=num_proc).shuffle(seed=34)
print(f"Total tokens: {len(tokenized_datasets) * tokenizer.model_max_length}")

# Model & training
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
model = BertForMaskedLM(config).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

# Training setup
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)
train_dataloader = DataLoader(tokenized_datasets, shuffle=True, batch_size=32, collate_fn=data_collator)
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

# Training loop
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.train()
for epoch in range(4):
    loop = tqdm(train_dataloader, desc=f"Epoch {epoch+1}")
    for batch in loop:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        loop.set_postfix(loss=loss.item())

model.save_pretrained("models/pre_training")
tokenizer.save_pretrained("models/pre_training")

