import time
import torch
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# --------------------------------------------------
# 1. Paths to fine‑tuned models (folder checkpoints)
# --------------------------------------------------
SENTIMENT_MODEL_DIR = "models/sentiment_analysis"      # contains config.json + model.safetensors
TOPIC_MODEL_DIR     = "models/topic_classification"    # same structure for topic classifier

# --------------------------------------------------
# 2. Device setup
# --------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------------------------------------------------
# 3. Load tokenizers & models
#    ── Fix for meta‑tensor error: load on CPU first, then .to(DEVICE) if CUDA
# --------------------------------------------------
# Sentiment (binary)
tokenizer_sentiment = AutoTokenizer.from_pretrained(SENTIMENT_MODEL_DIR)
model_sentiment = AutoModelForSequenceClassification.from_pretrained(
    SENTIMENT_MODEL_DIR,
    torch_dtype=torch.float32,   # ensure real tensors
    device_map=None              # load fully on CPU
)
if DEVICE.type == "cuda":
    model_sentiment.to(DEVICE)
model_sentiment.eval()

# Topic (10 classes)
tokenizer_topic = AutoTokenizer.from_pretrained(TOPIC_MODEL_DIR)
model_topic = AutoModelForSequenceClassification.from_pretrained(
    TOPIC_MODEL_DIR,
    torch_dtype=torch.float32,
    device_map=None
)
if DEVICE.type == "cuda":
    model_topic.to(DEVICE)
model_topic.eval()

# --------------------------------------------------
# 4. Helper functions
# --------------------------------------------------

def _preprocess(text: str, tokenizer, max_len: int = 512):
    """Tokenise a single text and return (input_ids, attention_mask) tensors."""
    enc = tokenizer(text,
                    add_special_tokens=True,
                    max_length=max_len,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt")
    return enc["input_ids"].to(DEVICE), enc["attention_mask"].to(DEVICE)

# --------------------------------------------------
# 5. Sentiment prediction
# --------------------------------------------------

def predict_sentiment(text: str):
    if not text.strip():
        return None
    time.sleep(1)
    input_ids, attn_mask = _preprocess(text, tokenizer_sentiment, max_len=256)
    with torch.no_grad():
        logits = model_sentiment(input_ids=input_ids, attention_mask=attn_mask).logits
        probs  = torch.softmax(logits, dim=1)
        pred   = torch.argmax(probs, dim=1).item()
        conf   = probs[0, pred].item()
    label = "Tích cực" if pred == 1 else "Tiêu cực"
    return label, conf

# --------------------------------------------------
# 6. Topic prediction
# --------------------------------------------------
CLASS_LABELS = [
    "Society & Culture",
    "Science & Mathematics",
    "Health",
    "Education & Reference",
    "Computers & Internet",
    "Sports",
    "Business & Finance",
    "Entertainment & Music",
    "Family & Relationships",
    "Politics & Government",
]

def predict_topic(text: str):
    if not text.strip():
        return None
    time.sleep(1)
    input_ids, attn_mask = _preprocess(text, tokenizer_topic, max_len=256)
    with torch.no_grad():
        logits = model_topic(input_ids=input_ids, attention_mask=attn_mask).logits
        probs  = torch.softmax(logits, dim=1)
        pred   = torch.argmax(probs, dim=1).item()
        conf   = probs[0, pred].item()
    return CLASS_LABELS[pred], conf

# --------------------------------------------------
# 7. Streamlit UI
# --------------------------------------------------

st.set_page_config(page_title="Hệ thống Dự đoán AI", layout="centered")

st.title("🧠 Hệ thống Dự đoán AI")
st.subheader("🔍 Phân tích cảm xúc & phân loại chủ đề")

tab_sentiment, tab_topic = st.tabs(["📦 Cảm xúc", "🏷️ Chủ đề"])

with tab_sentiment:
    st.markdown("Nhập văn bản mà bạn muốn phân tích cảm xúc (Tích cực / Tiêu cực).")
    txt_inp = st.text_area("✍️ Nhập văn bản", height=150, key="sentiment_input")
    if st.button("📊 Phân tích Cảm xúc"):
        with st.spinner("Đang phân tích..."):
            result = predict_sentiment(txt_inp)
        if result:
            label, conf = result
            st.success(f"**Kết quả:** {label}")
            st.info(f"Độ tin cậy: {conf*100:.1f}%")

with tab_topic:
    st.markdown("Nhập văn bản mà bạn muốn phân loại chủ đề (Công nghệ, Sức khỏe, ...).")
    txt_inp = st.text_area("✍️ Nhập văn bản", height=150, key="topic_input")
    if st.button("📚 Phân loại Chủ đề"):
        with st.spinner("Đang phân loại..."):
            result = predict_topic(txt_inp)
        if result:
            topic, conf = result
            st.success(f"**Chủ đề:** {topic}")
            st.info(f"Độ tin cậy: {conf*100:.1f}%")

st.markdown("---")
st.caption("💡 Đây là bản demo của ứng dụng.")
