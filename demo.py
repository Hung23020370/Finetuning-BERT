import streamlit as st
import torch
import time
from transformers import AutoTokenizer, AutoModelForSequenceClassification, BertForSequenceClassification

# Đường dẫn đến mô hình fine-tuned
model_path_sentiment = "model_sentiment_analysis"
model_path_topic = "model_topic classification/best_model.bin"

# Tự động chọn thiết bị GPU nếu có, ngược lại dùng CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load tokenizer và config
tokenizer_sentiment = AutoTokenizer.from_pretrained(model_path_sentiment)
tokenizer_topic = AutoTokenizer.from_pretrained("bert-base-uncased")

# Load mô hình từ 
model_sentiment = AutoModelForSequenceClassification.from_pretrained(model_path_sentiment)
model_topic = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=10)
model_topic.load_state_dict(torch.load(model_path_topic, map_location=torch.device('cpu'), weights_only=False))

model_sentiment.eval()
model_topic.eval()

# Dự đoán cảm xúc Amazon
def predict_amazon_sentiment(text):
    if not text.strip():
        return None

    time.sleep(2)  # Giả lập delay API

    def preprocess_text(text, tokenizer, max_len=512):
        encoding = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return encoding['input_ids'], encoding['attention_mask']
    
    input_ids, attention_mask = preprocess_text(text, tokenizer_sentiment)

    # Đưa input về đúng device
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device) 
       
    with torch.no_grad():
        outputs = model_sentiment(input_ids=input_ids, attention_mask=attention_mask)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)
        predicted_class = torch.argmax(probs, dim=1).item()
        confidence = probs[0][predicted_class].item()

    sentiment = "Tiêu cực"
    if predicted_class == 1:
        sentiment = "Tích cực"

    return sentiment, confidence


# Dự đoán chủ đề Yahoo
def predict_yahoo_topic(text):
    if not text.strip():
        return None

    time.sleep(2)  # Giả lập delay API

    def preprocess_text(text, tokenizer, max_len=512):
        encoding = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return encoding['input_ids'], encoding['attention_mask']
    
    input_ids, attention_mask = preprocess_text(text, tokenizer_sentiment)
    
    # Đưa input về đúng device
    input_ids = input_ids.to(torch.device('cpu'))
    attention_mask = attention_mask.to(torch.device('cpu'))

       
    with torch.no_grad():
        outputs = model_sentiment(input_ids=input_ids, attention_mask=attention_mask)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)
        predicted_class = torch.argmax(probs, dim=1).item()
        confidence = probs[0][predicted_class].item()

    best_topic = ""

    class_labels = [
    "Society & Culture",
    "Science & Mathematics",
    "Health",
    "Education & Reference",
    "Computers & Internet",
    "Sports",
    "Business & Finance",
    "Entertainment & Music",
    "Family & Relationships",
    "Politics & Government"
    ]   

    best_topic = class_labels[predicted_class]

    return best_topic, confidence

# Giao diện
st.set_page_config(page_title="Hệ thống Dự đoán AI", layout="centered")

st.title("🧠 Hệ thống Dự đoán AI")
st.subheader("🔍 Phân tích cảm xúc & phân loại chủ đề")

tab1, tab2 = st.tabs(["📦 Cảm xúc", "🏷️ Chủ đề"])

with tab1:
    st.markdown("Nhập văn bản mà bạn muốn phân tích cảm xúc (Tích cực / Trung tính).")
    amazon_input = st.text_area("✍️ Nhập văn bản", height=150, key="amazon_input")

    if st.button("📊 Phân tích Cảm xúc", key="amazon"):
        with st.spinner("Đang phân tích..."):
            result = predict_amazon_sentiment(amazon_input)
        if result:
            sentiment, confidence = result
            st.success(f"**Kết quả:** {sentiment}")
            st.info(f"Độ tin cậy: {confidence*100:.1f}%")

with tab2:
    st.markdown("Nhập văn bản mà bạn muốn phân loại chủ đề (Công nghệ, Sức khỏe, ...).")
    yahoo_input = st.text_area("✍️ Nhập văn bản", height=150, key="yahoo_input")

    if st.button("📚 Phân loại Chủ đề", key="yahoo"):
        with st.spinner("Đang phân loại..."):
            result = predict_yahoo_topic(yahoo_input)
        if result:
            topic, confidence = result
            st.success(f"**Chủ đề:** {topic}")
            st.info(f"Độ tin cậy: {confidence*100:.1f}%")

st.markdown("---")
st.caption("💡 Đây là bản demo của ứng dụng.")

 