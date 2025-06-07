import streamlit as st
import torch
import time
from transformers import AutoTokenizer, BertConfig, AutoModelForSequenceClassification

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_path_sentiment = "model_sentiment_analysis"
tokenizer_sentiment = AutoTokenizer.from_pretrained(model_path_sentiment)
config_sentiment = BertConfig.from_pretrained(model_path_sentiment, num_labels=2)
model_sentiment = AutoModelForSequenceClassification.from_pretrained(model_path_sentiment, config=config_sentiment)
model_sentiment.to(device)
model_sentiment.eval()  

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

    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)

    with torch.no_grad():
        outputs = model_sentiment(input_ids=input_ids, attention_mask=attention_mask)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)
        predicted_class = torch.argmax(probs, dim=1).item()
        confidence = probs[0][predicted_class].item()

    sentiment = "Tiêu cực" if predicted_class == 0 else "Tích cực"

    return sentiment, confidence


# Dự đoán chủ đề Yahoo
def predict_yahoo_topic(text):
    if not text.strip():
        return None

    time.sleep(2)  # Giả lập delay API

    topics = {
        "Công nghệ": ["computer", "software", "tech", "programming", "code", "app", "website"],
        "Sức khỏe": ["health", "medical", "doctor", "medicine", "sick", "disease", "treatment"],
        "Giáo dục": ["school", "study", "learn", "education", "student", "teacher", "university"],
        "Thể thao": ["sport", "game", "team", "player", "match", "football", "basketball"],
        "Giải trí": ["movie", "music", "show", "actor", "singer", "entertainment", "celebrity"],
        "Kinh doanh": ["business", "money", "work", "job", "company", "finance", "investment"],
    }

    text_lower = text.lower()
    best_topic = "Tổng quát"
    max_score = 0

    for topic, keywords in topics.items():
        score = sum(word in text_lower for word in keywords)
        if score > max_score:
            max_score = score
            best_topic = topic

    confidence = min(0.95, 0.5 + max_score * 0.15) if max_score > 0 else 0.4

    return best_topic, confidence

# Giao diện
st.set_page_config(page_title="Hệ thống Dự đoán AI", layout="centered")

st.title("🧠 Hệ thống Dự đoán AI")
st.subheader("🔍 Phân tích cảm xúc & phân loại chủ đề")

tab1, tab2 = st.tabs(["📦 Cảm xúc", "🏷️ Chủ đề"])

with tab1:
    st.markdown("Nhập văn bản mà bạn muốn phân tích cảm xúc (Tích cực / Trung tính).")
    amazon_input = st.text_area("✍️ Nhập văn bản", height=150)

    if st.button("📊 Phân tích Cảm xúc", key="amazon"):
        with st.spinner("Đang phân tích..."):
            result = predict_amazon_sentiment(amazon_input)
        if result:
            sentiment, confidence = result
            st.success(f"**Kết quả:** {sentiment}")
            st.info(f"Độ tin cậy: {confidence*100:.1f}%")

with tab2:
    st.markdown("Nhập văn bản mà bạn muốn phân loại chủ đề (Công nghệ, Sức khỏe, ...).")
    yahoo_input = st.text_area("✍️ Nhập văn bản", height=150)

    if st.button("📚 Phân loại Chủ đề", key="yahoo"):
        with st.spinner("Đang phân loại..."):
            result = predict_yahoo_topic(yahoo_input)
        if result:
            topic, confidence = result
            st.success(f"**Chủ đề:** {topic}")
            st.info(f"Độ tin cậy: {confidence*100:.1f}%")

st.markdown("---")
st.caption("💡 Đây là bản demo của ứng dụng.")
