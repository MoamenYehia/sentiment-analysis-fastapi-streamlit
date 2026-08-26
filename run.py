import streamlit as st
import os
from huggingface_hub import InferenceClient

# إعدادات مظهر الصفحة
st.set_page_config(
    page_title="AI Sentiment Analyzer",
    page_icon="🧠",
    layout="centered"
)

st.title("🧠 AI Sentiment Analysis")
st.caption("Powered by Hugging Face & Streamlit")

HF_TOKEN = st.secrets.get("HF_TOKEN", os.getenv("HF_TOKEN"))
if not HF_TOKEN:
    st.error("Hugging Face Token is missing. Please add it to Secrets!")
    st.stop()

# إعداد العميل والموديل
MODEL_ID = "cardiffnlp/twitter-roberta-base-sentiment-latest"
client = InferenceClient(api_key=HF_TOKEN)

# حقل إدخال النص
user_text = st.text_area(
    "Enter the sentence/review below:",
    placeholder="e.g. This product exceeded my expectations, absolutely loved it!"
)

# زر التحليل
if st.button("Analyze Sentiment", use_container_width=True):
    if not user_text.strip():
        st.warning("Please enter some text first.")
    else:
        with st.spinner("Analyzing text with neural network..."):
            try:
                response = client.text_classification(
                    text=user_text,
                    model=MODEL_ID
                )
                
                if isinstance(response, list) and len(response) > 0:
                    top_result = response[0]
                    label = top_result.label
                    score = float(top_result.score)

                    st.divider()
                    st.subheader("Analysis Result:")

                    if label.lower() == "positive":
                        st.success(f"**Sentiment:** {label} 🎉")
                    elif label.lower() == "negative":
                        st.error(f"**Sentiment:** {label} ⚠️")
                    else:
                        st.info(f"**Sentiment:** {label} ⚖️")

                    st.metric(label="Confidence Score", value=f"{score * 100:.2f}%")
                    st.progress(score)
                else:
                    st.error("Unexpected response from Hugging Face.")

            except Exception as e:
                st.error(f"Inference Error: {str(e)}")