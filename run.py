import streamlit as st
import requests


hf_token = st.secrets.get("HF_TOKEN", os.getenv("HF_TOKEN"))

st.set_page_config(
    page_title="AI Sentiment Analyzer",
    page_icon="🧠",
    layout="centered"
)

st.title("🧠 AI Sentiment Analysis")
st.caption("Powered by FastAPI & Hugging Face")

API_URL = "http://127.0.0.1:8000/analyze"

# حقل إدخال النص
user_text = st.text_area(
    "Enter the sentence/review below:",
    placeholder="e.g. This product exceeded my expectations, absolutely loved it!"
)

if st.button("Analyze Sentiment", use_container_width=True):
    if not user_text.strip():
        st.warning("Please enter some text first.")
    else:
        with st.spinner("Analyzing text with neural network..."):
            try:
                response = requests.post(
                    API_URL,
                    json={"text": user_text},
                    timeout=10
                )
                
                if response.status_code == 200:
                    data = response.json()
                    label = data["label"]
                    score = data["score"]
                    
                    st.divider()
                    st.subheader("Analysis Result:")
                    
                    if label.upper() == "POSITIVE":
                        st.success(f"**Sentiment:** {label} 🎉")
                    else:
                        st.error(f"**Sentiment:** {label} ⚠️")
                        
                    st.metric(label="Confidence Score", value=f"{score * 100:.2f}%")
                    st.progress(score)
                else:
                    st.error(f"Backend Error: {response.json().get('detail', 'Unknown error')}")
                    
            except requests.exceptions.ConnectionError:
                st.error("Could not connect to FastAPI server. Make sure it is running on port 8000!")