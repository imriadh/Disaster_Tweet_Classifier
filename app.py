import streamlit as st
import joblib
import re
import os

# --- Page Configuration ---
st.set_page_config(
    page_title="Disaster Tweet Classifier", 
    page_icon="🌪️", 
    layout="centered",
    initial_sidebar_state="expanded"
)

# --- Load Model and Vectorizer ---
@st.cache_resource
def load_resources():
    # Check if files exist in the current directory
    if not os.path.exists('linear_svm_model.pkl') or not os.path.exists('tfidf_vectorizer.pkl'):
        st.error("❌ Model files not found! Please ensure 'linear_svm_model.pkl' and 'tfidf_vectorizer.pkl' are in the same directory as app.py.")
        st.stop()
    
    vectorizer = joblib.load('tfidf_vectorizer.pkl')
    model = joblib.load('linear_svm_model.pkl')
    return vectorizer, model

vectorizer, model = load_resources()

# --- Preprocessing Function (Must match training) ---
def clean_text(text):
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# --- UI Layout ---
st.title("🌪️ Disaster Tweet Classifier")
st.markdown("""
**Project Overview:**  
This application uses a **Linear Support Vector Machine (SVM)** paired with **TF-IDF** 
feature extraction to predict whether a tweet describes a real disaster.
""")

st.divider()

# Input Section
st.subheader("Enter a Tweet to Analyze")
user_input = st.text_area(
    "Tweet Text:", 
    height=120, 
    placeholder="e.g., Forest fire near La Ronge Sask. Canada",
    help="Paste the text of a tweet here."
)

# Prediction Button
if st.button("🔍 Analyze Tweet", type="primary", use_container_width=True):
    if user_input.strip():
        # 1. Preprocess the text
        cleaned_text = clean_text(user_input)
        
        # 2. Vectorize the text
        text_vectorized = vectorizer.transform([cleaned_text])
        
        # 3. Predict
        prediction = model.predict(text_vectorized)[0]
        
        # 4. Display Result
        st.divider()
        if prediction == 1:
            st.error("🚨 **Prediction: REAL DISASTER**")
            st.info("This tweet is highly likely to be describing a real emergency or disaster.")
        else:
            st.success("✅ **Prediction: NOT a Real Disaster**")
            st.info("This tweet is likely metaphorical, sarcastic, or unrelated to a real emergency.")
            
        # Show cleaned text for transparency
        with st.expander("View Processed Text"):
            st.write(f"**Original:** {user_input}")
            st.write(f"**Cleaned:** {cleaned_text}")
    else:
        st.warning("⚠️ Please enter some text to analyze.")

# --- Sidebar Information ---
with st.sidebar:
    st.header("ℹ️ About the Model")
    st.markdown("""
    - **Algorithm:** Linear Support Vector Machine (LinearSVC)
    - **Feature Extraction:** TF-IDF (Unigrams + Bigrams)
    - **Training Data:** 7,613 labeled tweets
    - **Validation F1 Score:** 0.7737
    - **Primary Metric:** F1-Score (Balances Precision & Recall)
    """)
    
    st.divider()
    st.header(" Academic Project")
    st.markdown("""
    Built as part of a University NLP project to filter metaphorical and sarcastic language from genuine emergency reports.
    """)

# --- Footer ---
st.divider()
st.caption("Built with Streamlit | Model: Linear SVM (TF-IDF) | Academic Project")
