import streamlit as st
from transformers import pipeline
import os

# 1. Page Configuration (GUI Layout)
st.set_page_config(page_title="Quora Similarity Tool", page_icon="❓", layout="centered")

# 2. Sidebar for Project Info (SGSITS Branding)
with st.sidebar:
    st.title("Project Dashboard")
    st.markdown("---")
    st.write("**Student:** Jot Ajmani")
    st.write("**College:** SGSITS Indore")
    st.write("**Model:** Fine-tuned BERT")
    st.write("**Accuracy:** 81.75%")
    st.info("This tool uses NLP to detect if two questions have the same meaning.")

# 3. Model Loading Function (Using Absolute Path for Local Folder)
@st.cache_resource
def load_huggingface_model():
    # Yahan humne local folder ka jhanjhat khatam kar diya
    repo_id = "jotaj30/quora-bert-model"
    return pipeline("text-classification", model=repo_id)

pipe = load_huggingface_model()

# 4. Main UI
st.title("🔍 Quora Question Similarity Checker")
st.markdown("Enter two questions below to see if they are semantically identical.")

# Input fields
q1 = st.text_area("Question 1:", placeholder="Example: How to learn Python?")
q2 = st.text_area("Question 2:", placeholder="Example: What is the best way to study Python?")

# Action Button
if st.button("Analyze Similarity"):
    if q1 and q2:
        if pipe is not None:
            with st.spinner('BERT is analyzing the context...'):
                # Model Prediction
                raw_result = pipe({"text": q1, "text_pair": q2})
                
                # Handling list or dict output (Fixes KeyError: 0)
                if isinstance(raw_result, list):
                    result = raw_result[0]
                else:
                    result = raw_result
                
                label = result['label']
                score = result['score']

                st.divider()
                
                # Result Logic
                # (Note: Agar aapke model mein 1=Duplicate hai, toh LABEL_1 check karein)
                if label == "LABEL_1": 
                    st.success(f"✅ **Duplicate Found!** (Confidence: {score:.2f})")
                    st.balloons() # Visual effect
                else:
                    st.error(f"❌ **Not Duplicate** (Confidence: {score:.2f})")
        else:
            st.error("Model is not loaded. Please check the folder path.")
    else:
        st.warning("Please enter both questions first.")
