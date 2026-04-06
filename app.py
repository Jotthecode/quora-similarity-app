import streamlit as st
from transformers import pipeline
import pandas as pd
from annotated_text import annotated_text
import re

# 1. Page Configuration
st.set_page_config(page_title="Semantic Similarity AI", page_icon="🧠", layout="wide")

# Custom CSS for Professional Look
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    </style>
    """, unsafe_allow_html=True)

# 2. Sidebar Branding
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/en/2/23/SGSITS_Logo.png", width=80)
    st.title("🛡️ AI Security Lab")
    st.markdown("---")
    st.info("**Developer:** Jot Ajmani (SGSITS)\n\n**Model:** BERT-Base-Uncased\n\n**Accuracy:** 81.75%")

# 3. Model Loading
@st.cache_resource
def load_nlp_pipeline():
    repo_id = "jotaj30/quora-bert-model"
    return pipeline("text-classification", model=repo_id)

pipe = load_nlp_pipeline()

# 4. Header
st.title("🔍 Semantic Question Similarity Tool")
st.write("Using Transformer Architecture to identify semantic duplicates in Quora datasets.")

col1, col2 = st.columns(2)
with col1:
    q1 = st.text_area("Question A", placeholder="e.g. How to stay fit?", height=120)
with col2:
    q2 = st.text_area("Question B", placeholder="e.g. What is the secret to a healthy body?", height=120)

st.markdown("---")

# Logic must be INSIDE the button click
if st.button("🚀 Run Deep Semantic Analysis", use_container_width=True):
    if q1.strip() and q2.strip():
        with st.spinner('🔄 BERT is calculating attention weights...'):
            try:
                # Model Inference
                raw_result = pipe({"text": q1, "text_pair": q2})
                result = raw_result[0] if isinstance(raw_result, list) else raw_result
                label = result['label']
                score = result['score']
                is_duplicate = (label == "LABEL_1")

                # Layout for Results
                res_col1, res_col2 = st.columns([1, 1])

                with res_col1:
                    st.metric(label="Similarity Score", value=f"{score:.2%}")
                    if is_duplicate:
                        st.success("✅ Verdict: DUPLICATE INTENT")
                        st.balloons()
                    else:
                        st.error("❌ Verdict: UNIQUE INTENT")
                    
                    # --- RECOMMENDATIONS ADDED HERE ---
                    
                    if is_duplicate:
                        st.balloons()
                        st.info("### 💡 AI Recommendations")
                        all_text = (q1 + " " + q2).lower()
                        words = re.findall(r'\b\w{4,}\b', all_text) # Sirf 4+ letters waale words
                        stop_words = {'what', 'how', 'is', 'are', 'the', 'this', 'that', 'with', 'related', 'stay', 'being'}
                        dynamic_tags = list(set([word for word in words if word not in stop_words]))[:4]
                        tag_string = " ".join([f"#{tag.capitalize()}" for tag in dynamic_tags])
                        st.markdown(f"""
                        * **Merge Content:** These questions are semantically identical. We recommend merging them to avoid answer fragmentation.
                        * **Top Tags suggested:** {tag_string if tag_string else "#General #Knowledge"}
                        * **Action:** Redirect new users to the existing high-authority thread of Question A.
                        """)
    
    # Common useless words (Stopwords) ko ignore karna
    
                    else:
                        st.warning("""
                        * **Keep Separate:** These questions have distinct intents. Maintain separate threads.
                        * **Context Gap:** Question B covers a different niche. Consider adding specific tags to differentiate them further.
                        """)

                with res_col2:
                    st.write("### 📝 Explainable AI: Token Match")
                    words_q2 = re.findall(r'\w+', q2.lower())
                    annotated_content = []
                    for word in q1.split():
                        clean_word = re.sub(r'[^\w]', '', word.lower())
                        if clean_word in words_q2:
                            annotated_content.append((word, "Match", "#afa"))
                        else:
                            annotated_content.append(word + " ")
                    annotated_text(*annotated_content)

                # Metadata Table
                st.write("### 📊 Inference Metadata")
                df_meta = pd.DataFrame({
                    "Parameter": ["Model Type", "Softmax Probability", "Hardware"],
                    "Value": ["Transformer (BERT)", f"{score:.4f}", "CPU/Neural Engine"]
                })
                st.table(df_meta)

            except Exception as e:
                st.error(f"Inference Error: {e}")
    else:
        st.warning("Please enter both questions to proceed.")

# 5. Technical Documentation Section
st.markdown("---")
with st.expander("🛠️ Technical Methodology & Architecture"):
    st.write("""
    This system utilizes a **Cross-Encoder architecture**. Unlike Bi-Encoders that process sentences separately, 
    the Cross-Encoder processes both questions simultaneously through the attention layers of BERT.
    """)
