import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# Cache models to avoid reloading on every interaction
@st.cache_resource
def load_models():
    # Hugging Face BART model
    hf_summarizer = pipeline(
        "summarization", 
        model="facebook/bart-large-cnn",
        device=0 if torch.cuda.is_available() else -1
    )
    
    # T5 model
    t5_tokenizer = AutoTokenizer.from_pretrained("t5-small")
    t5_model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")
    
    return hf_summarizer, t5_tokenizer, t5_model

def summarize_hf(text, _summarizer):
    """Summarize using Hugging Face BART"""
    return _summarizer(text, max_length=150, min_length=50)[0]['summary_text']

def summarize_t5(text, _tokenizer, _model):
    """Summarize using T5"""
    inputs = _tokenizer.encode(
        "summarize: " + text,
        return_tensors="pt",
        max_length=512,
        truncation=True
    )
    outputs = _model.generate(
        inputs,
        max_length=150,
        min_length=50,
        length_penalty=2.0,
        num_beams=4,
        early_stopping=True
    )
    return _tokenizer.decode(outputs[0], skip_special_tokens=True)

# Streamlit UI
def main():
    st.set_page_config(page_title="Text Summarizer", layout="wide")
    st.title("📝 Summarization Model Comparison")
    st.markdown("Compare **Hugging Face (BART)** and **T5** summarization models")
    
    # Load models
    with st.spinner("Loading models..."):
        hf_summarizer, t5_tokenizer, t5_model = load_models()
    
    # Input text
    text = st.text_area(
        "Enter text to summarize:", 
        height=200,
        placeholder="Paste your article, essay, or long text here..."
    )
    
    if st.button("✨ Generate Summaries"):
        if not text.strip():
            st.warning("Please enter some text!")
        else:
            with st.spinner("Processing..."):
                col1, col2 = st.columns(2)
                
                # Hugging Face Summary
                with col1:
                    st.subheader("Hugging Face (BART)")
                    hf_summary = summarize_hf(text, hf_summarizer)
                    st.info(hf_summary)
                    st.caption(f"Length: {len(hf_summary.split())} words")
                
                # T5 Summary
                with col2:
                    st.subheader("T5 Model")
                    t5_summary = summarize_t5(text, t5_tokenizer, t5_model)
                    st.success(t5_summary)
                    st.caption(f"Length: {len(t5_summary.split())} words")
            
            # Original text stats
            st.divider()
            st.caption(f"Original text: {len(text.split())} words | {len(text)} characters")

if __name__ == "__main__":
    main()

