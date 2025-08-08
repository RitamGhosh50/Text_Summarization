import streamlit as st
from transformers AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import pipeline
import torch

# Load models (cached to avoid reloading on every interaction)
@st.cache_resource
def load_models():
    # Hugging Face model (e.g., BART)
    hf_summarizer = pipeline(
        "summarization", 
        model="facebook/bart-large-cnn",
        device=0 if torch.cuda.is_available() else -1
    )
    
    # T5 model
    t5_tokenizer = AutoTokenizer.from_pretrained("t5-small")
    t5_model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")
    
    return hf_summarizer, t5_tokenizer, t5_model

# Summarization functions
def summarize_hf(text, _summarizer):
    summary = _summarizer(
        text,
        max_length=150,
        min_length=50,
        do_sample=False
    )
    return summary[0]['summary_text']

def summarize_t5(text, _tokenizer, _model):
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
    st.title("📝 Text Summarization Comparison")
    st.markdown("Compare summaries from Hugging Face (BART) and T5 models")
    
    # Load models (shows loading spinner)
    with st.spinner("Loading models..."):
        hf_summarizer, t5_tokenizer, t5_model = load_models()
    
    # Text input
    input_text = st.text_area(
        "Enter text to summarize:",
        height=200,
        placeholder="Paste your article or long text here..."
    )
    
    # Process button
    if st.button("✨ Proceed"):
        if not input_text.strip():
            st.warning("Please enter some text to summarize")
        else:
            with st.spinner("Generating summaries..."):
                # Create two columns for results
                col1, col2 = st.columns(2)
                
                # Hugging Face summary
                with col1:
                    st.subheader("Hugging Face (BART)")
                    hf_summary = summarize_hf(input_text, hf_summarizer)
                    st.info(hf_summary)
                
                # T5 summary
                with col2:
                    st.subheader("T5 Model")
                    t5_summary = summarize_t5(input_text, t5_tokenizer, t5_model)
                    st.success(t5_summary)
                
                # Show token counts
                st.caption(f"Original length: {len(input_text.split())} tokens")
                st.caption(f"BART summary: {len(hf_summary.split())} tokens | T5 summary: {len(t5_summary.split())} tokens")

if __name__ == "__main__":

    main()
