import streamlit as st
import torch
from transformers import (
    pipeline,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    logging
)

# Reduce verbosity of transformers logging
logging.set_verbosity_error()

@st.cache_resource
def load_models():
    try:
        # Load BART model (Hugging Face)
        hf_model = pipeline(
            "summarization",
            model="facebook/bart-large-cnn",
            device="cpu"  # Streamlit Cloud uses CPU
        )
        
        # Load T5 model
        t5_tokenizer = AutoTokenizer.from_pretrained("t5-small")
        t5_model = AutoModelForSeq2SeqLM.from_pretrained(
            "t5-small",
            torch_dtype=torch.float32  # Ensure CPU compatibility
        )
        return hf_model, t5_tokenizer, t5_model
    except Exception as e:
        st.error(f"❌ Model loading failed: {str(e)}")
        return None, None, None

def main():
    st.title("🔍 Text Summarizer Comparison")
    st.caption("Comparing BART and T5 models")
    
    # Load models with progress
    with st.status("Loading models...", expanded=True) as status:
        hf_model, t5_tokenizer, t5_model = load_models()
        if None in (hf_model, t5_tokenizer, t5_model):
            st.stop()
        status.update(label="Models loaded!", state="complete")
    
    # Input text
    text = st.text_area("Enter text to summarize:", height=200)
    
    if st.button("Generate Summaries"):
        if not text.strip():
            st.warning("Please enter text first!")
        else:
            with st.spinner("Processing..."):
                try:
                    # BART Summary
                    bart_summary = hf_model(
                        text,
                        max_length=150,
                        min_length=50,
                        do_sample=False
                    )[0]['summary_text']
                    
                    # T5 Summary
                    t5_inputs = t5_tokenizer(
                        "summarize: " + text,
                        return_tensors="pt",
                        max_length=512,
                        truncation=True
                    )
                    t5_outputs = t5_model.generate(**t5_inputs, max_length=150)
                    t5_summary = t5_tokenizer.decode(
                        t5_outputs[0],
                        skip_special_tokens=True
                    )
                    
                    # Display results
                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader("BART Summary")
                        st.info(bart_summary)
                    with col2:
                        st.subheader("T5 Summary")
                        st.success(t5_summary)
                        
                except Exception as e:
                    st.error(f"Summarization failed: {str(e)}")

if __name__ == "__main__":
    main()
 

