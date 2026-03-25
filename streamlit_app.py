import streamlit as st
import os
import tempfile
import torch
from pathlib import Path
from torchtext.data.utils import get_tokenizer

from src.predict import load_config, tokenize_and_pad
from src.utils.helpers import load_checkpoint
from src.utils.preprocessing import FixedVocab
from src.models.rnn import RNNClassifier
from src.models.transformer import TransformerClassifier


@st.cache_data(show_spinner=False)
def read_class_labels(path: str):
    with open(path, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


@st.cache_data(show_spinner=False)
def load_model(config_path: str, checkpoint_path: str, device: str):
    config = load_config(config_path)
    model_type = config.get("model")
    if model_type not in ["transformer", "rnn"]:
        raise ValueError("Config model must be 'transformer' or 'rnn'.")

    max_len = config.get("max_len", 128)
    model_config = config.get(model_type, {})

    embed_dim = model_config.get("embed_dim") or config.get("embed_dim")
    hidden_dim = model_config.get("hidden_dim") or config.get("hidden_dim")
    num_layers = model_config.get("num_layers") or config.get("num_layers")
    dropout = model_config.get("dropout") or config.get("dropout", 0.0)

    checkpoint = load_checkpoint(None, None, checkpoint_path, device)
    vocab_stoi = checkpoint.get("vocab_stoi")
    if vocab_stoi is None:
        raise ValueError("Checkpoint must include vocab_stoi for inference")

    vocab = FixedVocab(vocab_stoi)
    num_classes = config.get("num_classes", 14)

    if model_type == "transformer":
        num_heads = model_config.get("num_heads") or config.get("num_heads")
        model = TransformerClassifier(
            vocab_size=len(vocab),
            embed_dim=embed_dim,
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_classes=num_classes,
            max_len=max_len,
            pad_index=vocab["<pad>"],
            dropout=dropout,
        )
    else:
        model = RNNClassifier(
            vocab_size=len(vocab),
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_classes=num_classes,
            pad_index=vocab["<pad>"],
            dropout=dropout,
        )

    model.to(device)
    load_checkpoint(model, None, checkpoint_path, device)
    model.eval()

    return config, model, vocab


def predict_text(model, vocab, text: str, tokenizer, max_len: int, device: str):
    seqs, mask = tokenize_and_pad([text], tokenizer, vocab, max_len, pad_index=vocab["<pad>"], device=device)
    with torch.no_grad():
        logits = model(seqs, mask)
        pred = torch.argmax(logits, dim=-1).item()
    return pred


def main():
    st.title("Text Classification Demo")

    st.markdown(
        """
        Enter text to classify using a saved model.
        """
    )

    base_dir = Path(".")
    config_file = st.text_input("Config file path", value=str(base_dir / "config.yml"))

    uploaded_checkpoint = st.file_uploader("Upload checkpoint file (browse)", type=["pth", "pt"])

    input_text = st.text_area("Text to classify", value="Enter text here...")
    run_predict = st.button("Predict")

    if run_predict:
        if not input_text.strip():
            st.error("Please enter text to classify.")
            return

        if not os.path.exists(config_file):
            st.error(f"Config file not found: {config_file}")
            return

        if uploaded_checkpoint is None:
            st.error("Please upload a checkpoint (.pth/.pt) before predicting.")
            return

        cfg = load_config(config_file)
        temp_ckpt_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pth")
        temp_ckpt_file.write(uploaded_checkpoint.read())
        temp_ckpt_file.flush()
        temp_ckpt_file.close()
        ckpt = temp_ckpt_file.name

        if not os.path.exists(ckpt):
            st.error(f"Checkpoint not found: {ckpt}")
            try:
                os.remove(temp_ckpt_file.name)
            except Exception:
                pass
            return

        label_path = base_dir / "data" / "classes.txt"
        if not os.path.exists(label_path):
            st.error(f"Default class file not found: {label_path}")
            return
        labels = read_class_labels(str(label_path))

        device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            config, model, vocab = load_model(config_file, ckpt, device)
            tokenizer = get_tokenizer("basic_english")
            max_len = config.get("max_len", 128)
            pred_idx = predict_text(model, vocab, input_text, tokenizer, max_len, device)
            label_name = labels[pred_idx+1] if 0 <= pred_idx < len(labels) else f"Class {pred_idx+1}"

            st.success("Prediction completed")
            st.write("**Predicted class:**", label_name)
        except Exception as e:
            st.error(f"Prediction error: {e}")
        finally:
            if temp_ckpt_file is not None:
                try:
                    os.remove(temp_ckpt_file.name)
                except Exception:
                    pass


if __name__ == "__main__":
    main()
