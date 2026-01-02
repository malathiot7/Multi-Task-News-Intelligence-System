import streamlit as st
import torch
import torch.nn as nn
import joblib
import json
import pickle
import pandas as pd
import os
import datetime
import re
import numpy as np
from transformers import (
    BertTokenizerFast,
    BertForSequenceClassification,
    BertForTokenClassification,
    BartTokenizer,
    BartForConditionalGeneration
)

# ----------------- DL CLASS DEFINITIONS -----------------
class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_dim*2, num_classes)

    def forward(self, x):
        emb = self.embedding(x)
        _, (h, _) = self.lstm(emb)
        return self.fc(torch.cat((h[-2], h[-1]), dim=1))

class BiLSTMNER(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_tags):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            bidirectional=True,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_dim * 2, num_tags)

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.lstm(x)
        x = self.fc(x)
        return x

class Seq2Seq(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.encoder = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.decoder = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, src, tgt):
        # Encode source
        src_emb = self.embedding(src)
        _, (h, c) = self.encoder(src_emb)

        # Decode target
        tgt_emb = self.embedding(tgt)
        out, _ = self.decoder(tgt_emb, (h, c))

        # Project to vocab
        return self.fc(out)


# ----------------- CONFIG -----------------
st.set_page_config(page_title="Multi-Task NLP System", layout="wide")

# ----------------- LOGIN -----------------
if "user" not in st.session_state:
    st.session_state.user = None

if st.session_state.user is None:
    st.title("🔐 Login")
    username = st.text_input("Username")
    if st.button("Login") and username:
        st.session_state.user = username
        st.rerun()  # Fixed for Streamlit 1.52
    st.stop()

# ----------------- MODEL LOADERS -----------------
@st.cache_resource
def load_ml_classifier():
    tfidf = joblib.load("models/classification/ml/tfidf.pkl")
    clf = joblib.load("models/classification/ml/logreg.pkl")
    le = joblib.load("models/classification/ml/label_encoder.pkl")
    return tfidf, clf, le

@st.cache_resource
def load_bert_classifier():
    path = "models/classification/transformer/bert_classifier"
    tokenizer = BertTokenizerFast.from_pretrained(path)
    model = BertForSequenceClassification.from_pretrained(path)
    model.eval()
    return tokenizer, model

@st.cache_resource
def load_rule_ner():
    with open("models/ner/ml/rule_patterns.json") as f:
        return json.load(f)

@st.cache_resource
def load_bert_ner():
    path = "models/ner/transformer/bert_ner"
    tokenizer = BertTokenizerFast.from_pretrained(path)
    model = BertForTokenClassification.from_pretrained(path)
    model.eval()
    id2tag = pickle.load(open("models/ner/dl/id2tag.pkl", "rb"))
    return tokenizer, model, id2tag

@st.cache_resource
def load_dl_classification():
    le = joblib.load("models/classification/ml/label_encoder.pkl")
    num_classes = len(le.classes_)
    model = BiLSTMClassifier(vocab_size=10000, embed_dim=100, hidden_dim=64, num_classes=num_classes)
    checkpoint_path = "models/classification/dl/bilstm_classifier.pt"
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(checkpoint)
    model.eval()
    word2idx = pickle.load(open("models/classification/dl/word2idx.pkl", "rb"))
    idx2label = pickle.load(open("models/classification/dl/idx2label.pkl", "rb"))
    return model, word2idx, idx2label

@st.cache_resource
def load_dl_ner():
    import torch, pickle, os

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    ner_dl_dir = os.path.join(BASE_DIR, "models", "ner", "dl")

    # Load vocab and tags
    with open(os.path.join(ner_dl_dir, "word2idx.pkl"), "rb") as f:
        word2idx = pickle.load(f)

    with open(os.path.join(ner_dl_dir, "idx2tag.pkl"), "rb") as f:
        idx2tag = pickle.load(f)

    VOCAB_SIZE = len(word2idx)
    NUM_TAGS = len(idx2tag)

    # 🔥 MUST MATCH TRAINING EXACTLY
    EMBED_DIM = 32
    HIDDEN_DIM = 32

    model_dl = BiLSTMNER(
        vocab_size=VOCAB_SIZE,
        embed_dim=EMBED_DIM,
        hidden_dim=HIDDEN_DIM,
        num_tags=NUM_TAGS
    )

    checkpoint_path = os.path.join(ner_dl_dir, "bilstm_ner.pt")
    model_dl.load_state_dict(
        torch.load(checkpoint_path, map_location="cpu")
    )

    model_dl.eval()
    return model_dl, word2idx, idx2tag



@st.cache_resource
def load_bart():
    path = "models/summarization/transformer/bart_summarizer"
    tokenizer = BartTokenizer.from_pretrained(path)
    model = BartForConditionalGeneration.from_pretrained(path)
    model.eval()
    return tokenizer, model

@st.cache_resource
def load_seq2seq():
    # Load vocab mapping used during training
    word2idx = pickle.load(open("models/summarization/dl/word2idx.pkl", "rb"))
    idx2word = pickle.load(open("models/summarization/dl/idx2word.pkl", "rb"))

    vocab_size = len(word2idx)
    embed_dim = 32   # must match training
    hidden_dim = 32  # must match training

    # Initialize model
    model = Seq2Seq(vocab_size=vocab_size, embed_dim=embed_dim, hidden_dim=hidden_dim)

    # Load checkpoint
    checkpoint_path = "models/summarization/dl/seq2seq_lstm.pt"
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(checkpoint)
    else:
        st.warning(f"Checkpoint not found at {checkpoint_path}. Model initialized randomly.")

    model.eval()
    return model, word2idx, idx2word


@st.cache_resource
def load_ml_summarizer():
    return pickle.load(open("models/summarization/ml/tfidf_extractive.pkl", "rb"))

# ----------------- LOGGING -----------------
def log_event(task, family, model_name, input_len, output_info):
    log_path = "logs.csv"
    row = {
        "user": st.session_state.user,
        "timestamp": datetime.datetime.now(),
        "task_type": task,
        "model_family": family,
        "model_name": model_name,
        "input_length": input_len,
        "output_info": output_info
    }
    df = pd.DataFrame([row])
    df.to_csv(log_path, mode="a", header=not os.path.exists(log_path), index=False)

# ----------------- TF-IDF Extractive Summarizer -----------------
def tfidf_extractive_summary(text, config):
    sentences = re.split(r'(?<=[.!?]) +', text)
    if len(sentences) <= config["top_k"]:
        return text
    from sklearn.feature_extraction.text import TfidfVectorizer
    X = TfidfVectorizer().fit_transform(sentences)
    scores = np.array(X.sum(axis=1)).ravel()
    top_idx = scores.argsort()[-config["top_k"]:][::-1]
    summary = " ".join([sentences[i] for i in sorted(top_idx)])
    return summary

# ----------------- UI -----------------
st.sidebar.title("Control Panel")
task = st.sidebar.selectbox("Task", ["Classification", "NER", "Summarization"])
model_family = st.sidebar.selectbox(
    "Model Type",
    ["From-Scratch ML", "From-Scratch DL", "Transformer"]
)
text = st.text_area("Input Text", height=200)

# ----------------- INFERENCE -----------------
if st.button("Run") and text.strip():

    # ---------------- Classification ----------------
    if task == "Classification":
        if model_family == "From-Scratch ML":
            tfidf, clf, le = load_ml_classifier()
            vec = tfidf.transform([text])
            probs = clf.predict_proba(vec)[0]
            pred = probs.argmax()
            label = le.inverse_transform([pred])[0]
            conf = probs[pred]
            st.success(f"Prediction: {label}")
            st.info(f"Confidence: {conf:.2f}")
            log_event(task, model_family, "logreg_tfidf", len(text), label)

        elif model_family == "From-Scratch DL":
            model_dl, word2idx, idx2label = load_dl_classification()
            tokens = text.lower().split()
            ids = [word2idx.get(t, 1) for t in tokens]
            max_len = 100
            ids = ids[:max_len] + [0]*(max_len-len(ids)) if len(ids)<max_len else ids[:max_len]
            tensor_input = torch.tensor([ids])
            with torch.no_grad():
                logits = model_dl(tensor_input)
                probs = torch.softmax(logits, dim=1)
            pred = torch.argmax(probs, dim=1).item()
            confidence = probs[0][pred].item()
            label = idx2label[pred]
            st.success(f"Prediction: {label}")
            st.info(f"Confidence: {confidence:.2f}")
            log_event(task, model_family, "bilstm_classifier", len(text), f"{label} ({confidence:.2f})")

        elif model_family == "Transformer":
            tokenizer, model = load_bert_classifier()
            inputs = tokenizer(text, return_tensors="pt", truncation=True)
            with torch.no_grad():
                outputs = model(**inputs)
                probs = torch.softmax(outputs.logits, dim=1)
            pred = probs.argmax().item()
            conf = probs[0][pred].item()
            st.success(f"Prediction: {pred}")
            st.info(f"Confidence: {conf:.2f}")
            log_event(task, model_family, "bert_classifier", len(text), pred)

    # ---------------- NER ----------------
    elif task == "NER":
        if model_family == "From-Scratch ML":
            patterns = load_rule_ner()
            entities = []
            for label, pattern in patterns.items():
                for m in re.finditer(pattern, text):
                    entities.append((m.group(), label))
            st.write("Entities:", entities)
            log_event(task, model_family, "rule_based", len(text), len(entities))

        elif task == "NER" and model_family == "From-Scratch DL":
            # Load model
            model_dl, word2idx, idx2tag = load_dl_ner()

            # Preprocess text
            tokens = text.lower().split()
            ids = [word2idx.get(tok, 1) for tok in tokens]  # 1 = <UNK>

            max_len = 20
            seq_len = len(ids)
            ids = ids[:max_len] + [0]*(max_len - len(ids)) if len(ids)<max_len else ids[:max_len]
            tensor_input = torch.tensor([ids])

            # Run model
            with torch.no_grad():
                output = model_dl(tensor_input)

            # Get predicted tags
            preds = torch.argmax(output, dim=2)[0][:seq_len]

            # Map tokens → tags
            entities = [(tok, idx2tag[p.item()]) for tok, p in zip(tokens, preds)]

            # Display table
            st.dataframe(pd.DataFrame(entities, columns=["Token", "Tag"]))

            # Highlight entities in text
            highlighted = text
            offset = 0
            for tok, tag in entities:
                if tag != "O":  # only highlight entities
                    start = highlighted.lower().find(tok, offset)
                    if start != -1:
                        end = start + len(tok)
                        highlighted = highlighted[:start] + f"**[{tok} ({tag})]**" + highlighted[end:]
                        offset = start + len(f"**[{tok} ({tag})]**")

            st.markdown("### Highlighted Text")
            st.markdown(highlighted)

            # Log event
            log_event(task, model_family, "bilstm_ner", len(text), len(entities))




        elif task == "NER" and model_family == "Transformer":
            tokenizer, model, id2tag = load_bert_ner()
            
            # Tokenize input text
            inputs = tokenizer(text, return_tensors="pt", truncation=True)
            
            with torch.no_grad():
                outputs = model(**inputs)
            
            # Get predicted tag IDs
            preds = torch.argmax(outputs.logits, dim=2)[0]
            
            # Convert token IDs to actual tokens
            tokens_bert = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
            
            # Map predicted IDs to tags with fallback
            entities = [
                (tok, id2tag.get(p.item(), f"TAG_{p.item()}"))
                for tok, p in zip(tokens_bert, preds)
                if tok not in ["[CLS]", "[SEP]"]
            ]
            
            # Show entities in a table
            if entities:
                st.success(f"Found {len(entities)} entities")
                st.dataframe(pd.DataFrame(entities, columns=["Token", "Tag"]))
            else:
                st.warning("No entities found")

            # Highlight entities in text
            highlighted = text
            offset = 0
            for tok, tag in entities:
                if tag != "O":
                    start = highlighted.find(tok, offset)
                    if start != -1:
                        end = start + len(tok)
                        highlighted = highlighted[:start] + f"**[{tok} ({tag})]**" + highlighted[end:]
                        offset = start + len(f"**[{tok} ({tag})]**")
            
            st.markdown("### Highlighted Text")
            st.markdown(highlighted)
            
            # Log event
            log_event(
                task="NER",
                family="Transformer",
                model_name="bert_ner",
                input_len=len(text),
                output_info=len(entities)
            )


    # ---------------- Summarization ----------------
    elif task == "Summarization":
        if model_family == "From-Scratch ML":
            config = load_ml_summarizer()
            summary = tfidf_extractive_summary(text, config)
            st.success(summary)
            log_event(task, model_family, "tfidf_extractive", len(text), len(summary))

        elif model_family == "From-Scratch DL":
            model_dl, word2idx, idx2word = load_seq2seq()

            # Tokenize input text
            tokens = text.lower().split()
            ids = [word2idx.get(t, 1) for t in tokens]  # 1 = <UNK>
            max_len = 100
            ids = ids[:max_len] + [0]*(max_len-len(ids)) if len(ids)<max_len else ids[:max_len]
            tensor_input_src = torch.tensor([ids])
            tensor_input_tgt = torch.tensor([ids])  # teacher forcing for now

            # Run model
            with torch.no_grad():
                output_logits = model_dl(tensor_input_src, tensor_input_tgt)

            # Convert logits to word IDs
            output_ids = torch.argmax(output_logits, dim=2)

            # Convert IDs to words
            summary = " ".join([idx2word[i.item()] for i in output_ids[0] if i.item() != 0])  # ignore <PAD>

            st.success(summary)
            log_event(task, model_family, "seq2seq_lstm", len(text), len(summary))


        elif model_family == "Transformer":
            tokenizer, model = load_bart()
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024)
            with torch.no_grad():
                summary_ids = model.generate(inputs["input_ids"], max_length=130, min_length=30, num_beams=4)
            summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            st.success(summary)
            log_event(task, model_family, "bart_summarizer", len(text), len(summary))


