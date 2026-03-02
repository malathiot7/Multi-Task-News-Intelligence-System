# Multi-Task-News-Intelligence-System
End-to-end multi-task NLP system for news analytics performing article classification, named entity recognition, and summarization using from-scratch ML/DL models and pretrained transformers. Deployed on AWS with Streamlit UI, S3 model storage, and RDS logging.

✅ Project Blueprint

Multi-Task News Intelligence System:

Classification, NER, and Summarization using From-Scratch Models & Pretrained Transformers
Cloud Deployment: 
Hugging Face, Streamlit, AWS EC2, S3, and RDS

📌 Problem Statement
Build an end-to-end multi-task NLP system that processes news articles to perform:

1.	Text Classification
Predict category → Politics, Business, Tech, Sports, Entertainment, etc.

3.	Named Entity Recognition (NER)
Extract entities → PERSON, ORG, LOC, DATE, PRODUCT, etc.

5.	Summarization
Generate concise summaries → Extractive + Abstractive
For each task you must build:
•	From-Scratch Traditional ML Models
•	From-Scratch DL Models
•	Pretrained Transformer Models
The system must be deployed as a unified Streamlit/Gradio web app on AWS EC2, using S3 for model storage and RDS for logging user activity


🎯 Objectives
•	Implement classification, NER, summarization pipelines under one system.
•	Build scratch baselines: ML (BoW/TF-IDF), DL (CNN/LSTM/BiLSTM/Seq2Seq).
•	Fine-tune transformers → BERT, DistilBERT, BART, T5, RoBERTa.
•	Compare feature representations (BoW vs TF-IDF vs Word2Vec).
•	Deploy full system → EC2 + S3 + RDS + Streamlit.
•	Log all user interactions (task, model, output, time, etc.).

🏗️ Approach & Architecture
1. Data Preparation
Dataset: Microsoft PENS – Personalized News Headlines / Articles

2. Preprocessing
Common text cleaning:

✔ Remove HTML, emojis, URLs
✔ Normalize punctuation
✔ Lower-casing (except transformer or NER models)
✔ Whitespace normalization

Tokenization & stopwords:
•	BoW/TF-IDF → remove stopwords
•	NER → keep casing + token boundaries

Labeling and sequences:
•	Classification → LabelEncoder
•	NER → BIO/BILOU tagging
•	Summarization → truncation + length control

Feature Representations
•	BoW / TF-IDF (CountVectorizer / TfidfVectorizer)
•	Word2Vec / GloVe embeddings
•	Transformer tokenization (BERT, T5, BART)

4. Exploratory Data Analysis
   
Classification EDA:
•	Category distribution
•	Per-category word counts
•	Word clouds / top keywords

NER EDA:
•	Entity type distribution
•	Examples of high-entity-density sentences

Summarization EDA:
•	Article vs summary lengths
•	Compression ratios

General text stats:
•	Vocabulary size
•	Frequent n-grams
•	TF-IDF heatmaps per topic

 6. Model Building
You will build 3 model families per task:

| Task           | ML Baseline       | Custom DL           | Transformer           |
| -------------- | ----------------- | ------------------- | --------------------- |
| Classification | LogReg, SVM, NB   | CNN / LSTM / BiLSTM | BERT, DistilBERT      |
| NER            | Rule-based        | BiLSTM / BiLSTM-CRF | BERT Token Classifier |
| Summarization  | TF-IDF / TextRank | Seq2Seq (LSTM)      | T5, BART              |


4.1 Text Classification

[1] ML Baselines (BoW / TF-IDF)
•	Logistic Regression
•	SVM
•	Multinomial Naive Bayes

[2] DL Baseline (Word2Vec + CNN/LSTM/BiLSTM)
•	Embedding layer (Word2Vec / GloVe / trainable)
•	CNN or LSTM/BiLSTM
•	Dropout + regularization
•	Early stopping

[3] Pretrained Transformers
•	BERT / DistilBERT / RoBERTa
•	Fine-tuning (Trainer API or custom loop)
________________________________________
4.2 Named Entity Recognition (NER)

[1] Rule-Based Baseline
•	Regex patterns for:
o	Capitalized names
o	Dates
o	Organizations
•	Used as a weak baseline

[2] DL Model: BiLSTM or BiLSTM-CRF
•	Word embeddings (Word2Vec/GloVe)
•	Optional char embeddings
•	BiLSTM → Linear → CRF

[3] Transformer NER
•	BERT-base-cased
•	RoBERTa-large-NER
•	Fine-tuning for token classification
________________________________________
4.3 Summarization

[1] Extractive Baseline
•	TF-IDF sentence scoring
•	TextRank (optional)
•	Top-k sentence selection

[2] Custom Seq2Seq (LSTM/GRU)
•	LSTM/GRU encoder
•	LSTM/GRU decoder with attention
•	Teacher forcing
•	Scheduled sampling

[3] Transformer Summarizers
•	T5-small / T5-base
•	BART-base
•	Pegasus (optional)
•	Evaluate with ROUGE
______________________________
📈 5. Evaluation Framework

| Task           | Metrics                            |
| -------------- | ---------------------------------- |
| Classification | Accuracy, Precision, Recall, F1    |
| NER            | Precision, Recall, F1 (per entity) |
| Summarization  | ROUGE-1, ROUGE-2, ROUGE-L          |

Classification
•	Accuracy, Precision, Recall, F1
•	Confusion matrix
•	Compare:
o	BoW vs TF-IDF
o	Word2Vec vs Transformer

NER
•	Precision, Recall, F1 (micro, macro, per entity)
•	Compare:
o	Rule-based vs BiLSTM vs BERT NER

Summarization
•	ROUGE-1, ROUGE-2, ROUGE-L
•	Human evaluation for coherence
•	Compare extractive vs Seq2Seq vs T5/BART

🖥️ 6. Unified Streamlit Application

Inputs:
•	Text box / file upload

Task selector:
•	Classification
•	NER
•	Summarization

Model selector:
•	From-Scratch ML
•	From-Scratch DL
•	Pretrained Transformer

Outputs:
•	Classification: label + confidence
•	NER: highlighted entities
•	Summarization: summary (with model comparison option)


☁️ 7. AWS Cloud Deployment

| AWS Service | Purpose                             |
| ----------- | ----------------------------------- |
| EC2         | Hosts Streamlit application         |
| S3          | Stores trained models & vectorizers |
| RDS         | Stores inference & user logs        |
| IAM         | Secure access control               |


7.1 RDS (PostgreSQL/MySQL) – User Interaction Logging

| Column Name  | Data Type | Description                          |
| ------------ | --------- | ------------------------------------ |
| id           | INT (PK)  | Unique log ID                        |
| user_id      | VARCHAR   | User identifier                      |
| timestamp    | TIMESTAMP | Inference time                       |
| task_type    | VARCHAR   | Classification / NER / Summarization |
| model_family | VARCHAR   | ML / DL / Transformer                |
| model_name   | VARCHAR   | Model used                           |
| input_length | INT       | Input text length                    |
| output_label | VARCHAR   | Predicted class / summary            |
| error_flag   | BOOLEAN   | Error indicator                      |


Store fields:
•	user_id
•	timestamp
•	task_type
•	model_family
•	model_name
•	input_length
•	output_label/summary_length
•	error_flag
Use:
•	SQLAlchemy / psycopg2 / mysqlclient
•	Credentials via env vars or AWS Secrets Manager


7.2 S3 – Model Artifact Storage

| Path                   | Contents                   |
| ---------------------- | -------------------------- |
| models/classification/ | ML, DL, Transformer models |
| models/ner/            | NER models                 |
| models/summarization/  | Summarization models       |
| artifacts/             | Vectorizers, encoders      |

Store:
•	ML models (.pkl)
•	DL weights (.pt)
•	Transformer checkpoints
•	Vectorizers, label encoders, tokenizers
•	Word2Vec models
Folder structure example:
s3://nlp-multitask/
    models/classification/
    models/ner/
    models/summarization/
    preprocessors/
Lazy loading recommended for speed.


7.3 EC2 – Application Hosting
Steps:
1.	Launch Ubuntu EC2
2.	Install Python, PyTorch, Transformers
3.	Pull project from GitHub
4.	Configure env variables
5.	Pull models from S3
6.	Connect to RDS (private subnet recommended)
7.	Run Streamlit on port 8501
8.	Optional: reverse proxy with Nginx + HTTPS

🏁 Expected Result

A production-style, cloud-deployed, multi-task NLP system with:
•	Robust classification
•	Accurate NER
•	High-quality summarization
•	Unified user-friendly interface
•	Reliable logging + analytics
•	Scalable architecture

📚 Project Evaluation Criteria

•	Functionality
•	Model performance
•	Deployment quality
•	UI/UX
•	Logging and monitoring
•	Documentation (README + diagrams)
•	Code quality and explainability

| Component        | Description                           |
| ---------------- | ------------------------------------- |
| Input Layer      | News article text (paste or upload)   |
| Preprocessing    | Cleaning, tokenization, vectorization |
| Task Selector    | Classification / NER / Summarization  |
| Model Layer      | ML, DL (from scratch), Transformer    |
| Inference Engine | Runs selected model                   |
| Output Layer     | Labels, entities, summaries           |
| Logging Layer    | Stores inference metadata             |
| UI               | Streamlit web application             |
| Cloud            | AWS EC2, S3, RDS                      |

FOLDER STRUCTURE

├── data/

├── notebooks/

├── models/

│   ├── classification/
|        |-ml
|        |-dl
|        |-Transformer

│   ├── ner/
|        |-ml
|        |-dl
|        |-Transformer

│   └── summarization/
|        |-ml
|        |-dl
|        |-Transformer

├── app/
│   └── news.py

├── requirements.txt
├── README.md


models/classification/ml/      -> tfidf.pkl, logreg.pkl, label_encoder.pkl

models/classification/dl/      -> bilstm_classifier.py, bilstm_classifier.pt, word2idx.pkl, idx2label.pkl

models/classification/transformer/ -> bert_classifier/...

ner/ml/                         -> rule_patterns.json

ner/dl/                         -> bilstm_ner.py, bilstm_ner.pt, word2idx.pkl, id2tag.pkl

ner/transformer/                -> bert_ner/..., id2tag.pkl

summarization/ml/               -> tfidf_extractive.pkl

summarization/dl/               -> seq2seq_lstm.py, seq2seq_lstm.pt, word2idx.pkl, idx2word.pkl

summarization/transformer/      -> bart_summarizer/...











