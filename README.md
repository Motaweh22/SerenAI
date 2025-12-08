# 📘 **README.md — Mental Health RAG System**

---

# 🧠 Mental Health RAG (Retrieval-Augmented Generation) System

*A safe, empathetic AI assistant designed to support users by retrieving real therapist advice instead of hallucinating.*

---

## 🎥 Demo Video

### ▶️ Inline Playback


[![Watch the Demo](https://img.icons8.com/?size=512&id=86188&format=png)](demo.mp4)

### 🔗 Direct Link

[📥 Click here to view the demo video](demo.mp4)

---

## 🌟 Overview

This project implements a **Retrieval-Augmented Generation (RAG)** system tailored for **mental health support**.
Instead of letting the LLM generate answers from scratch (which risks hallucinations or unsafe guidance), the system retrieves **real therapist answers** from a curated dataset and rewrites them in a supportive, empathetic tone.

### Why this matters:

* The LLM *is not the source of truth*
* Real therapist knowledge ensures grounding
* The LLM only **rephrases safely**
* Higher accuracy, lower risk, more trust

---

## 🔧 Key Features

* **Pair Embedding Retrieval** using instruction + merged therapist responses
* **Dense MPNet Embeddings** (all-mpnet-base-v2)
* **FAISS Vector Search** for fast semantic retrieval
* **Multiple LLMs via Unsloth** (Llama, Gemma, Mistral)
* **4-bit quantization** for fast inference
* **Safety prompting** to avoid harmful outputs
* **Streamlit UI** for interactive usage
* **Evaluation pipeline** for retrieval + LLM quality metrics

---

## 🏛️ System Architecture

```
                 ┌─────────────────────────┐
                 │       User Input        │
                 └─────────────┬──────────┘
                               ↓
               ┌────────────────────────────────┐
               │  MPNet Embedding Generation    │
               └─────────────┬──────────────────┘
                               ↓
                ┌──────────────────────────────┐
                │    FAISS Vector Retrieval    │
                │   (Top-k therapist answers)  │
                └──────────────┬───────────────┘
                               ↓
           ┌───────────────────────────────────────┐
           │ Retrieved Therapist Response (Grounded) │
           └────────────────────┬────────────────────┘
                               ↓
     ┌────────────────────────────────────────────────┐
     │  LLM Rephrase (Llama / Gemma / Mistral)        │
     │  + Safety Prompting                             │
     └────────────────────┬────────────────────────────┘
                          ↓
                ┌─────────────────────┐
                │   Final Safe Answer │
                └─────────────────────┘
```

---

## 📚 Dataset & Preprocessing

### Dataset structure:

```json
{
  "instruction": "User asked...",
  "responses": [
    "Therapist answer #1",
    "Therapist answer #2"
  ]
}
```

### Cleaning steps:

* Remove noise (formatting, Reddit artifacts, emojis, signatures)
* Merge all therapist responses into one answer block
* Build final pair text:

```
pair_text = instruction + merged_responses
```

### Why no chunking?

Chunking destroyed emotional meaning → retrieval performed poorly.
Pair embeddings produced extremely high recall (~0.93+).

---

## 🧠 Embedding Model

### **Model:** `sentence-transformers/all-mpnet-base-v2`

Chosen because:

* Excellent semantic understanding
* Strong performance on emotional text
* Stable and high-quality dense embeddings
* Perfect for RAG retrieval

---

## 🔍 Retrieval Engine

### **FAISS (Facebook AI Similarity Search)**

Used for:

* Storing dense vectors
* Fast similarity search
* Scaling to thousands of therapy responses

FAISS provides millisecond-level retrieval performance.

---

## 🤖 LLM Rewriting Models

Loaded using **Unsloth** for efficient 4-bit inference:

| Model            | Purpose                   |
| ---------------- | ------------------------- |
| **Llama 3.2 3B** | Fastest, lightweight      |
| **Gemma 3 4B**   | Most empathetic & natural |
| **Mistral 7B**   | Strong long-form clarity  |

The LLM **does NOT generate knowledge** —
it only **rewrites** the retrieved therapist answer safely.

---

## 📈 Evaluation

### Retrieval Metrics:

```
Recall@1   ≈ 0.93
Recall@3   ≈ 0.98
Recall@5   ≈ 0.98
MRR        ≈ 0.95+
```

### LLM Output Metrics:

* Cosine Similarity
* BERTScore
* ROUGE-L
* Safety classification

Our final system achieved strong grounding and minimal hallucination.

---

## 🧪 How to Run

### 1) Install dependencies

```bash
pip install -r requirements.txt
pip install unsloth transformers accelerate bitsandbytes
```

### 2) Build embeddings + FAISS index

```bash
python loader.py
```

### 3) Run the Streamlit app

```bash
streamlit run app.py
```

### 4) Run evaluation pipeline

```bash
python eval_pipeline.py
```

---

## 🗂️ Repository Structure

```
mental-health-rag/
│
├── app.py                # Streamlit UI
├── loader.py             # Data cleaning + embedding + FAISS index
├── retriever.py          # Semantic search logic
├── answer_service.py     # Final pipeline (retrieval → LLM → safety)
├── llm_client_unsloth.py # LLM loading (Llama/Gemma/Mistral)
├── llm_rephrase.py       # Rewriting layer
├── eval_pipeline.py      # Evaluation of retrieval & LLM
├── cleaned_dataset.json  # Cleaned dataset used for retrieval
├── demo.mp4              # Demo video file
└── README.md
```

---

## 🌱 Future Improvements

* Hybrid retrieval (dense + sparse + lexical fusion)
* Add ColBERT late-interaction retrieval
* Local safety classifier (emotion risk detection)
* Multilingual support
* Lightweight distilled encoder model
* Fine-tuning via SafeRLHF or supervised RAG tuning

---

## ⚠️ Disclaimer

This tool is **NOT a medical or psychological diagnostic system**.
It provides supportive, empathetic responses based on existing therapist advice.
Users in crisis should always seek help from a licensed professional.

---

## ❤️ Credits

Built using:

* Sentence Transformers
* FAISS
* Unsloth
* Transformers
* Streamlit
* Python ecosystem
