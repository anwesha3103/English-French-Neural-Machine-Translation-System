# 🌐 English → French Neural Machine Translator

<div align="center">

![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.12%2B-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-GRU-D00000?style=for-the-badge&logo=keras&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-Web%20App-000000?style=for-the-badge&logo=flask&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-22c55e?style=for-the-badge)

**A GRU-based sequence-to-sequence neural network that translates English to French**  
*with a clean Flask web interface*

[Features](#-features) • [Demo](#-demo) • [Installation](#-installation) • [Usage](#-usage) • [Architecture](#-model-architecture) • [Dataset](#-dataset)

</div>

---

## 🎯 Overview

This project implements **Neural Machine Translation (NMT)** using a GRU (Gated Recurrent Unit) encoder with word embeddings, trained on a parallel English-French corpus. The model is served through a minimal, elegant **Flask web application**.

> Type an English sentence → click Translate → get the French output instantly.

---

## ✨ Features

- 🧠 **GRU-based sequence model** with 256-dimensional word embeddings
- 📝 **Word-level tokenization** with Keras Tokenizer
- ⚖️ **Dropout regularization** to prevent overfitting
- 🌐 **Flask web interface** — clean, responsive, ready to deploy
- 💾 **Serialized model & tokenizers** via pickle for fast loading
- 🔁 **Easily retrainable** — just run `train.py`

---

## 🖥️ Demo

| Input (English) | Output (French) |
|---|---|
| `new jersey is sometimes quiet during autumn` | `new jersey est parfois calme en automne` |
| `the united states is usually hot during july` | `les états-unis est généralement chaud en juillet` |
| `california is never rainy during march` | `california est jamais pluvieux en mars` |

---

## 📁 Project Structure

```
english-french-translator/
├── app.py                    ← Flask web server
├── train.py                  ← Model training script
├── requirements.txt          ← Dependencies
├── .gitignore
├── README.md
│
├── templates/
│   └── index.html            ← Web UI (Jinja2 template)
│
└── models/                   ← Generated after training
    ├── simple_rnn_model.pkl
    ├── english_tokenizer.pkl
    ├── french_tokenizer.pkl
    └── model.h5
```

---

## 🚀 Installation

### 1 — Clone the repo

```bash
git clone https://github.com/yourusername/english-french-translator.git
cd english-french-translator
```

### 2 — Set up a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
```

### 3 — Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🧪 Usage

### Step 1 — Train the model

Downloads the dataset automatically and saves model + tokenizers:

```bash
python train.py
```

> ⏱️ Training takes ~5 minutes on CPU with `batch_size=1024, epochs=10`

### Step 2 — Launch the web app

```bash
python app.py
```

Open **http://localhost:5000** in your browser.

---

## 🏗️ Model Architecture

```
Input tokens (English)
        ↓
Embedding Layer (vocab_size → 256 dims)
        ↓
GRU (256 units, return_sequences=True)
        ↓
TimeDistributed Dense (1024 units, ReLU)
        ↓
Dropout (0.5)
        ↓
TimeDistributed Dense (french_vocab_size, Softmax)
        ↓
Output sequence (French tokens)
```

### Key Design Choices

| Component | Choice | Reason |
|---|---|---|
| Recurrent Layer | GRU | Faster than LSTM, comparable performance |
| Embeddings | 256-dim trainable | Learns semantic word representations |
| Loss | Sparse Categorical Crossentropy | Efficient for large vocabularies |
| Optimizer | Adam (lr=0.005) | Adaptive learning rate |
| Regularization | Dropout (0.5) | Reduces overfitting on small corpus |

---

## 📊 Dataset

**Small Vocabulary English-French corpus** hosted by [projjal1](https://github.com/projjal1/datasets)

| Property | Value |
|---|---|
| Source language | English |
| Target language | French |
| Domain | Simple declarative sentences |
| English vocabulary | ~227 unique words |
| French vocabulary | ~355 unique words |

The dataset uses a controlled vocabulary of common words about weather, geography, and states — making it ideal for learning NMT concepts.

---

## ⚙️ Configuration

All hyperparameters are at the top of `train.py`:

```python
LEARNING_RATE = 0.005
BATCH_SIZE    = 1024
EPOCHS        = 10
EMBED_DIM     = 256
GRU_UNITS     = 256
DENSE_UNITS   = 1024
DROPOUT_RATE  = 0.5
```

---

## 📦 Dependencies

```
tensorflow >= 2.12
keras      >= 2.12
flask      >= 2.3
numpy      >= 1.23
```

---

## 🗺️ Roadmap

- [ ] Add attention mechanism
- [ ] Support bidirectional GRU encoder
- [ ] Add beam search decoding
- [ ] Expand to larger dataset (WMT14)
- [ ] Deploy on Hugging Face Spaces

---

## 🙌 Acknowledgements

- Dataset by [projjal1](https://github.com/projjal1/datasets)
- Built with [TensorFlow](https://tensorflow.org) / [Keras](https://keras.io)
- Web interface powered by [Flask](https://flask.palletsprojects.com)

---

<div align="center">
Made with ❤️ · Neural Machine Translation from scratch
</div>
