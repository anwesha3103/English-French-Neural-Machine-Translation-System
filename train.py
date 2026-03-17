"""
train.py
========
Train the English → French GRU-based Neural Machine Translation model.
Run with: python train.py

Saves:
  - simple_rnn_model.pkl   (Keras model)
  - english_tokenizer.pkl  (English tokenizer)
  - french_tokenizer.pkl   (French tokenizer)
"""

import os
import pickle
import numpy as np
import collections
import tensorflow as tf

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import GRU, Dense, TimeDistributed, Dropout, Embedding
from keras.optimizers import Adam
from keras.losses import sparse_categorical_crossentropy


# ── Dataset URLs ──────────────────────────────────────────────────────────────

ENGLISH_URL = 'https://raw.githubusercontent.com/projjal1/datasets/master/small_vocab_en.txt'
FRENCH_URL  = 'https://raw.githubusercontent.com/projjal1/datasets/master/small_vocab_fr.txt'

# ── Hyperparameters ───────────────────────────────────────────────────────────

LEARNING_RATE = 0.005
BATCH_SIZE    = 1024
EPOCHS        = 10
EMBED_DIM     = 256
GRU_UNITS     = 256
DENSE_UNITS   = 1024
DROPOUT_RATE  = 0.5


# ── Data loading ──────────────────────────────────────────────────────────────

def load_data(path: str) -> list:
    with open(path, 'r') as f:
        return f.read().split('\n')


def tokenize(x):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(x)
    return tokenizer.texts_to_sequences(x), tokenizer


def pad(x, length=None):
    return pad_sequences(x, maxlen=length, padding='post')


def preprocess(x, y):
    preprocess_x, x_tk = tokenize(x)
    preprocess_y, y_tk = tokenize(y)
    preprocess_x = pad(preprocess_x)
    preprocess_y = pad(preprocess_y)
    preprocess_y = preprocess_y.reshape(*preprocess_y.shape, 1)
    return preprocess_x, preprocess_y, x_tk, y_tk


# ── Model ─────────────────────────────────────────────────────────────────────

def build_model(input_shape, english_vocab_size, french_vocab_size):
    model = Sequential([
        Embedding(english_vocab_size, EMBED_DIM,
                  input_length=input_shape[1],
                  input_shape=input_shape[1:]),
        GRU(GRU_UNITS, return_sequences=True),
        TimeDistributed(Dense(DENSE_UNITS, activation='relu')),
        Dropout(DROPOUT_RATE),
        TimeDistributed(Dense(french_vocab_size, activation='softmax')),
    ])
    model.compile(
        loss=sparse_categorical_crossentropy,
        optimizer=Adam(LEARNING_RATE),
        metrics=['accuracy'],
    )
    return model


# ── Training ──────────────────────────────────────────────────────────────────

def main():
    print("[train] Downloading dataset...")
    english_file = tf.keras.utils.get_file('small_vocab_en', ENGLISH_URL)
    french_file  = tf.keras.utils.get_file('small_vocab_fr', FRENCH_URL)

    english_sentences = load_data(english_file)
    french_sentences  = load_data(french_file)

    print(f"[train] Loaded {len(english_sentences)} sentence pairs.")

    # Vocab stats
    english_counter = collections.Counter(
        [word for s in english_sentences for word in s.split()])
    french_counter  = collections.Counter(
        [word for s in french_sentences  for word in s.split()])
    print(f"[train] English vocab: {len(english_counter)} | French vocab: {len(french_counter)}")

    # Preprocess
    preproc_en, preproc_fr, en_tk, fr_tk = preprocess(english_sentences, french_sentences)

    print(f"[train] Max EN length: {preproc_en.shape[1]}")
    print(f"[train] Max FR length: {preproc_fr.shape[1]}")

    # Reshape input
    tmp_x = pad(preproc_en, preproc_fr.shape[1])
    tmp_x = tmp_x.reshape((-1, preproc_fr.shape[-2]))

    # Build & train
    model = build_model(
        tmp_x.shape,
        len(en_tk.word_index) + 1,
        len(fr_tk.word_index) + 1,
    )
    model.summary()

    model.fit(tmp_x, preproc_fr,
              batch_size=BATCH_SIZE,
              epochs=EPOCHS,
              validation_split=0.2)

    # Save
    model.save('model.h5')
    pickle.dump(model,   open('simple_rnn_model.pkl', 'wb'))
    pickle.dump(en_tk,   open('english_tokenizer.pkl', 'wb'))
    pickle.dump(fr_tk,   open('french_tokenizer.pkl', 'wb'))

    print("\n[train]  Model and tokenizers saved successfully.")


if __name__ == '__main__':
    main()
