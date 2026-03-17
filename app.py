"""
app.py
======
Flask web server for English → French Neural Machine Translation.
Run with: python app.py
Then open: http://localhost:5000
"""

from flask import Flask, request, render_template
from keras.preprocessing.sequence import pad_sequences
import pickle
import numpy as np

app = Flask(__name__)

# ── Load model & tokenizers once at startup ───────────────────────────────────
translation_model   = pickle.load(open('simple_rnn_model.pkl', 'rb'))
english_tokenizer   = pickle.load(open('english_tokenizer.pkl', 'rb'))
french_tokenizer    = pickle.load(open('french_tokenizer.pkl', 'rb'))


# ── Helper functions ──────────────────────────────────────────────────────────

def logits_to_text(logits, tokenizer):
    """Convert model output logits to a human-readable string."""
    index_to_words = {idx: word for word, idx in tokenizer.word_index.items()}
    index_to_words[0] = ''
    return ' '.join([index_to_words[prediction] for prediction in np.argmax(logits, 1)])


def translate_text(text: str) -> str:
    """
    Translate an English sentence to French.

    Args:
        text: Raw English input string

    Returns:
        Translated French string
    """
    tokens   = [english_tokenizer.word_index.get(word, 0) for word in text.lower().split()]
    padded   = pad_sequences([tokens], maxlen=translation_model.input_shape[1], padding='post')
    logits   = translation_model.predict(padded[:1])[0]
    return logits_to_text(logits, french_tokenizer).strip()


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route('/', methods=['GET', 'POST'])
def index():
    translated_text = ''
    error = ''

    if request.method == 'POST':
        text = request.form.get('text', '').strip()
        if text:
            try:
                translated_text = translate_text(text)
            except Exception as e:
                error = f"Translation failed: {str(e)}"
        else:
            error = "Please enter some English text to translate."

    return render_template('index.html', translated_text=translated_text, error=error)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
