import re
import nltk
import string
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# Download resource sekali saja
nltk.download('punkt')
nltk.download('stopwords')

# Buat stemmer bahasa Indonesia
stemmer = StemmerFactory().create_stemmer()

# Stopwords bawaan + custom
stop_words = set(stopwords.words('indonesian'))
# Buat custom supaya kata penting tetap ada
custom_keep = {"pengguna", "aplikasi", "sistem", "penelitian"}  
stop_words = {w for w in stop_words if w not in custom_keep}

def preprocess_text(text):
    # 1. lowercase
    text = text.lower()
    # 2. hapus tanda baca & angka
    text = re.sub(r'[^a-z\s]', ' ', text)
    # 3. tokenisasi
    tokens = word_tokenize(text)
    # 4. hapus stopword & kata sangat pendek
    tokens = [t for t in tokens if t not in stop_words and len(t) > 2]
    # 5. stemming (misal: penggunaan -> guna)
    tokens = [stemmer.stem(t) for t in tokens]
    return tokens

def preprocess_sentences(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    kalimat_mentah = sent_tokenize(text)
    hasil = []
    for kalimat in kalimat_mentah:
        tokens = word_tokenize(kalimat)
        tokens = [t for t in tokens if t not in stop_words and len(t) > 2]
        tokens = [stemmer.stem(t) for t in tokens]
        if tokens:
            hasil.append(" ".join(tokens))
    return hasil
