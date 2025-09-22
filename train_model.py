import os
import fitz  # PyMuPDF
import re
import string
import nltk
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.model_selection import train_test_split

nltk.download('stopwords')

folder_path = r"C:\Project Skripsi\similarity-web-v2\dataset"

factory = StemmerFactory()
stemmer = factory.create_stemmer()
stop_words = set(stopwords.words('indonesian'))

def extract_text_from_pdf(pdf_path, max_pages=3):
    doc = fitz.open(pdf_path)
    text = ""
    for i in range(min(max_pages, len(doc))):
        text += doc[i].get_text()
    return text

def preprocess(text):
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.lower()
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words and len(word) > 2]
    tokens = [stemmer.stem(word) for word in tokens]
    return tokens

documents, file_names = [], []

for file in sorted(os.listdir(folder_path)):
    if file.endswith(".pdf"):
        path = os.path.join(folder_path, file)
        text = extract_text_from_pdf(path)
        tokens = preprocess(text)
        if tokens:
            documents.append(tokens)
            file_names.append(file)

# Split: 70% train, 10% validasi, 20% test
train_docs, temp_docs, train_files, temp_files = train_test_split(documents, file_names, test_size=0.3, random_state=42)
val_docs, test_docs, val_files, test_files = train_test_split(temp_docs, temp_files, test_size=2/3, random_state=42)

# Simpan ke file
import pickle
with open("data_split.pkl", "wb") as f:
    pickle.dump({
        "documents": documents,
        "file_names": file_names,
        "train_docs": train_docs,
        "val_docs": val_docs,
        "test_docs": test_docs
    }, f)
