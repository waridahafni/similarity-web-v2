# data_split_generator.py
import os
import fitz  # PyMuPDF
import pickle
from sklearn.model_selection import train_test_split
from preprocess import preprocess_sentences

folder_path = r"C:\Project Skripsi\similarity-web-v2\dataset"

def extract_text_from_pdf(pdf_path, max_pages=3):
    try:
        doc = fitz.open(pdf_path)
        return " ".join([doc[i].get_text() for i in range(min(max_pages, len(doc)))])
    except Exception as e:
        print(f"❌ Gagal baca {pdf_path}: {e}")
        return ""

documents, file_names = [], []

for file in sorted(os.listdir(folder_path)):
    if file.endswith(".pdf"):
        path = os.path.join(folder_path, file)
        print(f"📄 Proses: {file}")
        text = extract_text_from_pdf(path)
        kalimat_bersih = preprocess_sentences(text)
        if kalimat_bersih:
            documents.append(kalimat_bersih)
            file_names.append(file)

# Split data
train_docs, temp_docs, train_files, temp_files = train_test_split(
    documents, file_names, test_size=0.3, random_state=42)
val_docs, test_docs, val_files, test_files = train_test_split(
    temp_docs, temp_files, test_size=2/3, random_state=42)

# Simpan ke pickle
with open("data_split.pkl", "wb") as f:
    pickle.dump({
        "documents": documents,
        "file_names": file_names,
        "train_docs": train_docs,
        "val_docs": val_docs,
        "test_docs": test_docs
    }, f)

print("✅ data_split.pkl berhasil disimpan.")
