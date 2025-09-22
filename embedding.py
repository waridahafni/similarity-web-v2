import re
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import psycopg2
import os

INDONESIAN_STOPWORDS = set(StopWordRemoverFactory().get_stop_words())


def preprocess_text(text):
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = text.split()
    tokens = [word for word in tokens if word not in INDONESIAN_STOPWORDS]
    stemmed_tokens = [stemmer.stem(word) for word in tokens]
    return stemmed_tokens


def calculate_similarity(user_doc, db_docs, model):
    user_vector = model.infer_vector(preprocess_text(user_doc))
    similarities = []
    for db_doc in db_docs:
        db_vector = model.dv[str(db_doc['id'])]
        similarity = model.dv.similarity(user_vector, db_vector)
        similarity = float(similarity)  
        matched_text = highlight_similar_parts(user_doc, db_doc['file_text'])
        similarities.append((db_doc['id'], similarity, matched_text))
    return sorted(similarities, key=lambda x: x[1], reverse=True)[:5]


def highlight_similar_parts(text1, text2):
    tokens1 = set(preprocess_text(text1))
    tokens2 = preprocess_text(text2)

    highlighted = []
    for token in tokens2:
        if token in tokens1:
            highlighted.append(f"<mark>{token}</mark>")
        else:
            highlighted.append(token)

    return " ".join(highlighted)

def get_documents_from_db():
    conn = psycopg2.connect(
        "dbname=db_similarity user=postgres password=admin host=localhost"
    )
    cur = conn.cursor()
    cur.execute("SELECT id, title, file_text, vector FROM documents")
    documents = cur.fetchall()
    cur.close()
    conn.close()
    return [{'id': doc[0], 'title': doc[1], 'file_text': doc[2], 'vector': doc[3]} for doc in documents]

def save_to_history(user_id, file_name, file_text, similarities):

    conn = psycopg2.connect(
        "dbname=db_similarity user=postgres password=admin host=localhost"
    )
    cur = conn.cursor()
    for doc_id, similarity_score, matched_text in similarities:
        cur.execute(
            "INSERT INTO history (user_id, document_id, uploaded_file_name, uploaded_file_text, similarity_score, matched_text) "
            "VALUES (%s, %s, %s, %s, %s, %s)",
            (user_id, doc_id, file_name, file_text, similarity_score, matched_text)
        )
    conn.commit()
    cur.close()
    conn.close()

def prepare_and_train_model():
    documents = get_documents_from_db()
    tagged_data = [TaggedDocument(words=preprocess_text(doc['file_text']), tags=[
                                  str(doc['id'])]) for doc in documents]

    model = Doc2Vec(vector_size=20, window=2, min_count=1, workers=4)
    model.build_vocab(tagged_data)
    model.train(tagged_data, total_examples=model.corpus_count,
                epochs=model.epochs)

    model.save("d2v.model")
    return model



def load_model():
    if not os.path.exists("d2v.model"):
        raise FileNotFoundError(
            "Model file 'd2v.model' not found. Please train the model first.")
    return Doc2Vec.load("d2v.model")
