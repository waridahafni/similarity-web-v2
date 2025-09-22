import re
from sklearn.metrics.pairwise import cosine_similarity
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# ✅ Tokenizer kalimat sederhana (tanpa NLTK)
def split_and_clean_sentences(text):
    # Pecah berdasarkan akhir kalimat . ! ? + spasi
    raw_sentences = re.split(r'(?<=[.!?])\s+', text)
    cleaned_sentences = [re.sub(r'[^a-zA-Z\s]', ' ', s).lower().strip() for s in raw_sentences if s.strip()]
    return cleaned_sentences

# ✅ Tokenizer kata sederhana (tanpa NLTK)
def simple_tokenize(text):
    return re.findall(r'\b\w+\b', text.lower())

# ✅ Embedding kalimat per kalimat
def embed_sentences(sentences, model):
    return [model.infer_vector(simple_tokenize(s)) for s in sentences]

# ✅ Cari pasangan kalimat paling mirip
def get_top_similar_pairs(user_sents, user_vecs, doc_sents, doc_vecs, top_n=3, threshold=0.7):
    sim_matrix = cosine_similarity(user_vecs, doc_vecs)
    pairs = []

    for _ in range(top_n):
        i, j = divmod(sim_matrix.argmax(), sim_matrix.shape[1])
        sim_score = sim_matrix[i, j]
        if sim_score < threshold:
            break
        pairs.append((user_sents[i], doc_sents[j], float(sim_score)))
        sim_matrix[i, :] = -1
        sim_matrix[:, j] = -1

    return pairs

# ✅ Highlight kata-kata atau frasa mirip dalam teks
def highlight_matches(text, phrases):
    highlighted = text
    for phrase in sorted(phrases, key=len, reverse=True):
        if phrase.lower() in highlighted.lower():
            highlighted = re.sub(re.escape(phrase), f'<span style="background-color: yellow">{phrase}</span>', highlighted, flags=re.IGNORECASE)
    return highlighted

# ✅ Ambil stemmed tokens dari kalimat yang mirip
def get_stemmed_tokens_from_similar_sentences(user_text, db_text, model, top_n=3, threshold=0.72):
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()

    user_sents = split_and_clean_sentences(user_text)
    db_sents = split_and_clean_sentences(db_text)

    user_vecs = embed_sentences(user_sents, model)
    db_vecs = embed_sentences(db_sents, model)

    similar_pairs = get_top_similar_pairs(user_sents, user_vecs, db_sents, db_vecs, top_n=top_n, threshold=threshold)

    matched_tokens = set()
    for _, db_sent, _ in similar_pairs:
        tokens = simple_tokenize(db_sent)
        stemmed = [stemmer.stem(t.lower()) for t in tokens if len(t) > 2]
        matched_tokens.update(stemmed)

    return matched_tokens
