from vector_utils import update_document_vectors  # Pastikan fungsi ini diimpor dengan benar
from gensim.models.doc2vec import Doc2Vec

# Lanjutkan eksekusi
model = Doc2Vec.load('models/d2v.model')  
update_document_vectors(model)  # Memperbarui vektor dokumen di database
