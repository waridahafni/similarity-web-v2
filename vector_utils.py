import psycopg2
from psycopg2.extras import RealDictCursor

def update_document_vectors(model):
    conn = psycopg2.connect("dbname=db_similarity user=postgres password=admin host=localhost")
    cur = conn.cursor()

    cur.execute("SELECT id FROM documents")
    rows = cur.fetchall()

    for row in rows:
        doc_id = str(row[0])  # Pastikan ID dokumen dalam bentuk string
        if doc_id in model.dv:  # Cek apakah ID dokumen ada di model
            vector = model.dv[doc_id].tobytes()
            cur.execute("UPDATE documents SET vector = %s WHERE id = %s", (vector, doc_id))
        else:
            print(f"Document ID {doc_id} not found in model.")

    conn.commit()
    cur.close()
    conn.close()
    print("Vektor dokumen berhasil diperbarui.")
