from werkzeug.security import generate_password_hash

# Password yang ingin Anda hash
password_mentah = 'Hasibuan_123'

# Hasilkan password yang sudah di-hash
hashed_password = generate_password_hash(password_mentah)

# Cetak hasilnya
print(hashed_password)