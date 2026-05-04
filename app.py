import streamlit as st
import joblib
import numpy as np

# --- CONFIG DASHBOARD ---
# Mengatur judul tab browser dan icon
st.set_page_config(page_title="Komoku - SMS Spam Detection", page_icon="🛡️")

# --- LOAD MODEL & VECTORIZER ---
# Menggunakan cache_resource agar model hanya dimuat satu kali saat aplikasi dijalankan
@st.cache_resource 
def load_model_files():
    # Pastikan file .joblib berada di folder yang sama dengan app.py
    model = joblib.load('model_spam_nb.joblib')
    tfidf = joblib.load('tfidf_vectorizer.joblib')
    return model, tfidf

# Mencoba memuat model dan menangani error jika file tidak ditemukan
try:
    model, tfidf = load_model_files()
    is_model_ready = True
except Exception as e:
    st.error(f"Error: File model tidak ditemukan atau tidak kompatibel. Detail: {e}")
    is_model_ready = False

# --- UI STREAMLIT ---
st.title("🛡️ Spam & Phishing Detector")
st.write("Sistem Klasifikasi Teks Pesan Spam Menggunakan Algoritma Multinomial Naïve Bayes Bahasa Indonesia")

# Area Input Teks
user_input = st.text_area(
    "Masukkan pesan yang ingin diperiksa:", 
    placeholder="Contoh: Selamat! Anda menang hadiah 100jt, klik link ini bit.ly/hadiah-palsu",
    height=150
)

# Tombol Prediksi
if st.button("Periksa Pesan"):
    if not is_model_ready:
        st.error("Aplikasi tidak dapat dijalankan karena model gagal dimuat.")
    elif user_input.strip() == "":
        st.warning("Silakan masukkan teks terlebih dahulu.")
    else:
        # 1. Transformasi input teks menggunakan TF-IDF
        # Teks harus dimasukkan ke dalam list [user_input]
        input_vector = tfidf.transform([user_input])
        
        # 2. Prediksi menggunakan model
        prediction = model.predict(input_vector)
        
        # 3. Hitung Probabilitas (Tingkat Keyakinan)
        probability = model.predict_proba(input_vector)
        confidence = np.max(probability) * 100

        # 4. Tampilkan Hasil
        st.subheader("Hasil Analisis:")
        
        if prediction[0] == 1:
            st.error(f"⚠️ **PESAN INI TERINDIKASI SPAM / PHISHING**")
            st.write(f"**Tingkat Keyakinan:** {confidence:.2f}%")
            st.info("Saran: Jangan mengklik tautan apa pun atau memberikan data pribadi kepada pengirim.")
        else:
            st.success(f"✅ **PESAN INI AMAN (HAM)**")
            st.write(f"**Tingkat Keyakinan:** {confidence:.2f}%")

# --- FOOTER ---
st.markdown("---")
st.caption("Dikembangkan untuk Proyek Tugas Akhir Informatika - Universitas MercuBuana Yogyakarta")
