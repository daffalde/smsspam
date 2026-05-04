import streamlit as st
import pickle
import numpy as np

# --- CONFIG DASHBOARD ---
st.set_page_config(page_title="Komoku - SMS Spam Detection", page_icon="🛡️")

# --- LOAD MODEL & VECTORIZER ---
@st.cache_resource # Gunakan cache agar model tidak di-load ulang setiap kali klik tombol
def load_model():
    with open('model_spam_nb.pkl', 'rb') as m_file:
        model = pickle.load(m_file)
    with open('tfidf_vectorizer.pkl', 'rb') as t_file:
        tfidf = pickle.load(t_file)
    return model, tfidf

try:
    model, tfidf = load_model()
except FileNotFoundError:
    st.error("File model atau vectorizer tidak ditemukan. Pastikan file .pkl ada di folder yang sama.")

# --- UI STREAMLIT ---
st.title("🛡️ Komoku: Spam & Phishing Detector")
st.write("Sistem klasifikasi SMS untuk mendeteksi pesan berbahaya menggunakan Multinomial Naive Bayes.")

# Input teks dari pengguna
user_input = st.text_area("Masukkan pesan SMS yang ingin diperiksa:", placeholder="Contoh: Selamat Anda menang hadiah...")

if st.button("Periksa Pesan"):
    if user_input.strip() == "":
        st.warning("Silakan masukkan teks terlebih dahulu.")
    else:
        # 1. Transformasi input teks menggunakan TF-IDF yang sudah di-load
        input_vector = tfidf.transform([user_input])
        
        # 2. Prediksi menggunakan model
        prediction = model.predict(input_vector)
        
        # 3. Tampilkan Hasil
        st.subheader("Hasil Analisis:")
        if prediction[0] == 1:
            st.error("⚠️ PESAN INI TERINDIKASI SPAM / PHISHING")
            st.info("Saran: Jangan mengklik tautan apa pun atau memberikan data pribadi.")
        else:
            st.success("✅ PESAN INI AMAN (HAM)")

# --- FOOTER ---
st.markdown("---")
st.caption("Dikembangkan untuk Proyek Tugas Akhir Informatika - Universitas Mercu Buana Yogyakarta")