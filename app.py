import streamlit as st
import pandas as pd
import numpy as np
import joblib

# === Load model dan scaler ===
model = joblib.load("best_model.pkl")
scaler = joblib.load("scaler.pkl")

# === Daftar fitur yang digunakan saat training ===
features = ["Gender", "Age", "Height", "Weight", "FCVC", "FAF"]

# === Konfigurasi halaman ===
st.set_page_config(page_title="Prediksi Obesitas", layout="centered")
st.title("🔍 Prediksi Kategori Obesitas")
st.markdown("Masukkan data diri Anda untuk mengetahui kategori obesitas berdasarkan model machine learning.")

# === Input dari user ===
gender = st.radio("Jenis Kelamin", ["Male", "Female"])
age = st.slider("Usia", 10, 100, 25)
height = st.number_input("Tinggi Badan (meter)", min_value=1.0, max_value=2.5, value=1.7, step=0.01)
weight = st.number_input("Berat Badan (kg)", min_value=20.0, max_value=200.0, value=70.0, step=0.5)
fcvc = st.slider("Frekuensi Konsumsi Sayur (1: Jarang, 3: Sering)", 1.0, 3.0, 2.0)
faf = st.slider("Aktivitas Fisik Mingguan (jam/minggu)", 0.0, 3.0, 1.0)

# === Preprocessing input ===
gender = 1 if gender == "Male" else 0
input_data = pd.DataFrame([[gender, age, height, weight, fcvc, faf]], columns=features)

# === Scaling input ===
try:
    input_scaled = scaler.transform(input_data)
except ValueError:
    input_scaled = scaler.transform(input_data.values)

# === Prediksi ===
if st.button("🔮 Prediksi"):
    prediction = model.predict(input_scaled)
    st.success(f"Hasil Prediksi: **{prediction[0]}**")
    st.markdown("⚠️ Hasil prediksi ini tidak menggantikan diagnosis profesional medis.")

# === Footer ===
st.markdown("---")
st.caption("📘 Aplikasi ini dibuat untuk tugas akhir klasifikasi obesitas menggunakan machine learning.")
