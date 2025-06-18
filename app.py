import streamlit as st
import pandas as pd
import numpy as np
import joblib

# === Load model dan scaler ===
model = joblib.load("best_model.pkl")
scaler = joblib.load("scaler.pkl")

# === Judul Aplikasi ===
st.set_page_config(page_title="Prediksi Obesitas", layout="centered")
st.title("🔍 Prediksi Kategori Obesitas")
st.markdown("Masukkan data diri Anda untuk memprediksi kategori obesitas berdasarkan model machine learning.")

# === Input Data dari User ===
gender = st.radio("Jenis Kelamin", ["Male", "Female"])
age = st.slider("Usia", 10, 100, 25)
height = st.number_input("Tinggi Badan (meter)", min_value=1.0, max_value=2.5, value=1.7, step=0.01)
weight = st.number_input("Berat Badan (kg)", min_value=20.0, max_value=200.0, value=70.0, step=0.5)
fcvc = st.slider("Frekuensi Konsumsi Sayur (1: Jarang, 3: Sering)", 1.0, 3.0, 2.0)
faf = st.slider("Aktivitas Fisik (jam/minggu)", 0.0, 3.0, 1.0)

# === Preprocessing Input ===
gender = 1 if gender == "Male" else 0

# Data dalam urutan fitur sesuai training model
input_data = pd.DataFrame([[gender, age, height, weight, fcvc, faf]],
                          columns=["Gender", "Age", "Height", "Weight", "FCVC", "FAF"])

input_scaled = scaler.transform(input_data)

# === Prediksi ===
if st.button("🔮 Prediksi"):
    prediction = model.predict(input_scaled)
    st.success(f"Hasil Prediksi: **{prediction[0]}**")
    st.markdown("⚠️ Kategori prediksi ini hanya untuk keperluan edukasi dan bukan diagnosis medis.")

# === Footer ===
st.markdown("---")
st.caption("Dibuat dengan ❤️ oleh Mahasiswa Informatika | Streamlit Deployment")
