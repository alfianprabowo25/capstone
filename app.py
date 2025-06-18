import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model dan scaler
model = joblib.load("best_model.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="Prediksi Obesitas", layout="wide")
st.title("🔍 Prediksi Kategori Obesitas Berdasarkan Gaya Hidup")

# Input User
st.header("📝 Masukkan Data Anda")

col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Jenis Kelamin", ["Male", "Female"])
    age = st.slider("Usia", 10, 100, 25)
    height = st.number_input("Tinggi Badan (m)", min_value=1.0, max_value=2.5, value=1.70, step=0.01)
    weight = st.number_input("Berat Badan (kg)", min_value=30.0, max_value=200.0, value=70.0, step=0.5)
    favc = st.selectbox("Sering Mengonsumsi Makanan Berkalori Tinggi (FAVC)", ["yes", "no"])
    fcvc = st.slider("Frekuensi Konsumsi Sayur (1: Jarang - 3: Sering)", 1.0, 3.0, 2.0)
    ncp = st.slider("Jumlah Makanan Utama per Hari (NCP)", 1.0, 4.0, 3.0)
    caec = st.selectbox("Makan di Luar Waktu Makan Utama (CAEC)", ["no", "Sometimes", "Frequently", "Always"])

with col2:
    smoke = st.selectbox("Merokok (SMOKE)", ["yes", "no"])
    ch2o = st.slider("Konsumsi Air Harian (CH2O, liter)", 1.0, 3.0, 2.0)
    scc = st.selectbox("Apakah Mengontrol Kalori yang Dikonsumsi? (SCC)", ["yes", "no"])
    faf = st.slider("Aktivitas Fisik Mingguan (jam)", 0.0, 3.0, 1.0)
    tue = st.slider("Waktu di Depan Layar (jam/hari)", 0.0, 2.0, 1.0)
    calc = st.selectbox("Konsumsi Alkohol (CALC)", ["no", "Sometimes", "Frequently", "Always"])
    mtrans = st.selectbox("Moda Transportasi (MTRANS)", ["Public_Transportation", "Walking", "Automobile", "Motorbike", "Bike"])
    family_history = st.selectbox("Riwayat Keluarga Kelebihan Berat Badan", ["yes", "no"])

# Encode sesuai training (LabelEncoder)
encode_map = {
    "Gender": {"Male": 1, "Female": 0},
    "FAVC": {"yes": 1, "no": 0},
    "SCC": {"yes": 1, "no": 0},
    "SMOKE": {"yes": 1, "no": 0},
    "family_history_with_overweight": {"yes": 1, "no": 0},
    "CAEC": {"no": 0, "Sometimes": 1, "Frequently": 2, "Always": 3},
    "CALC": {"no": 0, "Sometimes": 1, "Frequently": 2, "Always": 3},
    "MTRANS": {
        "Public_Transportation": 0,
        "Walking": 1,
        "Automobile": 2,
        "Motorbike": 3,
        "Bike": 4
    }
}

# Buat DataFrame input
data = {
    "Age": age,
    "Gender": encode_map["Gender"][gender],
    "Height": height,
    "Weight": weight,
    "CALC": encode_map["CALC"][calc],
    "FAVC": encode_map["FAVC"][favc],
    "FCVC": fcvc,
    "NCP": ncp,
    "SCC": encode_map["SCC"][scc],
    "SMOKE": encode_map["SMOKE"][smoke],
    "CH2O": ch2o,
    "family_history_with_overweight": encode_map["family_history_with_overweight"][family_history],
    "FAF": faf,
    "TUE": tue,
    "CAEC": encode_map["CAEC"][caec],
    "MTRANS": encode_map["MTRANS"][mtrans]
}

input_df = pd.DataFrame([data])

# Scaling
input_scaled = scaler.transform(input_df.values)

# Prediksi
if st.button("🔮 Prediksi"):
    prediction = model.predict(input_scaled)
    st.success(f"Hasil Prediksi Kategori Obesitas: **{prediction[0]}**")
    st.markdown("⚠️ Prediksi ini bersifat estimasi dan bukan diagnosis medis.")

# Footer
st.markdown("---")
st.caption("Dibuat untuk Capstone Project: Klasifikasi Obesitas dengan 16 Fitur Gaya Hidup.")
