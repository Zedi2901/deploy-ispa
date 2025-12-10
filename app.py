import streamlit as st
import joblib
import numpy as np

# Load model
model = joblib.load("model_ispa.pkl")
scaler = joblib.load("scaler.pkl")

st.title("🩺 Prediksi Penyakit ISPA")

st.write("Masukkan data gejala pasien:")

umur = st.number_input("Umur", min_value=0, max_value=100)

gejala = [
    "Batuk Kering",
    "Batuk Berdahak",
    "Demam",
    "Pilek",
    "Hidung Tersumbat",
    "Sesak Napas",
    "Nyeri Tenggorokan",
    "Sakit Kepala",
    "Mual Muntah",
    "Nyeri Dada",
    "Suara Serak",
    "Kelelahan",
    "Berkeringat Malam",
    "Nafsu Makan Turun",
    "Hilang Penciuman",
    "Nyeri Saat Menelan"
]

input_data = [umur]

for g in gejala:
    val = st.selectbox(g, [0, 1])
    input_data.append(val)

if st.button("Prediksi"):
    data_np = np.array(input_data).reshape(1, -1)
    data_scaled = scaler.transform(data_np)
    hasil = model.predict(data_scaled)

    st.success(f"Hasil Prediksi: {hasil[0]}")
