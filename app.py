import streamlit as st
import joblib
import numpy as np
import pandas as pd

# =========================
# Konfigurasi Halaman
# =========================
st.set_page_config(
    page_title="Prediksi ISPA",
    page_icon="🩺",
    layout="centered"
)

# =========================
# Load Model & Scaler
# =========================
model = joblib.load("model_ispa.pkl")
scaler = joblib.load("scaler.pkl")

# =========================
# Session State
# =========================
if "riwayat" not in st.session_state:
    st.session_state.riwayat = []

# =========================
# Styling Modern
# =========================
st.markdown("""
<style>
.card {
    padding: 20px;
    border-radius: 16px;
    background: #f8f9fa;
    margin-bottom: 20px;
}
.result-positive {
    background-color: #d4edda;
    padding: 16px;
    border-radius: 12px;
}
.result-negative {
    background-color: #f8d7da;
    padding: 16px;
    border-radius: 12px;
}
.stButton>button {
    background-color: #2e86de;
    color: white;
    font-size: 18px;
    height: 3em;
    width: 100%;
    border-radius: 14px;
}
</style>
""", unsafe_allow_html=True)

# =========================
# Header
# =========================
st.markdown("<h1 style='text-align:center;'>🩺 Prediksi Penyakit ISPA</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Aplikasi klasifikasi ISPA berbasis Machine Learning</p>", unsafe_allow_html=True)

# =========================
# Input Section
# =========================
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.subheader("🧑‍⚕️ Data Pasien")

umur = st.number_input("Umur (tahun)", min_value=1, max_value=100)

st.subheader("🤒 Gejala Pasien")
gejala = [
    "Batuk Kering", "Batuk Berdahak", "Demam", "Pilek",
    "Hidung Tersumbat", "Sesak Napas", "Nyeri Tenggorokan", "Sakit Kepala",
    "Mual Muntah", "Nyeri Dada", "Suara Serak", "Kelelahan",
    "Berkeringat Malam", "Nafsu Makan Turun", "Hilang Penciuman", "Nyeri Saat Menelan"
]

col1, col2 = st.columns(2)
input_data = [umur]
display = {}

for i, g in enumerate(gejala):
    with col1 if i % 2 == 0 else col2:
        val = st.selectbox(g, ["Tidak", "Ya"])
        input_data.append(1 if val == "Ya" else 0)
        display[g] = val

st.markdown("</div>", unsafe_allow_html=True)

# =========================
# Ringkasan
# =========================
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.subheader("📋 Ringkasan Data")
st.write(f"**Umur:** {umur} tahun")
st.write(", ".join([f"{k}: {v}" for k, v in display.items()]))
st.markdown("</div>", unsafe_allow_html=True)

# =========================
# Prediksi
# =========================
if st.button("🔍 Prediksi Penyakit"):
    data = np.array(input_data).reshape(1, -1)
    data_scaled = scaler.transform(data)
    hasil = model.predict(data_scaled)[0]

    st.session_state.riwayat.append({
        "Umur": umur,
        "Hasil Prediksi": hasil
    })

    # Tampilan hasil
    css_class = "result-positive" if "ISPA" in str(hasil) else "result-negative"

    st.markdown(f"<div class='{css_class}'>", unsafe_allow_html=True)
    st.markdown(f"### ✅ Hasil Prediksi: **{hasil}**")
    st.markdown(
        "Berdasarkan data gejala yang dimasukkan, "
        "model machine learning memberikan hasil klasifikasi di atas. "
        "Hasil ini dapat digunakan sebagai **alat bantu** dan bukan pengganti diagnosis medis."
    )
    st.markdown("</div>", unsafe_allow_html=True)

# =========================
# Riwayat
# =========================
if st.session_state.riwayat:
    st.divider()
    st.subheader("🕒 Riwayat Prediksi")
    st.dataframe(pd.DataFrame(st.session_state.riwayat))

    if st.button("🔄 Reset Riwayat"):
        st.session_state.riwayat = []
        st.experimental_rerun()
