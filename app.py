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
# Session State (Riwayat)
# =========================
if "riwayat" not in st.session_state:
    st.session_state.riwayat = []

# =========================
# Styling
# =========================
st.markdown("""
<style>
.stButton>button {
    background-color: #2e86de;
    color: white;
    font-size: 18px;
    height: 3em;
    width: 100%;
    border-radius: 12px;
}
</style>
""", unsafe_allow_html=True)

# =========================
# Header
# =========================
st.markdown("<h1 style='text-align:center;'>🩺 Aplikasi Prediksi Penyakit ISPA</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Masukkan data pasien untuk mendapatkan hasil prediksi</p>", unsafe_allow_html=True)
st.divider()

# =========================
# Input Data
# =========================
st.subheader("🧑‍⚕️ Data Pasien")
umur = st.number_input("Umur", min_value=1, max_value=100)

st.subheader("🤒 Gejala Pasien")
gejala = [
    "Batuk Kering", "Batuk Berdahak", "Demam", "Pilek",
    "Hidung Tersumbat", "Sesak Napas", "Nyeri Tenggorokan", "Sakit Kepala",
    "Mual Muntah", "Nyeri Dada", "Suara Serak", "Kelelahan",
    "Berkeringat Malam", "Nafsu Makan Turun", "Hilang Penciuman", "Nyeri Saat Menelan"
]

col1, col2 = st.columns(2)
input_data = [umur]
input_display = {}

for i, g in enumerate(gejala):
    with col1 if i % 2 == 0 else col2:
        pilihan = st.selectbox(g, ["Tidak", "Ya"])
        nilai = 1 if pilihan == "Ya" else 0
        input_data.append(nilai)
        input_display[g] = pilihan

# =========================
# Ringkasan Input
# =========================
st.divider()
st.subheader("📋 Ringkasan Data Pasien")
st.write(f"**Umur:** {umur} tahun")
st.write(", ".join([f"{k}: {v}" for k, v in input_display.items()]))

# =========================
# Prediksi
# =========================
if st.button("🔍 Prediksi Penyakit"):
    data_np = np.array(input_data).reshape(1, -1)
    data_scaled = scaler.transform(data_np)

    prediksi = model.predict(data_scaled)[0]

    # Confidence / Probability
    proba = model.predict_proba(data_scaled)[0]
    kelas = model.classes_
    confidence = np.max(proba) * 100

    # Simpan riwayat
    st.session_state.riwayat.append({
        "Umur": umur,
        "Diagnosis": prediksi,
        "Confidence (%)": round(confidence, 2)
    })

    # =========================
    # Tampilan Hasil
    # =========================
    st.success(f"✅ **Hasil Prediksi: {prediksi}**")
    st.info(f"🔎 Tingkat Kepercayaan Model: **{confidence:.2f}%**")

    st.markdown(
        f"""
        **Penjelasan:**  
        Berdasarkan gejala klinis yang dimasukkan, model machine learning
        memprediksi bahwa pasien memiliki kemungkinan **{prediksi}**.
        """
    )

    # =========================
    # Visualisasi Probabilitas
    # =========================
    df_proba = pd.DataFrame({
        "Diagnosis": kelas,
        "Probabilitas (%)": proba * 100
    })

    st.subheader("📊 Distribusi Probabilitas Diagnosis")
    st.bar_chart(df_proba.set_index("Diagnosis"))

# =========================
# Riwayat Prediksi
# =========================
if st.session_state.riwayat:
    st.divider()
    st.subheader("🕒 Riwayat Prediksi")
    st.dataframe(pd.DataFrame(st.session_state.riwayat))
