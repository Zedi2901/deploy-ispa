import streamlit as st
import joblib
import numpy as np

# =========================
# Konfigurasi Halaman
# =========================
st.set_page_config(
    page_title="Prediksi ISPA",
    page_icon="🩺",
    layout="centered"
)

# =========================
# Load Model
# =========================
model = joblib.load("model_ispa.pkl")
scaler = joblib.load("scaler.pkl")

# =========================
# Styling Tambahan
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
        .stSelectbox label, .stNumberInput label {
            font-weight: 600;
        }
    </style>
""", unsafe_allow_html=True)

# =========================
# Header Aplikasi
# =========================
st.markdown("<h1 style='text-align: center;'>🩺 Aplikasi Prediksi Penyakit ISPA</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Masukkan data pasien untuk mendapatkan hasil prediksi.</p>", unsafe_allow_html=True)

st.divider()

# =========================
# Form Input
# =========================
st.subheader("🧑‍⚕️ Data Pasien")
umur = st.number_input("Umur", min_value=0, max_value=100)

st.subheader("🤒 Gejala Pasien")

gejala = [
    "Batuk Kering", "Batuk Berdahak", "Demam", "Pilek",
    "Hidung Tersumbat", "Sesak Napas", "Nyeri Tenggorokan", "Sakit Kepala",
    "Mual Muntah", "Nyeri Dada", "Suara Serak", "Kelelahan",
    "Berkeringat Malam", "Nafsu Makan Turun", "Hilang Penciuman", "Nyeri Saat Menelan"
]

col1, col2 = st.columns(2)
input_data = [umur]

for i, g in enumerate(gejala):
    with col1 if i % 2 == 0 else col2:
        pilihan = st.selectbox(g, ["Tidak", "Ya"])
        input_data.append(1 if pilihan == "Ya" else 0)

st.divider()

# =========================
# Tombol Prediksi
# =========================
if st.button("🔍 Prediksi Penyakit"):
    data_np = np.array(input_data).reshape(1, -1)
    data_scaled = scaler.transform(data_np)
    hasil = model.predict(data_scaled)

    st.success(f"✅ Hasil Prediksi: {hasil[0]}")
