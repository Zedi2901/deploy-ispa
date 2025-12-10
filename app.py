import streamlit as st
import joblib
import numpy as np

# =============================
# Page Config
# =============================
st.set_page_config(
    page_title="Sistem Prediksi ISPA",
    page_icon="🩺",
    layout="centered"
)

# =============================
# Load Model
# =============================
model = joblib.load("model_ispa.pkl")
scaler = joblib.load("scaler.pkl")

# =============================
# Custom CSS
# =============================
st.markdown("""
    <style>
        body {
            background-color: #f4f6f9;
        }
        .main-title {
            text-align: center;
            font-size: 40px;
            font-weight: bold;
            color: #2c3e50;
        }
        .sub-title {
            text-align: center;
            color: #7f8c8d;
            margin-bottom: 25px;
        }
        .card {
            background-color: white;
            padding: 20px;
            border-radius: 16px;
            box-shadow: 0px 4px 12px rgba(0,0,0,0.08);
            margin-bottom: 20px;
        }
        .stButton>button {
            background-color: #1abc9c;
            color: white;
            font-size: 18px;
            height: 3em;
            width: 100%;
            border-radius: 12px;
        }
        .result-success {
            background-color: #e8f8f5;
            padding: 20px;
            border-radius: 12px;
            font-size: 20px;
            color: #117864;
            font-weight: bold;
            text-align: center;
        }
        .result-warning {
            background-color: #fdecea;
            padding: 20px;
            border-radius: 12px;
            font-size: 20px;
            color: #922b21;
            font-weight: bold;
            text-align: center;
        }
    </style>
""", unsafe_allow_html=True)

# =============================
# Header
# =============================
st.markdown("<div class='main-title'>🩺 Sistem Prediksi Penyakit ISPA</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>Tampilan profesional untuk analisis gejala pasien</div>", unsafe_allow_html=True)

# =============================
# Form Input
# =============================
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.subheader("🧑‍⚕️ Data Pasien")

umur = st.number_input("Umur Pasien", min_value=0, max_value=100)

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

st.markdown("</div>", unsafe_allow_html=True)

# =============================
# Prediksi
# =============================
if st.button("🔍 ANALISIS PREDIKSI"):
    data_np = np.array(input_data).reshape(1, -1)
    data_scaled = scaler.transform(data_np)
    hasil = model.predict(data_scaled)[0]

    st.divider()
    st.subheader("📊 Hasil Analisis")

    # Mapping hasil ke label lebih manusiawi
    if hasil == 1:
        st.markdown("<div class='result-warning'>⚠️ Pasien Terindikasi ISPA</div>", unsafe_allow_html=True)
        st.info("Disarankan untuk melakukan pemeriksaan lanjutan ke fasilitas kesehatan.")
    else:
        st.markdown("<div class='result-success'>✅ Pasien Tidak Terindikasi ISPA</div>", unsafe_allow_html=True)
        st.success("Kondisi relatif aman berdasarkan gejala yang dimasukkan.")
