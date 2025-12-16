import streamlit as st
import joblib
import numpy as np
import pandas as pd

# =========================
# Page Config
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

if "riwayat" not in st.session_state:
    st.session_state.riwayat = []

# =========================
# Styling Advanced
# =========================
st.markdown("""
<style>
body {
    background-color: #f4f6f9;
}
.card {
    background: linear-gradient(135deg, #ffffff, #f1f3f6);
    padding: 24px;
    border-radius: 18px;
    box-shadow: 0 8px 20px rgba(0,0,0,0.05);
    margin-bottom: 24px;
}
.result-ispa {
    background: linear-gradient(135deg, #ffe5e5, #fff0f0);
    border-left: 6px solid #e74c3c;
    padding: 20px;
    border-radius: 16px;
}
.result-normal {
    background: linear-gradient(135deg, #e8f9f1, #f2fff9);
    border-left: 6px solid #2ecc71;
    padding: 20px;
    border-radius: 16px;
}
.badge {
    display: inline-block;
    padding: 6px 14px;
    border-radius: 20px;
    font-weight: 600;
    font-size: 14px;
}
.badge-red {
    background: #e74c3c;
    color: white;
}
.badge-green {
    background: #2ecc71;
    color: white;
}
.stButton>button {
    background: linear-gradient(90deg, #2e86de, #1b4f72);
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
st.markdown("<h1 style='text-align:center;'>🩺 Sistem Prediksi Penyakit ISPA</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Implementasi Machine Learning untuk Klasifikasi ISPA</p>", unsafe_allow_html=True)
st.divider()

# =========================
# Input
# =========================
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.subheader("🧑‍⚕️ Informasi Pasien")

umur = st.number_input("Umur Pasien (tahun)", min_value=1, max_value=100)

st.subheader("🤒 Gejala Klinis")
gejala = [
    "Batuk Kering", "Batuk Berdahak", "Demam", "Pilek",
    "Hidung Tersumbat", "Sesak Napas", "Nyeri Tenggorokan", "Sakit Kepala",
    "Mual Muntah", "Nyeri Dada", "Suara Serak", "Kelelahan",
    "Berkeringat Malam", "Nafsu Makan Turun", "Hilang Penciuman", "Nyeri Saat Menelan"
]

col1, col2 = st.columns(2)
input_data = [umur]
gejala_aktif = []

for i, g in enumerate(gejala):
    with col1 if i % 2 == 0 else col2:
        val = st.selectbox(g, ["Tidak", "Ya"])
        input_data.append(1 if val == "Ya" else 0)
        if val == "Ya":
            gejala_aktif.append(g)

st.markdown("</div>", unsafe_allow_html=True)

# =========================
# Ringkasan
# =========================
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.subheader("📋 Ringkasan Input")
st.write(f"**Umur:** {umur} tahun")

if gejala_aktif:
    st.write("**Gejala yang dialami:**")
    st.write("• " + " • ".join(gejala_aktif))
else:
    st.write("Tidak ada gejala yang dipilih.")

st.markdown("</div>", unsafe_allow_html=True)

# =========================
# Prediksi
# =========================
if st.button("🔍 Analisis & Prediksi"):
    X = np.array(input_data).reshape(1, -1)
    X_scaled = scaler.transform(X)
    hasil = model.predict(X_scaled)[0]

    st.session_state.riwayat.append({
        "Umur": umur,
        "Hasil Prediksi": hasil,
        "Jumlah Gejala": len(gejala_aktif)
    })

    if "ISPA" in str(hasil):
        st.markdown("<div class='result-ispa'>", unsafe_allow_html=True)
        st.markdown("<span class='badge badge-red'>RISIKO ISPA</span>", unsafe_allow_html=True)
        st.markdown("### ⚠️ Hasil Prediksi: **ISPA Terdeteksi**")
        st.markdown(
            "Berdasarkan pola gejala yang dimasukkan, sistem mengindikasikan "
            "kemungkinan adanya **Infeksi Saluran Pernapasan Akut (ISPA)**. "
            "Disarankan untuk melakukan pemeriksaan lanjutan ke tenaga medis."
        )
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='result-normal'>", unsafe_allow_html=True)
        st.markdown("<span class='badge badge-green'>KONDISI NORMAL</span>", unsafe_allow_html=True)
        st.markdown("### ✅ Hasil Prediksi: **Tidak Terindikasi ISPA**")
        st.markdown(
            "Berdasarkan data yang dimasukkan, sistem tidak menemukan indikasi kuat "
            "penyakit ISPA. Tetap jaga kesehatan dan pantau kondisi tubuh."
        )
        st.markdown("</div>", unsafe_allow_html=True)

# =========================
# Riwayat
# =========================
if st.session_state.riwayat:
    st.divider()
    st.subheader("🕒 Riwayat Prediksi")
    st.dataframe(pd.DataFrame(st.session_state.riwayat))
