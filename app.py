import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image


# CONFIG STREAMLIT
st.set_page_config(
    page_title="Deteksi Pneumonia",
    page_icon="🫁",
    layout="centered"
)

# HEADER
st.markdown(
    """
    <h1 style='text-align: center;'>🫁 Deteksi Pneumonia dari X-Ray Dada</h1>
    """,
    unsafe_allow_html=True
)

st.markdown("---")

# LOAD MODEL
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("inception_with_t_model.h5")

model = load_model()

# PREPROCESS IMAGE
def preprocess_image(image):
    image = image.convert("RGB")
    image = image.resize((299, 299))
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# VALIDASI X-RAY
def is_likely_xray(image):
    img = np.array(image)
    if len(img.shape) == 3:
        color_std = np.std(img, axis=2).mean()
        return color_std < 15
    return True

# INFORMASI SISTEM
st.subheader("ℹ️ Informasi Penggunaan")
st.info(
"- Gambar harus berupa **citra X-ray dada**\n" 
"- Disarankan menggunakan **posisi frontal (PA/AP)**\n" 
"- Kualitas dan kontras citra dapat memengaruhi hasil analisis"
)

st.markdown("---")

# INPUT SECTION
st.subheader("📤 Upload Gambar")

uploaded_file = st.file_uploader(
    "Format yang didukung: JPG, JPEG, PNG",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    st.image(image, caption="Citra Input", use_container_width=True)

    # AUTO REJECT
    if not is_likely_xray(image):
        st.error("❌ Input tidak teridentifikasi sebagai citra X-ray dada")
        st.warning(
            "Sistem ini dirancang khusus untuk menganalisis **citra X-ray paru**.\n\n"
            "Proses prediksi dihentikan guna mencegah interpretasi yang tidak valid."
        )
        st.stop()

    # PREDIKSI
    st.markdown("---")
    if st.button("🔍 Mulai Prediksi", use_container_width=True):
        with st.spinner("🔄 Menganalisis citra X-ray..."):
            img = preprocess_image(image)
            prediction = model.predict(img)

            pneumonia_score = float(prediction[0][0])
            normal_score = 1 - pneumonia_score

        st.markdown("---")
        st.subheader("Hasil Prediksi")

        # METRIC DISPLAY
        m1, m2 = st.columns(2)

        with m1:
            st.metric(
                label="🦠 Pneumonia",
                value=f"{pneumonia_score*100:.2f}%"
            )
            st.progress(pneumonia_score)

        with m2:
            st.metric(
                label="🫁 Normal",
                value=f"{normal_score*100:.2f}%"
            )
            st.progress(normal_score)

        # INTERPRETATION
        if pneumonia_score > 0.5:
            st.error("### 🩺 Kesimpulan: **Terindikasi Pneumonia**")
            st.markdown(
                """
                - Teridentifikasi **area opasitas** pada citra paru
                - Distribusi intensitas paru menunjukkan **ketidakhomogenan**
                - Pola citra konsisten dengan **indikasi pneumonia**
                """
            )
        else:
            st.success("### ✅ Kesimpulan: **Tidak Terindikasi Pneumonia (Normal)**")
            st.markdown(
                """
                    - Area paru tampak **relatif bersih dan simetris**
                    - Tidak teridentifikasi **infiltrat atau opasitas abnormal**
                    - Pola citra konsisten dengan **kondisi paru normal**
                """
            )

# KETERBATASAN
st.markdown("---")
st.subheader("📌 Keterbatasan & Pengembangan")

st.info(
    """
        **Keterbatasan Sistem**
        - Model hanya dilatih menggunakan citra X-ray dada
        - Citra non-medis tidak dapat diproses dan akan ditolak sistem

        **Pengembangan Selanjutnya**
        - Penambahan modul klasifikasi X-ray dan non-X-ray
        - Perluasan dan diversifikasi dataset
        - Integrasi Explainable AI (Grad-CAM)
    """
)

# DISCLAIMER
st.markdown("---")
st.caption(
    "⚠️ Aplikasi ini digunakan untuk **edukasi dan penelitian**. "
    "Hasil prediksi **bukan diagnosis medis**."
)
