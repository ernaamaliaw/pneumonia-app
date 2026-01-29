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

st.title("🫁 Deteksi Pneumonia dari X-Ray")
st.write(
    "Upload gambar **X-ray dada** untuk memprediksi apakah termasuk **Pneumonia** atau **Normal**."
)

st.markdown("---")

# LOAD MODEL
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(r"inception_with_t_model.h5")

model = load_model()

# IMAGE PREPROCESS
def preprocess_image(image):
    image = image.convert("RGB")
    image = image.resize((299, 299))  # GANTI dari 224 -> 299
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# FILE UPLOADER
uploaded_file = st.file_uploader(
    "Upload gambar X-ray (JPG / PNG)",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Gambar X-Ray", use_container_width=True)

    if st.button("🔍 Prediksi"):
        with st.spinner("Menganalisis gambar..."):
            img = preprocess_image(image)
            prediction = model.predict(img)

            score = prediction[0][0]

        st.markdown("---")

        if score > 0.5:
            st.error(
                f"**PNEUMONIA TERDETEKSI**\n\n"
                f"Confidence: **{score*100:.2f}%**"
            )
        else:
            st.success(
                f"**NORMAL (Tidak Pneumonia)**\n\n"
                f"Confidence: **{(1-score)*100:.2f}%**"
            )

# DISCLAIMER
st.markdown("---")
st.caption(
    "⚠️ Aplikasi ini hanya untuk **edukasi & penelitian**. "
    "Bukan pengganti diagnosis medis profesional."
)
