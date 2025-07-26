import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
from PIL import Image

model = tf.keras.models.load_model(r"D:\aitronxi\me\model_inceptionV3 (1).h5")  

category = {
    0: 'Bean', 1: 'Bitter_Gourd', 2: 'Bottle_Gourd', 3: 'Brinjal',
    4: 'Broccoli', 5: 'Cabbage', 6: 'Capsicum', 7: 'Carrot',
    8: 'Cauliflower', 9: 'Cucumber', 10: 'Papaya', 11: 'Potato',
    12: 'Pumpkin', 13: 'Radish', 14: 'Tomato'
}

st.markdown("<h3>🍅🥕 Vegetable Classification Using Transfer Learning</h3>", unsafe_allow_html=True)


uploaded_file = st.file_uploader("Upload a vegetable image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded Image', use_column_width=True)

    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = tf.expand_dims(img_array, axis=0)
    img_array = tf.cast(img_array, tf.float32) / 255.0

    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions[0])
    predicted_class_name = category[predicted_class_index]
    confidence = predictions[0][predicted_class_index]

    st.markdown(f"### Prediction: **{predicted_class_name}**")
    st.markdown(f"### Confidence: **{confidence*100:.2f}%**")

    with st.expander("See all class probabilities"):
        for i, prob in enumerate(predictions[0]):
            st.write(f"{category[i]}: {prob*100:.2f}%")