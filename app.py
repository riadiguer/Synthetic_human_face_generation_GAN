import streamlit as st
import numpy as np

from prediction import generate, discriminate, generate_and_display_images

epochs = 0

st.title("Image Generation App")

num_images_slider = st.slider("Number of Images to Generate", 1, 25, 5)  # Adjust range as needed

if st.button("Generate Images"):
    generate_and_display_images(square=int(np.sqrt(num_images_slider)), epochs=epochs)



