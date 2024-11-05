import joblib
import streamlit as st
import numpy as np

def generate(data):
  generator = joblib.load("generator.joblib")
  return generator.generate(data)

def discriminate(data):
  discriminator = joblib.load("discriminator.joblib")
  return discriminator.discriminate(data)

def generate_and_display_images(square=5, epochs=0, latent_dim=None):
    """Generates and displays images from your model using Streamlit.

    Args:
        square (int, optional): The number of images per row/column in the grid. Defaults to 5.
        epochs (int, optional): The epoch number to display in the title (if applicable). Defaults to 0.
        latent_dim (int, optional): The latent dimension of the noise vector. Defaults to None (requires manual input).
    """

    if latent_dim is None:
        latent_dim = st.number_input("Latent Dimension", min_value=1)

    noise = np.random.normal(0, 1, (square * square, 1, latent_dim))  # Generate noise for all images
    images = generate(noise)  # Generate images using your model

    # Ensure correct image data format (0-255 range)
    images = np.clip((images + 1) / 2, 0, 1) * 255  # Normalize and convert to uint8

    st.columns(square)  # Create a grid layout
    for i, image in enumerate(images):
        col = st.columns[i]
        with col:
            st.image(image.astype(np.uint8), caption=f"Image {i+1}" if epochs == 0 else f"Generated Image at Epoch: {epochs}")

    st.success("Images generated!")
