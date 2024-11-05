import tensorflow as tf
from tensorflow import keras
import streamlit as st
from tensorflow.keras.preprocessing import image
import numpy as np

# Load the image generator model
model = tf.keras.models.load_model("generator.h5")
discriminator = tf.keras.models.load_model("discriminator.h5")

def generate_image(num_images=4):
  # ... (rest of the function)

  # Generate images in a loop
  generated_images = []
  for _ in range(num_images):
    noise = tf.random.normal(shape=(1, 100))  # Generate noise vector for each image
    image = model.predict(noise)  # Predict image for each noise vector
    generated_images.append((image + 1) / 2.0)  # Post-process and scale

  return generated_images


def predict_real_fake(image_path):
  """
  Predicts whether an image is real or fake based on the discriminator model.

  Args:
      image_path: Path to the uploaded image file.

  Returns:
      A string indicating "real" or "fake" based on the model prediction.
  """
  # Preprocess the image
  img = image.load_img(image_path, target_size=(28, 28))  # Adjust image size based on model input
  x = image.img_to_array(img)
  x = np.expand_dims(x, axis=0)  # Add a batch dimension
  x = x / 255.0  # Normalize pixel values to 0-1

  # Predict using the discriminator model
  prediction = discriminator.predict(x)
  prediction = np.squeeze(prediction)  # Remove extra dimension from prediction

  # Determine and return the class label ("real" or "fake")
  class_label = "real" if prediction > 0.5 else "fake"
  return class_label

# Streamlit App
st.title("Image Classifier - Real or Fake?")

uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
  image_data = uploaded_file.read()
  with open("temp_image.jpg", "wb") as f:
    f.write(image_data)
  predicted_class = predict_real_fake("temp_image.jpg")
  st.image(image_data, channels="RGB", caption="Uploaded Image")
  st.write(f"Prediction: {predicted_class}")



# Streamlit App
st.title("AI Image Generator")

num_images_slider = st.slider("Number of Images", min_value=1, max_value=10, value=4)
generate_button = st.button("Generate Images")

if generate_button:
  generated_images = generate_image(num_images=num_images_slider)
  cols = st.columns(len(generated_images))  # Create columns for displaying images
  for i, image in enumerate(generated_images):
    cols[i].image(image, caption=f"Generated Image {i+1}")
