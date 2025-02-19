import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model

# Load your trained model
model = load_model('best_model.keras')

def load_image(image_file):
    # Convert the file to an opencv image.
    file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    return img

def preprocess_image(img):
    # Resize and normalize the image
    img = cv2.resize(img, (256, 256))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_array = img.astype(np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict_image(img_array):
    # Predict and return the label
    predictions = model.predict(img_array)
    predicted_class = 'PNEUMONIA' if (predictions > 0.5)[0][0] else 'Normal'
    return predicted_class

# Streamlit interface
st.title('Pneumonia Prediction App')

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
if uploaded_file is not None:
    # Load and display the uploaded image
    image = load_image(uploaded_file)
    st.image(image, caption='Uploaded Image', use_container_width=True)  # Updated to use_container_width
    st.write("")
    st.write("Classifying...")
    
    # Process the image and predict
    img_array = preprocess_image(image)
    result = predict_image(img_array)
    
    # Show the result
    st.success(f'The predicted result is: {result}')


