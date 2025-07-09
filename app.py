import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image

# Load your pre-trained model (ensure it's in the same directory or provide full path)
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('Prediction_Model.h5')  # <-- Update path if needed

model = load_model()

# Constants
input_height, input_width = 254, 254
input_channels = 1

# Index to letter mapping
index_to_letter = {
    0: '1', 1: '2', 2: '3', 3: '4', 4: '5', 5: '6', 6: '7', 7: '8', 8: '9',
    9: 'A', 10: 'B', 11: 'C', 12: 'D', 13: 'E', 14: 'F', 15: 'G', 16: 'H',
    17: 'I', 18: 'J', 19: 'K', 20: 'L', 21: 'M', 22: 'N', 23: 'O', 24: 'P',
    25: 'Q', 26: 'R', 27: 'S', 28: 'T', 29: 'U', 30: 'V', 31: 'W', 32: 'X',
    33: 'Y', 34: 'Z'
}

# Streamlit UI
st.title("🤖 Real-time Hand Sign Prediction using CNN")
run = st.checkbox('Start Webcam')

frame_window = st.image([])  # For live video display
prediction_text = st.empty()

# Webcam logic
if run:
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        st.error("Webcam not found or cannot be accessed.")
    else:
        while True:
            ret, frame = cap.read()
            if not ret:
                st.warning("Failed to grab frame from webcam.")
                break

            # Preprocessing
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            resized_frame = cv2.resize(gray_frame, (input_height, input_width))
            normalized_frame = resized_frame / 255.0
            input_frame = np.expand_dims(normalized_frame, axis=(0, -1))

            # Prediction
            prediction = model.predict(input_frame)
            predicted_index = np.argmax(prediction)
            predicted_letter = index_to_letter.get(predicted_index, 'Unknown')

            # Overlay prediction
            cv2.putText(frame, f'Predicted: {predicted_letter}', (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Display in Streamlit
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_window.image(frame_rgb)
            prediction_text.markdown(f"### 🔤 Predicted Sign: *{predicted_letter}*")

        cap.release()
else:
    st.info("☝ Check the box above to start webcam.")