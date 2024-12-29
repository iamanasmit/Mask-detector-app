import streamlit as st
import pickle
import cv2
import numpy as np

# Load the pre-trained model
model = pickle.load(open('model.pkl', 'rb'))

# Set up the Streamlit app title
st.title('Mask Detection App')

# Function to detect mask
def detect_mask(img):
    img_resized = cv2.resize(img, (224, 224))
    img_resized = np.expand_dims(img_resized, axis=0) / 255.0  # Normalize the image
    prediction = model.predict(img_resized)[0][0]
    return prediction

# Instructions for the user
st.markdown("### Instructions:")
st.markdown("1. Click 'Start Camera' to activate the webcam.")
st.markdown("2. The app will predict whether a mask is detected or not.")
st.markdown("3. Press 'q' to exit the camera feed.")

# Button to start camera feed
if st.button('Start Camera'):
    cap = cv2.VideoCapture(0)
    stframe = st.empty()  # Create a placeholder for video frames

    stop_button = st.button('Stop Camera')  # Add stop button

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to capture video. Please check your camera.")
            break

        # Process the frame
        prediction = detect_mask(frame)
        label = 'Mask Detected' if prediction > 0.5 else 'No Mask'

        # Display the result on the frame
        color = (0, 255, 0) if label == 'Mask Detected' else (0, 0, 255)
        cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        # Show the frame in Streamlit
        stframe.image(frame, channels="BGR")

        # Break loop when 'Stop Camera' button is pressed
        if stop_button:
            break

    cap.release()
    cv2.destroyAllWindows()
