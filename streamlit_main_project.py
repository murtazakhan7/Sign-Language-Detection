import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import joblib
import speech_recognition as sr
import webbrowser
import os
import win32com.client

# Initialize the speaker
speaker = win32com.client.Dispatch("SAPI.SpVoice")

# Function to make the AI speak
def speak(user):
    speaker.Speak(user)

# Load model and scaler
@st.cache_resource
def load_model_scaler():
    model = joblib.load('logistic_model.pkl')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

# MediaPipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Map labels back to characters
def map_label_to_character(label):
    if 0 <= label <= 9:
        return str(label)  # For digits 0-9
    else:
        return chr(label - 10 + ord('a'))  # For letters a-z

# Extract hand landmarks
def extract_landmarks(results, frame_shape):
    """
    Extract 63 hand landmarks relative to the bounding box of the hand.
    """
    if results.multi_hand_landmarks:
        landmarks = []
        h, w, _ = frame_shape  # Get frame dimensions

        for hand_landmarks in results.multi_hand_landmarks:
            # Calculate bounding box around the hand
            x_min, y_min = w, h
            x_max, y_max = 0, 0

            for lm in hand_landmarks.landmark:
                x, y = int(lm.x * w), int(lm.y * h)
                x_min, y_min = min(x_min, x), min(y_min, y)
                x_max, y_max = max(x_max, x), max(y_max, y)

            # Crop dimensions
            x_min, y_min = max(x_min, 0), max(y_min, 0)
            x_max, y_max = min(x_max, w), min(y_max, h)

            # Normalize landmarks relative to bounding box
            for lm in hand_landmarks.landmark:
                rel_x = (lm.x * w - x_min) / (x_max - x_min)  # Relative X
                rel_y = (lm.y * h - y_min) / (y_max - y_min)  # Relative Y
                landmarks.extend([rel_x, rel_y, lm.z])  # Append Z unmodified

            return np.array(landmarks).reshape(1, -1)  # Return flattened landmarks
    return None


# Predict function
def predict_sign(frame, model, scaler):
    """
    Process the frame to extract hand landmarks and predict the gesture.
    """
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    landmarks = extract_landmarks(results)

    if landmarks is not None:
        landmarks_scaled = scaler.transform(landmarks)  # Scale input
        prediction = model.predict(landmarks_scaled)[0]
        return map_label_to_character(prediction)
    return None

# Streamlit App
# Streamlit App with Optimizations
def main():
    
    predicted_char = str
    st.title("Optimized Sign Language Recognition")
    st.write("Predicts hand gestures (0-9, a-z) with improved speed and accuracy.")

    model, scaler = load_model_scaler()
    run_camera = st.checkbox("Start Camera", value=False)

    if run_camera:
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        stframe = st.empty()

        frame_skip = 2
        frame_counter = 0

        while run_camera:
            ret, frame = cap.read()
            if not ret:
                st.error("Unable to access the camera.")
                break

            # Skip frames to reduce lag
            frame_counter += 1
            if frame_counter % frame_skip != 0:
                continue
      
            # Predict using landmarks
            landmarks = extract_landmarks(hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)), frame.shape)
            if landmarks is not None:
                scaled_landmarks = scaler.transform(landmarks)
                prediction = model.predict(scaled_landmarks)[0]
                predicted_char = map_label_to_character(prediction)
                # predicted.append(predicted_char)

                # Display Prediction
                cv2.putText(frame, f"Prediction: {predicted_char}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
            
            # Stream video
            stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", use_column_width=True)
            speak(predicted_char) 
        cap.release()
    st.write("Uncheck the box to stop the camera.")

if __name__ == "__main__":
    
    main()

