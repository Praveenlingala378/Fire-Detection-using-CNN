import cv2
import numpy as np
import tensorflow as tf
import pygame

# Load the trained CNN model
model = tf.keras.models.load_model("C:/Projects/FIREDETECTPROJECT/Real-Time-Fire-Detection-CNN-master/Real-Time-Fire-Detection-CNN-master/Trained Models/best_weights.h5")

# Print the summary of the model's architecture
model.summary()

# Initialize Pygame mixer for sound
pygame.mixer.init()

# Load the alert sound file
alert_sound = pygame.mixer.Sound("Scripts\mixkit-alert-alarm-1005.wav")  # Replace 'alert.wav' with your sound file

# Define a function to play the alert sound
def play_alert_sound():
    alert_sound.play()

# Define a function to preprocess an image frame
def preprocess_frame(frame):
    frame = cv2.resize(frame, (150, 150))  # Resize to match the model's input size
    frame = frame / 255.0  # Normalize pixel values to [0, 1]
    return frame

# Open the default camera (0 is usually the default camera, but it may vary)
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()
    if not ret:
        break
    
    # Preprocess the frame
    preprocessed_frame = preprocess_frame(frame)
    
    # Expand dimensions to match the input shape expected by the model
    preprocessed_frame = np.expand_dims(preprocessed_frame, axis=0)
    
    # Make predictions using the loaded model
    predictions = model.predict(preprocessed_frame)
    
    # Get the class label with the highest probability
    class_label = np.argmax(predictions, axis=1)
    
    # Define class labels (you can modify these based on your model)
    class_labels = ["FIRE", "SMOKE", "NEUTRAL"]
    
    # Display the class label on the frame
    cv2.putText(frame, f"Class: {class_labels[class_label[0]]}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    # Check if the model predicts fire
    if class_label[0] == 1:  # Adjust the class index as needed
        # Play the alert sound
        play_alert_sound()
    
    # Display the processed frame
    cv2.imshow('Fire Detection', frame)
    
    # Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam capture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
