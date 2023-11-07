import cv2
import numpy as np
import tensorflow as tf

# Load the trained CNN model
model = tf.keras.models.load_model("C:/Projects/FIREDETECTPROJECT/Real-Time-Fire-Detection-CNN-master/Real-Time-Fire-Detection-CNN-master/Trained Models/best_weights.h5")

# Define a function to preprocess an image frame
def preprocess_frame(frame):
    frame = cv2.resize(frame, (224, 224))  # Resize to match the model's input size
    frame = frame / 255.0  # Normalize pixel values to [0, 1]
    return frame

# Open a video capture stream (0 is the default camera, or you can specify a video file)
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the video stream
    ret, frame = cap.read()
    if not ret:
        break
    
    # Preprocess the frame
    preprocessed_frame = preprocess_frame(frame)
    
    # Expand dimensions to match the input shape expected by the model
    preprocessed_frame = np.expand_dims(preprocessed_frame, axis=0)
    
    # Make predictions using the loaded model
    predictions = model.predict(preprocessed_frame)
    
    # Check if the model predicts fire
    if predictions[0][0] > 0.5:  # Adjust the threshold as needed
        # Draw a bounding box or overlay text on the frame to indicate fire
        cv2.putText(frame, "Fire Detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    # Display the processed frame
    cv2.imshow('Fire Detection', frame)
    
    # Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()

