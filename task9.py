import cv2
import torch
import torch.nn as nn
from torchvision import transforms
import numpy as np
from picamera2 import Picamera2
from torchvision.models import mobilenet_v3_small
import os
import time

# Load the trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize the model
model = mobilenet_v3_small(weights=None, num_classes=2).to(device)
model.load_state_dict(torch.load('best_model.pth', map_location=device))
model.eval()  # Set model to evaluation mode

# Define the face detector (Haar Cascade)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Define the transform to apply to the cropped image before passing it into the model
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((64, 64)),  # Resize to match the input size expected by the model
    transforms.Grayscale(num_output_channels=3),  # Convert to 3 channels (for RGB compatibility)
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize like ImageNet
])

# Initialize Picamera2
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"format": 'XRGB8888', "size": (640, 480)}))
picam2.start()

# Dictionary for teammates
labels = {0: 'Person 1', 1: 'Person 2'}

# Function to capture an image, crop face, and classify
def capture_and_classify():
    frame = picam2.capture_array()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # If no faces are detected
    if len(faces) == 0:
        print("No face detected!")
        return None, None

    # Crop and classify each face
    for (x, y, w, h) in faces:
        # Crop the face from the image
        face = gray[y:y + h, x:x + w]
        
        # Transform the face image to match model input requirements
        face_tensor = transform(face).unsqueeze(0).to(device)

        # Get the prediction from the model
        with torch.no_grad():
            output = model(face_tensor)
            _, predicted = torch.max(output, 1)
            predicted_label = labels[predicted.item()]

        # Draw rectangle around face and display predicted label
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, f"Prediction: {predicted_label}", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    return frame, predicted_label if len(faces) > 0 else None


# Main loop
try:
    while True:
        # Capture and classify the image
        frame, predicted_label = capture_and_classify()

        if frame is not None:
            # Display the frame with the face and prediction
            cv2.imshow('Teammate Classification', frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    pass

finally:
    # Cleanup
    cv2.destroyAllWindows()
    picam2.stop()
