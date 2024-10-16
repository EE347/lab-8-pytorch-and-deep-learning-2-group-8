from picamera2 import Picamera2
import cv2
import os
import time

# Folder paths and configuration
base_dir = 'data'
train_dir = os.path.join(base_dir, 'train')
test_dir = os.path.join(base_dir, 'test')

# Labels for each person
labels = {0: 'Person_1', 1: 'Person_2'}

# Create folder structure if it does not exist
for split in ['train', 'test']:
    for label in labels.keys():
        path = os.path.join(base_dir, split, str(label))
        os.makedirs(path, exist_ok=True)

# Initialize Picamera2
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"format": 'XRGB8888', "size": (640, 480)}))
picam2.start()

# Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Function to capture images for a given person
def capture_images(label, num_train=50, num_test=10):
    count = 0
    total_images = num_train + num_test
    split_point = num_train  # First 50 for training, next 10 for testing

    while count < total_images:
        # Capture frame
        frame = picam2.capture_array()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the image
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            # Crop and resize the face to 64x64
            face = gray[y:y + h, x:x + w]
            face_resized = cv2.resize(face, (64, 64))

            # Determine save location (train or test)
            if count < split_point:
                folder = os.path.join(train_dir, str(label))
            else:
                folder = os.path.join(test_dir, str(label))

            # Save the image
            image_path = os.path.join(folder, f"{label}_{count}.jpg")
            cv2.imwrite(image_path, face_resized)
            count += 1
            print(f"Saved {image_path}")

            # Display the frame with a rectangle around the detected face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.imshow("Capture", frame)

            # Wait a moment for the next capture
            cv2.waitKey(500)

            # Exit the inner loop to prevent multiple detections from the same frame
            break

        # Display the frame in a window
        cv2.imshow("Capture", frame)

        # Exit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            return count

    return count

# Capture images for each person
print("Capturing images for Person 1. Press 'q' to quit.")
capture_images(label=0)

print("Capturing images for Person 2. Press 'q' to quit.")
capture_images(label=1)

# Clean up
cv2.destroyAllWindows()
picam2.stop()

