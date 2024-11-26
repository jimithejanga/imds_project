import cv2
import os
import time

# Initialize the camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not access the camera.")
    exit()

# Create directory to save images
output_dir = 'captured_images'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Capture exactly 40 images
image_count = 0
max_images = 40  # Set the limit
print("Capturing 40 images. Press 'q' to stop prematurely.")

while image_count < max_images:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame. Exiting...")
        break

    # Save frame as an image
    image_path = os.path.join(output_dir, f'image_{image_count}.jpg')
    try:
        cv2.imwrite(image_path, frame)
        print(f"Captured and saved: {image_path}")
    except Exception as e:
        print(f"Error saving image: {e}")
    
    image_count += 1

    # Display the frame
    cv2.imshow('Capturing Images', frame)

    # Optional: Add delay for controlled frame capture rate
    time.sleep(0.1)  # Capture a frame every 0.1 seconds

    # Allow manual exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Capture stopped by user.")
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
print(f"Image capture complete. {image_count} images saved in {output_dir}.")
