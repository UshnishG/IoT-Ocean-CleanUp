from ultralytics import YOLO
import cv2
import numpy as np

# Load the trained model
model = YOLO("best.pt")  # Ensure best.pt is in the same directory

# Load the image
image_path = "Test_Images\\Images\\test_image.jpg"  # Change this to your image file
image = cv2.imread(image_path)

# Run inference
results = model(image)

# Extract detected objects
detections = results[0]  # Get the first result (assuming single image input)

# Create a dictionary to store object counts
object_counts = {}

# Loop through detections
for box in detections.boxes:
    class_id = int(box.cls[0].item())  # Get class ID
    class_name = model.names[class_id]  # Convert to label name
    confidence = box.conf[0].item()  # Confidence score

    # Update object count dictionary
    object_counts[class_name] = object_counts.get(class_name, 0) + 1

# Print detected objects and their counts
print("\n🔍 Detected Objects:")
for obj, count in object_counts.items():
    print(f"{obj}: {count}")

# Display detections one by one
for i, box in enumerate(detections.boxes):
    class_id = int(box.cls[0].item())  # Get class ID
    class_name = model.names[class_id]  # Get class name
    confidence = box.conf[0].item()  # Confidence score

    # Get box coordinates
    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

    # Create a copy of the original image
    temp_image = image.copy()

    # Draw the detected box with label
    cv2.rectangle(temp_image, (x1, y1), (x2, y2), (0, 255, 0), 3)
    cv2.putText(temp_image, f"{class_name} ({confidence:.2f})", (x1, y1 - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Show the image with the detected object
    cv2.imshow(f"Detection {i+1}: {class_name}", temp_image)
    cv2.waitKey(1000)  # Wait 1 second before showing next detection
    cv2.destroyAllWindows()

# Finally, show the full detection result
final_image = results[0].plot()
cv2.imshow("Final Detection", final_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
