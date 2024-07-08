import ultralytics
from ultralytics import YOLO
import cv2
import os

model = YOLO("C:/VI/BDA/Final/project/runs/segment/train2/weights/best.pt")

# Path to the folder containing input images
input_folder = "C:/VI/BDA/Final/project/Images"
# Path to the folder to save cropped objects
output_folder = "C:/VI/BDA/Final/project/Cropped_images"

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Get a list of all files in the input folder
image_files = [os.path.join(input_folder, file) for file in os.listdir(input_folder) if file.endswith(('.jpg', '.jpeg', '.png'))]

for image_path in image_files:
    # Perform object detection on each image
    results = model(image_path)

    # Load the image
    img = cv2.imread(image_path)

    # Check if any objects are detected
    if results:
        boxes = results[0].boxes.xyxy.tolist()

        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box
            # Crop the object using the bounding box coordinates
            ultralytics_crop_object = img[int(y1):int(y2), int(x1):int(x2)]
            # Save the cropped object as an image in the output folder
            filename = os.path.splitext(os.path.basename(image_path))[0]  # Extract filename without extension
            cv2.imwrite(os.path.join(output_folder, f'ultralytics_crop_{filename}_{i}.jpg'), ultralytics_crop_object)
    else:
        print(f"No objects detected in {image_path}")

print("Cropping completed. Cropped objects saved in:", output_folder)
