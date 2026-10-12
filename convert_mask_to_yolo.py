import os
import cv2

# CHANGE THESE PATHS
IMAGE_FOLDER = "dataset/train/img"
MASK_FOLDER = "dataset/train/label"
OUTPUT_FOLDER = "dataset/train/label"

"


CLASS_ID = 0  # change if multiple classes

for filename in os.listdir(MASK_FOLDER):
    if filename.endswith(".png"):
        mask_path = os.path.join(MASK_FOLDER, filename)
        image_path = os.path.join(IMAGE_FOLDER, filename)

        mask = cv2.imread(mask_path, 0)
        image = cv2.imread(image_path)

        if mask is None or image is None:
            continue

        h, w = mask.shape

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        label_lines = []

        for cnt in contours:
            x, y, bw, bh = cv2.boundingRect(cnt)

            x_center = (x + bw / 2) / w
            y_center = (y + bh / 2) / h
            width = bw / w
            height = bh / h

            label_lines.append(f"{CLASS_ID} {x_center} {y_center} {width} {height}")

        # Save .txt
        txt_filename = filename.replace(".png", ".txt")
        with open(os.path.join(OUTPUT_FOLDER, txt_filename), "w") as f:
            for line in label_lines:
                f.write(line + "\n")

print("Conversion Complete!")