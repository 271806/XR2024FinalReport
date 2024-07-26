import sys
import cv2
import torch
import torchvision
import numpy as np
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

def print_versions():
    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"torchvision version: {torchvision.__version__}")
    print(f"NumPy version: {np.__version__}")
    print(f"OpenCV version: {cv2.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")


# print_versions()

"""Device"""
device = "cuda" if torch.cuda.is_available() else "cpu"

"""Paths"""
checkpoint = "checkpoints/sam_vit_h_4b8939.pth"
pic = "pics/snack2.jpg"
output_img_with_masks = "output/output_with_masks.png"
output_only_masks = "output/only_masks.png"
output_masks = "output/output_masks.txt"

"""Model"""
model_type = "vit_h" # Huge

try:
    sam = sam_model_registry[model_type](checkpoint=checkpoint)
    sam.to(device=device)
    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=8,                # Adjust as needed
        pred_iou_thresh=0.89,              # Adjust as needed
        stability_score_thresh=0.96,       # Adjust as needed
        crop_n_layers=0,                   # Adjust as needed
        crop_n_points_downscale_factor=8,  # Adjust as needed
        min_mask_region_area=2900000,      # Increase this to filter out small masks
    )

except Exception as e:
    print(f"Error initializing SAM model: {str(e)}")
    sys.exit(1)

"""Images"""
try:
    image = cv2.imread(pic)
    if image is None:
        raise ValueError(f"Image not found or failed to load at path: {pic}")
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_rgb = np.array(image_rgb, dtype=np.uint8)
except Exception as e:
    print(f"Error loading or processing image: {str(e)}")
    sys.exit(1)

print(f"Type of image_rgb: {type(image_rgb)}")
print(f"Shape of image_rgb: {image_rgb.shape}")
print(f"Dtype of image_rgb: {image_rgb.dtype}")

try:
    masks = mask_generator.generate(image_rgb)
    print(f"Number of masks generated: {len(masks)}")
    if masks:
        print(f"Keys in first mask: {masks[0].keys()}")
except Exception as e:
    print(f"Error generating masks: {str(e)}")
    sys.exit(1)

# Predefined set of vibrant colors
colors = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255),
    (255, 255, 0), (255, 0, 255), (0, 255, 255),
    (128, 0, 0), (0, 128, 0), (0, 0, 128),
    (128, 128, 0), (128, 0, 128), (0, 128, 128)
]

try:
    overlay = np.zeros_like(image)
    output = image.copy()

    for i, mask in enumerate(masks):
        color = colors[i % len(colors)]
        colored_mask = np.zeros_like(image, dtype=np.uint8)
        colored_mask[mask["segmentation"]] = color
        
        # Draw mask boundary
        contours, _ = cv2.findContours(mask["segmentation"].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(colored_mask, contours, -1, (255, 255, 255), 1)

        overlay = cv2.addWeighted(overlay, 1, colored_mask, 0.5, 0)
        if (i+1) % 10 == 0:
            print(f"Processed {i+1}/{len(masks)} masks")

    # Save the image with masks overlaid on the original image
    output_with_masks = cv2.addWeighted(image, 0.5, overlay, 0.5, 0)
    cv2.imwrite(output_img_with_masks, cv2.cvtColor(output_with_masks, cv2.COLOR_RGB2BGR))
    
    # Save the image with only the visualized masks
    cv2.imwrite(output_only_masks, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    with open(output_masks, "w") as f:
        for mask in masks:
            f.write(f"{mask}\n")

    print(f"Image with masks saved to {output_img_with_masks}")
    print(f"Only masks image saved to {output_only_masks}")
    print(f"Masks details saved to {output_masks}")
except Exception as e:
    print(f"Error applying masks or saving output: {str(e)}")
    sys.exit(1)
    

import os

# Create a directory to store cropped images
output_crops_dir = "output/crops"
os.makedirs(output_crops_dir, exist_ok=True)

try:
    for i, mask in enumerate(masks):
        # Get the bounding box
        bbox = mask['bbox']  # [x, y, width, height]
        
        # Crop the image
        x, y, w, h = bbox
        cropped_image = image_rgb[y:y+h, x:x+w]
        
        # Apply the mask to the cropped image
        mask_array = mask['segmentation'][y:y+h, x:x+w]
        cropped_image[~mask_array] = [0, 0, 0]  # Set background to black
        
        # Save the cropped image
        output_path = os.path.join(output_crops_dir, f"crop_{i}.png")
        cv2.imwrite(output_path, cv2.cvtColor(cropped_image, cv2.COLOR_RGB2BGR))
        
        if (i+1) % 10 == 0:
            print(f"Saved {i+1}/{len(masks)} cropped images")

    print(f"All cropped images saved to {output_crops_dir}")
except Exception as e:
    print(f"Error cropping and saving individual masks: {str(e)}")
    sys.exit(1)

