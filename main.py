import cv2
import numpy as np
import os  # Added for path handling
from utils.blending import pyramid_blending
from utils.color_adjustment import match_colors
from utils.helpers import load_images, show_image

def create_blend_mask(img_shape, blend_width=100):
    """Create a gradual blend mask with proper shape handling"""
    if isinstance(img_shape, tuple):  # If passed full shape (height, width, channels)
        height, width = img_shape[:2]
    else:  # If passed an image array
        height, width = img_shape.shape[:2]
    
    mask = np.zeros((height, width), dtype=np.float32)
    center = width // 2
    
    # Create smooth transition
    for x in range(center - blend_width, center + blend_width):
        if x < center:
            mask[:, x] = (x - (center - blend_width)) / blend_width
        else:
            mask[:, x] = 1 - (x - center) / blend_width
            
    return mask

def ensure_output_folder_exists():
    """Creates output folder if it doesn't exist"""
    os.makedirs('images/output', exist_ok=True)

def main():
    # 1. Ensure output folder exists
    ensure_output_folder_exists()

    # 2. Load satellite images
    try:
        img1, img2 = load_images('images/input/sat1.jpg', 'images/input/sat2.jpg')
    except Exception as e:
        print(f"Error loading images: {e}")
        return

    # Verify images loaded correctly
    if img1 is None or img2 is None:
        print("Error: Could not load one or both images")
        return

    print(f"Loaded images with shapes: {img1.shape} and {img2.shape}")

    # 3. Create blend mask
   mask = create_blend_mask(img1.shape) 

    # 4. Color correction
    print("Applying color correction...")
    img2_corrected = match_colors(img2, img1, mask)

    # 5. Pyramid blending
    print("Performing pyramid blending...")
    result = pyramid_blending(img1, img2_corrected, mask, levels=5)

    # 6. Save and show results
    output_path = 'images/output/composite.jpg'
    cv2.imwrite(output_path, result)
    print(f"Composite image saved to: {output_path}")

    # Display results
    show_image(result, "Final Composition")

    # Show intermediate steps
    if input("Show intermediate steps? (y/n): ").lower() == 'y':
        show_image(img1, "Original Image 1")
        show_image(img2, "Original Image 2")
        show_image(cv2.merge([mask*255]*3), "Blend Mask")
        show_image(img2_corrected, "Color-Corrected Image 2")

if __name__ == "__main__":
    main()
