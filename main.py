import cv2
import numpy as np
import os
from pathlib import Path
from utils.blending import pyramid_blending
from utils.color_adjustment import match_colors
from utils.helpers import show_image

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

def ensure_folders_exist():
    """Creates required folders if they don't exist"""
    os.makedirs('images/input', exist_ok=True)
    os.makedirs('images/output', exist_ok=True)

def load_images(path1, path2):
    """Improved image loader with path validation"""
    img1_path = Path(path1).absolute()
    img2_path = Path(path2).absolute()
    
    print(f"Attempting to load:\n- {img1_path}\n- {img2_path}")
    
    if not img1_path.exists():
        raise FileNotFoundError(f"Image not found: {img1_path}")
    if not img2_path.exists():
        raise FileNotFoundError(f"Image not found: {img2_path}")
    
    img1 = cv2.imread(str(img1_path))
    img2 = cv2.imread(str(img2_path))
    
    if img1 is None or img2 is None:
        raise ValueError("Files exist but couldn't be read (may be corrupt or wrong format)")
    
    return img1, img2

def main():
    # 1. Ensure folder structure exists
    ensure_folders_exist()
    
    # 2. Load images with detailed error handling
    try:
        img1, img2 = load_images('images/input/sat1.jpg', 'images/input/sat2.jpg')
        print(f"Successfully loaded images with shapes: {img1.shape} and {img2.shape}")
    except Exception as e:
        print(f"ERROR: {e}")
        print("Contents of images/input:", os.listdir('images/input'))
        return

    # 3. Create blend mask
    mask = create_blend_mask(img1.shape)

    # 4. Color correction
    print("Applying color correction...")
    img2_corrected = match_colors(img2, img1, mask)

    # 5. Pyramid blending
    print("Performing pyramid blending...")
    result = pyramid_blending(img1, img2_corrected, mask, levels=5)

    # 6. Save results
    output_path = 'images/output/composite.jpg'
    cv2.imwrite(output_path, result)
    print(f"Composite image saved to: {Path(output_path).absolute()}")

    # 7. Display results
    show_image(result, "Final Composition")

    # Optional debug views
    if input("Show intermediate steps? (y/n): ").lower() == 'y':
        show_image(img1, "Original Image 1")
        show_image(img2, "Original Image 2")
        show_image(cv2.merge([mask*255]*3), "Blend Mask")
        show_image(img2_corrected, "Color-Corrected Image 2")

if __name__ == "__main__":
    main()
