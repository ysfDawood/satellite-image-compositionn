import cv2
import numpy as np
import os
from pathlib import Path
from utils.blending import pyramid_blending
from utils.color_adjustment import match_colors
from utils.helpers import show_image

def create_blend_mask(img_shape, blend_width=100):
    """Create a 3-channel blend mask for smooth transitions"""
    if isinstance(img_shape, tuple):  # If passed full shape (height, width, channels)
        height, width = img_shape[:2]
    else:  # If passed an image array
        height, width = img_shape.shape[:2]
    
    # Create 3-channel mask
    mask = np.zeros((height, width, 3), dtype=np.float32)
    center = width // 2
    
    # Create smooth transition (applied to all channels)
    for x in range(center - blend_width, center + blend_width):
        if x < center:
            val = (x - (center - blend_width)) / blend_width
        else:
            val = 1 - (x - center) / blend_width
        mask[:, x, :] = val  # Apply to all channels
            
    return mask

def ensure_folders_exist():
    """Creates required folders if they don't exist"""
    os.makedirs('images/input', exist_ok=True)
    os.makedirs('images/output', exist_ok=True)

def load_images(path1, path2):
    """Robust image loader with detailed error reporting"""
    img1_path = Path(path1).absolute()
    img2_path = Path(path2).absolute()
    
    print(f"Attempting to load:\n- {img1_path}\n- {img2_path}")
    
    # Verify files exist
    if not img1_path.exists():
        raise FileNotFoundError(f"Image not found: {img1_path}")
    if not img2_path.exists():
        raise FileNotFoundError(f"Image not found: {img2_path}")
    
    # Read images
    img1 = cv2.imread(str(img1_path))
    img2 = cv2.imread(str(img2_path))
    
    # Verify images were properly read
    if img1 is None or img2 is None:
        raise ValueError(
            "Files exist but couldn't be read. Possible causes:\n"
            "- Corrupt file\n"
            "- Unsupported format\n"
            "- Permission issues"
        )
    
    # Verify image dimensions match
    if img1.shape != img2.shape:
        raise ValueError(
            f"Image size mismatch:\n"
            f"Image 1: {img1.shape}\n"
            f"Image 2: {img2.shape}\n"
            "Both images must have identical dimensions"
        )
    
    return img1, img2

def main():
    try:
        # 1. Ensure folder structure exists
        ensure_folders_exist()
        
        # 2. Load images with validation
        img1, img2 = load_images('images/input/sat1.jpg', 'images/input/sat2.jpg')
        print(f"Successfully loaded images with shapes: {img1.shape} and {img2.shape}")

        # 3. Create 3-channel blend mask
        mask = create_blend_mask(img1.shape)
        print(f"Created blend mask with shape: {mask.shape}")

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
            show_image((mask * 255).astype(np.uint8), "Blend Mask")
            show_image(img2_corrected, "Color-Corrected Image 2")

    except Exception as e:
        print(f"\nERROR: {e}")
        print("\nTroubleshooting tips:")
        print("- Verify the image files exist in images/input/")
        print("- Check file extensions aren't duplicated (e.g., sat1.jpg.jpg)")
        print("- Ensure images are valid (try opening them in another viewer)")
        print("- Confirm both images have identical dimensions")
        return

if __name__ == "__main__":
    main()
