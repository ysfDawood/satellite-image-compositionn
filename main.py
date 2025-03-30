import cv2
import numpy as np
import os
from pathlib import Path
from utils.blending import pyramid_blending
from utils.color_adjustment import match_colors
from utils.helpers import show_image

def create_blend_mask(img_shape, blend_width=100):
    """Create a single-channel uint8 mask (CV_8UC1) for OpenCV compatibility"""
    if isinstance(img_shape, tuple):
        height, width = img_shape[:2]
    else:
        height, width = img_shape.shape[:2]
    
    mask = np.zeros((height, width), dtype=np.uint8)  # Critical: uint8 single channel
    center = width // 2
    
    for x in range(center - blend_width, center + blend_width):
        if x < center:
            val = int(255 * (x - (center - blend_width)) / blend_width)
        else:
            val = int(255 * (1 - (x - center) / blend_width))
        mask[:, x] = np.clip(val, 0, 255)
            
    return mask

def ensure_folders_exist():
    """Create required folder structure"""
    os.makedirs('images/input', exist_ok=True)
    os.makedirs('images/output', exist_ok=True)

def load_images(path1, path2):
    """Robust image loading with detailed error reporting"""
    img1_path = Path(path1).absolute()
    img2_path = Path(path2).absolute()
    
    print(f"Image paths:\n- {img1_path}\n- {img2_path}")
    
    # Verify files exist
    if not img1_path.exists() or not img2_path.exists():
        missing = [p for p in [img1_path, img2_path] if not p.exists()]
        raise FileNotFoundError(f"Missing files:\n" + "\n".join(map(str, missing)))
    
    # Load images
    img1 = cv2.imread(str(img1_path))
    img2 = cv2.imread(str(img2_path))
    
    # Verify successful load
    if img1 is None or img2 is None:
        raise ValueError(
            "Images failed to load. Possible causes:\n"
            "- Corrupt file\n"
            "- Invalid format\n"
            "- Permission issues"
        )
    
    # Verify dimensions match
    if img1.shape != img2.shape:
        raise ValueError(
            f"Image size mismatch:\n"
            f"Image 1: {img1.shape}\n"
            f"Image 2: {img2.shape}"
        )
    
    return img1, img2

def main():
    try:
        # 1. Initialize folders
        ensure_folders_exist()
        
        # 2. Load and validate images
        img1, img2 = load_images('images/input/sat1.jpg', 'images/input/sat2.jpg')
        print(f"Loaded images with shapes: {img1.shape} and {img2.shape}")

        # 3. Create proper blend mask (uint8)
        mask = create_blend_mask(img1.shape)
        print(f"Mask properties - Type: {mask.dtype}, Shape: {mask.shape}, Range: [{np.min(mask)}-{np.max(mask)}]")

        # 4. Color correction (uses uint8 mask directly)
        print("Applying color correction...")
        img2_corrected = match_colors(img2, img1, mask)

        # 5. Pyramid blending (convert to float32 0-1 range)
        print("Performing pyramid blending...")
        mask_float = mask.astype(np.float32) / 255.0
        result = pyramid_blending(img1, img2_corrected, mask_float, levels=5)

        # 6. Save results
        output_path = 'images/output/composite.jpg'
        cv2.imwrite(output_path, result)
        print(f"Successfully saved to: {Path(output_path).absolute()}")

        # 7. Display results
        show_image(result, "Final Composition")

        # debug views
        if input("Show intermediate steps? (y/n): ").lower() == 'y':
            show_image(img1, "Original Image 1")
            show_image(img2, "Original Image 2")
            show_image(cv2.merge([mask]*3), "Blend Mask (uint8)")
            show_image((mask_float * 255).astype(np.uint8), "Blend Mask (float32 scaled)")
            show_image(img2_corrected, "Color-Corrected Image 2")

    except Exception as e:
        print(f"\nERROR: {str(e)}")
        print("\nTroubleshooting:")
        print("1. Verify both sat1.jpg and sat2.jpg exist in images/input/")
        print("2. Check images open normally in other software")
        print(f"3. Current input contents: {os.listdir('images/input') if os.path.exists('images/input') else 'input folder missing'}")
        print("4. Run verification command:")
        print(f"   python -c \"import cv2; print('sat1 readable:', cv2.imread('images/input/sat1.jpg') is not None)\"")

if __name__ == "__main__":
    main()
