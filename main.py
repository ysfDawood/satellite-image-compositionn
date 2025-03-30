import cv2
import numpy as np
import os
from pathlib import Path
from utils.blending import pyramid_blending
from utils.helpers import show_image

def create_blend_mask(img_shape, blend_width=100):
    """Create a single-channel uint8 mask (CV_8UC1)"""
    if isinstance(img_shape, tuple):
        height, width = img_shape[:2]
    else:
        height, width = img_shape.shape[:2]
    
    mask = np.zeros((height, width), dtype=np.uint8)
    center = width // 2
    
    for x in range(center - blend_width, center + blend_width):
        if x < center:
            val = int(255 * (x - (center - blend_width)) / blend_width)
        else:
            val = int(255 * (1 - (x - center) / blend_width))
        mask[:, x] = np.clip(val, 0, 255)
            
    return mask

def match_colors(source, target, mask, threshold=0.1):
    """Color matching with proper mask handling"""
    # Convert mask to float32 and threshold
    mask_float = (mask.astype(np.float32) / 255.0 > threshold).astype(np.float32)
    
    # Convert to 3-channel by broadcasting
    if len(mask_float.shape) == 2:
        mask_float = np.stack([mask_float]*3, axis=-1)
    
    # Color space conversion
    source_lab = cv2.cvtColor(source, cv2.COLOR_BGR2LAB)
    target_lab = cv2.cvtColor(target, cv2.COLOR_BGR2LAB)
    
    # Calculate statistics (using first channel only)
    s_mean, s_std = cv2.meanStdDev(source_lab, mask=mask_float[:,:,0])
    t_mean, t_std = cv2.meanStdDev(target_lab, mask=mask_float[:,:,0])
    
    # Color transfer
    result_lab = source_lab.copy()
    for c in range(3):
        result_lab[:,:,c] = ((source_lab[:,:,c] - s_mean[c]) * (t_std[c] / (s_std[c] + 1e-6))) + t_mean[c]
    
    return cv2.cvtColor(result_lab, cv2.COLOR_LAB2BGR)

def ensure_folders_exist():
    os.makedirs('images/input', exist_ok=True)
    os.makedirs('images/output', exist_ok=True)

def load_images(path1, path2):
    img1_path = Path(path1).absolute()
    img2_path = Path(path2).absolute()
    
    print(f"Loading:\n- {img1_path}\n- {img2_path}")
    
    if not img1_path.exists() or not img2_path.exists():
        missing = [p for p in [img1_path, img2_path] if not p.exists()]
        raise FileNotFoundError("Missing files:\n" + "\n".join(map(str, missing)))
    
    img1 = cv2.imread(str(img1_path))
    img2 = cv2.imread(str(img2_path))
    
    if img1 is None or img2 is None:
        raise ValueError("Failed to load images - may be corrupt")
    
    if img1.shape != img2.shape:
        raise ValueError(f"Size mismatch: {img1.shape} vs {img2.shape}")
    
    return img1, img2

def main():
    try:
        ensure_folders_exist()
        img1, img2 = load_images('images/input/sat1.jpg', 'images/input/sat2.jpg')
        print(f"Loaded images: {img1.shape}, {img2.shape}")

        mask = create_blend_mask(img1.shape)
        print(f"Mask created: {mask.dtype}, {mask.shape}")

        print("Applying color correction...")
        img2_corrected = match_colors(img2, img1, mask)

        print("Blending images...")
        mask_float = mask.astype(np.float32) / 255.0
        result = pyramid_blending(img1, img2_corrected, mask_float, levels=5)

        output_path = 'images/output/composite.jpg'
        cv2.imwrite(output_path, result)
        print(f"Saved result to: {Path(output_path).absolute()}")

        show_image(result, "Final Result")
        
        if input("Show steps? (y/n): ").lower() == 'y':
            show_image(img1, "Image 1")
            show_image(img2, "Image 2")
            show_image(cv2.merge([mask]*3), "Blend Mask")
            show_image(img2_corrected, "Color Adjusted")

    except Exception as e:
        print(f"\nERROR: {str(e)}")
        print("\nTroubleshooting:")
        print("1. Verify both sat1.jpg and sat2.jpg exist")
        print("2. Check images open in other software")
        print(f"3. Input contents: {os.listdir('images/input') if os.path.exists('images/input') else 'Missing input folder'}")

if __name__ == "__main__":
    main()
