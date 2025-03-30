import cv2
import numpy as np
import os
from pathlib import Path
from utils.helpers import show_image

def create_blend_mask(img_shape, blend_width=100):
    """Create a single-channel uint8 mask with smooth transition"""
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
    """Color correction with proper mask handling"""
    _, binary_mask = cv2.threshold(mask, int(255*threshold), 255, cv2.THRESH_BINARY)
    
    source_lab = cv2.cvtColor(source, cv2.COLOR_BGR2LAB)
    target_lab = cv2.cvtColor(target, cv2.COLOR_BGR2LAB)
    
    s_mean, s_std = cv2.meanStdDev(source_lab, mask=binary_mask)
    t_mean, t_std = cv2.meanStdDev(target_lab, mask=binary_mask)
    
    result_lab = source_lab.copy()
    for c in range(3):
        result_lab[:,:,c] = ((source_lab[:,:,c] - s_mean[c]) * (t_std[c] / (s_std[c] + 1e-6))) + t_mean[c]
    
    return cv2.cvtColor(result_lab, cv2.COLOR_LAB2BGR)

def pyramid_blending(img1, img2, mask, levels=5):
    """Fixed pyramid blending with precise size alignment"""
    # Ensure mask is 3-channel for multiplication
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR).astype(np.float32) / 255.0
    
    # Generate Gaussian pyramids
    G1 = img1.copy().astype(np.float32)
    G2 = img2.copy().astype(np.float32)
    GM = mask.copy()
    
    gp1 = [G1]
    gp2 = [G2]
    gpM = [GM]
    
    for _ in range(levels):
        G1 = cv2.pyrDown(G1)
        G2 = cv2.pyrDown(G2)
        GM = cv2.pyrDown(GM)
        gp1.append(G1)
        gp2.append(G2)
        gpM.append(GM)
    
    # Generate Laplacian pyramids
    lp1 = [gp1[-1]]
    lp2 = [gp2[-1]]
    for i in range(levels, 0, -1):
        # Calculate size explicitly to avoid mismatches
        expanded = cv2.pyrUp(gp1[i])
        h, w = gp1[i-1].shape[:2]
        expanded = cv2.resize(expanded, (w, h))
        L1 = gp1[i-1] - expanded
        
        expanded = cv2.pyrUp(gp2[i])
        expanded = cv2.resize(expanded, (w, h))
        L2 = gp2[i-1] - expanded
        
        lp1.append(L1)
        lp2.append(L2)
    
    # Blend pyramids
    LS = []
    for l1, l2, gm in zip(lp1, lp2, reversed(gpM)):
        # Resize mask to match current pyramid level
        h, w = l1.shape[:2]
        gm_resized = cv2.resize(gm, (w, h))
        ls = l1 * (1 - gm_resized) + l2 * gm_resized
        LS.append(ls)
    
    # Reconstruct
    ls_ = LS[0]
    for i in range(1, len(LS)):
        h, w = LS[i].shape[:2]
        ls_ = cv2.pyrUp(ls_)
        ls_ = cv2.resize(ls_, (w, h))
        ls_ = ls_ + LS[i]
    
    return np.clip(ls_, 0, 255).astype(np.uint8)

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
        raise ValueError("Failed to load images - check if they're valid image files")
    
    if img1.shape != img2.shape:
        print(f"Warning: Size mismatch ({img1.shape} vs {img2.shape}), resizing...")
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    
    return img1, img2

def main():
    try:
        ensure_folders_exist()
        img1, img2 = load_images('images/input/sat1.jpg', 'images/input/sat2.jpg')
        print(f"Loaded images: {img1.shape}, {img2.shape}")

        mask = create_blend_mask(img1.shape)
        print(f"Mask properties - Type: {mask.dtype}, Shape: {mask.shape}")

        print("Applying color correction...")
        img2_corrected = match_colors(img2, img1, mask)

        print("Performing pyramid blending...")
        result = pyramid_blending(img1, img2_corrected, mask, levels=5)

        output_path = 'images/output/composite.jpg'
        cv2.imwrite(output_path, result)
        print(f"Successfully saved to: {Path(output_path).absolute()}")

        show_image(result, "Final Result")
        
        if input("Show steps? (y/n): ").lower() == 'y':
            show_image(img1, "Image 1")
            show_image(img2, "Image 2")
            show_image(cv2.merge([mask]*3), "Blend Mask")
            show_image(img2_corrected, "Color Adjusted")

    except Exception as e:
        print(f"\nERROR: {str(e)}")
        print("\nTroubleshooting:")
        print("1. Verify both sat1.jpg and sat2.jpg exist in images/input/")
        print("2. Check images are valid (try opening them in another viewer)")
        print(f"3. Input folder contents: {os.listdir('images/input') if os.path.exists('images/input') else 'Missing input folder'}")
        print("4. Run verification:")
        print(f"   python -c \"import cv2; print('sat1 readable:', cv2.imread('images/input/sat1.jpg') is not None)")

if __name__ == "__main__":
    main()
