import cv2
import numpy as np
import os
from pathlib import Path
from typing import List

def create_blend_mask(img_shape: tuple, blend_width: int = 100, direction: str = 'horizontal') -> np.ndarray:
    """
    Create a smooth gradient mask for blending.
    Args:
        img_shape: (height, width) or image array
        blend_width: Width of the transition zone (in pixels)
        direction: 'horizontal' (left-right) or 'vertical' (top-bottom)
    """
    if isinstance(img_shape, tuple):
        height, width = img_shape[:2]
    else:
        height, width = img_shape.shape[:2]
    
    mask = np.zeros((height, width), dtype=np.uint8)
    
    if direction == 'horizontal':
        center = width // 2
        for x in range(center - blend_width, center + blend_width):
            if x < center:
                val = int(255 * (x - (center - blend_width)) / blend_width)
            else:
                val = int(255 * (1 - (x - center) / blend_width))
            mask[:, x] = np.clip(val, 0, 255)
    else:  # vertical
        center = height // 2
        for y in range(center - blend_width, center + blend_width):
            if y < center:
                val = int(255 * (y - (center - blend_width)) / blend_width)
            else:
                val = int(255 * (1 - (y - center) / blend_width))
            mask[y, :] = np.clip(val, 0, 255)
    
    return mask

def match_colors(source: np.ndarray, target: np.ndarray, mask: np.ndarray, threshold: float = 0.1) -> np.ndarray:
    """Match colors of source to target using mask-defined regions."""
    _, binary_mask = cv2.threshold(mask, int(255*threshold), 255, cv2.THRESH_BINARY)
    
    source_lab = cv2.cvtColor(source, cv2.COLOR_BGR2LAB)
    target_lab = cv2.cvtColor(target, cv2.COLOR_BGR2LAB)
    
    s_mean, s_std = cv2.meanStdDev(source_lab, mask=binary_mask)
    t_mean, t_std = cv2.meanStdDev(target_lab, mask=binary_mask)
    
    result_lab = source_lab.copy()
    for c in range(3):
        result_lab[:,:,c] = ((source_lab[:,:,c] - s_mean[c]) * (t_std[c] / (s_std[c] + 1e-6))) + t_mean[c]
    
    return cv2.cvtColor(result_lab, cv2.COLOR_LAB2BGR)

def pyramid_blending(img1: np.ndarray, img2: np.ndarray, mask: np.ndarray, levels: int = 5) -> np.ndarray:
    """Blend two images using Laplacian pyramids with exact size alignment."""
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR).astype(np.float32) / 255.0
    
    # Build Gaussian pyramids
    gp1 = [img1.astype(np.float32)]
    gp2 = [img2.astype(np.float32)]
    gpM = [mask.copy()]
    
    for _ in range(levels):
        gp1.append(cv2.pyrDown(gp1[-1]))
        gp2.append(cv2.pyrDown(gp2[-1]))
        gpM.append(cv2.pyrDown(gpM[-1]))
    
    # Build Laplacian pyramids
    lp1 = [gp1[-1]]
    lp2 = [gp2[-1]]
    for i in range(levels, 0, -1):
        size = (gp1[i-1].shape[1], gp1[i-1].shape[0])
        lp1.append(gp1[i-1] - cv2.resize(cv2.pyrUp(gp1[i]), size))
        lp2.append(gp2[i-1] - cv2.resize(cv2.pyrUp(gp2[i]), size))
    
    # Blend pyramids
    LS = []
    for l1, l2, gm in zip(lp1, lp2, reversed(gpM)):
        gm_resized = cv2.resize(gm, (l1.shape[1], l1.shape[0]))
        LS.append(l1 * (1 - gm_resized) + l2 * gm_resized)
    
    # Reconstruct
    ls_ = LS[0]
    for i in range(1, levels+1):
        ls_ = cv2.resize(cv2.pyrUp(ls_), (LS[i].shape[1], LS[i].shape[0]))
        ls_ += LS[i]
    
    return np.clip(ls_, 0, 255).astype(np.uint8)

def load_images(image_paths: List[str]) -> List[np.ndarray]:
    """Load and validate N images, auto-resizing to match the first image's dimensions."""
    if len(image_paths) < 2:
        raise ValueError("At least 2 images required")
    
    images = []
    base_shape = None
    
    for path in image_paths:
        img = cv2.imread(str(Path(path)))
        if img is None:
            raise FileNotFoundError(f"Failed to load {path}")
        
        if base_shape is None:
            base_shape = img.shape[:2]
            images.append(img)
        else:
            if img.shape[:2] != base_shape:
                img = cv2.resize(img, (base_shape[1], base_shape[0]))
            images.append(img)
    
    return images

def blend_images_sequential(images: List[np.ndarray], blend_width: int = 100, direction: str = 'horizontal') -> np.ndarray:
    """Blend N images sequentially with color correction."""
    composite = images[0].copy()
    
    for i in range(1, len(images)):
        mask = create_blend_mask(composite.shape, blend_width, direction)
        corrected_img = match_colors(images[i], composite, mask)
        composite = pyramid_blending(composite, corrected_img, mask)
    
    return composite

def main():
    try:
        # Configuration
        input_dir = 'images/input'
        output_dir = 'images/output'
        os.makedirs(input_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)
        
        # Get all images in input directory (sorted alphabetically)
        image_paths = sorted([os.path.join(input_dir, f) for f in os.listdir(input_dir) 
                            if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        
        if len(image_paths) < 2:
            raise ValueError(f"Need at least 2 images in {input_dir}")
        
        print(f"Found {len(image_paths)} images to blend:")
        for path in image_paths:
            print(f"- {Path(path).name}")
        
        # Load and validate
        images = load_images(image_paths)
        print(f"\nLoaded images with shape: {images[0].shape}")
        
        # Blend all images
        print("\nBlending images sequentially...")
        final_composite = blend_images_sequential(images, blend_width=150, direction='horizontal')
        
        # Save result
        output_path = os.path.join(output_dir, 'composite.jpg')
        cv2.imwrite(output_path, final_composite)
        print(f"\nSuccess! Saved to: {Path(output_path).absolute()}")
        
        # Preview
        if input("\nShow result? (y/n): ").lower() == 'y':
            cv2.imshow('Composite', final_composite)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    
    except Exception as e:
        print(f"\nERROR: {e}")
        print("\nTroubleshooting:")
        print(f"1. Ensure {input_dir} contains 2+ images (JPEG/PNG)")
        print(f"2. Current contents: {os.listdir(input_dir) if os.path.exists(input_dir) else 'Directory missing'}")

if __name__ == "__main__":
    main()
