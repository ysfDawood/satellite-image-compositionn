import cv2
import numpy as np
from utils.blending import pyramid_blending
from utils.color_adjustment import match_colors
from utils.helpers import load_images, show_image

def create_blend_mask(img_shape, blend_width=100):
    """Create a gradual blend mask for smooth transitions"""
    mask = np.zeros(img_shape[:2], dtype=np.float32)
    center = img_shape[1] // 2
    for x in range(center - blend_width, center + blend_width):
        if x < center:
            mask[:, x] = (x - (center - blend_width)) / blend_width
        else:
            mask[:, x] = 1 - (x - center) / blend_width
    return mask

def main():
    # Load satellite images (replace with your image paths)
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

    # Create blend mask (gradual transition at center)
    mask = create_blend_mask(img1)

    # Color correction (match img2 colors to img1 in overlap region)
    print("Applying color correction...")
    img2_corrected = match_colors(img2, img1, mask)

    # Pyramid blending
    print("Performing pyramid blending...")
    result = pyramid_blending(img1, img2_corrected, mask, levels=5)

    # Save and show results
    output_path = 'images/output/composite.jpg'
    cv2.imwrite(output_path, result)
    print(f"Composite image saved to: {output_path}")

    # Display results
    show_image(result, "Final Composition")

    # Optional: Show intermediate steps
    if input("Show intermediate steps? (y/n): ").lower() == 'y':
        show_image(img1, "Original Image 1")
        show_image(img2, "Original Image 2")
        show_image(cv2.merge([mask*255]*3), "Blend Mask")
        show_image(img2_corrected, "Color-Corrected Image 2")

if __name__ == "__main__":
    main()
