import cv2
import numpy as np

def pyramid_blending(img1, img2, mask, levels=5):
    """
    Blend two images using Laplacian pyramid blending.
    
    Args:
        img1: Background image (numpy array)
        img2: Foreground image (numpy array)
        mask: Blend mask (float32, 0-1 range where 1=foreground)
        levels: Number of pyramid levels (default=5)
    
    Returns:
        Blended image (uint8)
    """
    # Convert images to float32 for calculations
    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)
    mask = mask.astype(np.float32)

    # 1. Build Gaussian pyramids
    gp1 = [img1]  # Gaussian pyramid for img1
    gp2 = [img2]  # Gaussian pyramid for img2
    gpM = [mask]  # Gaussian pyramid for mask
    
    for _ in range(levels):
        # Downsample images and mask
        img1 = cv2.pyrDown(img1)
        img2 = cv2.pyrDown(img2)
        mask = cv2.pyrDown(mask)
        
        gp1.append(img1)
        gp2.append(img2)
        gpM.append(mask)

    # 2. Build Laplacian pyramids
    lp1 = [gp1[levels-1]]  # Laplacian pyramid for img1
    lp2 = [gp2[levels-1]]  # Laplacian pyramid for img2
    
    for i in range(levels-1, 0, -1):
        # Upsample and subtract to get Laplacian level
        up1 = cv2.pyrUp(gp1[i])
        up2 = cv2.pyrUp(gp2[i])
        
        # Handle size mismatches from pyramid operations
        h, w = gp1[i-1].shape[:2]
        up1 = cv2.resize(up1, (w, h))
        up2 = cv2.resize(up2, (w, h))
        
        lp1.append(cv2.subtract(gp1[i-1], up1))
        lp2.append(cv2.subtract(gp2[i-1], up2))

    # 3. Blend each pyramid level
    LS = []
    for l1, l2, gm in zip(lp1, lp2, reversed(gpM)):
        # Ensure mask matches image channels
        if len(gm.shape) == 2:
            gm = cv2.merge([gm, gm, gm])
        
        # Blend images using the mask
        blended = l1 * (1 - gm) + l2 * gm
        LS.append(blended)

    # 4. Reconstruct the final image
    ls_ = LS[0]
    for i in range(1, levels+1):
        ls_ = cv2.pyrUp(ls_)
        
        # Resize to match next level
        h, w = LS[i].shape[:2]
        ls_ = cv2.resize(ls_, (w, h))
        
        ls_ = cv2.add(ls_, LS[i])

    # Convert back to 8-bit and clip values
    return np.clip(ls_, 0, 255).astype(np.uint8)
