import cv2
import numpy as np

def match_colors(source, target, mask, threshold=0.1):
    """Adjust colors of source image to match target in masked region."""
    mask = mask > threshold
    if len(mask.shape) == 2:
        mask = cv2.merge([mask, mask, mask])

    # Convert to LAB color space (better for color adjustments)
    source_lab = cv2.cvtColor(source, cv2.COLOR_BGR2LAB)
    target_lab = cv2.cvtColor(target, cv2.COLOR_BGR2LAB)

    # Calculate mean and std for each channel
    s_mean, s_std = cv2.meanStdDev(source_lab, mask=mask.astype('uint8'))
    t_mean, t_std = cv2.meanStdDev(target_lab, mask=mask.astype('uint8'))

    # Color transfer
    result_lab = source_lab.copy()
    for c in range(3):
        result_lab[:,:,c] = ((source_lab[:,:,c] - s_mean[c]) * (t_std[c] / (s_std[c] + 1e-6))) + t_mean[c]

    # Convert back to BGR
    result = cv2.cvtColor(result_lab, cv2.COLOR_LAB2BGR)
    return np.clip(result, 0, 255).astype('uint8')
