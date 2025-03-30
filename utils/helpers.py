import cv2
import matplotlib.pyplot as plt

def load_images(path1, path2):
    """Load and resize two images to the same dimensions."""
    img1 = cv2.imread(path1)
    img2 = cv2.imread(path2)
    
    if img1 is None or img2 is None:
        raise ValueError("Could not load one or both images")

    # Resize img2 to match img1's dimensions
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    
    return img1, img2

def show_image(img, title="Image"):
    """Display an image using matplotlib."""
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis('off')
    plt.show()
