# Satellite Image Composition

Python implementation for compositing satellite images with feathering and color adjustment.

## Features
- Multi-band pyramid blending
- Color correction between images
- Smooth transition between overlapping regions

## Requirements
Python 3.8+ with packages listed in `requirements.txt`

## Usage
1. Place satellite images in `images/input/`
2. Run `main.py`
3. Results will be saved in `images/output/`

## Techniques Used
- Laplacian pyramid blending
- LAB color space adjustment
- Mask-based region blending
