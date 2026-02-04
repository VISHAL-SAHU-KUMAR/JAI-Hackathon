"""
TinyWorld OCR - Preprocessing Module
Classical computer vision techniques for image cleanup
NO ML models used here - pure algorithmic preprocessing
"""

import cv2
import numpy as np
from scipy import ndimage


def preprocess_image(image_path):
    """
    Apply classical CV preprocessing to clean up the image
    
    Args:
        image_path: Path to input image
        
    Returns:
        Cleaned binary image
    """
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    
    # Convert to grayscale
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    
    # Step 1: Denoise using bilateral filter (preserves edges)
    denoised = cv2.bilateralFilter(gray, 9, 75, 75)
    
    # Step 2: Enhance contrast using CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)
    
    # Step 3: Adaptive thresholding
    binary = cv2.adaptiveThreshold(
        enhanced, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=11,
        C=2
    )
    
    # Step 4: Morphological operations to remove noise
    kernel = np.ones((2, 2), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    # Step 5: Deskew if needed
    binary = deskew(binary)
    
    return binary


def deskew(image):
    """
    Detect and correct skew angle using Hough transform
    
    Args:
        image: Binary image
        
    Returns:
        Deskewed image
    """
    # Detect edges
    edges = cv2.Canny(image, 50, 150, apertureSize=3)
    
    # Detect lines using Hough transform
    lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=100)
    
    if lines is None:
        return image
    
    # Calculate dominant angle
    angles = []
    for rho, theta in lines[:, 0]:
        angle = np.degrees(theta) - 90
        if -45 < angle < 45:
            angles.append(angle)
    
    if not angles:
        return image
    
    # Use median angle
    median_angle = np.median(angles)
    
    # Only correct if skew is significant
    if abs(median_angle) > 0.5:
        # Rotate image
        (h, w) = image.shape
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
        rotated = cv2.warpAffine(
            image, M, (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        )
        return rotated
    
    return image


def remove_borders(image, border_size=10):
    """
    Remove border artifacts
    
    Args:
        image: Binary image
        border_size: Size of border to remove
        
    Returns:
        Image with borders removed
    """
    h, w = image.shape
    if h > 2 * border_size and w > 2 * border_size:
        image[:border_size, :] = 0
        image[-border_size:, :] = 0
        image[:, :border_size] = 0
        image[:, -border_size:] = 0
    return image


def normalize_size(image, target_height=800):
    """
    Normalize image size while maintaining aspect ratio
    
    Args:
        image: Input image
        target_height: Target height in pixels
        
    Returns:
        Resized image
    """
    h, w = image.shape
    if h > target_height:
        ratio = target_height / h
        new_w = int(w * ratio)
        image = cv2.resize(image, (new_w, target_height), interpolation=cv2.INTER_AREA)
    return image


def invert_if_needed(image):
    """
    Ensure text is white on black background
    
    Args:
        image: Binary image
        
    Returns:
        Correctly oriented binary image
    """
    # Count white pixels
    white_pixels = np.sum(image > 127)
    total_pixels = image.shape[0] * image.shape[1]
    
    # If more than 50% white, invert
    if white_pixels > total_pixels * 0.5:
        image = cv2.bitwise_not(image)
    
    return image


def full_preprocess(image_path, target_height=800):
    """
    Complete preprocessing pipeline
    
    Args:
        image_path: Path to input image
        target_height: Target height for normalization
        
    Returns:
        Fully preprocessed binary image
    """
    # Load and basic preprocessing
    binary = preprocess_image(image_path)
    
    # Normalize size
    binary = normalize_size(binary, target_height)
    
    # Remove borders
    binary = remove_borders(binary)
    
    # Ensure correct orientation
    binary = invert_if_needed(binary)
    
    return binary


if __name__ == '__main__':
    # Test preprocessing
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python preprocessor.py <image_path>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    processed = full_preprocess(image_path)
    
    # Save result
    output_path = 'preprocessed_output.png'
    cv2.imwrite(output_path, processed)
    print(f"Preprocessed image saved to: {output_path}")
    
    # Show size
    print(f"Output size: {processed.shape}")