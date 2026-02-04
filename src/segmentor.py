"""
TinyWorld OCR - Character Segmentation Module
Uses connected component analysis to segment characters
NO ML models - pure algorithmic approach
"""

import cv2
import numpy as np
from collections import namedtuple

BoundingBox = namedtuple('BoundingBox', ['x', 'y', 'w', 'h', 'img'])


def segment_lines(binary_image):
    """
    Segment image into text lines using horizontal projection
    
    Args:
        binary_image: Preprocessed binary image
        
    Returns:
        List of line images
    """
    # Horizontal projection
    h_projection = np.sum(binary_image, axis=1)
    
    # Find line boundaries
    threshold = np.max(h_projection) * 0.1
    in_line = False
    line_starts = []
    line_ends = []
    
    for i, val in enumerate(h_projection):
        if not in_line and val > threshold:
            line_starts.append(i)
            in_line = True
        elif in_line and val <= threshold:
            line_ends.append(i)
            in_line = False
    
    # Handle case where line continues to end
    if in_line:
        line_ends.append(len(h_projection))
    
    # Extract line images
    lines = []
    for start, end in zip(line_starts, line_ends):
        # Add padding
        start = max(0, start - 5)
        end = min(binary_image.shape[0], end + 5)
        line_img = binary_image[start:end, :]
        if line_img.shape[0] > 10:  # Minimum height
            lines.append(line_img)
    
    return lines


def segment_words(line_image, min_gap=15):
    """
    Segment line into words using vertical projection
    
    Args:
        line_image: Binary image of a text line
        min_gap: Minimum gap between words
        
    Returns:
        List of word images
    """
    # Vertical projection
    v_projection = np.sum(line_image, axis=0)
    
    # Find gaps
    threshold = np.max(v_projection) * 0.05
    gaps = v_projection < threshold
    
    # Find word boundaries
    words = []
    in_word = False
    word_start = 0
    gap_count = 0
    
    for i, is_gap in enumerate(gaps):
        if not in_word and not is_gap:
            word_start = i
            in_word = True
            gap_count = 0
        elif in_word:
            if is_gap:
                gap_count += 1
                if gap_count >= min_gap:
                    # End of word
                    word_img = line_image[:, word_start:i-gap_count]
                    if word_img.shape[1] > 5:  # Minimum width
                        words.append(word_img)
                    in_word = False
                    gap_count = 0
            else:
                gap_count = 0
    
    # Handle last word
    if in_word:
        word_img = line_image[:, word_start:]
        if word_img.shape[1] > 5:
            words.append(word_img)
    
    return words


def segment_characters(word_image):
    """
    Segment word into characters using connected components
    
    Args:
        word_image: Binary image of a word
        
    Returns:
        List of BoundingBox objects for each character
    """
    # Find connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        word_image, connectivity=8
    )
    
    # Extract character bounding boxes
    characters = []
    
    for i in range(1, num_labels):  # Skip background (label 0)
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]
        
        # Filter out noise (too small components)
        if w < 3 or h < 8 or area < 20:
            continue
        
        # Filter out very large components (probably merged characters)
        if w > word_image.shape[1] * 0.8 or h > word_image.shape[0] * 0.9:
            continue
        
        # Extract character image
        char_img = word_image[y:y+h, x:x+w]
        
        characters.append(BoundingBox(x, y, w, h, char_img))
    
    # Sort by x position (left to right)
    characters.sort(key=lambda box: box.x)
    
    return characters


def merge_close_components(characters, max_gap=3):
    """
    Merge components that are very close (likely parts of same character)
    
    Args:
        characters: List of BoundingBox objects
        max_gap: Maximum gap to merge
        
    Returns:
        Merged list of BoundingBox objects
    """
    if len(characters) < 2:
        return characters
    
    merged = []
    current = characters[0]
    
    for next_char in characters[1:]:
        gap = next_char.x - (current.x + current.w)
        
        if gap <= max_gap:
            # Merge
            new_x = current.x
            new_y = min(current.y, next_char.y)
            new_w = (next_char.x + next_char.w) - current.x
            new_h = max(current.y + current.h, next_char.y + next_char.h) - new_y
            
            # Create merged image
            merged_img = np.zeros((new_h, new_w), dtype=np.uint8)
            merged_img[current.y-new_y:current.y-new_y+current.h,
                      current.x-new_x:current.x-new_x+current.w] = current.img
            merged_img[next_char.y-new_y:next_char.y-new_y+next_char.h,
                      next_char.x-new_x:next_char.x-new_x+next_char.w] = next_char.img
            
            current = BoundingBox(new_x, new_y, new_w, new_h, merged_img)
        else:
            merged.append(current)
            current = next_char
    
    merged.append(current)
    return merged


def segment_image(binary_image):
    """
    Complete segmentation pipeline: image → lines → words → characters
    
    Args:
        binary_image: Preprocessed binary image
        
    Returns:
        List of lists of character images (organized by lines and words)
    """
    all_chars = []
    
    # Segment into lines
    lines = segment_lines(binary_image)
    
    for line_img in lines:
        line_chars = []
        
        # Segment line into words
        words = segment_words(line_img)
        
        for word_img in words:
            # Segment word into characters
            chars = segment_characters(word_img)
            
            # Merge close components
            chars = merge_close_components(chars)
            
            # Extract just the images
            char_images = [char.img for char in chars]
            line_chars.extend(char_images)
            
            # Add space after word
            line_chars.append(None)  # None represents space
        
        all_chars.append(line_chars)
    
    return all_chars


def visualize_segmentation(binary_image, output_path='segmentation_debug.png'):
    """
    Visualize segmentation for debugging
    
    Args:
        binary_image: Preprocessed binary image
        output_path: Path to save visualization
    """
    # Create color image for visualization
    vis_img = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2BGR)
    
    lines = segment_lines(binary_image)
    
    for line_img in lines:
        words = segment_words(line_img)
        
        for word_img in words:
            chars = segment_characters(word_img)
            
            # Draw bounding boxes
            for char in chars:
                # Random color for each character
                color = tuple(np.random.randint(0, 255, 3).tolist())
                cv2.rectangle(vis_img, (char.x, char.y),
                            (char.x + char.w, char.y + char.h),
                            color, 2)
    
    cv2.imwrite(output_path, vis_img)
    print(f"Segmentation visualization saved to: {output_path}")


if __name__ == '__main__':
    # Test segmentation
    import sys
    from preprocessor import full_preprocess
    
    if len(sys.argv) < 2:
        print("Usage: python segmentor.py <image_path>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    # Preprocess
    binary = full_preprocess(image_path)
    
    # Segment
    segments = segment_image(binary)
    
    print(f"Found {len(segments)} lines")
    for i, line in enumerate(segments):
        char_count = sum(1 for c in line if c is not None)
        print(f"  Line {i+1}: {char_count} characters")
    
    # Visualize
    visualize_segmentation(binary)