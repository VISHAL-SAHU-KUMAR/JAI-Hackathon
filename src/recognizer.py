"""
TinyWorld OCR - Character Recognition Module
Uses trained tiny CNN to recognize individual characters
"""

import numpy as np
import cv2
import json
import os


class CharacterRecognizer:
    """Character recognition using tiny CNN model"""
    
    def __init__(self, model_path='models/char_classifier.h5', 
                 mapping_path='models/char_mapping.json'):
        """
        Initialize recognizer
        
        Args:
            model_path: Path to trained model
            mapping_path: Path to character mapping JSON
        """
        self.model = None
        self.idx_to_char = {}
        self.char_to_idx = {}
        self.charset = ""
        self.img_size = 28
        
        # Load model
        self._load_model(model_path, mapping_path)
    
    def _load_model(self, model_path, mapping_path):
        """Load model and character mappings"""
        try:
            import tensorflow as tf
            from tensorflow import keras
            
            # Load model
            if os.path.exists(model_path):
                self.model = keras.models.load_model(model_path)
                print(f"Model loaded from: {model_path}")
            else:
                raise FileNotFoundError(f"Model not found: {model_path}")
            
            # Load character mapping
            if os.path.exists(mapping_path):
                with open(mapping_path, 'r') as f:
                    mapping = json.load(f)
                    self.idx_to_char = {int(k): v for k, v in mapping['idx_to_char'].items()}
                    self.char_to_idx = mapping['char_to_idx']
                    self.charset = mapping['charset']
                print(f"Character mapping loaded: {len(self.charset)} characters")
            else:
                raise FileNotFoundError(f"Mapping not found: {mapping_path}")
                
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def preprocess_char_image(self, char_img):
        """
        Preprocess character image for recognition
        
        Args:
            char_img: Character image (binary)
            
        Returns:
            Preprocessed image ready for model
        """
        # Ensure grayscale
        if len(char_img.shape) == 3:
            char_img = cv2.cvtColor(char_img, cv2.COLOR_BGR2GRAY)
        
        # Resize to model input size with padding to preserve aspect ratio
        h, w = char_img.shape
        
        # Calculate padding to make it square
        size = max(h, w)
        padded = np.zeros((size, size), dtype=np.uint8)
        
        # Center the character
        y_offset = (size - h) // 2
        x_offset = (size - w) // 2
        padded[y_offset:y_offset+h, x_offset:x_offset+w] = char_img
        
        # Resize to target size
        resized = cv2.resize(padded, (self.img_size, self.img_size), 
                           interpolation=cv2.INTER_AREA)
        
        # Normalize to [0, 1]
        normalized = resized.astype(np.float32) / 255.0
        
        # Reshape for model
        normalized = normalized.reshape(1, self.img_size, self.img_size, 1)
        
        return normalized
    
    def recognize_char(self, char_img, top_k=3):
        """
        Recognize a single character
        
        Args:
            char_img: Character image
            top_k: Return top K predictions
            
        Returns:
            List of (character, confidence) tuples
        """
        if char_img is None:
            return [(' ', 1.0)]  # Space character
        
        # Preprocess
        processed = self.preprocess_char_image(char_img)
        
        # Predict
        predictions = self.model.predict(processed, verbose=0)[0]
        
        # Get top K predictions
        top_indices = np.argsort(predictions)[-top_k:][::-1]
        results = []
        
        for idx in top_indices:
            char = self.idx_to_char[idx]
            confidence = float(predictions[idx])
            results.append((char, confidence))
        
        return results
    
    def recognize_text(self, char_images, use_confidence_threshold=True, 
                      confidence_threshold=0.5):
        """
        Recognize text from list of character images
        
        Args:
            char_images: List of character images
            use_confidence_threshold: Whether to use confidence filtering
            confidence_threshold: Minimum confidence to accept prediction
            
        Returns:
            Recognized text string
        """
        text = ""
        confidences = []
        
        for char_img in char_images:
            predictions = self.recognize_char(char_img, top_k=1)
            char, confidence = predictions[0]
            
            if use_confidence_threshold and confidence < confidence_threshold:
                char = '?'  # Unknown character
            
            text += char
            confidences.append(confidence)
        
        # Calculate average confidence
        avg_confidence = np.mean(confidences) if confidences else 0.0
        
        return text, avg_confidence
    
    def recognize_lines(self, segmented_lines):
        """
        Recognize text from segmented lines
        
        Args:
            segmented_lines: List of lists of character images
            
        Returns:
            List of recognized text lines
        """
        recognized_lines = []
        
        for line_chars in segmented_lines:
            text, confidence = self.recognize_text(line_chars)
            recognized_lines.append({
                'text': text,
                'confidence': confidence
            })
        
        return recognized_lines
    
    def batch_recognize(self, char_images):
        """
        Batch recognition for efficiency
        
        Args:
            char_images: List of character images
            
        Returns:
            List of characters
        """
        if not char_images:
            return []
        
        # Preprocess all images
        processed_batch = []
        for char_img in char_images:
            if char_img is None:
                processed_batch.append(None)
            else:
                processed = self.preprocess_char_image(char_img)
                processed_batch.append(processed)
        
        # Separate None (spaces) from actual images
        valid_indices = [i for i, img in enumerate(processed_batch) if img is not None]
        valid_images = [processed_batch[i] for i in valid_indices]
        
        results = [' '] * len(char_images)  # Default to space
        
        if valid_images:
            # Stack into batch
            batch = np.vstack(valid_images)
            
            # Predict batch
            predictions = self.model.predict(batch, verbose=0)
            
            # Decode predictions
            for idx, pred in zip(valid_indices, predictions):
                char_idx = np.argmax(pred)
                results[idx] = self.idx_to_char[char_idx]
        
        return results


# Utility functions for testing
def test_recognizer(model_path='models/char_classifier.h5'):
    """Test character recognizer"""
    from PIL import Image, ImageDraw, ImageFont
    
    # Create recognizer
    recognizer = CharacterRecognizer(model_path)
    
    # Test with synthetic character
    test_char = 'A'
    img = Image.new('L', (28, 28), color=255)
    draw = ImageDraw.Draw(img)
    
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20)
    except:
        font = ImageFont.load_default()
    
    draw.text((5, 2), test_char, fill=0, font=font)
    img_array = np.array(img)
    
    # Recognize
    predictions = recognizer.recognize_char(img_array, top_k=5)
    
    print(f"\nTesting recognition for character '{test_char}':")
    print("\nTop 5 predictions:")
    for char, conf in predictions:
        print(f"  {char}: {conf*100:.2f}%")
    
    return recognizer


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    else:
        model_path = 'models/char_classifier.h5'
    
    test_recognizer(model_path)