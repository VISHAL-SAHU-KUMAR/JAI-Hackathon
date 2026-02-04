#!/usr/bin/env python3
"""
TinyWorld OCR - Main Inference Script
Complete offline OCR pipeline: Image → Text

Usage:
    python inference.py <image_path> [--output output.txt] [--no-correction]
"""

import sys
import os
import time
import argparse

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from preprocessor import full_preprocess
from segmentor import segment_image
from recognizer import CharacterRecognizer
from postprocessor import PostProcessor


class TinyWorldOCR:
    """Complete OCR pipeline"""
    
    def __init__(self, model_path='models/char_classifier.h5',
                 mapping_path='models/char_mapping.json',
                 dictionary_path='dictionaries/common_words.txt'):
        """
        Initialize OCR system
        
        Args:
            model_path: Path to character classifier model
            mapping_path: Path to character mapping JSON
            dictionary_path: Path to dictionary for spell correction
        """
        print("=" * 60)
        print("TinyWorld OCR - Ultra-Lightweight Offline OCR")
        print("=" * 60)
        
        # Initialize components
        print("\n[1/3] Loading character recognizer...")
        self.recognizer = CharacterRecognizer(model_path, mapping_path)
        
        print("[2/3] Loading post-processor...")
        self.postprocessor = PostProcessor(dictionary_path)
        
        print("[3/3] System ready!")
        print("=" * 60)
    
    def process_image(self, image_path, use_spell_correction=True, verbose=True):
        """
        Process image and extract text
        
        Args:
            image_path: Path to input image
            use_spell_correction: Whether to apply spell correction
            verbose: Print progress messages
            
        Returns:
            Dictionary with results
        """
        start_time = time.time()
        
        if verbose:
            print(f"\nProcessing: {image_path}")
            print("-" * 60)
        
        # Step 1: Preprocessing
        if verbose:
            print("[Step 1/4] Preprocessing image...")
        preprocess_start = time.time()
        binary_image = full_preprocess(image_path)
        preprocess_time = time.time() - preprocess_start
        if verbose:
            print(f"  ✓ Preprocessing complete ({preprocess_time:.2f}s)")
        
        # Step 2: Segmentation
        if verbose:
            print("[Step 2/4] Segmenting characters...")
        segment_start = time.time()
        segmented_lines = segment_image(binary_image)
        segment_time = time.time() - segment_start
        
        # Count characters
        total_chars = sum(len([c for c in line if c is not None]) 
                         for line in segmented_lines)
        if verbose:
            print(f"  ✓ Found {len(segmented_lines)} lines, {total_chars} characters ({segment_time:.2f}s)")
        
        # Step 3: Recognition
        if verbose:
            print("[Step 3/4] Recognizing text...")
        recognize_start = time.time()
        
        recognized_lines = []
        raw_text = ""
        
        for line_chars in segmented_lines:
            # Batch recognize for efficiency
            chars = self.recognizer.batch_recognize(line_chars)
            line_text = ''.join(chars)
            recognized_lines.append(line_text)
            raw_text += line_text + '\n'
        
        recognize_time = time.time() - recognize_start
        if verbose:
            print(f"  ✓ Recognition complete ({recognize_time:.2f}s)")
        
        # Step 4: Post-processing
        if verbose:
            print("[Step 4/4] Post-processing...")
        postprocess_start = time.time()
        
        corrected_lines = []
        for line in recognized_lines:
            corrected = self.postprocessor.post_process(
                line, 
                use_spell_correction=use_spell_correction
            )
            corrected_lines.append(corrected)
        
        final_text = '\n'.join(corrected_lines)
        postprocess_time = time.time() - postprocess_start
        if verbose:
            print(f"  ✓ Post-processing complete ({postprocess_time:.2f}s)")
        
        # Total time
        total_time = time.time() - start_time
        
        if verbose:
            print("-" * 60)
            print(f"Total processing time: {total_time:.2f}s")
            print("=" * 60)
        
        return {
            'raw_text': raw_text.strip(),
            'corrected_text': final_text.strip(),
            'lines': corrected_lines,
            'num_lines': len(segmented_lines),
            'num_characters': total_chars,
            'timing': {
                'preprocessing': preprocess_time,
                'segmentation': segment_time,
                'recognition': recognize_time,
                'postprocessing': postprocess_time,
                'total': total_time
            }
        }
    
    def process_batch(self, image_paths, output_dir='outputs', 
                     use_spell_correction=True):
        """
        Process multiple images
        
        Args:
            image_paths: List of image paths
            output_dir: Directory to save outputs
            use_spell_correction: Whether to apply spell correction
            
        Returns:
            List of results
        """
        os.makedirs(output_dir, exist_ok=True)
        
        results = []
        for i, image_path in enumerate(image_paths):
            print(f"\n{'='*60}")
            print(f"Processing image {i+1}/{len(image_paths)}")
            print(f"{'='*60}")
            
            result = self.process_image(image_path, use_spell_correction)
            results.append(result)
            
            # Save output
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            output_path = os.path.join(output_dir, f"{base_name}_ocr.txt")
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(result['corrected_text'])
            
            print(f"\nOutput saved to: {output_path}")
        
        return results


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='TinyWorld OCR - Ultra-lightweight offline OCR'
    )
    parser.add_argument('image', help='Input image path')
    parser.add_argument('--output', '-o', help='Output text file path')
    parser.add_argument('--no-correction', action='store_true',
                       help='Disable spell correction')
    parser.add_argument('--show-raw', action='store_true',
                       help='Show raw OCR output before correction')
    parser.add_argument('--model', default='models/char_classifier.h5',
                       help='Path to model file')
    parser.add_argument('--quiet', '-q', action='store_true',
                       help='Suppress progress messages')
    
    args = parser.parse_args()
    
    # Check if image exists
    if not os.path.exists(args.image):
        print(f"Error: Image not found: {args.image}")
        sys.exit(1)
    
    # Initialize OCR
    try:
        ocr = TinyWorldOCR(model_path=args.model)
    except Exception as e:
        print(f"Error initializing OCR: {e}")
        sys.exit(1)
    
    # Process image
    use_correction = not args.no_correction
    result = ocr.process_image(args.image, use_correction, verbose=not args.quiet)
    
    # Display results
    if not args.quiet:
        print("\n" + "=" * 60)
        print("RESULTS")
        print("=" * 60)
        
        if args.show_raw:
            print("\nRaw OCR Output:")
            print("-" * 60)
            print(result['raw_text'])
            print("-" * 60)
        
        print("\nFinal Text:")
        print("-" * 60)
        print(result['corrected_text'])
        print("-" * 60)
        
        print(f"\nStatistics:")
        print(f"  Lines: {result['num_lines']}")
        print(f"  Characters: {result['num_characters']}")
        print(f"  Processing time: {result['timing']['total']:.2f}s")
    
    # Save output
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(result['corrected_text'])
        if not args.quiet:
            print(f"\nOutput saved to: {args.output}")
    else:
        # Just print to stdout if no output file specified
        if args.quiet:
            print(result['corrected_text'])
    
    return 0


if __name__ == '__main__':
    sys.exit(main())