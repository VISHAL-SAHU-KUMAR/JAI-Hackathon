#!/usr/bin/env python3
"""
Verify that the TinyWorld OCR system meets the <10 MB requirement
"""

import os
import json


def get_dir_size(path):
    """Calculate total size of directory"""
    total = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            if os.path.exists(filepath):
                total += os.path.getsize(filepath)
    return total


def get_file_size(path):
    """Get size of single file"""
    if os.path.exists(path):
        return os.path.getsize(path)
    return 0


def format_size(size_bytes):
    """Format bytes to human readable"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} TB"


def verify_size():
    """Verify total system size"""
    print("=" * 70)
    print("TinyWorld OCR - Size Verification")
    print("=" * 70)
    
    # Check for model file (either .h5 or .bin)
    model_path = 'models/char_classifier.h5'
    if not os.path.exists(model_path):
        model_path = 'models/char_classifier.bin'
    
    components = {
        'Model Weights': model_path,
        'Character Mapping': 'models/char_mapping.json',
        'Dictionary': 'dictionaries/common_words.txt',
        'Preprocessor': 'src/preprocessor.py',
        'Segmentor': 'src/segmentor.py',
        'Recognizer': 'src/recognizer.py',
        'Post-processor': 'src/postprocessor.py',
        'Inference Script': 'inference.py',
        'Training Script': 'train.py',
    }
    
    print("\nComponent Sizes:")
    print("-" * 70)
    
    total_size = 0
    for name, path in components.items():
        size = get_file_size(path)
        total_size += size
        status = "✓" if size > 0 else "✗"
        print(f"{status} {name:.<50} {format_size(size):>15}")
    
    print("-" * 70)
    print(f"{'TOTAL (Essential Components)':.<50} {format_size(total_size):>15}")
    print("=" * 70)
    
    # Check against limit
    limit_mb = 10 * 1024 * 1024  # 10 MB in bytes
    limit_percentage = (total_size / limit_mb) * 100
    
    print(f"\nSize Limit: {format_size(limit_mb)}")
    print(f"Current Size: {format_size(total_size)} ({limit_percentage:.1f}% of limit)")
    
    if total_size < limit_mb:
        print("\n✓ SUCCESS: System is under 10 MB limit!")
        remaining = limit_mb - total_size
        print(f"  Remaining budget: {format_size(remaining)}")
    else:
        print("\n✗ FAILED: System exceeds 10 MB limit!")
        excess = total_size - limit_mb
        print(f"  Excess: {format_size(excess)}")
    
    print("=" * 70)
    
    # Additional info
    print("\nOptional Components (not counted in limit):")
    print("-" * 70)
    
    optional = {
        'README': 'README.md',
        'Report': 'REPORT.md',
        'Requirements': 'requirements.txt',
        'Test Script': 'test_ocr.py',
    }
    
    optional_size = 0
    for name, path in optional.items():
        size = get_file_size(path)
        optional_size += size
        if size > 0:
            print(f"  {name:.<50} {format_size(size):>15}")
    
    print("-" * 70)
    print(f"{'Total (with optional)':.<52} {format_size(total_size + optional_size):>15}")
    print("=" * 70)
    
    # Create size report
    report = {
        'total_size_bytes': total_size,
        'total_size_mb': total_size / (1024 * 1024),
        'limit_mb': 10,
        'under_limit': total_size < limit_mb,
        'percentage_used': limit_percentage,
        'components': {
            name: get_file_size(path) 
            for name, path in components.items()
        }
    }
    
    with open('size_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    print("\nDetailed size report saved to: size_report.json")


if __name__ == '__main__':
    verify_size()
