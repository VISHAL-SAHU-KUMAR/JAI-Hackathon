"""
TinyWorld OCR - Training Script
Trains an ultra-lightweight character classifier (<1 MB)
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image, ImageDraw, ImageFont
import random
import os

# Character set: A-Z, a-z, 0-9, and common symbols
CHARSET = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.,!?'-:; "
CHAR_TO_IDX = {char: idx for idx, char in enumerate(CHARSET)}
IDX_TO_CHAR = {idx: char for idx, char in enumerate(CHARSET)}

IMG_SIZE = 28
NUM_CLASSES = len(CHARSET)


def generate_synthetic_char(char, size=(28, 28)):
    """Generate a synthetic character image with noise and distortion"""
    # Create larger image first for better quality
    img = Image.new('L', (56, 56), color=255)
    draw = ImageDraw.Draw(img)
    
    # Use default font (works on all systems)
    try:
        # Try to use a better font if available
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 40)
    except:
        font = ImageFont.load_default()
    
    # Draw character
    draw.text((10, 5), char, fill=0, font=font)
    
    # Add random transformations
    angle = random.uniform(-15, 15)
    img = img.rotate(angle, fillcolor=255)
    
    # Resize to target size
    img = img.resize(size, Image.Resampling.LANCZOS)
    
    # Add noise
    img_array = np.array(img, dtype=np.float32)
    noise = np.random.normal(0, random.uniform(5, 15), img_array.shape)
    img_array = np.clip(img_array + noise, 0, 255)
    
    # Random brightness
    brightness = random.uniform(0.7, 1.3)
    img_array = np.clip(img_array * brightness, 0, 255)
    
    return img_array.astype(np.uint8)


def create_dataset(num_samples_per_char=1000):
    """Create synthetic training dataset"""
    print("Generating synthetic dataset...")
    
    X = []
    y = []
    
    for char in CHARSET:
        char_idx = CHAR_TO_IDX[char]
        for _ in range(num_samples_per_char):
            img = generate_synthetic_char(char, (IMG_SIZE, IMG_SIZE))
            X.append(img)
            y.append(char_idx)
    
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)
    
    # Normalize
    X = X / 255.0
    X = X.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    
    # Shuffle
    indices = np.random.permutation(len(X))
    X = X[indices]
    y = y[indices]
    
    print(f"Dataset created: {X.shape[0]} samples")
    return X, y


def create_tiny_model():
    """Create ultra-lightweight CNN model"""
    model = keras.Sequential([
        # First conv block - 8 filters
        keras.layers.Conv2D(8, 3, activation='relu', 
                           input_shape=(IMG_SIZE, IMG_SIZE, 1),
                           padding='same'),
        keras.layers.MaxPooling2D(2),
        keras.layers.Dropout(0.2),
        
        # Second conv block - 16 filters
        keras.layers.Conv2D(16, 3, activation='relu', padding='same'),
        keras.layers.MaxPooling2D(2),
        keras.layers.Dropout(0.3),
        
        # Third conv block - 32 filters
        keras.layers.Conv2D(32, 3, activation='relu', padding='same'),
        keras.layers.MaxPooling2D(2),
        
        # Dense layers
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dropout(0.4),
        keras.layers.Dense(NUM_CLASSES, activation='softmax')
    ])
    
    return model


def train_model():
    """Train the character classifier"""
    print("=" * 60)
    print("TinyWorld OCR - Training Character Classifier")
    print("=" * 60)
    
    # Create dataset
    X, y = create_dataset(num_samples_per_char=1000)
    
    # Split into train/val
    split_idx = int(0.9 * len(X))
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    print(f"\nTrain samples: {len(X_train)}")
    print(f"Val samples: {len(X_val)}")
    
    # Create model
    model = create_tiny_model()
    
    # Print model summary
    print("\nModel Architecture:")
    model.summary()
    
    # Calculate model size
    param_count = model.count_params()
    model_size_mb = (param_count * 4) / (1024 * 1024)  # 4 bytes per float32
    print(f"\nEstimated model size: {model_size_mb:.2f} MB")
    
    # Compile
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3)
    ]
    
    # Train
    print("\nTraining...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=30,
        batch_size=128,
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate
    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
    print(f"\nFinal validation accuracy: {val_acc*100:.2f}%")
    
    # Save model
    os.makedirs('models', exist_ok=True)
    model.save('models/char_classifier.h5')
    print("\nModel saved to: models/char_classifier.h5")
    
    # Save character mapping
    import json
    with open('models/char_mapping.json', 'w') as f:
        json.dump({
            'char_to_idx': CHAR_TO_IDX,
            'idx_to_char': IDX_TO_CHAR,
            'charset': CHARSET
        }, f)
    
    return model, history


if __name__ == '__main__':
    model, history = train_model()
    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)