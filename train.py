import os
import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models

from config import MODEL_DIR, LABELS_PATH, TFLITE_MODEL_PATH, MODEL_INPUT_SIZE


def load_dataset(data_dir: str, image_size=(224, 224)):
    data_dir = Path(data_dir)
    classes = sorted([d.name for d in data_dir.iterdir() if d.is_dir()])
    print(f"Found classes: {classes}")

    images = []
    labels = []
    for idx, cls in enumerate(classes):
        cls_dir = data_dir / cls
        for p in cls_dir.rglob('*'):
            if p.suffix.lower() not in {'.jpg', '.jpeg', '.png'}:
                continue
            try:
                img = Image.open(p).convert('RGB').resize((image_size[1], image_size[0]))
                img = np.array(img, dtype=np.float32) / 255.0
                images.append(img)
                labels.append(idx)
            except Exception as e:
                print(f"Skip {p}: {e}")
    X = np.stack(images).astype(np.float32)
    y = np.array(labels, dtype=np.int64)
    print(f"Loaded {len(X)} images")
    return X, y, classes


def build_model(num_classes: int, input_shape=(224, 224, 3)):
    base = tf.keras.applications.MobileNetV2(
        input_shape=input_shape, include_top=False, weights='imagenet')
    base.trainable = False  # freeze for initial training
    inp = layers.Input(shape=input_shape)
    x = base(inp, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    out = layers.Dense(num_classes, activation='softmax')(x)
    model = models.Model(inp, out)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def fine_tune(model, base_lr=1e-4):
    # Unfreeze top layers for fine-tuning
    base_model = model.layers[1]
    base_model.trainable = True
    # fine-tune last N layers
    for layer in base_model.layers[:-30]:
        layer.trainable = False
    model.compile(optimizer=tf.keras.optimizers.Adam(base_lr),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def representative_dataset_gen(X_train):
    for i in range(min(300, len(X_train))):
        sample = X_train[i:i+1]
        yield [sample.astype(np.float32)]


def export_tflite(model, X_train, labels):
    os.makedirs(MODEL_DIR, exist_ok=True)
    # Save labels
    with open(LABELS_PATH, 'w', encoding='utf-8') as f:
        for l in labels:
            f.write(f"{l}\n")
    # TFLite conversion with int8 full quantization if possible
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = lambda: representative_dataset_gen(X_train)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8
    tflite_model = converter.convert()
    with open(TFLITE_MODEL_PATH, 'wb') as f:
        f.write(tflite_model)
    print(f"Saved TFLite model to {TFLITE_MODEL_PATH}")


def main():
    parser = argparse.ArgumentParser(description='Train plant disease classifier (healthy/powdery/rust) and export TFLite.')
    parser.add_argument('--data', required=True, help='Path to dataset directory with subfolders healthy/, powdery/, rust/')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--fine_tune_epochs', type=int, default=5)
    args = parser.parse_args()

    X, y, classes = load_dataset(args.data, image_size=MODEL_INPUT_SIZE)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15, stratify=y, random_state=42)

    model = build_model(num_classes=len(classes), input_shape=(MODEL_INPUT_SIZE[0], MODEL_INPUT_SIZE[1], 3))
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True),
    ]

    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=args.epochs, batch_size=32, callbacks=callbacks)

    # Fine-tune
    model = fine_tune(model)
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=args.fine_tune_epochs, batch_size=16, callbacks=callbacks)

    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
    print(f"Validation accuracy: {val_acc:.4f}")

    export_tflite(model, X_train, classes)


if __name__ == '__main__':
    main()
