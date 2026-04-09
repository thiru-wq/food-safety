"""
train.py  –  Food Safety AI  –  Training Script
=================================================

Dataset structure (auto-detected):
    dataset/dataset/
        fresh_bread/
        fresh_dairy/
        fresh_fruits/
        fresh_vegetables/
        spoiled_bread/
        spoiled_dairy/
        spoiled_fruits/
        spoiled_vegetables/

Categories are merged into 2 binary labels:
    fresh_*   →  class 0  (EDIBLE)
    spoiled_* →  class 1  (NOT EDIBLE)

Usage:
    python train.py
    python train.py --model resnet50 --epochs 20 --batch-size 32

Output:  models/food_model.keras
"""

import argparse
import os
import shutil
import random
from pathlib import Path

import numpy as np

# ─────────────────────────────────────────────────
BASE_DIR    = Path(__file__).parent
RAW_DIR     = BASE_DIR / "dataset" / "dataset"   # 8-category source
PREPARED_DIR= BASE_DIR / "dataset" / "prepared"  # binary train/test splits
MODEL_DIR   = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)
MODEL_PATH  = MODEL_DIR / "food_model.keras"
IMG_SIZE    = (224, 224)
TEST_SPLIT  = 0.20   # 20% held out for testing
SEED        = 42


# ────────────────────────────────────────────────────
#  Step 1: Prepare binary dataset  (fresh / rotten)
# ────────────────────────────────────────────────────
def prepare_dataset(force: bool = False):
    """
    Flatten 8 categories into binary fresh / rotten splits.
    Writes to dataset/prepared/{train,test}/{fresh,rotten}/
    """
    if PREPARED_DIR.exists() and not force:
        # Count existing files
        existing = list(PREPARED_DIR.rglob("*.*"))
        if len(existing) > 100:
            print(f"[INFO] Prepared dataset already exists ({len(existing)} files). Skipping re-preparation.")
            print(f"[INFO] Use --reprepare flag to redo this step.")
            return

    print("[INFO] Preparing binary dataset from 8 categories …")
    random.seed(SEED)

    for split in ["train", "test"]:
        for cls in ["fresh", "rotten"]:
            (PREPARED_DIR / split / cls).mkdir(parents=True, exist_ok=True)

    total_fresh, total_rotten = 0, 0

    for category_dir in sorted(RAW_DIR.iterdir()):
        if not category_dir.is_dir():
            continue
        name = category_dir.name.lower()
        if name.startswith("fresh"):
            label = "fresh"
        elif name.startswith("spoiled"):
            label = "rotten"
        else:
            print(f"[WARN] Unknown category '{name}', skipping.")
            continue

        images = [
            f for f in category_dir.iterdir()
            if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        ]
        random.shuffle(images)
        split_idx = int(len(images) * (1 - TEST_SPLIT))
        train_imgs = images[:split_idx]
        test_imgs  = images[split_idx:]

        for img in train_imgs:
            dst = PREPARED_DIR / "train" / label / f"{category_dir.name}_{img.name}"
            if not dst.exists():
                shutil.copy2(img, dst)
        for img in test_imgs:
            dst = PREPARED_DIR / "test" / label / f"{category_dir.name}_{img.name}"
            if not dst.exists():
                shutil.copy2(img, dst)

        if label == "fresh":
            total_fresh += len(images)
        else:
            total_rotten += len(images)

        print(f"  [OK] {name:25s}  ->  {label:6s}  ({len(train_imgs)} train / {len(test_imgs)} test)")

    # Summary
    train_f = len(list((PREPARED_DIR / "train" / "fresh").iterdir()))
    train_r = len(list((PREPARED_DIR / "train" / "rotten").iterdir()))
    test_f  = len(list((PREPARED_DIR / "test"  / "fresh").iterdir()))
    test_r  = len(list((PREPARED_DIR / "test"  / "rotten").iterdir()))

    print(f"\n[INFO] Prepared dataset summary:")
    print(f"  Train  -> fresh: {train_f}  |  rotten: {train_r}")
    print(f"  Test   -> fresh: {test_f}   |  rotten: {test_r}")
    print(f"  Total images: {train_f+train_r+test_f+test_r}\n")


# ────────────────────────────────────────────────────
#  Step 2: Build Model
# ────────────────────────────────────────────────────
def build_model(backbone: str):
    import tensorflow as tf
    from tensorflow.keras import layers, models
    from tensorflow.keras.applications import MobileNetV2, ResNet50

    if backbone == "mobilenetv2":
        base = MobileNetV2(input_shape=(*IMG_SIZE, 3), include_top=False, weights="imagenet")
        print("[INFO] Using MobileNetV2 backbone (ImageNet pretrained)")
    else:
        base = ResNet50(input_shape=(*IMG_SIZE, 3), include_top=False, weights="imagenet")
        print("[INFO] Using ResNet50 backbone (ImageNet pretrained)")

    base.trainable = False  # freeze for phase 1

    inputs  = tf.keras.Input(shape=(*IMG_SIZE, 3))
    x       = base(inputs, training=False)
    x       = layers.GlobalAveragePooling2D()(x)
    x       = layers.BatchNormalization()(x)
    x       = layers.Dense(256, activation="relu")(x)
    x       = layers.Dropout(0.4)(x)
    x       = layers.Dense(128, activation="relu")(x)
    x       = layers.Dropout(0.3)(x)
    # Output: [fresh_prob, rotten_prob]
    outputs = layers.Dense(2, activation="softmax")(x)

    model = models.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model, base


# ────────────────────────────────────────────────────
#  Step 3: Data Generators
# ────────────────────────────────────────────────────
def get_generators(batch_size: int):
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    train_aug = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        zoom_range=0.25,
        horizontal_flip=True,
        vertical_flip=False,
        brightness_range=[0.75, 1.25],
        fill_mode="nearest",
    )
    val_aug = ImageDataGenerator(rescale=1.0 / 255)

    CLASSES = ["fresh", "rotten"]

    train_ds = train_aug.flow_from_directory(
        str(PREPARED_DIR / "train"),
        target_size=IMG_SIZE,
        batch_size=batch_size,
        class_mode="categorical",
        classes=CLASSES,
        shuffle=True,
        seed=SEED,
    )
    test_ds = val_aug.flow_from_directory(
        str(PREPARED_DIR / "test"),
        target_size=IMG_SIZE,
        batch_size=batch_size,
        class_mode="categorical",
        classes=CLASSES,
        shuffle=False,
    )
    return train_ds, test_ds


# ────────────────────────────────────────────────────
#  Step 4: Train
# ────────────────────────────────────────────────────
def train(args):
    import tensorflow as tf
    from tensorflow.keras.callbacks import (
        EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
    )

    print(f"\n{'='*55}")
    print(f"  Food Safety AI  –  Training Pipeline")
    print(f"{'='*55}")
    print(f"  Backbone   : {args.model}")
    print(f"  Epochs     : {args.epochs}  (+ {args.epochs//2} fine-tune)")
    print(f"  Batch size : {args.batch_size}")
    print(f"  Output     : {MODEL_PATH}")
    print(f"{'='*55}\n")

    # Prepare dataset
    prepare_dataset(force=args.reprepare)

    # Data generators
    train_ds, test_ds = get_generators(args.batch_size)
    print(f"[INFO] Class indices: {train_ds.class_indices}")

    # Build model
    model, base = build_model(args.model)
    model.summary()

    callbacks_phase1 = [
        EarlyStopping(patience=5, restore_best_weights=True, monitor="val_accuracy", verbose=1),
        ReduceLROnPlateau(patience=3, factor=0.5, min_lr=1e-7, verbose=1),
        ModelCheckpoint(
            str(MODEL_PATH), save_best_only=True,
            monitor="val_accuracy", verbose=1
        ),
    ]

    # ── Phase 1: Train classification head only ──────────
    print("\n" + "-"*55)
    print("  Phase 1: Training classification head (backbone frozen)")
    print("-"*55)
    history1 = model.fit(
        train_ds,
        epochs=args.epochs,
        validation_data=test_ds,
        callbacks=callbacks_phase1,
    )
    best_phase1 = max(history1.history.get("val_accuracy", [0]))
    print(f"\n  [OK] Phase 1 best val_accuracy: {best_phase1*100:.2f}%")

    # ── Phase 2: Fine-tune top layers of backbone ─────────
    print("\n" + "-"*55)
    print("  Phase 2: Fine-tuning top 40 backbone layers")
    print("-"*55)
    base.trainable = True
    for layer in base.layers[:-40]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-5),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    callbacks_phase2 = [
        EarlyStopping(patience=5, restore_best_weights=True, monitor="val_accuracy", verbose=1),
        ReduceLROnPlateau(patience=3, factor=0.3, min_lr=1e-8, verbose=1),
        ModelCheckpoint(
            str(MODEL_PATH), save_best_only=True,
            monitor="val_accuracy", verbose=1
        ),
    ]

    history2 = model.fit(
        train_ds,
        epochs=args.epochs // 2,
        validation_data=test_ds,
        callbacks=callbacks_phase2,
    )
    best_phase2 = max(history2.history.get("val_accuracy", [0]))
    print(f"\n  [OK] Phase 2 best val_accuracy: {best_phase2*100:.2f}%")

    # ── Final Evaluation ──────────────────────────────────
    print("\n" + "-"*55)
    print("  Final Evaluation on Test Set")
    print("-"*55)
    loss, acc = model.evaluate(test_ds, verbose=1)
    print(f"\n  📊 Test Accuracy : {acc*100:.2f}%")
    print(f"  📉 Test Loss     : {loss:.4f}")

    # ── Per-class report ──────────────────────────────────
    print("\n" + "-"*55)
    print("  Generating Classification Report ...")
    print("-"*55)
    try:
        from sklearn.metrics import classification_report, confusion_matrix
        test_ds.reset()
        y_pred_probs = model.predict(test_ds, verbose=1)
        y_pred = np.argmax(y_pred_probs, axis=1)
        y_true = test_ds.classes
        print("\n" + classification_report(y_true, y_pred, target_names=["fresh", "rotten"]))
        cm = confusion_matrix(y_true, y_pred)
        print(f"  Confusion Matrix:\n{cm}\n")
    except ImportError:
        print("[INFO] Install scikit-learn for a detailed classification report.")

    print(f"\n{'='*55}")
    print(f"  [OK] Model saved -> {MODEL_PATH}")
    print(f"  Restart the FastAPI server to switch to real inference.")
    print(f"{'='*55}\n")


# ────────────────────────────────────────────────────
#  Entry Point
# ────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Food Safety AI")
    parser.add_argument("--model",      default="mobilenetv2",
                        choices=["mobilenetv2", "resnet50"],
                        help="Backbone architecture (default: mobilenetv2)")
    parser.add_argument("--epochs",     type=int, default=15,
                        help="Epochs for phase 1 (default: 15)")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size (default: 32)")
    parser.add_argument("--reprepare",  action="store_true",
                        help="Force re-copying images into prepared/ folder")
    train(parser.parse_args())
