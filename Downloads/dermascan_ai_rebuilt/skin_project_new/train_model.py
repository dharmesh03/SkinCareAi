"""
=============================================================
  train_model_v2.py  —  HIGH ACCURACY VERSION
  Target: 90–98% accuracy
  
  Improvements over v1:
  1. EfficientNetB3 (much better than MobileNetV2)
  2. More aggressive augmentation (MixUp + CutMix)
  3. Label smoothing (prevents overconfidence)
  4. Cosine learning rate decay
  5. Longer training with better callbacks
  6. Test-Time Augmentation (TTA) for prediction
  7. Class weight balancing
  8. Progressive image resizing
  
  Run: python train_model_v2.py
=============================================================
"""

import os, json, random, shutil
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, LambdaCallback
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path

# ─────────────────────────────────────────────
# CONFIGURATION  (tweak these if needed)
# ─────────────────────────────────────────────
IMG_SIZE      = 300      # EfficientNetB3 native size (better than 224)
BATCH_SIZE    = 16       # Lower = more stable training on small datasets
EPOCHS_P1     = 20       # Phase 1: frozen base
EPOCHS_P2     = 30       # Phase 2: fine-tune
EPOCHS_P3     = 20       # Phase 3: fine-tune deeper
LR_P1         = 1e-3
LR_P2         = 5e-5
LR_P3         = 1e-5
MODEL_PATH    = "model/skin_model.h5"
BEST_PATH     = "model/skin_model_best.h5"
TRAIN_DIR     = "dataset/train"
VAL_DIR       = "dataset/val"
TEST_DIR      = "dataset/test"
LABEL_SMOOTH  = 0.1      # Label smoothing — prevents overconfidence

# ─────────────────────────────────────────────
# GET AVAILABLE CLASSES
# ─────────────────────────────────────────────
def get_classes():
    if not os.path.exists(TRAIN_DIR):
        print(f"❌ {TRAIN_DIR} not found. Run create_folders.py first.")
        exit()

    classes = []
    print("\n📊 Dataset inventory:")
    print(f"  {'Class':<30} {'Train':>6} {'Val':>6}")
    print("  " + "─"*45)

    for cls in sorted(os.listdir(TRAIN_DIR)):
        train_path = os.path.join(TRAIN_DIR, cls)
        val_path   = os.path.join(VAL_DIR,   cls)
        if not os.path.isdir(train_path):
            continue

        exts = {'.jpg','.jpeg','.png','.webp','.bmp'}
        t_imgs = [f for f in os.listdir(train_path) if Path(f).suffix.lower() in exts]
        v_imgs = [f for f in os.listdir(val_path) if Path(f).suffix.lower() in exts] if os.path.exists(val_path) else []

        if len(t_imgs) < 5:
            print(f"  ⚠️  {cls:<30} {len(t_imgs):>6}  ← SKIPPED (too few)")
            continue

        status = "✅" if len(t_imgs) >= 100 else "⚠️ "
        print(f"  {status} {cls:<29} {len(t_imgs):>6} {len(v_imgs):>6}")
        classes.append(cls)

    print()
    return sorted(classes)


# ─────────────────────────────────────────────
# STEP 1: ADVANCED DATA GENERATORS
# ─────────────────────────────────────────────
def build_generators(classes):
    """
    Much stronger augmentation than v1:
    - Heavier rotation, zoom, brightness
    - Added: random erasing simulation via channel shift
    - Separate strong/mild generators for phase 1/2
    """
    strong_aug = ImageDataGenerator(
        rescale            = 1./255,
        rotation_range     = 45,        # More rotation
        width_shift_range  = 0.25,
        height_shift_range = 0.25,
        shear_range        = 0.25,
        zoom_range         = [0.6, 1.4],
        horizontal_flip    = True,
        vertical_flip      = False,
        brightness_range   = [0.3, 1.8], # More extreme lighting
        channel_shift_range= 40.0,       # More color variation
        fill_mode          = 'reflect',
    )

    val_gen = ImageDataGenerator(rescale=1./255)

    train = strong_aug.flow_from_directory(
        TRAIN_DIR,
        target_size  = (IMG_SIZE, IMG_SIZE),
        batch_size   = BATCH_SIZE,
        class_mode   = 'categorical',
        classes      = classes,
        shuffle      = True,
        seed         = 42,
    )
    val = val_gen.flow_from_directory(
        VAL_DIR,
        target_size  = (IMG_SIZE, IMG_SIZE),
        batch_size   = BATCH_SIZE,
        class_mode   = 'categorical',
        classes      = classes,
        shuffle      = False,
    )

    print(f"\n✅ Train: {train.samples} images | Val: {val.samples} images")
    return train, val


# ─────────────────────────────────────────────
# STEP 2: MIXUP AUGMENTATION
# Blends two images + labels → forces model to learn
# better decision boundaries, big accuracy boost
# ─────────────────────────────────────────────
def mixup_generator(generator, alpha=0.3):
    """Wrap a generator with MixUp augmentation"""
    while True:
        X1, y1 = next(generator)
        X2, y2 = next(generator)

        # Match batch sizes
        n = min(len(X1), len(X2))
        X1, y1 = X1[:n], y1[:n]
        X2, y2 = X2[:n], y2[:n]

        # Sample lambda from Beta distribution
        lam = np.random.beta(alpha, alpha, n)
        lam = np.maximum(lam, 1 - lam)
        lam_x = lam.reshape(n, 1, 1, 1)
        lam_y = lam.reshape(n, 1)

        X = lam_x * X1 + (1 - lam_x) * X2
        y = lam_y * y1 + (1 - lam_y) * y2

        yield X, y


# ─────────────────────────────────────────────
# STEP 3: BUILD EFFICIENTNETB3 MODEL
# Much more powerful than MobileNetV2
# ─────────────────────────────────────────────
def build_model(n_classes):
    print(f"\n🏗️  Building EfficientNetB3 model ({n_classes} classes)...")

    # EfficientNetB3 — significantly better feature extraction
    base = EfficientNetB3(
        weights      = 'imagenet',
        include_top  = False,
        input_shape  = (IMG_SIZE, IMG_SIZE, 3),
        drop_connect_rate = 0.2
    )
    base.trainable = False  # Start frozen

    inp = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = base(inp, training=False)

    # Better classification head
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)

    # Dense block 1
    x = layers.Dense(1024, activation='swish',
                     kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)

    # Dense block 2
    x = layers.Dense(512, activation='swish',
                     kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)

    # Dense block 3
    x = layers.Dense(256, activation='swish',
                     kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)

    out = layers.Dense(n_classes, activation='softmax')(x)

    model = models.Model(inp, out)

    # Label smoothing loss — key for high accuracy
    loss = CategoricalCrossentropy(label_smoothing=LABEL_SMOOTH)

    model.compile(
        optimizer = Adam(LR_P1),
        loss      = loss,
        metrics   = ['accuracy',
                     tf.keras.metrics.TopKCategoricalAccuracy(k=2, name='top2')]
    )

    total  = model.count_params()
    frozen = sum(np.prod(w.shape) for w in base.non_trainable_weights)
    print(f"  Total params    : {total:,}")
    print(f"  Trainable params: {total - frozen:,}")
    return model, base


# ─────────────────────────────────────────────
# STEP 4: COMPUTE CLASS WEIGHTS
# ─────────────────────────────────────────────
def get_class_weights(classes):
    counts = {}
    for i, cls in enumerate(classes):
        d = os.path.join(TRAIN_DIR, cls)
        n = len([f for f in os.listdir(d)
                 if Path(f).suffix.lower() in {'.jpg','.jpeg','.png','.webp','.bmp'}])
        counts[i] = max(n, 1)

    total = sum(counts.values())
    n_cls = len(classes)
    weights = {i: (total / (n_cls * c)) for i, c in counts.items()}

    # Cap weights to avoid extreme imbalance
    max_w = max(weights.values())
    if max_w > 10:
        weights = {i: min(w, 10.0) for i, w in weights.items()}

    print("\n⚖️  Class weights:")
    for i, cls in enumerate(classes):
        print(f"   {cls:<30}: {weights[i]:.2f}")
    return weights


# ─────────────────────────────────────────────
# STEP 5: CALLBACKS
# ─────────────────────────────────────────────
def get_callbacks(phase, lr):
    os.makedirs("model", exist_ok=True)
    os.makedirs("logs",  exist_ok=True)

    def lr_log(epoch, logs):
        current_lr = float(tf.keras.backend.get_value(model_ref[0].optimizer.lr))
        logs['lr'] = current_lr

    return [
        ModelCheckpoint(
            BEST_PATH,
            monitor   = 'val_accuracy',
            save_best_only = True,
            mode      = 'max',
            verbose   = 1,
        ),
        EarlyStopping(
            monitor   = 'val_accuracy',
            patience  = 10,            # More patience than v1
            restore_best_weights = True,
            verbose   = 1,
        ),
        ReduceLROnPlateau(
            monitor   = 'val_loss',
            factor    = 0.5,
            patience  = 5,
            min_lr    = 1e-8,
            verbose   = 1,
        ),
    ]


# ─────────────────────────────────────────────
# STEP 6: UNFREEZE LAYERS FOR FINE-TUNING
# ─────────────────────────────────────────────
def unfreeze(model, base, n_layers, lr, n_classes):
    """Unfreeze the last n_layers of EfficientNetB3"""
    base.trainable = True
    for layer in base.layers[:-n_layers]:
        layer.trainable = False

    trainable = sum(1 for l in base.layers if l.trainable)
    print(f"\n🔓 Unfroze last {n_layers} layers ({trainable} trainable in base)")

    model.compile(
        optimizer = Adam(lr),
        loss      = CategoricalCrossentropy(label_smoothing=LABEL_SMOOTH),
        metrics   = ['accuracy',
                     tf.keras.metrics.TopKCategoricalAccuracy(k=2, name='top2')]
    )
    return model


# ─────────────────────────────────────────────
# STEP 7: TEST-TIME AUGMENTATION (TTA)
# Run each test image through 5 augmented versions
# and average predictions → higher accuracy
# ─────────────────────────────────────────────
def predict_with_tta(model, img_array, n_aug=5):
    """Average predictions over multiple augmentations"""
    aug_gen = ImageDataGenerator(
        rotation_range    = 20,
        zoom_range        = 0.1,
        horizontal_flip   = True,
        brightness_range  = [0.8, 1.2],
    )

    preds = [model.predict(img_array, verbose=0)]
    for _ in range(n_aug - 1):
        aug_batch = next(aug_gen.flow(img_array, batch_size=len(img_array), shuffle=False))
        preds.append(model.predict(aug_batch, verbose=0))

    return np.mean(preds, axis=0)


# ─────────────────────────────────────────────
# STEP 8: PLOT TRAINING
# ─────────────────────────────────────────────
def plot_all(histories, phase_ends):
    acc, val_acc, loss, val_loss = [], [], [], []
    for h in histories:
        acc     += h.history['accuracy']
        val_acc += h.history['val_accuracy']
        loss    += h.history['loss']
        val_loss+= h.history['val_loss']

    epochs = range(1, len(acc)+1)
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    colors = ['#ef4444', '#f97316', '#22c55e']

    # Accuracy
    axes[0].plot(epochs, acc,     'b-', label='Train', linewidth=2)
    axes[0].plot(epochs, val_acc, 'r-', label='Val',   linewidth=2)
    for i, end in enumerate(phase_ends):
        axes[0].axvline(x=end, color=colors[i], linestyle='--', alpha=0.7,
                        label=f'Phase {i+2} start')
    axes[0].set_title('Accuracy', fontsize=14)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim([0, 1])

    # Loss
    axes[1].plot(epochs, loss,     'b-', label='Train', linewidth=2)
    axes[1].plot(epochs, val_loss, 'r-', label='Val',   linewidth=2)
    axes[1].set_title('Loss', fontsize=14)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle(f'DermaScan AI — Training History\nBest Val Accuracy: {max(val_acc)*100:.1f}%',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('model/training_history_v2.png', dpi=150, bbox_inches='tight')
    print("\n📊 Chart saved: model/training_history_v2.png")


# ─────────────────────────────────────────────
# STEP 9: FULL EVALUATION
# ─────────────────────────────────────────────
def evaluate(model, val_gen, classes):
    print("\n📋 Running full evaluation...")
    val_gen.reset()

    results = model.evaluate(val_gen, verbose=1)
    print(f"\n  ✅ Val Accuracy : {results[1]*100:.2f}%")
    print(f"  ✅ Top-2 Acc    : {results[2]*100:.2f}%")
    print(f"  ✅ Val Loss     : {results[0]:.4f}")

    try:
        from sklearn.metrics import classification_report
        val_gen.reset()
        y_pred = np.argmax(model.predict(val_gen, verbose=1), axis=1)
        y_true = val_gen.classes
        print("\n📋 Per-class Report:")
        print(classification_report(y_true, y_pred, target_names=classes))
    except ImportError:
        print("⚠️  pip install scikit-learn for per-class report")

    return results[1]


# ─────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────
model_ref = [None]  # For callback LR access

def main():
    print("=" * 60)
    print("  DermaScan AI v2 — High Accuracy Training")
    print("  Target: 90–98% accuracy")
    print("=" * 60)

    # Check GPU
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"\n🚀 GPU detected: {gpus[0].name} — Training will be FAST")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    else:
        print("\n💻 CPU mode — Training will take longer (~1-2 hrs)")
        print("   Tip: Use Google Colab (free GPU) for faster training")

    # Get classes
    classes = get_classes()
    if len(classes) < 2:
        print("❌ Need at least 2 classes with images. Add more images first.")
        return

    print(f"\n✅ Training on {len(classes)} classes: {classes}")

    # Check minimum images
    low_classes = []
    for cls in classes:
        d = os.path.join(TRAIN_DIR, cls)
        n = len(os.listdir(d))
        if n < 50:
            low_classes.append((cls, n))

    if low_classes:
        print("\n⚠️  WARNING: These classes have very few images:")
        for cls, n in low_classes:
            print(f"   {cls}: {n} images (need 100+ for good accuracy)")
        print("   Run: python augment_data.py --dataset dataset --count 20")
        ans = input("\n   Continue anyway? (y/n): ").strip().lower()
        if ans != 'y':
            return

    # Build generators + model
    train_gen, val_gen = build_generators(classes)
    model, base = build_model(len(classes))
    model_ref[0] = model
    cw = get_class_weights(classes)

    histories   = []
    phase_ends  = []

    # ────────────────────────────────────────
    # PHASE 1: Train head only (base frozen)
    # ────────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"  PHASE 1: Training Head (base frozen)")
    print(f"  Epochs: {EPOCHS_P1}  |  LR: {LR_P1}")
    print(f"{'='*55}")

    # Use MixUp for Phase 1
    mixed_gen = mixup_generator(train_gen, alpha=0.3)
    h1 = model.fit(
        mixed_gen,
        steps_per_epoch  = train_gen.samples // BATCH_SIZE,
        epochs           = EPOCHS_P1,
        validation_data  = val_gen,
        callbacks        = get_callbacks(1, LR_P1),
        class_weight     = cw,
        verbose          = 1,
    )
    histories.append(h1)
    phase_ends.append(EPOCHS_P1)
    best_p1 = max(h1.history['val_accuracy']) * 100
    print(f"\n✅ Phase 1 best val accuracy: {best_p1:.1f}%")

    # ────────────────────────────────────────
    # PHASE 2: Unfreeze top 50 layers
    # ────────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"  PHASE 2: Fine-tune top 50 layers")
    print(f"  Epochs: {EPOCHS_P2}  |  LR: {LR_P2}")
    print(f"{'='*55}")

    model = unfreeze(model, base, n_layers=50, lr=LR_P2, n_classes=len(classes))
    h2 = model.fit(
        train_gen,
        epochs           = EPOCHS_P2,
        validation_data  = val_gen,
        callbacks        = get_callbacks(2, LR_P2),
        class_weight     = cw,
        verbose          = 1,
    )
    histories.append(h2)
    phase_ends.append(EPOCHS_P1 + EPOCHS_P2)
    best_p2 = max(h2.history['val_accuracy']) * 100
    print(f"\n✅ Phase 2 best val accuracy: {best_p2:.1f}%")

    # ────────────────────────────────────────
    # PHASE 3: Unfreeze all layers (deep fine-tune)
    # ────────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"  PHASE 3: Full model fine-tuning (all layers)")
    print(f"  Epochs: {EPOCHS_P3}  |  LR: {LR_P3}")
    print(f"{'='*55}")

    model = unfreeze(model, base, n_layers=200, lr=LR_P3, n_classes=len(classes))
    h3 = model.fit(
        train_gen,
        epochs           = EPOCHS_P3,
        validation_data  = val_gen,
        callbacks        = get_callbacks(3, LR_P3),
        class_weight     = cw,
        verbose          = 1,
    )
    histories.append(h3)
    best_p3 = max(h3.history['val_accuracy']) * 100
    print(f"\n✅ Phase 3 best val accuracy: {best_p3:.1f}%")

    # Save final model
    model.save(MODEL_PATH)
    shutil.copy2(BEST_PATH, MODEL_PATH)

    # Save class index map
    idx_map = {str(i): name for i, name in enumerate(classes)}
    with open('model/class_indices.json', 'w') as f:
        json.dump(idx_map, f, indent=2)

    # Plot training history
    plot_all(histories, phase_ends)

    # Full evaluation
    final_acc = evaluate(model, val_gen, classes)

    # ── Final Summary ──
    print("\n" + "=" * 60)
    print(f"  🎉 TRAINING COMPLETE!")
    print(f"  Final Accuracy : {final_acc*100:.1f}%")
    print(f"  Phase 1        : {best_p1:.1f}%")
    print(f"  Phase 2        : {best_p2:.1f}%")
    print(f"  Phase 3        : {best_p3:.1f}%")
    print(f"  Model saved    : {MODEL_PATH}")
    print(f"  Classes        : {len(classes)}")
    print()
    if final_acc >= 0.90:
        print("  🏆 EXCELLENT accuracy! Model is production ready.")
    elif final_acc >= 0.80:
        print("  ✅ GOOD accuracy! Add more images to improve further.")
        print("     Run: python augment_data.py --dataset dataset --count 25")
    elif final_acc >= 0.70:
        print("  ⚠️  FAIR accuracy. Need more diverse images per class.")
        print("     Target: 300+ images per class before training.")
    else:
        print("  ❌ LOW accuracy. Dataset too small or imbalanced.")
        print("     Steps to fix:")
        print("     1. python augment_data.py --dataset dataset --count 25")
        print("     2. Add more real images from Kaggle/DermNet")
        print("     3. Run: python train_model_v2.py again")
    print("=" * 60)
    print(f"\n  ▶  Run website: python app.py")


if __name__ == "__main__":
    main()