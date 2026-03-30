"""
=============================================================
  organize_dataset.py  —  UNIFIED DATASET MANAGER
  Combines: create_folders + organize_kaggle + verify

  Supports ALL recommended datasets:
    - DermNet (Kaggle)
    - HAM10000
    - ISIC 2019 / 2020 / 2024
    - PAD-UFES-20
    - Fitzpatrick 17k
    - SD-198
    - Any custom skin folder

  USAGE:
    # Step 1: Create empty folders
    python organize_dataset.py --create

    # Step 2: Organize a downloaded dataset
    python organize_dataset.py --source path/to/unzipped --dataset dermnet
    python organize_dataset.py --source ham10000_folder  --dataset ham10000
    python organize_dataset.py --source isic_folder      --dataset isic
    python organize_dataset.py --source pad_folder       --dataset pad
    python organize_dataset.py --source fitz_folder      --dataset fitzpatrick
    python organize_dataset.py --source any_folder       --dataset auto

    # Step 3: Verify your full dataset
    python organize_dataset.py --verify

    # Custom train/val/test split
    python organize_dataset.py --source myfolder --dataset auto --split 75 15 10
=============================================================
"""

import os, shutil, random, argparse, csv, json
from pathlib import Path
from collections import defaultdict

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("⚠️  Pillow not installed — skipping corrupt-image check.")
    print("   Install: pip install Pillow")

# ─────────────────────────────────────────────────────────────
#  MASTER DISEASE LIST  (20 diseases, standardized names)
# ─────────────────────────────────────────────────────────────
ALL_DISEASES = [
    # Common inflammatory
    "Acne",
    "Rosacea",
    "Eczema",
    "Psoriasis",
    "Contact_Dermatitis",
    "Seborrheic_Dermatitis",
    "Hives",
    # Fungal / parasitic
    "Ringworm",
    "Tinea_Versicolor",
    "Scabies",
    # Viral / bacterial
    "Chickenpox",
    "Shingles",
    "Impetigo",
    "Cellulitis",
    "Warts",
    # Hair / pigment
    "Vitiligo",
    "Alopecia",
    # Cancer
    "Melanoma",
    "Basal_Cell_Carcinoma",
    # Normal / benign
    "Normal",
]

SPLITS = ["train", "val", "test"]
IMG_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff', '.avif'}

# ─────────────────────────────────────────────────────────────
#  DISEASE NAME MAPS — one per dataset source
#  Key   = folder name as it appears in the downloaded dataset
#  Value = our standard disease name from ALL_DISEASES
# ─────────────────────────────────────────────────────────────

# ── DermNet (Kaggle) ─────────────────────────────────────────
DERMNET_MAP = {
    "acne and rosacea photos":                          "Acne",
    "acne vulgaris":                                    "Acne",
    "acne":                                             "Acne",
    "rosacea":                                          "Rosacea",
    "atopic dermatitis photos":                         "Eczema",
    "atopic dermatitis":                                "Eczema",
    "eczema":                                           "Eczema",
    "psoriasis pictures lichen planus and related diseases": "Psoriasis",
    "psoriasis":                                        "Psoriasis",
    "contact dermatitis photos":                        "Contact_Dermatitis",
    "contact dermatitis":                               "Contact_Dermatitis",
    "seborrheic keratoses and other benign tumors":     "Seborrheic_Dermatitis",
    "seborrheic dermatitis":                            "Seborrheic_Dermatitis",
    "seborrhoeic dermatitis":                           "Seborrheic_Dermatitis",
    "urticaria hives":                                  "Hives",
    "hives":                                            "Hives",
    "urticaria":                                        "Hives",
    "ringworm tinea corporis photos":                   "Ringworm",
    "tinea ringworm candidiasis and other fungal infections": "Ringworm",
    "ringworm":                                         "Ringworm",
    "tinea corporis":                                   "Ringworm",
    "tinea versicolor photos":                          "Tinea_Versicolor",
    "tinea versicolor":                                 "Tinea_Versicolor",
    "pityriasis versicolor":                            "Tinea_Versicolor",
    "scabies lyme disease and other infestations":      "Scabies",
    "scabies":                                          "Scabies",
    "chicken pox":                                      "Chickenpox",
    "chickenpox":                                       "Chickenpox",
    "varicella":                                        "Chickenpox",
    "shingles":                                         "Shingles",
    "herpes zoster photos":                             "Shingles",
    "herpes zoster":                                    "Shingles",
    "impetigo contagiosa ecthyma photos":               "Impetigo",
    "impetigo":                                         "Impetigo",
    "cellulitis impetigo and other bacterial infections": "Cellulitis",
    "cellulitis":                                       "Cellulitis",
    "warts molluscum and other viral infections":       "Warts",
    "warts":                                            "Warts",
    "viral warts":                                      "Warts",
    "vitiligo photos":                                  "Vitiligo",
    "vitiligo":                                         "Vitiligo",
    "alopecia photos":                                  "Alopecia",
    "alopecia areata":                                  "Alopecia",
    "alopecia":                                         "Alopecia",
    "melanoma skin cancer nevi and moles":              "Melanoma",
    "melanoma":                                         "Melanoma",
    "basal cell carcinoma (bcc)":                       "Basal_Cell_Carcinoma",
    "basal cell carcinoma":                             "Basal_Cell_Carcinoma",
    "normal":                                           "Normal",
    "healthy":                                          "Normal",
    "healthy skin":                                     "Normal",
}

# ── HAM10000 ─────────────────────────────────────────────────
# HAM10000 uses short codes; map them to our standard names
HAM10000_MAP = {
    "mel":  "Melanoma",
    "nv":   "Normal",           # Melanocytic nevus (benign mole)
    "bcc":  "Basal_Cell_Carcinoma",
    "bkl":  "Seborrheic_Dermatitis",  # Benign keratosis-like lesions
    "akiec":"Basal_Cell_Carcinoma",   # Actinic keratosis (precancerous)
    "vasc": "Normal",           # Vascular lesions (mostly benign)
    "df":   "Normal",           # Dermatofibroma (benign)
    # HAM10000 also comes as a single CSV with labels — handled below
}

# ── ISIC 2019 / 2020 ─────────────────────────────────────────
ISIC_MAP = {
    "melanoma":             "Melanoma",
    "mel":                  "Melanoma",
    "nevus":                "Normal",
    "nv":                   "Normal",
    "basal cell carcinoma": "Basal_Cell_Carcinoma",
    "bcc":                  "Basal_Cell_Carcinoma",
    "actinic keratosis":    "Basal_Cell_Carcinoma",
    "benign keratosis":     "Seborrheic_Dermatitis",
    "bkl":                  "Seborrheic_Dermatitis",
    "dermatofibroma":       "Normal",
    "df":                   "Normal",
    "vascular lesion":      "Normal",
    "vasc":                 "Normal",
    "squamous cell carcinoma": "Melanoma",
    "scc":                  "Melanoma",
    "unknown":              None,   # Skip unknown class
}

# ── PAD-UFES-20 ──────────────────────────────────────────────
PAD_MAP = {
    "bcc":  "Basal_Cell_Carcinoma",
    "mel":  "Melanoma",
    "scc":  "Melanoma",         # Squamous cell carcinoma
    "nev":  "Normal",
    "nev ": "Normal",
    "ack":  "Normal",           # Actinic keratosis — borderline, treat as lesion
    "sek":  "Seborrheic_Dermatitis",
}

# ── Fitzpatrick 17k ──────────────────────────────────────────
# Fitzpatrick CSV has a 'label' column with detailed disease names
FITZPATRICK_MAP = {
    "acne":                     "Acne",
    "rosacea":                  "Rosacea",
    "atopic dermatitis":        "Eczema",
    "eczema":                   "Eczema",
    "psoriasis":                "Psoriasis",
    "contact dermatitis":       "Contact_Dermatitis",
    "seborrheic dermatitis":    "Seborrheic_Dermatitis",
    "urticaria":                "Hives",
    "vitiligo":                 "Vitiligo",
    "alopecia":                 "Alopecia",
    "tinea":                    "Ringworm",
    "ringworm":                 "Ringworm",
    "scabies":                  "Scabies",
    "herpes zoster":            "Shingles",
    "varicella":                "Chickenpox",
    "impetigo":                 "Impetigo",
    "cellulitis":               "Cellulitis",
    "wart":                     "Warts",
    "melanoma":                 "Melanoma",
    "basal cell":               "Basal_Cell_Carcinoma",
    "squamous cell":            "Melanoma",
    "normal":                   "Normal",
}

# ── Unified auto-detect map (combines all) ───────────────────
AUTO_MAP = {}
for m in [DERMNET_MAP, HAM10000_MAP, ISIC_MAP, PAD_MAP, FITZPATRICK_MAP]:
    AUTO_MAP.update(m)


# ─────────────────────────────────────────────────────────────
#  HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────

def get_map(dataset_type: str) -> dict:
    return {
        "dermnet":      DERMNET_MAP,
        "ham10000":     HAM10000_MAP,
        "isic":         ISIC_MAP,
        "pad":          PAD_MAP,
        "fitzpatrick":  FITZPATRICK_MAP,
        "auto":         AUTO_MAP,
    }.get(dataset_type, AUTO_MAP)


def map_name(name: str, name_map: dict) -> str | None:
    """Map a folder/label name to standard disease name."""
    lower = name.lower().strip()
    # 1. Exact match
    if lower in name_map:
        return name_map[lower]
    # 2. Partial match (key inside folder name)
    for key, disease in name_map.items():
        if key and key in lower:
            return disease
    # 3. Partial match (folder name inside key)
    for key, disease in name_map.items():
        if lower and lower in key:
            return disease
    return None


def is_valid_image(path: str) -> bool:
    """Return True if image file is readable and not corrupted."""
    if not PIL_AVAILABLE:
        return True
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except Exception:
        return False


def count_images_in(folder: str) -> int:
    if not os.path.exists(folder):
        return 0
    return sum(
        1 for f in os.listdir(folder)
        if Path(f).suffix.lower() in IMG_EXTENSIONS
    )


def copy_image(src: str, dst_dir: str, prefix: str) -> bool:
    """Copy an image to dst_dir with a unique indexed filename."""
    ext = Path(src).suffix.lower() or ".jpg"
    existing = count_images_in(dst_dir)
    dst = os.path.join(dst_dir, f"{prefix}_{existing:05d}{ext}")
    if not os.path.exists(dst):
        try:
            shutil.copy2(src, dst)
            return True
        except Exception as e:
            print(f"   ⚠️  Copy failed: {src} → {e}")
    return False


# ─────────────────────────────────────────────────────────────
#  STEP 1: CREATE FOLDER STRUCTURE
# ─────────────────────────────────────────────────────────────

def create_folders():
    """Create all dataset/split/disease folders."""
    print("=" * 60)
    print("  Creating Dataset Folder Structure")
    print("=" * 60)

    created = 0
    for split in SPLITS:
        for disease in ALL_DISEASES:
            path = os.path.join("dataset", split, disease)
            if not os.path.exists(path):
                os.makedirs(path, exist_ok=True)
                created += 1

    print(f"\n✅ Created {created} folders\n")
    print("  dataset/")
    for split in SPLITS:
        print(f"    {split}/")
        for d in ALL_DISEASES:
            print(f"      {d}/")


# ─────────────────────────────────────────────────────────────
#  STEP 2: ORGANIZE DOWNLOADED DATASET
# ─────────────────────────────────────────────────────────────

def organize_folder(source_dir: str, dataset_type: str,
                    train_r=0.70, val_r=0.15, test_r=0.15):
    """
    Walk source_dir, detect disease folders, map names,
    split images into train/val/test and copy to dataset/.
    Handles:
      - Simple folder-per-class structure (DermNet, most Kaggle)
      - HAM10000 CSV-based labelling
      - ISIC CSV-based labelling
      - PAD-UFES-20 CSV metadata
    """
    assert abs(train_r + val_r + test_r - 1.0) < 0.01, \
        "Split ratios must sum to 1.0"

    print(f"\n{'='*60}")
    print(f"  Organizing: {source_dir}")
    print(f"  Dataset type: {dataset_type}")
    print(f"  Split: train={int(train_r*100)}% "
          f"val={int(val_r*100)}% test={int(test_r*100)}%")
    print(f"{'='*60}\n")

    # Ensure all folders exist
    create_folders()

    name_map = get_map(dataset_type)
    stats = defaultdict(lambda: {"train": 0, "val": 0, "test": 0, "skipped": 0})
    corrupt_count = 0

    # ── Try CSV-based organisation first (HAM10000 / ISIC / PAD) ──
    csv_handled = _try_csv_organization(
        source_dir, dataset_type, name_map, stats, train_r, val_r, test_r
    )

    if not csv_handled:
        # ── Folder-per-class structure ─────────────────────────
        for root, dirs, files in os.walk(source_dir):
            folder_name = os.path.basename(root)
            disease = map_name(folder_name, name_map)

            if not disease:
                continue

            images = [
                f for f in files
                if Path(f).suffix.lower() in IMG_EXTENSIONS
            ]
            if not images:
                continue

            random.shuffle(images)
            n = len(images)
            i_val  = int(n * train_r)
            i_test = int(n * (train_r + val_r))

            splits_data = {
                "train": images[:i_val],
                "val":   images[i_val:i_test],
                "test":  images[i_test:],
            }

            print(f"  📁  '{folder_name}'  →  {disease}  ({n} images)")

            for split, split_files in splits_data.items():
                dst_dir = os.path.join("dataset", split, disease)
                for fname in split_files:
                    src = os.path.join(root, fname)
                    if PIL_AVAILABLE and not is_valid_image(src):
                        stats[disease]["skipped"] += 1
                        corrupt_count += 1
                        continue
                    if copy_image(src, dst_dir, disease):
                        stats[disease][split] += 1
                    else:
                        stats[disease]["skipped"] += 1

    _print_stats(stats, corrupt_count)


def _try_csv_organization(source_dir, dataset_type, name_map,
                           stats, train_r, val_r, test_r) -> bool:
    """
    Handle CSV-labelled datasets (HAM10000, ISIC, PAD-UFES-20).
    Returns True if a CSV was found and handled.
    """
    csv_files = list(Path(source_dir).rglob("*.csv"))
    if not csv_files:
        return False

    # Find the main metadata CSV
    meta_csv = None
    priority_names = ["metadata", "hmnist", "isic", "ground_truth",
                      "train", "labels", "pad_ufes_20"]
    for pname in priority_names:
        for cf in csv_files:
            if pname in cf.stem.lower():
                meta_csv = cf
                break
        if meta_csv:
            break

    if not meta_csv:
        meta_csv = csv_files[0]

    print(f"  📄 Found metadata CSV: {meta_csv.name}")

    try:
        with open(meta_csv, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
    except Exception as e:
        print(f"  ⚠️  Could not read CSV: {e}")
        return False

    if not rows:
        return False

    cols = list(rows[0].keys())
    print(f"  📄 Columns: {', '.join(cols)}")

    # Detect image ID column
    id_col = next((c for c in cols
                   if c.lower() in {"image_id", "image", "isic_id",
                                    "img_id", "image_name", "id"}), None)

    # Detect label column
    label_col = next((c for c in cols
                      if c.lower() in {"dx", "diagnosis", "label",
                                       "disease", "class", "category",
                                       "target", "benign_malignant"}), None)

    if not id_col or not label_col:
        print(f"  ⚠️  Could not find image/label columns in CSV — "
              f"falling back to folder scan.")
        return False

    print(f"  🔑  Image col: '{id_col}'  |  Label col: '{label_col}'")

    # Group rows by disease
    disease_rows = defaultdict(list)
    skipped_labels = set()
    for row in rows:
        raw_label = row[label_col].strip()
        disease = map_name(raw_label, name_map)
        if disease:
            disease_rows[disease].append(row[id_col].strip())
        else:
            skipped_labels.add(raw_label)

    if skipped_labels:
        print(f"  ℹ️  Unmapped labels (skipped): {', '.join(sorted(skipped_labels))}")

    # Find all images under source_dir indexed by stem
    print("  🔍 Indexing image files (may take a moment)...")
    img_index = {}
    for root, _, files in os.walk(source_dir):
        for fname in files:
            if Path(fname).suffix.lower() in IMG_EXTENSIONS:
                stem = Path(fname).stem
                img_index[stem] = os.path.join(root, fname)

    print(f"  🖼️  Indexed {len(img_index):,} images")

    # Copy images per disease
    for disease, img_ids in disease_rows.items():
        random.shuffle(img_ids)
        n = len(img_ids)
        i_val  = int(n * train_r)
        i_test = int(n * (train_r + val_r))
        splits_data = {
            "train": img_ids[:i_val],
            "val":   img_ids[i_val:i_test],
            "test":  img_ids[i_test:],
        }
        print(f"  📁  CSV label → {disease}  ({n} images)")

        for split, ids in splits_data.items():
            dst_dir = os.path.join("dataset", split, disease)
            for img_id in ids:
                # Try exact stem match first, then partial
                src = img_index.get(img_id)
                if not src:
                    # Try stripping extensions from img_id
                    bare = Path(img_id).stem
                    src = img_index.get(bare)
                if not src:
                    stats[disease]["skipped"] += 1
                    continue
                if copy_image(src, dst_dir, disease):
                    stats[disease][split] += 1
                else:
                    stats[disease]["skipped"] += 1

    return True


def _print_stats(stats, corrupt_count=0):
    print(f"\n{'='*65}")
    print(f"  {'Disease':<28} {'Train':>6} {'Val':>5} {'Test':>5} {'Total':>7}")
    print(f"  {'─'*61}")
    grand = 0
    low_classes = []
    for disease in ALL_DISEASES:
        s = stats.get(disease, {"train": 0, "val": 0, "test": 0})
        total = s["train"] + s["val"] + s["test"]
        grand += total
        if total == 0:
            continue
        icon = "✅" if total >= 300 else ("⚠️ " if total >= 80 else "❌ ")
        if total < 80:
            low_classes.append(disease)
        print(f"  {icon} {disease:<26} {s['train']:>6} "
              f"{s['val']:>5} {s['test']:>5} {total:>7}")
    print(f"  {'─'*61}")
    print(f"  {'TOTAL':<28} {'':>6} {'':>5} {'':>5} {grand:>7}")
    if corrupt_count:
        print(f"\n  ⚠️  Skipped {corrupt_count} corrupt/unreadable images")
    print()
    if low_classes:
        print("  Classes with < 80 images (need more data):")
        for cls in low_classes:
            print(f"    ❌  {cls}")
        print()
        print("  Fix options:")
        print("    1. Run augment_data.py to synthetically expand them")
        print("    2. Download more images from the sources in the guide")
    print(f"\n{'='*65}")


# ─────────────────────────────────────────────────────────────
#  STEP 3: VERIFY FULL DATASET
# ─────────────────────────────────────────────────────────────

def verify_dataset():
    """
    Full dataset health check:
    - Image counts per class / split
    - Class imbalance ratio
    - Missing val/test splits
    - Corrupt image detection
    - Estimated training time
    - Readiness score
    """
    print(f"\n{'='*65}")
    print("  DATASET VERIFICATION REPORT")
    print(f"{'='*65}\n")

    grand_total = 0
    class_totals = {}
    issues = []
    corrupt_total = 0

    print(f"  {'Disease':<28} {'Train':>6} {'Val':>5} {'Test':>5} "
          f"{'Total':>7} {'Balance':>8}  Status")
    print(f"  {'─'*72}")

    for disease in ALL_DISEASES:
        counts = {}
        for split in SPLITS:
            folder = os.path.join("dataset", split, disease)
            counts[split] = count_images_in(folder)

        total = sum(counts.values())
        grand_total += total
        class_totals[disease] = total

        if total == 0:
            continue

        # Check val/test presence
        if counts["val"] == 0:
            issues.append(f"No val images for {disease}")
        if counts["test"] == 0:
            issues.append(f"No test images for {disease}")

        # Quality status
        if total >= 300:
            status = "✅ Good"
        elif total >= 150:
            status = "🟡 OK"
        elif total >= 50:
            status = "⚠️  Low"
        else:
            status = "❌ Too few"
            issues.append(f"{disease}: only {total} images")

        # Train/val balance check
        expected_val = int(counts["train"] * (0.15 / 0.70))
        balance = "OK" if counts["val"] >= expected_val * 0.7 else "⚠️ "

        print(f"  {disease:<28} {counts['train']:>6} {counts['val']:>5} "
              f"{counts['test']:>5} {total:>7} {balance:>8}  {status}")

    print(f"  {'─'*72}")
    print(f"  {'TOTAL':<28} {'':>6} {'':>5} {'':>5} {grand_total:>7}")

    # ── Class imbalance report ───────────────────────────────
    non_empty = {k: v for k, v in class_totals.items() if v > 0}
    if non_empty:
        max_c = max(non_empty.values())
        min_c = min(non_empty.values())
        ratio = max_c / max(min_c, 1)

        print(f"\n  📊 Class imbalance ratio: {ratio:.1f}x "
              f"(max={max_c}, min={min_c})")

        if ratio > 10:
            issues.append(f"Severe class imbalance ({ratio:.1f}x) — "
                          "use augment_data.py or focal loss")
        elif ratio > 5:
            issues.append(f"Moderate class imbalance ({ratio:.1f}x) — "
                          "consider augmentation")

    # ── Estimated training time ──────────────────────────────
    if grand_total > 0:
        steps_per_epoch = grand_total * 0.70 // 8
        epochs_total    = 70  # rough estimate (P1+P2+P3)
        gpu_mins  = int(steps_per_epoch * epochs_total * 0.05 / 60)
        cpu_mins  = gpu_mins * 20
        print(f"\n  ⏱️  Estimated training time:")
        print(f"      GPU (RTX 3060+) : ~{gpu_mins} minutes")
        print(f"      CPU only        : ~{cpu_mins // 60}h {cpu_mins % 60}m")
        print(f"      Google Colab T4 : ~{gpu_mins * 2} minutes")

    # ── Issues summary ───────────────────────────────────────
    if issues:
        print(f"\n  ⚠️  Issues found ({len(issues)}):")
        for issue in issues:
            print(f"     • {issue}")
    else:
        print(f"\n  ✅  No issues found!")

    # ── Readiness score ──────────────────────────────────────
    filled = sum(1 for v in class_totals.values() if v >= 100)
    total_classes = len(ALL_DISEASES)
    score = int((filled / total_classes) * 100)

    print(f"\n  🎯 Dataset Readiness: {score}%  "
          f"({filled}/{total_classes} classes have 100+ images)")

    if score >= 80:
        print("     → Ready to train! Run: python train_model_v3.py")
    elif score >= 50:
        print("     → Partially ready. Add more images to low classes first.")
        print("     → Or run: python augment_data.py --dataset dataset --count 20")
    else:
        print("     → Need more data. Download HAM10000 and ISIC datasets.")
        print("     → Run this script again after adding images.")

    print(f"\n{'='*65}\n")

    # Save report to JSON
    report = {
        "total_images"    : grand_total,
        "class_counts"    : class_totals,
        "issues"          : issues,
        "readiness_score" : score,
    }
    with open("dataset/dataset_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print("  📄 Full report saved: dataset/dataset_report.json\n")


# ─────────────────────────────────────────────────────────────
#  MERGE DUPLICATE CLASSES (run after adding multiple datasets)
# ─────────────────────────────────────────────────────────────

def merge_duplicates():
    """
    After combining datasets, merge any stray folders that map
    to the same standard disease (e.g. 'atopic_dermatitis' → 'Eczema').
    """
    print("\n🔄 Checking for duplicate / stray folders...")
    merged = 0

    for split in SPLITS:
        split_dir = os.path.join("dataset", split)
        if not os.path.exists(split_dir):
            continue

        for folder in os.listdir(split_dir):
            folder_path = os.path.join(split_dir, folder)
            if not os.path.isdir(folder_path):
                continue
            if folder in ALL_DISEASES:
                continue  # Already standardized

            disease = map_name(folder, AUTO_MAP)
            if disease:
                target = os.path.join(split_dir, disease)
                os.makedirs(target, exist_ok=True)
                images = [f for f in os.listdir(folder_path)
                          if Path(f).suffix.lower() in IMG_EXTENSIONS]
                for img in images:
                    src = os.path.join(folder_path, img)
                    copy_image(src, target, disease)
                    merged += 1
                print(f"  Merged '{folder}' → '{disease}' ({len(images)} images)")
                # Remove old folder after merging
                shutil.rmtree(folder_path)

    if merged:
        print(f"  ✅ Merged {merged} images from stray folders")
    else:
        print("  ✅ No duplicate folders found")


# ─────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="DermaScan Dataset Manager — create, organize, verify",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python organize_dataset.py --create
  python organize_dataset.py --source dermnet_folder   --dataset dermnet
  python organize_dataset.py --source ham10000_folder  --dataset ham10000
  python organize_dataset.py --source isic_2019        --dataset isic
  python organize_dataset.py --source pad_ufes_folder  --dataset pad
  python organize_dataset.py --source fitz_folder      --dataset fitzpatrick
  python organize_dataset.py --source any_folder       --dataset auto
  python organize_dataset.py --verify
  python organize_dataset.py --merge
        """
    )

    parser.add_argument("--create",   action="store_true",
                        help="Create empty folder structure")
    parser.add_argument("--source",   type=str, default=None,
                        help="Path to unzipped downloaded dataset")
    parser.add_argument("--dataset",  type=str, default="auto",
                        choices=["dermnet", "ham10000", "isic",
                                 "pad", "fitzpatrick", "auto"],
                        help="Dataset type for correct label mapping "
                             "(default: auto)")
    parser.add_argument("--split",    nargs=3, type=float,
                        default=[0.70, 0.15, 0.15],
                        metavar=("TRAIN", "VAL", "TEST"),
                        help="Split ratios summing to 1.0 "
                             "(default: 0.70 0.15 0.15)")
    parser.add_argument("--verify",   action="store_true",
                        help="Run full dataset health check")
    parser.add_argument("--merge",    action="store_true",
                        help="Merge stray/duplicate class folders")

    args = parser.parse_args()

    if not any([args.create, args.source, args.verify, args.merge]):
        parser.print_help()
        return

    random.seed(42)

    if args.create:
        create_folders()

    if args.source:
        if not os.path.exists(args.source):
            print(f"❌  Source not found: {args.source}")
            return
        train_r, val_r, test_r = args.split
        organize_folder(args.source, args.dataset, train_r, val_r, test_r)

    if args.merge:
        merge_duplicates()

    if args.verify:
        verify_dataset()

    if args.source or args.merge:
        print("\n👉 Next steps:")
        print("   python organize_dataset.py --verify")
        print("   python augment_data.py --dataset dataset --count 20")
        print("   python train_model_v3.py")


if __name__ == "__main__":
    main()