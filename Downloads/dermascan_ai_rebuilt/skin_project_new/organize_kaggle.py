"""
=============================================================
  organize_kaggle.py
  After downloading a Kaggle dataset ZIP, run this to
  automatically sort images into:
    dataset/train/<Disease>/
    dataset/val/<Disease>/
    dataset/test/<Disease>/

  USAGE:
    python organize_kaggle.py --source path/to/unzipped/folder
    python organize_kaggle.py --source dermnet_unzipped --split 70 15 15
=============================================================
"""

import os, shutil, random, argparse
from pathlib import Path
from PIL import Image

# ── Map any folder name → our standard disease name ──────
DISEASE_NAME_MAP = {
    # Acne
    "acne and rosacea photos":          "Acne",
    "acne":                             "Acne",
    "acne vulgaris":                    "Acne",
    # Eczema
    "atopic dermatitis photos":         "Eczema",
    "eczema":                           "Eczema",
    "atopic dermatitis":                "Eczema",
    # Melanoma
    "melanoma skin cancer nevi and moles": "Melanoma",
    "melanoma":                         "Melanoma",
    "mel":                              "Melanoma",
    # Normal
    "normal":                           "Normal",
    "healthy skin":                     "Normal",
    "nevus":                            "Normal",
    "nv":                               "Normal",
    "benign":                           "Normal",
    # Psoriasis
    "psoriasis pictures lichen planus and related diseases": "Psoriasis",
    "psoriasis":                        "Psoriasis",
    # Ringworm
    "ringworm tinea corporis photos":   "Ringworm",
    "tinea ringworm candidiasis and other fungal infections": "Ringworm",
    "ringworm":                         "Ringworm",
    "tinea corporis":                   "Ringworm",
    # Vitiligo
    "vitiligo photos":                  "Vitiligo",
    "vitiligo":                         "Vitiligo",
    # Chickenpox
    "chicken pox":                      "Chickenpox",
    "chickenpox":                       "Chickenpox",
    "varicella":                        "Chickenpox",
    # Shingles
    "shingles":                         "Shingles",
    "herpes zoster":                    "Shingles",
    "herpes zoster photos":             "Shingles",
    # Impetigo
    "impetigo contagiosa ecthyma photos": "Impetigo",
    "impetigo":                         "Impetigo",
    # Cellulitis
    "cellulitis impetigo and other bacterial infections": "Cellulitis",
    "cellulitis":                       "Cellulitis",
    # Rosacea
    "rosacea":                          "Rosacea",
    # Hives
    "urticaria hives":                  "Hives",
    "hives":                            "Hives",
    "urticaria":                        "Hives",
    # Seborrheic Dermatitis
    "seborrheic keratoses and other benign tumors": "Seborrheic_Dermatitis",
    "seborrheic dermatitis":            "Seborrheic_Dermatitis",
    "seborrhoeic dermatitis":           "Seborrheic_Dermatitis",
    # Contact Dermatitis
    "contact dermatitis photos":        "Contact_Dermatitis",
    "contact dermatitis":               "Contact_Dermatitis",
    # Warts
    "warts molluscum and other viral infections": "Warts",
    "warts":                            "Warts",
    "viral warts":                      "Warts",
    # Scabies
    "scabies lyme disease and other infestations": "Scabies",
    "scabies":                          "Scabies",
    # BCC
    "basal cell carcinoma (bcc)":       "Basal_Cell_Carcinoma",
    "basal cell carcinoma":             "Basal_Cell_Carcinoma",
    "bcc":                              "Basal_Cell_Carcinoma",
    # Alopecia
    "alopecia photos":                  "Alopecia",
    "alopecia":                         "Alopecia",
    "alopecia areata":                  "Alopecia",
    # Tinea Versicolor
    "tinea versicolor photos":          "Tinea_Versicolor",
    "tinea versicolor":                 "Tinea_Versicolor",
    "pityriasis versicolor":            "Tinea_Versicolor",
}

ALL_DISEASES = list(set(DISEASE_NAME_MAP.values()))
IMG_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.avif', '.tiff'}


def is_valid_image(path: str) -> bool:
    """Check if file is a valid, non-corrupt image"""
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except:
        return False


def map_folder_name(folder_name: str) -> str | None:
    """Map any folder name to our standard disease name"""
    lower = folder_name.lower().strip()
    # Exact match first
    if lower in DISEASE_NAME_MAP:
        return DISEASE_NAME_MAP[lower]
    # Partial match
    for key, disease in DISEASE_NAME_MAP.items():
        if key in lower or lower in key:
            return disease
    return None


def find_and_organize(source_dir: str, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15):
    """
    Walk through any Kaggle dataset folder.
    Find image folders, map to disease names, split into train/val/test.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 0.01, "Ratios must sum to 1"

    print(f"\n🔍 Scanning: {source_dir}")
    print(f"   Split: train={int(train_ratio*100)}%  val={int(val_ratio*100)}%  test={int(test_ratio*100)}%\n")

    # Create output folders
    for split in ["train", "val", "test"]:
        for disease in ALL_DISEASES:
            os.makedirs(os.path.join("dataset", split, disease), exist_ok=True)

    stats = {d: {"train": 0, "val": 0, "test": 0} for d in ALL_DISEASES}
    unmatched_folders = []

    # Walk the source directory
    for root, dirs, files in os.walk(source_dir):
        folder_name = os.path.basename(root)
        disease = map_folder_name(folder_name)

        if not disease:
            continue

        # Get all valid images in this folder
        images = [
            f for f in files
            if Path(f).suffix.lower() in IMG_EXTENSIONS
        ]

        if not images:
            continue

        random.shuffle(images)
        n = len(images)
        train_end = int(n * train_ratio)
        val_end   = int(n * (train_ratio + val_ratio))

        split_groups = {
            "train": images[:train_end],
            "val":   images[train_end:val_end],
            "test":  images[val_end:]
        }

        print(f"  📁 '{folder_name}' → {disease}  ({n} images)")

        for split, split_files in split_groups.items():
            out_dir = os.path.join("dataset", split, disease)
            for i, fname in enumerate(split_files):
                src = os.path.join(root, fname)
                ext = Path(fname).suffix.lower() or ".jpg"
                # Unique filename: disease_split_index.jpg
                existing = len(os.listdir(out_dir))
                dst = os.path.join(out_dir, f"{disease}_{existing+i:04d}{ext}")
                if not os.path.exists(dst):
                    try:
                        shutil.copy2(src, dst)
                        stats[disease][split] += 1
                    except Exception as e:
                        pass

    # Print final summary
    print("\n" + "=" * 65)
    print(f"  {'Disease':<28} {'Train':>6} {'Val':>5} {'Test':>5} {'Total':>7}")
    print("─" * 65)
    grand = 0
    for disease in sorted(ALL_DISEASES):
        s = stats[disease]
        total = s["train"] + s["val"] + s["test"]
        grand += total
        if total == 0:
            continue
        status = "✅" if total >= 200 else "⚠️ "
        print(f"  {status} {disease:<26} {s['train']:>6} {s['val']:>5} {s['test']:>5} {total:>7}")
    print("─" * 65)
    print(f"  {'TOTAL':<28} {'':>6} {'':>5} {'':>5} {grand:>7}")
    print()
    print("✅ Organization complete!")
    print("👉 Next: python augment_data.py --dataset dataset --count 15")
    print("👉 Then: python train_model.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Organize Kaggle skin dataset")
    parser.add_argument("--source", required=True, help="Path to unzipped Kaggle dataset folder")
    parser.add_argument("--split", nargs=3, type=float, default=[0.70, 0.15, 0.15],
                        metavar=("TRAIN", "VAL", "TEST"),
                        help="Train/val/test split ratios (default: 0.70 0.15 0.15)")
    args = parser.parse_args()

    train_r, val_r, test_r = args.split
    find_and_organize(args.source, train_r, val_r, test_r)
