"""
=============================================================
  create_folders.py
  Creates ALL dataset folders for 20 skin diseases
  Run: python create_folders.py
=============================================================
"""
import os

# ── All 20 Skin Diseases ──────────────────────────────────
ALL_DISEASES = [
    # Original 7
    "Acne",
    "Eczema",
    "Melanoma",
    "Normal",
    "Psoriasis",
    "Ringworm",
    "Vitiligo",
    # 13 New Common Diseases
    "Chickenpox",
    "Shingles",
    "Impetigo",
    "Cellulitis",
    "Rosacea",
    "Hives",
    "Seborrheic_Dermatitis",
    "Contact_Dermatitis",
    "Warts",
    "Scabies",
    "Basal_Cell_Carcinoma",
    "Alopecia",
    "Tinea_Versicolor",
]

SPLITS = ["train", "val", "test"]

def create_all_folders():
    print("=" * 55)
    print("  Creating Dataset Folder Structure")
    print("=" * 55)

    for split in SPLITS:
        for disease in ALL_DISEASES:
            path = os.path.join("dataset", split, disease)
            os.makedirs(path, exist_ok=True)

    print("\n✅ Folder structure created!\n")
    print("dataset/")
    for split in SPLITS:
        print(f"  {split}/")
        for disease in ALL_DISEASES:
            print(f"    {disease}/")
    print()

def count_images():
    print("\n📊 Current Image Count per Class:")
    print(f"{'Disease':<28} {'Train':>6} {'Val':>5} {'Test':>5} {'Total':>7} Status")
    print("─" * 65)

    grand = 0
    for disease in ALL_DISEASES:
        counts = []
        for split in SPLITS:
            folder = os.path.join("dataset", split, disease)
            if os.path.exists(folder):
                imgs = [
                    f for f in os.listdir(folder)
                    if f.lower().endswith(('.jpg','.jpeg','.png','.webp','.bmp','.avif'))
                ]
                counts.append(len(imgs))
            else:
                counts.append(0)
        total = sum(counts)
        grand += total
        if total >= 200:   status = "✅ Good"
        elif total >= 80:  status = "⚠️  Low"
        elif total >= 1:   status = "❌ Too few"
        else:              status = "⬜ Empty"
        print(f"  {disease:<26} {counts[0]:>6} {counts[1]:>5} {counts[2]:>5} {total:>7}   {status}")

    print("─" * 65)
    print(f"  {'TOTAL':<26} {'':>6} {'':>5} {'':>5} {grand:>7}")
    print()

if __name__ == "__main__":
    create_all_folders()
    count_images()
    print("👉 Next: run   python download_dataset.py")
    print("         then: python split_and_verify.py")
