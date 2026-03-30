"""
=============================================================
  download_datasets.py  —  AUTO DATASET DOWNLOADER
  Downloads all recommended skin disease datasets automatically

  Supports:
    ✅ HAM10000          (Kaggle)
    ✅ ISIC 2019         (Kaggle)
    ✅ ISIC 2020         (Kaggle)
    ✅ DermNet           (Kaggle)
    ✅ Skin Disease 19k  (Kaggle)
    ✅ PAD-UFES-20       (Direct download - Mendeley)
    ✅ Fitzpatrick 17k   (GitHub + image scraping)

  SETUP (one-time):
    pip install kaggle requests tqdm Pillow gdown

    Then get your Kaggle API key:
    1. Go to https://www.kaggle.com/settings
    2. Click "Create New Token" → downloads kaggle.json
    3. Place it at: ~/.kaggle/kaggle.json  (Linux/Mac)
                or: C:\\Users\\YOU\\.kaggle\\kaggle.json  (Windows)

  USAGE:
    # Download everything (recommended)
    python download_datasets.py --all

    # Download specific datasets
    python download_datasets.py --ham10000
    python download_datasets.py --isic
    python download_datasets.py --dermnet
    python download_datasets.py --pad
    python download_datasets.py --fitzpatrick
    python download_datasets.py --skin19k

    # Download + auto-organize into dataset/ folders
    python download_datasets.py --all --organize

    # Check what's already downloaded
    python download_datasets.py --status
=============================================================
"""

import os, sys, json, shutil, zipfile, argparse, time, random
import subprocess
from pathlib import Path
from datetime import datetime

# ── Try importing optional libraries ──────────────────────────
try:
    import requests
    from tqdm import tqdm
    REQUESTS_OK = True
except ImportError:
    REQUESTS_OK = False

try:
    from PIL import Image
    PIL_OK = True
except ImportError:
    PIL_OK = False

# ─────────────────────────────────────────────────────────────
#  DATASET REGISTRY
#  All datasets with their Kaggle slugs / direct URLs
# ─────────────────────────────────────────────────────────────
DATASETS = {

    "ham10000": {
        "name":        "HAM10000 — Skin Lesion Analysis",
        "type":        "kaggle",
        "kaggle_slug": "kmader/skin-lesion-analysis-toward-melanoma-detection",
        "size_mb":     2800,
        "images":      10015,
        "classes":     7,
        "description": "Gold standard dermoscopy dataset. Best quality labels.",
        "folder":      "downloads/ham10000",
        "priority":    1,
    },

    "isic2019": {
        "name":        "ISIC 2019 — Skin Lesion Classification",
        "type":        "kaggle",
        "kaggle_slug": "andrewmvd/isic-2019",
        "size_mb":     8500,
        "images":      25331,
        "classes":     8,
        "description": "Large dermoscopy benchmark. 8 well-balanced classes.",
        "folder":      "downloads/isic2019",
        "priority":    2,
    },

    "skin19k": {
        "name":        "Skin Disease Dataset (19k images, 23 types)",
        "type":        "kaggle",
        "kaggle_slug": "ismailpromus/skin-diseases-image-dataset",
        "size_mb":     1200,
        "images":      19000,
        "classes":     23,
        "description": "Broad clinical photos. Covers eczema, psoriasis, etc.",
        "folder":      "downloads/skin19k",
        "priority":    3,
    },

    "dermnet": {
        "name":        "DermNet — 23 Skin Disease Classes",
        "type":        "kaggle",
        "kaggle_slug": "shubhamgoel27/dermnet",
        "size_mb":     1100,
        "images":      15557,
        "classes":     23,
        "description": "DermNet NZ clinical photos. Good class variety.",
        "folder":      "downloads/dermnet",
        "priority":    4,
    },

    "isic_labelled": {
        "name":        "ISIC Labelled (8-class, ready-to-use)",
        "type":        "kaggle",
        "kaggle_slug": "riyaelizashaju/isic-skin-disease-image-dataset-labelled",
        "size_mb":     3200,
        "images":      25331,
        "classes":     8,
        "description": "Pre-labelled ISIC data. Easy drop-in, no CSV needed.",
        "folder":      "downloads/isic_labelled",
        "priority":    5,
    },

    "pad": {
        "name":        "PAD-UFES-20 — Smartphone Skin Photos",
        "type":        "direct",
        "urls": [
            # Mendeley direct download (may require browser)
            "https://data.mendeley.com/public-files/datasets/zr7vgbcyr2/files/3b1fc7b0-b2c7-4e20-af38-9eef85d6e487/file_downloaded",
        ],
        "kaggle_fallback": "mahdavi1202/skin-cancer",
        "size_mb":     600,
        "images":      2298,
        "classes":     6,
        "description": "Real smartphone photos at varied angles. Critical for app accuracy.",
        "folder":      "downloads/pad_ufes",
        "priority":    6,
    },

    "fitzpatrick": {
        "name":        "Fitzpatrick 17k — Diverse Skin Tones",
        "type":        "github_csv",
        "csv_url":     "https://raw.githubusercontent.com/mattgroh/fitzpatrick17k/main/fitzpatrick17k.csv",
        "size_mb":     4000,
        "images":      16577,
        "classes":     114,
        "description": "Only dataset with Fitzpatrick skin type labels (I–VI). Prevents bias.",
        "folder":      "downloads/fitzpatrick",
        "priority":    7,
        "note":        "Downloads images from source URLs in CSV (slower)",
    },

    "isic2020": {
        "name":        "ISIC 2020 — Melanoma Classification",
        "type":        "kaggle",
        "kaggle_slug": "cdeotte/jpeg-melanoma-256x256",
        "size_mb":     5500,
        "images":      33126,
        "classes":     2,
        "description": "Large melanoma vs benign. Best for cancer detection accuracy.",
        "folder":      "downloads/isic2020",
        "priority":    8,
    },
}

DOWNLOAD_LOG = "downloads/download_log.json"


# ─────────────────────────────────────────────────────────────
#  KAGGLE SETUP CHECK
# ─────────────────────────────────────────────────────────────
def check_kaggle_setup() -> bool:
    """Verify Kaggle API credentials are in place."""
    kaggle_paths = [
        Path.home() / ".kaggle" / "kaggle.json",
        Path(os.environ.get("KAGGLE_CONFIG_DIR", "")) / "kaggle.json",
    ]
    for p in kaggle_paths:
        if p.exists():
            # Check permissions (should be 600 on Unix)
            if os.name != "nt":
                mode = oct(p.stat().st_mode)[-3:]
                if mode != "600":
                    os.chmod(p, 0o600)
                    print(f"  🔒 Fixed permissions on {p}")
            print(f"  ✅ Kaggle credentials found: {p}")
            return True

    # Check environment variables
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        print("  ✅ Kaggle credentials found in environment variables")
        return True

    return False


def setup_kaggle_guide():
    """Print step-by-step Kaggle setup instructions."""
    print("""
  ╔══════════════════════════════════════════════════════════╗
  ║         Kaggle API Setup (one-time, 2 minutes)          ║
  ╚══════════════════════════════════════════════════════════╝

  1. Open: https://www.kaggle.com/settings/account

  2. Scroll to "API" section → click "Create New Token"
     → This downloads: kaggle.json

  3. Place the file here:
     • Linux/Mac : ~/.kaggle/kaggle.json
     • Windows   : C:\\Users\\<YourName>\\.kaggle\\kaggle.json

  4. Run this script again!

  ─── OR set environment variables ────────────────────────
  Windows (PowerShell):
    $env:KAGGLE_USERNAME = "your_username"
    $env:KAGGLE_KEY      = "your_api_key"

  Linux/Mac:
    export KAGGLE_USERNAME="your_username"
    export KAGGLE_KEY="your_api_key"
  ──────────────────────────────────────────────────────────
""")


def install_kaggle() -> bool:
    """Auto-install kaggle package if missing."""
    try:
        import kaggle
        return True
    except ImportError:
        print("  📦 Installing kaggle package...")
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "kaggle", "-q"],
            capture_output=True
        )
        if result.returncode == 0:
            print("  ✅ kaggle installed")
            return True
        print("  ❌ Failed to install kaggle. Run: pip install kaggle")
        return False


# ─────────────────────────────────────────────────────────────
#  PROGRESS BAR DOWNLOADER
# ─────────────────────────────────────────────────────────────
def download_file(url: str, dest_path: str, description: str = "") -> bool:
    """Download a file with a progress bar."""
    if not REQUESTS_OK:
        print("  ❌ requests/tqdm not installed. Run: pip install requests tqdm")
        return False

    try:
        response = requests.get(url, stream=True, timeout=30,
                                 headers={"User-Agent": "Mozilla/5.0"})
        response.raise_for_status()

        total = int(response.headers.get("content-length", 0))
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)

        desc = description or Path(dest_path).name
        with open(dest_path, "wb") as f, tqdm(
            desc=f"  ⬇️  {desc}",
            total=total,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                bar.update(len(chunk))
        return True

    except Exception as e:
        print(f"  ❌ Download failed: {e}")
        return False


# ─────────────────────────────────────────────────────────────
#  EXTRACT ZIP / TAR
# ─────────────────────────────────────────────────────────────
def extract_archive(archive_path: str, dest_dir: str) -> bool:
    """Extract zip or tar archive."""
    print(f"  📦 Extracting {Path(archive_path).name}...")
    os.makedirs(dest_dir, exist_ok=True)

    try:
        if archive_path.endswith(".zip"):
            with zipfile.ZipFile(archive_path, "r") as z:
                z.extractall(dest_dir)
        elif archive_path.endswith((".tar.gz", ".tgz")):
            import tarfile
            with tarfile.open(archive_path, "r:gz") as t:
                t.extractall(dest_dir)
        elif archive_path.endswith(".tar"):
            import tarfile
            with tarfile.open(archive_path, "r") as t:
                t.extractall(dest_dir)
        else:
            print(f"  ⚠️  Unknown archive format: {archive_path}")
            return False

        print(f"  ✅ Extracted to: {dest_dir}")
        return True

    except Exception as e:
        print(f"  ❌ Extraction failed: {e}")
        return False


# ─────────────────────────────────────────────────────────────
#  LOAD / SAVE DOWNLOAD LOG
# ─────────────────────────────────────────────────────────────
def load_log() -> dict:
    os.makedirs("downloads", exist_ok=True)
    if os.path.exists(DOWNLOAD_LOG):
        with open(DOWNLOAD_LOG) as f:
            return json.load(f)
    return {}


def save_log(log: dict):
    os.makedirs("downloads", exist_ok=True)
    with open(DOWNLOAD_LOG, "w") as f:
        json.dump(log, f, indent=2)


def mark_done(key: str, folder: str):
    log = load_log()
    log[key] = {
        "status":      "completed",
        "folder":      folder,
        "downloaded_at": datetime.now().isoformat(),
    }
    save_log(log)


def is_done(key: str) -> bool:
    log = load_log()
    return log.get(key, {}).get("status") == "completed"


# ─────────────────────────────────────────────────────────────
#  KAGGLE DOWNLOADER
# ─────────────────────────────────────────────────────────────
def download_kaggle(key: str, info: dict, force: bool = False) -> bool:
    """Download a dataset from Kaggle using the API."""
    folder = info["folder"]

    if not force and is_done(key):
        print(f"  ✓  {info['name']} already downloaded. (use --force to re-download)")
        return True

    if not install_kaggle():
        return False

    if not check_kaggle_setup():
        setup_kaggle_guide()
        return False

    os.makedirs(folder, exist_ok=True)
    slug = info["kaggle_slug"]

    print(f"\n  ⬇️  Downloading: {info['name']}")
    print(f"      Kaggle slug : {slug}")
    print(f"      Size        : ~{info['size_mb']} MB")
    print(f"      Destination : {folder}/")

    cmd = [
        sys.executable, "-m", "kaggle", "datasets", "download",
        "-d", slug,
        "-p", folder,
        "--unzip",
    ]

    try:
        result = subprocess.run(cmd, capture_output=False, text=True)
        if result.returncode == 0:
            print(f"  ✅ Downloaded: {info['name']}")
            mark_done(key, folder)
            return True
        else:
            print(f"  ❌ Kaggle download failed for: {slug}")
            return False

    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False


# ─────────────────────────────────────────────────────────────
#  PAD-UFES-20 DOWNLOADER (Mendeley / Kaggle fallback)
# ─────────────────────────────────────────────────────────────
def download_pad(force: bool = False) -> bool:
    info = DATASETS["pad"]
    folder = info["folder"]

    if not force and is_done("pad"):
        print(f"  ✓  PAD-UFES-20 already downloaded.")
        return True

    print(f"\n  ⬇️  Downloading: {info['name']}")
    print(f"      Source: Mendeley → Kaggle fallback")
    print(f"      Size  : ~{info['size_mb']} MB")

    os.makedirs(folder, exist_ok=True)

    # Try Kaggle fallback (most reliable)
    print("  💡 Using Kaggle mirror (most reliable for PAD-UFES-20)...")
    result = download_kaggle("pad", {
        **info,
        "kaggle_slug": info["kaggle_fallback"],
        "name": "PAD-UFES-20 (Kaggle mirror)",
    }, force)

    if result:
        mark_done("pad", folder)
        return True

    # Fallback: print manual instructions
    print("""
  ⚠️  Auto-download failed for PAD-UFES-20.
  Manual download (2 minutes):

  1. Open: https://data.mendeley.com/datasets/zr7vgbcyr2/1
  2. Click "Download All" (opens a zip)
  3. Extract to: downloads/pad_ufes/
  4. Run:  python organize_dataset.py --source downloads/pad_ufes --dataset pad
  """)
    return False


# ─────────────────────────────────────────────────────────────
#  FITZPATRICK 17K DOWNLOADER
#  Downloads the CSV from GitHub, then fetches each image URL
# ─────────────────────────────────────────────────────────────
def download_fitzpatrick(max_images: int = 5000, force: bool = False) -> bool:
    """
    Downloads Fitzpatrick 17k images from their source URLs.
    max_images limits download for speed (5000 = good sample).
    Full dataset = 16,577 images.
    """
    info   = DATASETS["fitzpatrick"]
    folder = info["folder"]

    if not force and is_done("fitzpatrick"):
        print(f"  ✓  Fitzpatrick 17k already downloaded.")
        return True

    if not REQUESTS_OK:
        print("  ❌ requests/tqdm required. Run: pip install requests tqdm")
        return False

    print(f"\n  ⬇️  Downloading: {info['name']}")
    print(f"      Max images  : {max_images} (of 16,577 total)")
    print(f"      Note        : Downloads from original source URLs in CSV")

    import csv
    os.makedirs(folder, exist_ok=True)

    # Download the CSV metadata
    csv_path = os.path.join(folder, "fitzpatrick17k.csv")
    print(f"\n  📄 Fetching metadata CSV...")
    if not download_file(info["csv_url"], csv_path, "fitzpatrick17k.csv"):
        print("  ❌ Could not download Fitzpatrick CSV")
        return False

    # Read CSV and group by disease label
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    print(f"  📊 Found {len(rows):,} entries in CSV")

    # Map label column (may vary: 'label', 'three_partition_label', etc.)
    cols = list(rows[0].keys()) if rows else []
    label_col = next((c for c in cols if "label" in c.lower() and "fitzpatrick" not in c.lower()), None)
    url_col   = next((c for c in cols if "url" in c.lower()), None)
    id_col    = next((c for c in cols if c.lower() in {"md5hash", "id", "image_id", "hasher"}), None)

    if not url_col:
        print(f"  ❌ Cannot find URL column. Found columns: {cols}")
        return False

    print(f"  🔑 Label col: '{label_col}' | URL col: '{url_col}'")

    # Limit and shuffle
    random.shuffle(rows)
    rows = rows[:max_images]

    # Download images grouped by disease
    success, failed = 0, 0
    disease_map = _load_fitzpatrick_disease_map()

    for i, row in enumerate(rows):
        url        = row.get(url_col, "").strip()
        label_raw  = row.get(label_col, "unknown").strip().lower() if label_col else "unknown"
        img_id     = row.get(id_col, f"img_{i:05d}").strip() if id_col else f"img_{i:05d}"

        if not url:
            failed += 1
            continue

        # Map label to standard disease
        disease = _map_fitzpatrick_label(label_raw, disease_map)
        if not disease:
            disease = "Normal"

        img_dir = os.path.join(folder, disease)
        os.makedirs(img_dir, exist_ok=True)
        img_path = os.path.join(img_dir, f"{img_id}.jpg")

        if os.path.exists(img_path):
            success += 1
            continue

        try:
            resp = requests.get(url, timeout=15,
                                headers={"User-Agent": "Mozilla/5.0"},
                                stream=True)
            if resp.status_code == 200:
                with open(img_path, "wb") as f:
                    for chunk in resp.iter_content(8192):
                        f.write(chunk)
                success += 1
            else:
                failed += 1
        except Exception:
            failed += 1

        # Progress update every 100 images
        if (i + 1) % 100 == 0:
            pct = (i + 1) / len(rows) * 100
            print(f"  ⬇️  Progress: {i+1}/{len(rows)} ({pct:.0f}%)  "
                  f"✅{success}  ❌{failed}", end="\r")
        time.sleep(0.05)  # Be polite to servers

    print(f"\n  ✅ Fitzpatrick: {success} downloaded, {failed} failed")
    mark_done("fitzpatrick", folder)
    return True


def _load_fitzpatrick_disease_map() -> dict:
    """Return a mapping of Fitzpatrick label patterns → standard disease names."""
    return {
        "acne":                 "Acne",
        "rosacea":              "Rosacea",
        "eczema":               "Eczema",
        "atopic":               "Eczema",
        "dermatitis":           "Eczema",
        "psoriasis":            "Psoriasis",
        "contact":              "Contact_Dermatitis",
        "seborrheic":           "Seborrheic_Dermatitis",
        "urticaria":            "Hives",
        "hives":                "Hives",
        "vitiligo":             "Vitiligo",
        "alopecia":             "Alopecia",
        "tinea":                "Ringworm",
        "ringworm":             "Ringworm",
        "fungal":               "Ringworm",
        "scabies":              "Scabies",
        "herpes zoster":        "Shingles",
        "shingles":             "Shingles",
        "varicella":            "Chickenpox",
        "chickenpox":           "Chickenpox",
        "impetigo":             "Impetigo",
        "cellulitis":           "Cellulitis",
        "wart":                 "Warts",
        "molluscum":            "Warts",
        "melanoma":             "Melanoma",
        "basal cell":           "Basal_Cell_Carcinoma",
        "squamous":             "Melanoma",
        "normal":               "Normal",
        "nevus":                "Normal",
    }


def _map_fitzpatrick_label(label: str, disease_map: dict) -> str | None:
    label = label.lower().strip()
    for key, disease in disease_map.items():
        if key in label:
            return disease
    return None


# ─────────────────────────────────────────────────────────────
#  STATUS CHECKER
# ─────────────────────────────────────────────────────────────
def show_status():
    log = load_log()
    print(f"\n{'='*65}")
    print(f"  DOWNLOAD STATUS")
    print(f"{'='*65}\n")
    print(f"  {'Dataset':<32} {'Size':>7} {'Images':>8}  Status")
    print(f"  {'─'*58}")

    total_dl = 0
    for key, info in sorted(DATASETS.items(),
                             key=lambda x: x[1]["priority"]):
        done    = is_done(key)
        folder  = info["folder"]
        status  = "✅ Done" if done else "⬜ Not downloaded"

        # Count images if downloaded
        img_count = 0
        if done and os.path.exists(folder):
            for _, _, files in os.walk(folder):
                img_count += sum(
                    1 for f in files
                    if Path(f).suffix.lower() in
                    {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
                )
            total_dl += img_count

        count_str = f"{img_count:,}" if img_count > 0 else f"~{info['images']:,}"
        print(f"  {info['name']:<32} {info['size_mb']:>5}MB "
              f"{count_str:>8}  {status}")

    print(f"  {'─'*58}")
    if total_dl:
        print(f"  Total downloaded images: {total_dl:,}")
    print()

    kaggle_ok = check_kaggle_setup()
    if not kaggle_ok:
        print("  ⚠️  Kaggle API not configured yet.")
        print("      Run: python download_datasets.py --setup")
    else:
        print("  ✅ Kaggle API ready")

    print(f"\n{'='*65}\n")


# ─────────────────────────────────────────────────────────────
#  AUTO-ORGANIZE AFTER DOWNLOAD
# ─────────────────────────────────────────────────────────────
def auto_organize():
    """Run organize_dataset.py for each downloaded dataset."""
    log = load_log()
    organize_script = "organize_dataset.py"

    if not os.path.exists(organize_script):
        print("  ⚠️  organize_dataset.py not found. Skipping auto-organize.")
        print("      Place organize_dataset.py in the same folder.")
        return

    dataset_type_map = {
        "ham10000":      "ham10000",
        "isic2019":      "isic",
        "isic2020":      "isic",
        "isic_labelled": "isic",
        "skin19k":       "auto",
        "dermnet":       "dermnet",
        "pad":           "pad",
        "fitzpatrick":   "fitzpatrick",
    }

    print(f"\n{'='*55}")
    print(f"  AUTO-ORGANIZING DOWNLOADED DATASETS")
    print(f"{'='*55}\n")

    for key, entry in log.items():
        if entry.get("status") != "completed":
            continue
        folder = entry.get("folder", "")
        if not folder or not os.path.exists(folder):
            continue

        dtype = dataset_type_map.get(key, "auto")
        print(f"  📁 Organizing {key} from {folder}...")

        cmd = [
            sys.executable, organize_script,
            "--source", folder,
            "--dataset", dtype,
        ]
        result = subprocess.run(cmd, capture_output=False, text=True)
        if result.returncode != 0:
            print(f"  ⚠️  organize failed for {key}")

    # Final verify
    print(f"\n  Running dataset verification...")
    subprocess.run([sys.executable, organize_script, "--verify"])


# ─────────────────────────────────────────────────────────────
#  INSTALL DEPENDENCIES
# ─────────────────────────────────────────────────────────────
def install_dependencies():
    packages = ["kaggle", "requests", "tqdm", "Pillow"]
    print("  📦 Installing required packages...")
    for pkg in packages:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", pkg, "-q"],
            capture_output=True
        )
        icon = "✅" if result.returncode == 0 else "❌"
        print(f"  {icon} {pkg}")
    print()


# ─────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="DermaScan — Auto Dataset Downloader",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Quick start:
  python download_datasets.py --install     # install packages
  python download_datasets.py --setup       # show Kaggle setup guide
  python download_datasets.py --status      # check what's downloaded
  python download_datasets.py --all         # download everything
  python download_datasets.py --all --organize  # download + organize
        """
    )

    # Actions
    parser.add_argument("--all",         action="store_true", help="Download all datasets")
    parser.add_argument("--ham10000",    action="store_true", help="Download HAM10000")
    parser.add_argument("--isic",        action="store_true", help="Download ISIC 2019 + labelled")
    parser.add_argument("--isic2020",    action="store_true", help="Download ISIC 2020 (large)")
    parser.add_argument("--dermnet",     action="store_true", help="Download DermNet")
    parser.add_argument("--skin19k",     action="store_true", help="Download Skin Disease 19k")
    parser.add_argument("--pad",         action="store_true", help="Download PAD-UFES-20")
    parser.add_argument("--fitzpatrick", action="store_true", help="Download Fitzpatrick 17k")

    # Options
    parser.add_argument("--organize",    action="store_true", help="Auto-run organize_dataset.py after download")
    parser.add_argument("--force",       action="store_true", help="Re-download even if already done")
    parser.add_argument("--status",      action="store_true", help="Show download status")
    parser.add_argument("--setup",       action="store_true", help="Show Kaggle setup guide")
    parser.add_argument("--install",     action="store_true", help="Install required packages")
    parser.add_argument("--fitz-max",    type=int, default=5000,
                        help="Max images to download from Fitzpatrick 17k (default: 5000)")

    args = parser.parse_args()

    print("=" * 65)
    print("  DermaScan AI — Auto Dataset Downloader")
    print("=" * 65)

    if not any(vars(args).values()):
        parser.print_help()
        return

    if args.install:
        install_dependencies()

    if args.setup:
        setup_kaggle_guide()
        return

    if args.status:
        show_status()
        return

    # Determine which datasets to download
    to_download = []
    if args.all:
        to_download = list(DATASETS.keys())
    else:
        if args.ham10000:    to_download.append("ham10000")
        if args.isic:        to_download += ["isic2019", "isic_labelled"]
        if args.isic2020:    to_download.append("isic2020")
        if args.dermnet:     to_download.append("dermnet")
        if args.skin19k:     to_download.append("skin19k")
        if args.pad:         to_download.append("pad")
        if args.fitzpatrick: to_download.append("fitzpatrick")

    if not to_download:
        print("  Nothing selected. Use --all or a specific dataset flag.")
        return

    # Sort by priority
    to_download = sorted(to_download,
                         key=lambda k: DATASETS[k]["priority"])

    # Print plan
    total_mb = sum(DATASETS[k]["size_mb"] for k in to_download)
    total_imgs = sum(DATASETS[k]["images"] for k in to_download)
    print(f"\n  📋 Download plan ({len(to_download)} datasets):")
    for k in to_download:
        info = DATASETS[k]
        done_str = " (already done)" if is_done(k) and not args.force else ""
        print(f"     {info['priority']}. {info['name']:<35} "
              f"~{info['size_mb']}MB{done_str}")
    print(f"\n  Total size  : ~{total_mb / 1024:.1f} GB")
    print(f"  Total images: ~{total_imgs:,}")
    print()

    # Check disk space (rough)
    try:
        import shutil as sh
        free_bytes = sh.disk_usage(".").free
        free_gb = free_bytes / (1024 ** 3)
        needed_gb = total_mb / 1024
        if free_gb < needed_gb * 1.5:
            print(f"  ⚠️  Low disk space: {free_gb:.1f} GB free, "
                  f"~{needed_gb:.1f} GB needed")
            print("      Consider downloading datasets one at a time.\n")
        else:
            print(f"  💾 Disk space: {free_gb:.1f} GB free — OK\n")
    except Exception:
        pass

    # Confirm
    if args.all and not args.force:
        ans = input("  Start download? (y/n): ").strip().lower()
        if ans != "y":
            print("  Cancelled.")
            return

    # ── Download loop ────────────────────────────────────────
    results = {}
    start_time = time.time()

    for key in to_download:
        info = DATASETS[key]
        print(f"\n{'─'*65}")
        print(f"  [{to_download.index(key)+1}/{len(to_download)}] "
              f"{info['name']}")
        print(f"  {info['description']}")
        print(f"{'─'*65}")

        if key == "pad":
            ok = download_pad(force=args.force)
        elif key == "fitzpatrick":
            ok = download_fitzpatrick(max_images=args.fitz_max, force=args.force)
        else:
            ok = download_kaggle(key, info, force=args.force)

        results[key] = ok

    # ── Summary ──────────────────────────────────────────────
    elapsed = int(time.time() - start_time)
    mins, secs = divmod(elapsed, 60)

    print(f"\n{'='*65}")
    print(f"  DOWNLOAD SUMMARY  ({mins}m {secs}s)")
    print(f"{'='*65}")
    for key, ok in results.items():
        icon = "✅" if ok else "❌"
        print(f"  {icon}  {DATASETS[key]['name']}")

    failed = [k for k, ok in results.items() if not ok]
    if failed:
        print(f"\n  ⚠️  {len(failed)} failed. Try again with --force flag")
        print(f"      or check your Kaggle credentials.")

    # ── Auto-organize ─────────────────────────────────────────
    if args.organize:
        print()
        auto_organize()

    print(f"\n  👉 Next steps:")
    if not args.organize:
        print(f"     python organize_dataset.py --source downloads/<folder> --dataset <type>")
        print(f"     python organize_dataset.py --verify")
    print(f"     python augment_data.py --dataset dataset --count 20")
    print(f"     python train_model_v3.py")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    main()