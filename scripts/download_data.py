"""
scripts/download_data.py
=========================
Automated data download for ESP Predictive Maintenance project.

Supports:
  - pump_sensor  : Kaggle Pump Sensor Dataset (requires Kaggle API key)
  - cmapss       : NASA CMAPSS Turbofan Dataset (direct download)
  - synthetic    : Generate synthetic ESP data locally (no download needed)

Setup for Kaggle:
  1. Go to kaggle.com → Account → API → Create New Token
  2. Move kaggle.json to ~/.kaggle/kaggle.json
  3. chmod 600 ~/.kaggle/kaggle.json

Usage:
  python scripts/download_data.py --dataset pump_sensor
  python scripts/download_data.py --dataset cmapss
  python scripts/download_data.py --dataset synthetic --n_wells 50
  python scripts/download_data.py --dataset all
"""

import argparse
import os
import sys
import zipfile
import subprocess
import urllib.request
from pathlib import Path


RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)


def download_pump_sensor():
    """Download Kaggle Pump Sensor Dataset."""
    print("\n── Pump Sensor Dataset (Kaggle) ──────────────────────────────")
    try:
        import kaggle
    except ImportError:
        print("ERROR: kaggle package not installed. Run: pip install kaggle")
        sys.exit(1)

    out_dir = RAW_DIR / "pump_sensor"
    out_dir.mkdir(exist_ok=True)

    print("Downloading from Kaggle (nphantawee/pump-sensor-data)...")
    subprocess.run([
        "kaggle", "datasets", "download",
        "-d", "nphantawee/pump-sensor-data",
        "-p", str(out_dir),
    ], check=True)

    # Unzip
    zip_path = out_dir / "pump-sensor-data.zip"
    if zip_path.exists():
        print("Extracting...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(out_dir)
        zip_path.unlink()

    # Rename for consistency
    csv_files = list(out_dir.glob("*.csv"))
    if csv_files:
        src = csv_files[0]
        dst = RAW_DIR / "pump_sensor.csv"
        src.rename(dst)
        print(f"✓ Saved to {dst}")
    else:
        print("WARNING: No CSV found after extraction. Check the download.")


def download_cmapss():
    """
    Download NASA CMAPSS dataset.

    The dataset is publicly available from NASA's prognostics data repository.
    If the direct URL changes, visit:
    https://data.nasa.gov/dataset/CMAPSS-Jet-Engine-Simulated-Data
    """
    print("\n── NASA CMAPSS Dataset ──────────────────────────────────────")
    out_dir = RAW_DIR / "cmapss"
    out_dir.mkdir(exist_ok=True)

    # Mirror URL (NASA dataset is hosted on various academic mirrors)
    # Primary: NASA Prognostics Center
    url = "https://data.nasa.gov/api/views/ff5v-kuh6/files/CMAPSSData.zip"
    zip_path = out_dir / "CMAPSSData.zip"

    print(f"Downloading from: {url}")
    try:
        urllib.request.urlretrieve(url, zip_path, _download_progress)
        print()
    except Exception as e:
        print(f"\nPrimary download failed: {e}")
        print("Alternative: Download manually from")
        print("  https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/")
        print("  Then extract to data/raw/cmapss/")
        return

    print("Extracting CMAPSS files...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(out_dir)
    zip_path.unlink()

    # Expected files: train_FD001.txt, test_FD001.txt, RUL_FD001.txt, etc.
    files = list(out_dir.rglob("*.txt"))
    print(f"✓ Extracted {len(files)} files to {out_dir}")
    for f in sorted(files)[:8]:
        print(f"   {f.name}")


def download_synthetic(n_wells: int = 50, timesteps: int = 5000, seed: int = 42):
    """Generate synthetic ESP data locally."""
    print("\n── Synthetic ESP Data (no download needed) ──────────────")
    # Add parent directory to path to import from src
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from src.data.synthetic_generator import generate_esp_dataset

    print(f"Generating {n_wells} simulated wells × {timesteps} timesteps ...")
    df = generate_esp_dataset(
        n_wells=n_wells,
        timesteps_per_well=timesteps,
        failure_prob=0.6,
        random_seed=seed,
    )
    out_path = RAW_DIR / "synthetic_esp.csv"
    df.to_csv(out_path, index=False)
    print(f"✓ Generated {len(df):,} rows → {out_path}")
    print(f"  Failure rate: {(df['machine_status']=='BROKEN').mean():.1%}")
    print("  Failure modes:", df["failure_mode"].unique().tolist())


def _download_progress(block_num, block_size, total_size):
    """Simple progress callback for urllib."""
    downloaded = block_num * block_size
    pct = min(downloaded / total_size * 100, 100) if total_size > 0 else 0
    bar = "█" * int(pct // 5) + "░" * (20 - int(pct // 5))
    print(f"\r  [{bar}] {pct:.1f}% ({downloaded/1e6:.1f} MB)", end="", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Download ESP predictive maintenance datasets")
    parser.add_argument(
        "--dataset",
        choices=["pump_sensor", "cmapss", "synthetic", "all"],
        default="synthetic",
        help="Dataset to download (default: synthetic — no API key needed)"
    )
    parser.add_argument("--n_wells", type=int, default=50,
                        help="Number of wells for synthetic dataset")
    parser.add_argument("--timesteps", type=int, default=5000,
                        help="Timesteps per well for synthetic dataset")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"Output directory: {RAW_DIR.resolve()}")

    if args.dataset in ("pump_sensor", "all"):
        download_pump_sensor()
    if args.dataset in ("cmapss", "all"):
        download_cmapss()
    if args.dataset in ("synthetic", "all"):
        download_synthetic(args.n_wells, args.timesteps, args.seed)

    print("\n✓ Done. Next step: open notebooks/01_EDA_and_Domain_Context.ipynb")


if __name__ == "__main__":
    main()
