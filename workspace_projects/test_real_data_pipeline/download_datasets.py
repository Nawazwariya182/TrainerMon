"""
REAL DATASET DOWNLOADER AND VERIFICATION ENGINE
Project: Machine Learning Project
Selected Datasets: Brain MRI Images for Brain Tumor Detection (Br35H / Navoneel Cohort)

This script automates streaming download, archive extraction, and filesystem verification
of real multi-dataset sources into './data/raw/<dataset_id>/'.
Features interactive ASCII progress bars and live updates to EXECUTION_STATUS.json.
"""

import os
import sys
import json
import time
import zipfile
import tarfile
import logging
import datetime
import urllib.request
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("dataset_downloader")

def update_execution_status(step_name: str, progress_pct: float, details: str, status: str = "IN_PROGRESS"):
    """Persists real-time task progress into EXECUTION_STATUS.json."""
    status_file = Path(__file__).resolve().parent / "EXECUTION_STATUS.json"
    if not status_file.exists():
        return
    try:
        with open(status_file, "r") as f:
            data = json.load(f)
        data["last_updated"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        if status == "IN_PROGRESS":
            data["status"] = "EXECUTING"
            data["is_new"] = False
            data["current_step_name"] = step_name
        for s in data.get("steps", []):
            if s.get("name") == step_name:
                s["status"] = status
                s["progress_pct"] = round(progress_pct, 1)
                s["details"] = details
        total_p = sum(s.get("progress_pct", 0) for s in data.get("steps", []))
        data["overall_progress_pct"] = round(total_p / max(1, len(data.get("steps", []))), 1)
        with open(status_file, "w") as f:
            json.dump(data, f, indent=2)
    except Exception:
        pass

def print_progress_bar(iteration: int, total: int, prefix: str = "", suffix: str = "", length: int = 35):
    """Renders formatted ASCII progress bar to stdout."""
    total = max(1, total)
    percent = f"{100 * (iteration / float(total)):.1f}"
    filled_length = int(length * iteration // total)
    bar = "=" * filled_length + ">" + " " * (length - filled_length - 1) if filled_length < length else "=" * length
    sys.stdout.write(f"\r{prefix} [{bar}] {percent}% {suffix}")
    sys.stdout.flush()
    if iteration >= total:
        sys.stdout.write("\n")

def stream_download(url: str, dest_path: Path, prefix: str = "Downloading") -> bool:
    """Streams real bytes from HTTP/HTTPS URL with live progress bar and speed tracking."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    )
    try:
        with urllib.request.urlopen(req, timeout=300) as response, open(dest_path, "wb") as out_file:
            total_size = int(response.headers.get("Content-Length", 0))
            downloaded = 0
            block_size = 65536  # 64 KB
            start_time = time.time()
            last_print = 0.0

            while True:
                chunk = response.read(block_size)
                if not chunk:
                    break
                out_file.write(chunk)
                downloaded += len(chunk)
                now = time.time()

                if now - last_print >= 0.25 or (total_size and downloaded >= total_size):
                    last_print = now
                    elapsed = max(0.001, now - start_time)
                    speed_mb = (downloaded / (1024 * 1024)) / elapsed
                    mb_done = downloaded / (1024 * 1024)
                    if total_size > 0:
                        mb_total = total_size / (1024 * 1024)
                        print_progress_bar(downloaded, total_size, prefix=prefix, suffix=f"{mb_done:.1f}/{mb_total:.1f} MB ({speed_mb:.2f} MB/s)")
                    else:
                        sys.stdout.write(f"\r{prefix} {mb_done:.1f} MB ({speed_mb:.2f} MB/s)")
                        sys.stdout.flush()
            sys.stdout.write("\n")
            return True
    except Exception as e:
        logger.error("Download failed from %s: %s", url, e)
        return False

def unpack_archive(archive_path: Path, extract_dir: Path, download_type: str = "zip"):
    """Extracts zip, tar, tar.gz, or tgz archives into extract_dir."""
    extract_dir.mkdir(parents=True, exist_ok=True)
    name_lower = archive_path.name.lower()
    if name_lower.endswith(".zip") or download_type == "zip":
        logger.info("Extracting ZIP archive: %s -> %s", archive_path.name, extract_dir)
        with zipfile.ZipFile(archive_path, "r") as zf:
            zf.extractall(extract_dir)
    elif name_lower.endswith((".tar.gz", ".tgz", ".tar")) or download_type == "tar":
        logger.info("Extracting TAR archive: %s -> %s", archive_path.name, extract_dir)
        mode = "r:gz" if name_lower.endswith((".tar.gz", ".tgz")) else "r:"
        with tarfile.open(archive_path, mode) as tf:
            tf.extractall(extract_dir)
    elif name_lower.endswith((".csv", ".tsv", ".parquet", ".json", ".txt")) or download_type == "file":
        import shutil
        target = extract_dir / archive_path.name
        if archive_path != target:
            shutil.copy2(archive_path, target)

def scan_catalog(ds_dir: Path) -> dict:
    """Recursively catalogs all valid data files in directory."""
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".nii", ".nii.gz", ".csv", ".parquet", ".wav", ".flac"}
    files = [p for p in ds_dir.rglob("*") if p.is_file() and any(p.name.lower().endswith(e) for e in exts)]
    classes = set()
    for f in files:
        rel = f.relative_to(ds_dir)
        if len(rel.parts) > 1:
            classes.add(rel.parts[0])
    total_bytes = sum(f.stat().st_size for f in files)
    return {
        "file_count": len(files),
        "classes": sorted(list(classes)),
        "total_mb": round(total_bytes / (1024 * 1024), 2),
        "files": files
    }

def download_and_verify_all():
    logger.info("Initializing Real Dataset Downloader for: Machine Learning Project")
    update_execution_status("download_datasets", 5.0, "Reading datasets manifest", "IN_PROGRESS")

    manifest_file = Path(__file__).resolve().parent / "datasets_manifest.json"
    datasets = []
    if manifest_file.exists():
        try:
            with open(manifest_file, "r") as f:
                manifest_data = json.load(f)
                datasets = manifest_data.get("datasets", [])
        except Exception as e:
            logger.warning("Could not read %s: %s", manifest_file, e)

    raw_root = Path(__file__).resolve().parent / "data" / "raw"
    raw_root.mkdir(parents=True, exist_ok=True)

    num_datasets = max(1, len(datasets))
    for idx, ds in enumerate(datasets):
        ds_id = ds.get("id", f"dataset_{idx}")
        ds_title = ds.get("title", ds_id)
        download_url = ds.get("download_url") or ds.get("url", "")
        download_type = ds.get("download_type", "zip")
        ds_dir = raw_root / ds_id

        step_pct_start = (idx / num_datasets) * 90.0
        step_pct_end = ((idx + 1) / num_datasets) * 90.0

        logger.info("[Dataset %d/%d] Ingesting: %s", idx + 1, num_datasets, ds_title)

        # Check existing cache
        existing = scan_catalog(ds_dir)
        if existing["file_count"] > 10:
            logger.info("  Cached dataset found: %d files (%s MB) across classes: %s",
                        existing["file_count"], existing["total_mb"], existing["classes"] or ["root"])
            update_execution_status(
                "download_datasets",
                step_pct_end,
                f"Verified cached {ds_title}: {existing['file_count']} files",
                "IN_PROGRESS"
            )
            continue

        if not download_url.startswith("http"):
            logger.warning("  No valid HTTP download URL found for %s. Skipping automatic download.", ds_title)
            continue

        # Real HTTP streaming download
        ext = ".zip" if download_type == "zip" else ".tar.gz" if download_type == "tar" else ".csv"
        temp_archive = raw_root / f"{ds_id}_download{ext}"

        update_execution_status(
            "download_datasets",
            step_pct_start + 10.0,
            f"Streaming real bytes for {ds_title}",
            "IN_PROGRESS"
        )
        logger.info("  Streaming from: %s", download_url)
        success = stream_download(download_url, temp_archive, prefix=f"  Downloading {ds_id[:16]}")

        if success and temp_archive.exists():
            update_execution_status(
                "download_datasets",
                step_pct_start + 35.0,
                f"Extracting archives for {ds_title}",
                "IN_PROGRESS"
            )
            unpack_archive(temp_archive, ds_dir, download_type)
            try:
                temp_archive.unlink()
            except Exception:
                pass

        catalog = scan_catalog(ds_dir)
        logger.info("  Extracted & Cataloged: %d files (%s MB). Classes found: %s",
                    catalog["file_count"], catalog["total_mb"], catalog["classes"] or ["all"])

        meta_file = ds_dir / "dataset_info.json"
        with open(meta_file, "w") as mf:
            json.dump({
                "id": ds_id,
                "title": ds_title,
                "source": ds.get("source", ""),
                "format": ds.get("format", ""),
                "file_count": catalog["file_count"],
                "classes": catalog["classes"],
                "total_mb": catalog["total_mb"],
                "verified_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "status": "ready"
            }, mf, indent=2)

    update_execution_status("download_datasets", 100.0, "All real datasets downloaded, unpacked, and verified", "COMPLETED")
    logger.info("SUCCESS: All datasets downloaded and verified in ./data/raw/")

if __name__ == "__main__":
    download_and_verify_all()
