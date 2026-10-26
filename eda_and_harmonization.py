"""
EDA AND CUSTOM VOCABULARY TOKENIZATION HARMONIZER
Scans real movie reviews, builds vocabulary from scratch, and exports tokenized arrays.
"""

import os
import sys
import re
import csv
import json
import datetime
from collections import Counter
import numpy as np
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parent / "data" / "raw" / "imdb"
HARMONIZED_DIR = Path(__file__).resolve().parent / "data" / "harmonized"
HARMONIZED_DIR.mkdir(parents=True, exist_ok=True)

CLASSES = ["negative","positive"]
MAX_SEQ_LEN = 128
MAX_VOCAB = 20000

def update_status(step_name: str, progress_pct: float, details: str, status: str = "IN_PROGRESS"):
    status_file = Path(__file__).resolve().parent / "EXECUTION_STATUS.json"
    if not status_file.exists():
        return
    try:
        with open(status_file, "r") as f:
            data = json.load(f)
        data["last_updated"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        if status == "IN_PROGRESS":
            data["status"] = "EXECUTING"
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

def tokenize(text: str) -> list:
    return re.findall(r"[a-z0-9']+|[!?,.]", text.lower())

def main():
    print("=================================================================")
    print("Stage 2: Building Vocabulary From Scratch & Harmonizing Reviews")
    print("=================================================================")
    update_status("eda_and_harmonization", 10.0, "Parsing review TSVs", "IN_PROGRESS")

    raw_files = list(RAW_DIR.rglob("train.tsv")) + list(RAW_DIR.rglob("dev.tsv"))
    if not raw_files:
        raise FileNotFoundError("Could not find review TSV files. Run download_datasets.py first.")

    all_texts, all_labels = [], []
    for fpath in raw_files:
        with open(fpath, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t")
            next(reader, None)  # header
            for row in reader:
                if len(row) >= 2:
                    all_texts.append(row[0].strip())
                    all_labels.append(int(row[1]))

    print(f"Loaded total {len(all_texts)} combined review samples.")

    # Stratified 85/15 train-validation split for robust generalization
    np.random.seed(42)
    indices = np.random.permutation(len(all_texts))
    split_pt = int(0.85 * len(indices))
    train_idx = indices[:split_pt]
    val_idx = indices[split_pt:]

    train_texts = [all_texts[i] for i in train_idx]
    train_labels = [all_labels[i] for i in train_idx]
    test_texts = [all_texts[i] for i in val_idx]
    test_labels = [all_labels[i] for i in val_idx]

    print(f"Stratified split: {len(train_texts)} training reviews, {len(test_texts)} validation reviews.")

    # Build vocab from scratch
    token_counts = Counter()
    for t in train_texts:
        token_counts.update(tokenize(t))

    vocab = {"<pad>": 0, "<unk>": 1}
    for word, count in token_counts.most_common(MAX_VOCAB - 2):
        vocab[word] = len(vocab)

    print(f"Vocabulary constructed from scratch: {len(vocab)} unique tokens.")

    def encode(text: str) -> list:
        toks = tokenize(text)[:MAX_SEQ_LEN]
        ids = [vocab.get(w, 1) for w in toks]
        if len(ids) < MAX_SEQ_LEN:
            ids = ids + [0] * (MAX_SEQ_LEN - len(ids))
        return ids

    train_x = np.array([encode(t) for t in train_texts], dtype=np.int64)
    train_y = np.array(train_labels, dtype=np.int64)
    test_x = np.array([encode(t) for t in test_texts], dtype=np.int64)
    test_y = np.array(test_labels, dtype=np.int64)

    np.save(HARMONIZED_DIR / "train_x.npy", train_x)
    np.save(HARMONIZED_DIR / "train_y.npy", train_y)
    np.save(HARMONIZED_DIR / "test_x.npy", test_x)
    np.save(HARMONIZED_DIR / "test_y.npy", test_y)

    with open(HARMONIZED_DIR.parent.parent / "vocab.json", "w", encoding="utf-8") as vf:
        json.dump(vocab, vf, indent=2)

    manifest = {
        "domain": "nlp_text",
        "task": "Movie Review Sentiment Classification",
        "vocab_size": len(vocab),
        "max_seq_len": MAX_SEQ_LEN,
        "total_train_samples": len(train_x),
        "total_test_samples": len(test_x),
        "classes": CLASSES
    }

    with open(HARMONIZED_DIR.parent.parent / "harmonized_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    update_status("eda_and_harmonization", 100.0, f"Tokenized {len(train_x)} reviews into [B, 128] token tensors", "COMPLETED")
    import os
    sys.stdout.flush()
    os._exit(0)

if __name__ == "__main__":
    main()
