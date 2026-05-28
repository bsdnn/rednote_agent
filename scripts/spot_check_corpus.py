"""Randomly sample 10% of each generated corpus for manual review.

Usage: python scripts/spot_check_corpus.py
Prints sampled entries to stdout for human review.
"""
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / "backend" / "data" / "corpus"


def sample(path: Path, fraction: float = 0.1) -> None:
    if not path.exists():
        print(f"SKIP: {path} not found")
        return
    data = json.loads(path.read_text(encoding="utf-8"))
    n = max(3, int(len(data) * fraction))
    random.seed(42)
    samples = random.sample(data, min(n, len(data)))
    print(f"\n=== {path.name} — {n}/{len(data)} samples ===")
    for i, rec in enumerate(samples, 1):
        print(f"\n[{i}] {json.dumps(rec, ensure_ascii=False, indent=2)}")


if __name__ == "__main__":
    sample(CORPUS / "ingredients.json")
    sample(CORPUS / "posts.json")
