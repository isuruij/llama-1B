"""
Download polyglots/MADLAD_CulturaX_cleaned from HuggingFace and save as
a plain .txt file (one document per line) for the CPT script.

Usage:
    python prepare_sinhala_data.py --output_dir ./sinhala_data
"""

import os
import argparse
from datasets import load_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="./sinhala_data", type=str)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, "sinhala_corpus.txt")

    print("Loading polyglots/MADLAD_CulturaX_cleaned ...")
    ds = load_dataset("polyglots/MADLAD_CulturaX_cleaned", split="train", streaming=True)

    print(f"Writing to {out_path} ...")
    count = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for example in ds:
            text = example["text"].strip()
            if text:
                f.write(text + "\n")
                count += 1
            if count % 500_000 == 0 and count > 0:
                print(f"  Written {count:,} lines ...")

    print(f"Done. Total lines: {count:,}")
    print(f"Saved to: {out_path}")


if __name__ == "__main__":
    main()
