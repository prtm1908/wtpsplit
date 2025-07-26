
import random
from pathlib import Path
from typing import List, Tuple

import torch

# Import the ground-truth sentence segmentation logic from evals
try:
    from evals import create_ground_truth_sentences, normalize_text_for_comparison
except ImportError as e:
    raise ImportError("Could not import create_ground_truth_sentences from evals. Make sure the sat package is on PYTHONPATH.") from e


def remove_single_newlines(text: str) -> List[str]:
    """Process raw text to merge lines within paragraphs while preserving paragraph breaks.

    Args:
        text: Raw text containing single and double newlines.

    Returns:
        List of cleaned paragraph strings (single-line per paragraph).
    """
    paragraphs = text.split("\n\n")  # Paragraphs are separated by blank lines.
    cleaned: List[str] = []
    for paragraph in paragraphs:
        if not paragraph.strip():
            continue  # Skip empty paragraphs
        # Remove leading/trailing whitespace on each line, drop empty lines, join with spaces.
        lines = [line.strip() for line in paragraph.split("\n") if line.strip()]
        cleaned.append(" ".join(lines))
    return cleaned


def paragraphs_to_sentences(paragraphs: List[str]) -> List[str]:
    """Convert paragraphs to a flat list of sentences using ground-truth segmentation."""
    sentences: List[str] = []
    for p in paragraphs:
        sentences.extend(create_ground_truth_sentences(p))
    return sentences


def train_test_split_paragraphs(paragraphs: List[str], train_ratio: float = 0.8) -> Tuple[List[str], List[str]]:
    """Randomly split paragraphs into train and test subsets."""
    rng = random.Random(42)  # deterministic
    shuffled = paragraphs.copy()
    rng.shuffle(shuffled)
    split_idx = int(len(shuffled) * train_ratio)
    return shuffled[:split_idx], shuffled[split_idx:]


def build_data_dict(language_code: str, dataset_name: str, train_data: List[str], test_data: List[str]):
    """Wrap the data into the nested dictionary structure expected by downstream code."""
    return {
        language_code: {
            "sentence": {
                dataset_name: {
                    "meta": {
                        "train_data": train_data,
                    },
                    "data": test_data,
                }
            }
        }
    }


def main():
    script_dir = Path(__file__).resolve().parent
    txt_path = script_dir / "war_and_peace.txt"

    if not txt_path.exists():
        raise FileNotFoundError(f"Expected text file not found: {txt_path}")

    print(f"Reading source text from: {txt_path}")
    raw_text = txt_path.read_text(encoding="utf-8")

    # Step 1: Clean paragraphs (merge lines, preserve paragraph breaks)
    paragraphs_initial = remove_single_newlines(raw_text)
    print(f"Paragraphs after cleaning: {len(paragraphs_initial)}")

    # Step 2: Filter out paragraphs with <10 words
    paragraphs = [p for p in paragraphs_initial if len(p.split()) >= 10]
    print(f"Paragraphs after filtering (<10 words removed): {len(paragraphs)}")

    if len(paragraphs) == 0:
        raise ValueError("No paragraphs left after filtering. Adjust the word threshold or check the input text.")

    # Step 3: Train/Test split at the paragraph level
    train_paragraphs, test_paragraphs = train_test_split_paragraphs(paragraphs, train_ratio=0.8)
    print(f"Train paragraphs: {len(train_paragraphs)} | Test paragraphs: {len(test_paragraphs)}")

    # Step 4: From train paragraphs generate sentences, then normalize
    train_sentences_raw = paragraphs_to_sentences(train_paragraphs)
    train_data = [normalize_text_for_comparison(s) for s in train_sentences_raw if normalize_text_for_comparison(s)]
    print(f"Train sentences after segmentation & normalization: {len(train_data)}")

    # Step 5: For test paragraphs treat whole paragraph as one sample, normalize
    test_data = [normalize_text_for_comparison(p) for p in test_paragraphs if normalize_text_for_comparison(p)]
    print(f"Test data (paragraph-level) after normalization: {len(test_data)}")

    # Step 4: Build dictionary and save
    language_code = "en"  # War and Peace is in English
    dataset_name = "war_and_peace"
    data_dict = build_data_dict(language_code, dataset_name, train_data, test_data)

    out_path = script_dir / f"{dataset_name}.pth"
    torch.save(data_dict, str(out_path))
    print(f"Saved processed dataset to: {out_path}")


if __name__ == "__main__":
    main()

