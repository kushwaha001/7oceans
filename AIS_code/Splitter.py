
import os
import re
from typing import Optional


def extract_date_stem(filename: str) -> Optional[str]:
    """
    Extracts a YYYY-MM-DD date from a filename.

    Handles patterns like:
        ais-2022-01-01.csv
        Gulf_of_Mexico_ais-2022-01-01.csv
        AIS_2022_01_01.csv
        2022-01-01_macro_raw.parquet
        cleaned_Gulf_of_Mexico_ais-2022-01-01.csv
    """
    # Try YYYY-MM-DD first (most common)
    match = re.search(r"(\d{4}-\d{2}-\d{2})", filename)
    if match:
        return match.group(1)

    # Try YYYY_MM_DD (underscore-separated)
    match = re.search(r"(\d{4})_(\d{2})_(\d{2})", filename)
    if match:
        return f"{match.group(1)}-{match.group(2)}-{match.group(3)}"

    return None


def scan_date_stems(
    data_dir: str,
    extension: str = ".csv",
) -> list[str]:
    """
    Scans a directory for files matching the given extension,
    extracts date stems, deduplicates, and returns them sorted
    chronologically.

    Parameters
    ----------
    data_dir : str
        Path to the directory containing data files.
    extension : str
        File extension to match. Examples:
            ".csv"                  — for raw AIS CSV files
            "_macro_raw.parquet"    — for macro raw parquets

    Returns
    -------
    list[str] — sorted unique date stems like ["2022-01-01", "2022-01-02", ...]
    """
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Directory not found: {data_dir}")

    stems: set[str] = set()

    for fname in os.listdir(data_dir):
        if not fname.endswith(extension):
            continue
        stem = extract_date_stem(fname)
        if stem is not None:
            stems.add(stem)

    return sorted(stems)


def auto_split(
    data_dir: str,
    extension: str = ".csv",
    ratios: tuple[float, float, float] = (0.70, 0.15, 0.15),
    min_val: int = 1,
    min_test: int = 1,
    verbose: bool = True,
) -> dict:
    """
    Scans a data directory, extracts date stems, and splits them
    chronologically into train / val / test.

    The split is CHRONOLOGICAL (not random) — training always gets the
    earliest dates, test always gets the latest. This prevents data
    leakage from future days into the training set.

    Parameters
    ----------
    data_dir : str
        Path to the directory containing data files.

    extension : str
        File extension to match:
            ".csv"                  — for raw AIS CSV directories
            "_macro_raw.parquet"    — for macro raw parquet directories

    ratios : tuple of 3 floats
        (train_ratio, val_ratio, test_ratio). Must sum to ~1.0.
        Default: (0.70, 0.15, 0.15)

    min_val : int
        Minimum number of files to allocate to validation.
        Default: 1. Set to 0 if you don't need a validation split.

    min_test : int
        Minimum number of files to allocate to test.
        Default: 1. Set to 0 if you don't need a test split.

    verbose : bool
        If True, prints a human-readable summary.

    Returns
    -------
    dict with keys:
        "train"   : list[str] — date stems for training
        "val"     : list[str] — date stems for validation
        "test"    : list[str] — date stems for test
        "all"     : list[str] — all date stems (sorted)
        "summary" : dict      — human-readable stats
    """
    # ── validate ratios ──────────────────────────────────────────────
    train_r, val_r, test_r = ratios
    total_r = train_r + val_r + test_r
    if abs(total_r - 1.0) > 0.01:
        raise ValueError(
            f"Ratios must sum to ~1.0, got {train_r} + {val_r} + {test_r} = {total_r}"
        )

    # ── scan directory ───────────────────────────────────────────────
    all_stems = scan_date_stems(data_dir, extension)
    n_total = len(all_stems)

    if n_total == 0:
        raise FileNotFoundError(
            f"No files with extension '{extension}' found in: {data_dir}"
        )

    # ── compute split sizes ──────────────────────────────────────────
    #
    # Strategy:
    #   1. Compute ideal sizes from ratios
    #   2. Guarantee minimums for val and test (if enough files exist)
    #   3. Training gets everything that's left
    #
    # Edge cases:
    #   1 file  → train=1, val=0, test=0 (with warning)
    #   2 files → train=1, val=1, test=0 OR train=1, val=0, test=1
    #   3 files → train=1, val=1, test=1
    #   N files → ratio-based with minimum guarantees

    if n_total == 1:
        # Only one file — everything is training
        n_train, n_val, n_test = 1, 0, 0
    elif n_total == 2:
        # Two files — train + val (test gets 0)
        n_train, n_val, n_test = 1, 1, 0
    elif n_total <= 5:
        # Very few files — give 1 to val, 1 to test, rest to train
        n_test  = min(min_test, 1)
        n_val   = min(min_val, 1)
        n_train = n_total - n_val - n_test
    else:
        # Normal case — ratio-based with minimum guarantees
        n_test  = max(min_test, round(n_total * test_r))
        n_val   = max(min_val, round(n_total * val_r))
        n_train = n_total - n_val - n_test

        # Safety: if rounding left train with 0, fix it
        if n_train <= 0:
            n_train = 1
            leftover = n_total - 1
            n_val  = leftover // 2
            n_test = leftover - n_val

    # ── slice chronologically ────────────────────────────────────────
    train_stems = all_stems[:n_train]
    val_stems   = all_stems[n_train : n_train + n_val]
    test_stems  = all_stems[n_train + n_val:]

    # ── build summary ────────────────────────────────────────────────
    summary = {
        "data_dir":      data_dir,
        "extension":     extension,
        "total_files":   n_total,
        "ratios":        f"{train_r:.0%} / {val_r:.0%} / {test_r:.0%}",
        "train_count":   len(train_stems),
        "val_count":     len(val_stems),
        "test_count":    len(test_stems),
        "train_range":   f"{train_stems[0]} → {train_stems[-1]}" if train_stems else "—",
        "val_range":     f"{val_stems[0]} → {val_stems[-1]}" if val_stems else "—",
        "test_range":    f"{test_stems[0]} → {test_stems[-1]}" if test_stems else "—",
    }

    # ── print summary ────────────────────────────────────────────────
    if verbose:
        print(f"\n{'=' * 60}")
        print(f"  AUTO SPLIT — {n_total} files from {os.path.basename(data_dir)}")
        print(f"  Ratios: {summary['ratios']}")
        print(f"{'=' * 60}")
        print(f"  Train : {len(train_stems):>4} files   {summary['train_range']}")
        print(f"  Val   : {len(val_stems):>4} files   {summary['val_range']}")
        print(f"  Test  : {len(test_stems):>4} files   {summary['test_range']}")
        print(f"{'=' * 60}\n")

        if n_total == 1:
            print("  ⚠ WARNING: Only 1 file found — using it all for training.")
            print("    No validation or test split is possible.\n")
        elif n_total == 2:
            print("  ⚠ WARNING: Only 2 files found — no test split available.\n")
        elif n_total <= 5:
            print(f"  ⚠ NOTE: Only {n_total} files — using minimum splits "
                  f"(1 val, 1 test). Ratios not fully applied.\n")

    return {
        "train":   train_stems,
        "val":     val_stems,
        "test":    test_stems,
        "all":     all_stems,
        "summary": summary,
    }


def auto_split_parquet(
    parquet_dir: str,
    ratios: tuple[float, float, float] = (0.70, 0.15, 0.15),
    verbose: bool = True,
) -> dict:
    """
    Convenience wrapper for macro parquet directories.
    Equivalent to: auto_split(parquet_dir, extension="_macro_raw.parquet", ...)
    """
    return auto_split(
        data_dir=parquet_dir,
        extension="_macro_raw.parquet",
        ratios=ratios,
        verbose=verbose,
    )


# ─────────────────────────────────────────────────────────────────────
#  Self-test — run this file directly to test with a directory
# ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python splitter.py <data_dir> [extension]")
        print("  Example: python splitter.py D:\\AIS_project\\data\\CSV_files .csv")
        print("  Example: python splitter.py D:\\AIS_project\\processed\\macro_raw _macro_raw.parquet")
        sys.exit(1)

    data_dir  = sys.argv[1]
    extension = sys.argv[2] if len(sys.argv) > 2 else ".csv"

    result = auto_split(data_dir=data_dir, extension=extension)

    print(f"Train stems ({len(result['train'])}):")
    for s in result["train"][:5]:
        print(f"  {s}")
    if len(result["train"]) > 5:
        print(f"  ... and {len(result['train']) - 5} more")

    print(f"\nVal stems ({len(result['val'])}):")
    for s in result["val"]:
        print(f"  {s}")

    print(f"\nTest stems ({len(result['test'])}):")
    for s in result["test"]:
        print(f"  {s}")