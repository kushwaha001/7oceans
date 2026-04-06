import polars as pl
import os
import torch
import numpy as np
import geopandas as gpd
import geodatasets
import matplotlib.pyplot as plt
from shapely.geometry import LineString, box
from typing import Optional, List, Tuple
from tqdm import tqdm
import json
import math
import glob
from .Analysis_system import Analysis


class MicroAnalysis(Analysis):

    # ── Gulf of Mexico geographic constants ──────────────────────────────
    GULF_LAT_MIN: float = 17.4068
    GULF_LAT_MAX: float = 31.4648
    GULF_LON_MIN: float = -98.0539
    GULF_LON_MAX: float = -80.4330
    GULF_LAT_RANGE: float = GULF_LAT_MAX - GULF_LAT_MIN
    GULF_LON_RANGE: float = GULF_LON_MAX - GULF_LON_MIN

    # ── Polars schema overrides for raw AIS CSVs ─────────────────────────
    SCHEMA_OVERRIDES = {
        "mmsi":        pl.Int64,
        "latitude":    pl.Float64,
        "longitude":   pl.Float64,
        "sog":         pl.Float64,
        "cog":         pl.Float64,
        "heading":     pl.Float64,
        "vessel_type": pl.Int32,
        "length":      pl.Float64,
        "width":       pl.Float64,
        "draft":       pl.Float64,
    }

    def __init__(
        self,
        datasets_location: str,
        dataset_dir: str,
        analysis_result_dir: str,
        min_lat: float = 17.4068,
        min_lon: float = -98.0539,
        max_lat: float = 31.4648,
        max_lon: float = -80.4330,
        window_size_hours: int = 1,
        min_pings_per_window: int = 3,
        max_sog: float = 30.0,
        max_length: float = 400.0,
        max_draft: float = 25.0,
    ):
        super().__init__(datasets_location, dataset_dir, analysis_result_dir)
        self.min_lat = min_lat
        self.min_lon = min_lon
        self.max_lat = max_lat
        self.max_lon = max_lon
        self.window_size_hours = window_size_hours
        self.min_pings_per_window = min_pings_per_window
        self.max_sog = max_sog
        self.max_length = max_length
        self.max_draft = max_draft

    # ================================================================== #
    #  METHOD 1 — extract_vessel_windows (unchanged logic, minor cleanup) #
    # ================================================================== #
    def extract_vessel_windows(
        self,
        dataset_dir: str,
        for_group: bool = False,
    ) -> list[dict]:
        """
        Takes one (or many) daily CSVs, groups rows by MMSI, sorts by
        timestamp, slices into hour-long windows, and returns all valid
        windows that meet the minimum ping count.

        Returns list[dict] with keys: mmsi, date, window_hour, ping_count, df.
        """
        if for_group:
            csv_files = sorted([
                os.path.join(self.datasets_location, f)
                for f in os.listdir(self.datasets_location)
                if f.endswith(".csv")
            ])
            if not csv_files:
                print("No CSV files found in datasets_location.")
                return []
        else:
            target = dataset_dir if dataset_dir is not None else self.dataset_dir
            if not os.path.exists(target):
                raise FileNotFoundError(f"CSV file not found: {target}")
            csv_files = [target]

        all_windows: list[dict] = []

        for csv_path in tqdm(csv_files, desc="Extracting vessel windows", unit="file"):
            df = pl.read_csv(
                csv_path,
                schema_overrides=self.SCHEMA_OVERRIDES,
                infer_schema_length=1000,
            )

            df = df.with_columns(
                pl.col("base_date_time")
                .str.to_datetime("%Y-%m-%d %H:%M:%S")
                .alias("base_date_time")
            )

            date_str = self._extract_date_from_path(csv_path)
            if date_str is None:
                first_ts = df["base_date_time"][0]
                date_str = str(first_ts.date()) if first_ts is not None else "unknown"

            for mmsi_val, vessel_df in df.group_by("mmsi"):
                mmsi_int = int(mmsi_val[0])  # type: ignore

                vessel_df = vessel_df.sort("base_date_time")

                vessel_df = vessel_df.with_columns(
                    pl.col("base_date_time")
                    .dt.truncate(f"{self.window_size_hours}h")
                    .alias("window_key")
                )

                for window_key_val, window_df in vessel_df.group_by("window_key"):
                    window_ts = window_key_val[0]  # type: ignore
                    window_hour = int(window_ts.hour)  # type: ignore
                    ping_count = window_df.height

                    if ping_count < self.min_pings_per_window:
                        continue

                    window_df = window_df.sort("base_date_time")
                    window_df = window_df.drop("window_key")

                    all_windows.append({
                        "mmsi":        mmsi_int,
                        "date":        date_str,
                        "window_hour": window_hour,
                        "ping_count":  ping_count,
                        "df":          window_df,
                    })

            tqdm.write(f"  {os.path.basename(csv_path)}: extracted windows so far = {len(all_windows):,}")

        print(f"\nextract_vessel_windows complete")
        print(f"  CSVs processed   : {len(csv_files):,}")
        print(f"  Valid windows    : {len(all_windows):,}")

        return all_windows

    # ================================================================== #
    #  FULLY REWRITTEN: _build_ping_features — vectorized with Polars    #
    #                                                                     #
    #  OLD: Python for-loop over every ping (row-by-row), calling        #
    #       math.sin / math.cos per element → extremely slow on 10M+    #
    #       rows.                                                        #
    #  NEW: All 11 features computed as Polars column expressions in one #
    #       pass. sin/cos/radians run in Rust, not Python. Returns a     #
    #       NumPy array directly (no intermediate Python lists).         #
    #                                                                     #
    #  Expected speedup: 50–100× per window.                             #
    # ================================================================== #
    def _build_ping_features(
        self,
        df: pl.DataFrame,
        norm_stats: dict,
    ) -> np.ndarray:
        """
        Vectorized feature builder. Returns numpy array of shape
        [N_pings, 11] (float32), ready for torch.from_numpy().

        Feature order (unchanged from original):
            0  lat_norm
            1  lon_norm
            2  sog_norm
            3  cog_sin
            4  cog_cos
            5  heading_sin
            6  heading_cos
            7  delta_t_norm
            8  length_norm
            9  draft_norm
            10 type_norm
        """
        MEAN_LENGTH   = float(norm_stats["MEAN_LENGTH"])
        MAX_TYPE_CODE = float(norm_stats["MAX_TYPE_CODE"])
        LAT_MIN       = self.GULF_LAT_MIN
        LAT_RANGE     = self.GULF_LAT_RANGE
        LON_MIN       = self.GULF_LON_MIN
        LON_RANGE     = self.GULF_LON_RANGE
        MAX_SOG       = self.max_sog
        MAX_LENGTH    = self.max_length
        MAX_DRAFT     = self.max_draft

        df = df.sort("base_date_time")

        # All 11 features as a single .select() — runs entirely in Rust
        DEG2RAD = math.pi / 180.0

        features_df = df.select([
            # Feature 1: lat_norm
            ((pl.col("latitude").fill_null(0.0) - LAT_MIN) / LAT_RANGE)
            .clip(0.0, 1.0).alias("f01_lat"),

            # Feature 2: lon_norm
            ((pl.col("longitude").fill_null(0.0) - LON_MIN) / LON_RANGE)
            .clip(0.0, 1.0).alias("f02_lon"),

            # Feature 3: sog_norm
            (pl.col("sog").fill_null(0.0).fill_nan(0.0) / MAX_SOG)
            .clip(0.0, 1.0).alias("f03_sog"),

            # Feature 4: cog_sin
            pl.when(pl.col("cog").is_null() | (pl.col("cog") == 360.0))
            .then(0.0)
            .otherwise((pl.col("cog") * DEG2RAD).sin())
            .alias("f04_cog_sin"),

            # Feature 5: cog_cos
            pl.when(pl.col("cog").is_null() | (pl.col("cog") == 360.0))
            .then(1.0)
            .otherwise((pl.col("cog") * DEG2RAD).cos())
            .alias("f05_cog_cos"),

            # Feature 6: heading_sin
            pl.when(pl.col("heading").is_null() | (pl.col("heading") == 511.0))
            .then(0.0)
            .otherwise((pl.col("heading") * DEG2RAD).sin())
            .alias("f06_heading_sin"),

            # Feature 7: heading_cos
            pl.when(pl.col("heading").is_null() | (pl.col("heading") == 511.0))
            .then(1.0)
            .otherwise((pl.col("heading") * DEG2RAD).cos())
            .alias("f07_heading_cos"),

            # Feature 8: delta_t_norm (forward-looking: time_next - time_current)
            # Last row gets 0.0 (filled by fill_null).
            (
                (pl.col("base_date_time").shift(-1) - pl.col("base_date_time"))
                .dt.total_seconds()
                .fill_null(0.0)
                / 3600.0
            ).clip(0.0, None).alias("f08_delta_t"),

            # Feature 9: length_norm (0 or null → MEAN_LENGTH)
            (
                pl.when(pl.col("length").is_null() | (pl.col("length") == 0.0))
                .then(MEAN_LENGTH)
                .otherwise(pl.col("length"))
                / MAX_LENGTH
            ).clip(0.0, 1.0).alias("f09_length"),

            # Feature 10: draft_norm
            (pl.col("draft").fill_null(0.0).fill_nan(0.0) / MAX_DRAFT)
            .clip(0.0, 1.0).alias("f10_draft"),

            # Feature 11: type_norm
            (
                pl.col("vessel_type").fill_null(0).cast(pl.Float64)
                / max(MAX_TYPE_CODE, 1.0)
            ).clip(0.0, 1.0).alias("f11_type"),
        ])

        # Convert directly to numpy float32 — no Python list intermediary
        return features_df.to_numpy().astype(np.float32)

    # ================================================================== #
    #  METHOD 3 — compute_micro_norm_stats                                #
    #  OPTIMIZED: uses scan_csv (lazy) instead of read_csv (eager)        #
    #  for the stats that only need column-level aggregates.              #
    # ================================================================== #
    def compute_micro_norm_stats(
        self,
        output_dir: str,
        training_date_stems: Optional[list[str]] = None,
        dataset_dir: Optional[str] = None,
        for_group: bool = True,
    ) -> dict:
        os.makedirs(output_dir, exist_ok=True)

        if for_group:
            csv_files = sorted([
                os.path.join(self.datasets_location, f)
                for f in os.listdir(self.datasets_location)
                if f.endswith(".csv")
            ])
            if not csv_files:
                raise FileNotFoundError(
                    f"No CSV files found in: {self.datasets_location}"
                )
        else:
            target = dataset_dir if dataset_dir is not None else self.dataset_dir
            csv_files = [target]

        if training_date_stems is not None:
            stem_set = set(training_date_stems)
            csv_files = [
                f for f in csv_files
                if self._extract_date_from_path(f) in stem_set
            ]
            if not csv_files:
                raise ValueError(
                    "No CSV files matched the provided training_date_stems."
                )

        print(f"Computing micro norm stats from {len(csv_files)} training CSV(s)...")

        length_sum:   float = 0.0
        length_count: int   = 0
        draft_sum:    float = 0.0
        draft_count:  int   = 0
        max_type_code: int  = 0

        for csv_path in tqdm(csv_files, desc="Scanning training CSVs", unit="file"):
            # ── OPTIMIZED: lazy scan with only the columns we need ────────
            # Old code did pl.read_csv() which loads ALL columns eagerly.
            # New code selects only 3 columns via lazy scan.
            stats_df = (
                pl.scan_csv(csv_path, schema_overrides=self.SCHEMA_OVERRIDES, infer_schema_length=1000)
                .select(["length", "draft", "vessel_type"])
                .collect()
            )

            # ── length
            length_col = stats_df["length"].cast(pl.Float64, strict=False).fill_null(0.0).fill_nan(0.0)
            valid_lengths = length_col.filter(length_col > 0.0)
            if len(valid_lengths) > 0:
                length_sum += float(valid_lengths.sum())  # type: ignore
                length_count += len(valid_lengths)

            # ── draft
            draft_col = stats_df["draft"].cast(pl.Float64, strict=False).fill_null(0.0).fill_nan(0.0)
            valid_drafts = draft_col.filter(draft_col > 0.0)
            if len(valid_drafts) > 0:
                draft_sum += float(valid_drafts.sum())  # type: ignore
                draft_count += len(valid_drafts)

            # ── vessel type
            type_col = stats_df["vessel_type"].cast(pl.Int32, strict=False).drop_nulls()
            if len(type_col) > 0:
                file_max_type = int(type_col.max())  # type: ignore
                if file_max_type > max_type_code:
                    max_type_code = file_max_type

            tqdm.write(
                f"  {os.path.basename(csv_path)}: "
                f"lengths={len(valid_lengths)}  drafts={len(valid_drafts)}  "
                f"max_type_so_far={max_type_code}"
            )

        mean_length = (length_sum / length_count) if length_count > 0 else 0.0
        mean_draft  = (draft_sum / draft_count)   if draft_count > 0  else 0.0

        if max_type_code == 0:
            max_type_code = 99

        stats = {
            "MEAN_LENGTH":   round(mean_length, 4),
            "MEAN_DRAFT":    round(mean_draft, 4),
            "MAX_TYPE_CODE": max_type_code,
            "GULF_LAT_MIN":  self.GULF_LAT_MIN,
            "GULF_LAT_MAX":  self.GULF_LAT_MAX,
            "GULF_LON_MIN":  self.GULF_LON_MIN,
            "GULF_LON_MAX":  self.GULF_LON_MAX,
            "MAX_SOG":       self.max_sog,
            "MAX_LENGTH":    self.max_length,
            "MAX_DRAFT":     self.max_draft,
        }

        stats_path = os.path.abspath(os.path.join(output_dir, "micro_norm_stats.json"))
        with open(stats_path, "w", encoding="utf-8") as fh:
            json.dump(stats, fh, indent=2)

        print(f"\nMicro norm stats computed and saved → {stats_path}")
        print(f"  MEAN_LENGTH    : {mean_length:.4f}")
        print(f"  MEAN_DRAFT     : {mean_draft:.4f}")
        print(f"  MAX_TYPE_CODE  : {max_type_code}")
        print(f"  Training CSVs  : {len(csv_files):,}")

        return stats

    # ================================================================== #
    #  FULLY REWRITTEN (v2): create_micro_tensors — SINGLE-PASS          #
    #                                                                     #
    #  v1 bottleneck (still ~10 min/CSV):                                #
    #    extract_vessel_windows() → Python double group_by (2 min)       #
    #    then loop 101k windows calling _build_ping_features (8 min)     #
    #                                                                     #
    #  v2 approach (~1-2 min/CSV):                                       #
    #    1. Read CSV once                                                #
    #    2. Compute ALL 11 features on the FULL 3M-row DataFrame         #
    #       in ONE vectorized pass (Rust, ~5 sec)                        #
    #    3. group_by (mmsi, window_key) to collect pre-computed features  #
    #    4. Filter by min_pings, convert to tensors                      #
    #                                                                     #
    #  Expected speedup: ~5-10× over v1 (30 min vs 5 hrs for 30 days)  #
    # ================================================================== #
    def create_micro_tensors(
        self,
        output_dir: str,
        norm_stats_path: str,
        for_group: bool = False,
        dataset_dir: Optional[str] = None,
        date_stems: Optional[list[str]] = None,
    ) -> list[str]:
        """
        Processes daily CSVs into per-day PyTorch bundles.

        Each bundle: {date_stem}_micro_bundle.pt
        Each index:  {date_stem}_micro_index.json

        Returns list of absolute paths to all created files.
        """
        os.makedirs(output_dir, exist_ok=True)

        if not os.path.exists(norm_stats_path):
            raise FileNotFoundError(
                f"micro_norm_stats.json not found at: {norm_stats_path}\n"
                "Run compute_micro_norm_stats() on training data first."
            )

        with open(norm_stats_path, "r", encoding="utf-8") as fh:
            norm_stats = json.load(fh)

        if for_group:
            csv_files = sorted([
                os.path.join(self.datasets_location, f)
                for f in os.listdir(self.datasets_location)
                if f.endswith(".csv")
            ])
            if not csv_files:
                print("No CSV files found in datasets_location.")
                return []
        else:
            target = dataset_dir if dataset_dir is not None else self.dataset_dir
            if not os.path.exists(target):
                raise FileNotFoundError(f"CSV file not found: {target}")
            csv_files = [target]

        if date_stems is not None:
            stem_set = set(date_stems)
            csv_files = [
                f for f in csv_files
                if self._extract_date_from_path(f) in stem_set
            ]

        if not csv_files:
            print("No matching CSV files found.")
            return []

        created_files: list[str] = []
        total_windows_saved = 0

        for csv_path in tqdm(csv_files, desc="Creating micro tensors", unit="file"):

            date_stem = self._extract_date_from_path(csv_path)
            if date_stem is None:
                date_stem = os.path.splitext(os.path.basename(csv_path))[0]

            # ── SINGLE-PASS: compute features + group in one go ──────────
            daily_tensors, daily_metadata = self._process_csv_vectorized(
                csv_path=csv_path,
                norm_stats=norm_stats,
                date_stem=date_stem,
            )

            if not daily_tensors:
                tqdm.write(f"  [{date_stem}] No valid windows — skipping.")
                continue

            # ── save DAILY BUNDLE (.pt) ──────────────────────────────────
            bundle = {
                "date":     date_stem,
                "windows":  daily_tensors,
                "metadata": daily_metadata,
            }
            bundle_fname = f"{date_stem}_micro_bundle.pt"
            bundle_path  = os.path.abspath(os.path.join(output_dir, bundle_fname))
            torch.save(bundle, bundle_path)
            created_files.append(bundle_path)

            # ── save daily JSON index ────────────────────────────────────
            daily_index = {
                "date":           date_stem,
                "macro_tensor":   f"{date_stem}_macro.pt",
                "micro_bundle":   bundle_fname,
                "total_windows":  len(daily_tensors),
                "vessel_windows": daily_metadata,
            }
            index_path = os.path.join(output_dir, f"{date_stem}_micro_index.json")
            with open(index_path, "w", encoding="utf-8") as fh:
                json.dump(daily_index, fh, indent=2, ensure_ascii=False)
            created_files.append(index_path)

            total_windows_saved += len(daily_tensors)
            tqdm.write(
                f"  [{date_stem}] {len(daily_tensors):,} windows → {bundle_fname}"
            )

        # ── summary ──────────────────────────────────────────────────────
        n_bundles = sum(1 for f in created_files if f.endswith("_micro_bundle.pt"))
        print(f"\ncreate_micro_tensors complete")
        print(f"  CSVs processed       : {len(csv_files):,}")
        print(f"  Daily bundles saved  : {n_bundles:,}")
        print(f"  Total windows saved  : {total_windows_saved:,}")
        print(f"  Output dir           : {output_dir}")

        return created_files

    # ================================================================== #
    #  NEW: _process_csv_vectorized — the fast single-pass engine        #
    #                                                                     #
    #  Instead of:                                                       #
    #    extract_vessel_windows (Python group_by, 2 min)                 #
    #    → loop 101k windows calling _build_ping_features (8 min)        #
    #  Does:                                                             #
    #    1. Read CSV → compute 11 features on ALL 3M rows at once (5s)  #
    #    2. group_by(mmsi, hour) → collect pre-computed lists (20s)      #
    #    3. Stack into tensors with numpy (30s)                          #
    #  Total: ~1-2 min instead of ~10 min per CSV.                      #
    # ================================================================== #
    def _process_csv_vectorized(
        self,
        csv_path: str,
        norm_stats: dict,
        date_stem: str,
    ) -> tuple[list[torch.Tensor], list[dict]]:
        """
        Single-pass processing of one daily CSV into tensors + metadata.
        Computes all features on the full DataFrame, then groups into windows.
        """
        MEAN_LENGTH   = float(norm_stats["MEAN_LENGTH"])
        MAX_TYPE_CODE = float(norm_stats["MAX_TYPE_CODE"])
        DEG2RAD       = math.pi / 180.0

        # ── Step 1: read CSV and parse timestamp ─────────────────────────
        df = pl.read_csv(
            csv_path,
            schema_overrides=self.SCHEMA_OVERRIDES,
            infer_schema_length=1000,
        )

        df = df.with_columns(
            pl.col("base_date_time")
            .str.to_datetime("%Y-%m-%d %H:%M:%S")
            .alias("base_date_time")
        )

        # Drop rows with null MMSI
        df = df.filter(pl.col("mmsi").is_not_null())

        if df.is_empty():
            return [], []

        # ── Step 3: sort by (mmsi, time) — needed for delta_t ───────────
        df = df.sort(["mmsi", "base_date_time"])

        # ── Step 4: compute ALL 11 features on the ENTIRE DataFrame ──────
        #    This runs in Rust on 3M rows — takes seconds, not minutes.
        #    delta_t uses .over() to compute within each (mmsi, window).
        df = df.with_columns([
            # F1: lat_norm
            ((pl.col("latitude").fill_null(0.0) - self.GULF_LAT_MIN) / self.GULF_LAT_RANGE)
            .clip(0.0, 1.0).alias("_f01"),

            # F2: lon_norm
            ((pl.col("longitude").fill_null(0.0) - self.GULF_LON_MIN) / self.GULF_LON_RANGE)
            .clip(0.0, 1.0).alias("_f02"),

            # F3: sog_norm
            (pl.col("sog").fill_null(0.0).fill_nan(0.0) / self.max_sog)
            .clip(0.0, 1.0).alias("_f03"),

            # F4: cog_sin
            pl.when(pl.col("cog").is_null() | (pl.col("cog") == 360.0))
            .then(0.0)
            .otherwise((pl.col("cog") * DEG2RAD).sin())
            .alias("_f04"),

            # F5: cog_cos
            pl.when(pl.col("cog").is_null() | (pl.col("cog") == 360.0))
            .then(1.0)
            .otherwise((pl.col("cog") * DEG2RAD).cos())
            .alias("_f05"),

            # F6: heading_sin
            pl.when(pl.col("heading").is_null() | (pl.col("heading") == 511.0))
            .then(0.0)
            .otherwise((pl.col("heading") * DEG2RAD).sin())
            .alias("_f06"),

            # F7: heading_cos
            pl.when(pl.col("heading").is_null() | (pl.col("heading") == 511.0))
            .then(1.0)
            .otherwise((pl.col("heading") * DEG2RAD).cos())
            .alias("_f07"),

            # F8: delta_t_norm (forward-looking WITHIN each window group)
            #     .over() respects current sort order so shift(-1) gets
            #     the next row's time within the same (mmsi, window).
            (
                (
                    pl.col("base_date_time").shift(-1).over(["mmsi"])
                    - pl.col("base_date_time")
                )
                .dt.total_seconds()
                .fill_null(0.0)
                / 3600.0
            ).clip(0.0, None).alias("_f08"),

            # F9: length_norm (0 or null → MEAN_LENGTH)
            (
                pl.when(pl.col("length").is_null() | (pl.col("length") == 0.0))
                .then(MEAN_LENGTH)
                .otherwise(pl.col("length"))
                / self.max_length
            ).clip(0.0, 1.0).alias("_f09"),

            # F10: draft_norm
            (pl.col("draft").fill_null(0.0).fill_nan(0.0) / self.max_draft)
            .clip(0.0, 1.0).alias("_f10"),

            # F11: type_norm
            (
                pl.col("vessel_type").fill_null(0).cast(pl.Float64)
                / max(MAX_TYPE_CODE, 1.0)
            ).clip(0.0, 1.0).alias("_f11"),
        ])

        # ── Step 5: group by (mmsi, window_key) and collect features ─────
        #    Each group becomes one window. We collect pre-computed feature
        #    values as lists, then stack them into tensors below.
        _FCOLS = ["_f01", "_f02", "_f03", "_f04", "_f05", "_f06",
                  "_f07", "_f08", "_f09", "_f10", "_f11"]

        grouped = (
            df
            .group_by(["mmsi"], maintain_order=True)
            .agg(
                [pl.col(c) for c in _FCOLS]
                + [pl.len().alias("_pc")]
            )
            .filter(pl.col("_pc") >= 10)   # min 10 pings per day (v2.0 requirement)
        )

        if grouped.is_empty():
            return [], []

        # ── Step 6: convert grouped rows to tensors ──────────────────────
        #    This is still a Python loop, but it's just numpy stacking —
        #    no Polars operations, no DataFrame creation per window.
        mmsi_col  = grouped["mmsi"].to_list()
        pc_col    = grouped["_pc"].to_list()
        feat_cols = [grouped[c].to_list() for c in _FCOLS]

        n_windows = grouped.height
        daily_tensors:  list[torch.Tensor] = []
        daily_metadata: list[dict]         = []

        for i in range(n_windows):
            # Stack the 11 feature lists for this window into [N_pings, 11]
            features = np.column_stack([fc[i] for fc in feat_cols]).astype(np.float32)
            tensor   = torch.from_numpy(features)

            if tensor.shape[0] == 0:
                continue

            # Skip windows with NaN/Inf (rare edge cases)
            if torch.isnan(tensor).any() or torch.isinf(tensor).any():
                continue

            bundle_index = len(daily_tensors)
            daily_tensors.append(tensor)
            daily_metadata.append({
                "mmsi":         int(mmsi_col[i]),
                "ping_count":   int(pc_col[i]),
                "bundle_index": bundle_index,
            })

        return daily_tensors, daily_metadata

    # ================================================================== #
    #  UPDATED: validate_micro_tensors — works with daily bundles        #
    # ================================================================== #
    def validate_micro_tensors(
        self,
        tensor_dir: str,
        expected_feature_dim: int = 11,
        value_min: float = -1.1,
        value_max: float = 1.1,
        check_index_files: bool = True,
    ) -> dict:
        """
        Quality assurance for the new daily bundle format.
        Validates every *_micro_bundle.pt file in tensor_dir.
        """
        bundle_files = sorted([
            f for f in os.listdir(tensor_dir)
            if f.endswith("_micro_bundle.pt")
        ])

        errors: dict       = {}
        passed_windows: int = 0
        failed_windows: int = 0
        total_windows:  int = 0

        for fname in tqdm(bundle_files, desc="Validating micro bundles", unit="file"):
            fpath = os.path.join(tensor_dir, fname)

            try:
                bundle = torch.load(fpath, map_location="cpu", weights_only=False)
            except Exception as e:
                errors[fname] = [f"Failed to load: {e}"]
                failed_windows += 1
                continue

            windows  = bundle.get("windows", [])
            metadata = bundle.get("metadata", [])

            if len(windows) != len(metadata):
                errors[fname] = [
                    f"Mismatch: {len(windows)} tensors vs {len(metadata)} metadata entries"
                ]
                failed_windows += len(windows)
                continue

            file_errors: list[str] = []

            for i, tensor in enumerate(windows):
                total_windows += 1
                window_errors: list[str] = []

                if tensor.ndim != 2:
                    window_errors.append(f"Window {i}: wrong dims {tensor.ndim}")
                elif tensor.shape[1] != expected_feature_dim:
                    window_errors.append(
                        f"Window {i}: wrong feature dim {tensor.shape[1]}"
                    )

                if torch.isnan(tensor).any():
                    window_errors.append(f"Window {i}: contains NaN")
                if torch.isinf(tensor).any():
                    window_errors.append(f"Window {i}: contains Inf")

                actual_min = float(tensor.min().item())
                actual_max = float(tensor.max().item())
                if actual_min < value_min:
                    window_errors.append(
                        f"Window {i}: min {actual_min:.6f} < {value_min}"
                    )
                if actual_max > value_max:
                    window_errors.append(
                        f"Window {i}: max {actual_max:.6f} > {value_max}"
                    )

                if tensor.shape[0] < self.min_pings_per_window:
                    window_errors.append(
                        f"Window {i}: too few pings {tensor.shape[0]}"
                    )

                if window_errors:
                    file_errors.extend(window_errors)
                    failed_windows += 1
                else:
                    passed_windows += 1

            if file_errors:
                errors[fname] = file_errors
                tqdm.write(f"  FAIL [{fname}]: {len(file_errors)} issue(s)")

        # ── index file cross-check ───────────────────────────────────────
        orphaned_bundles: list[str] = []
        missing_bundles:  list[str] = []

        if check_index_files:
            index_files = sorted([
                f for f in os.listdir(tensor_dir)
                if f.endswith("_micro_index.json")
            ])

            indexed_bundles: set[str] = set()
            for idx_fname in index_files:
                idx_path = os.path.join(tensor_dir, idx_fname)
                with open(idx_path, "r", encoding="utf-8") as fh:
                    idx_data = json.load(fh)
                bundle_name = idx_data.get("micro_bundle")
                if bundle_name:
                    indexed_bundles.add(bundle_name)

            bundle_set = set(bundle_files)
            orphaned_bundles = sorted(list(bundle_set - indexed_bundles))
            missing_bundles  = sorted(list(indexed_bundles - bundle_set))

        result = {
            "total_bundles":    len(bundle_files),
            "total_windows":    total_windows,
            "passed_windows":   passed_windows,
            "failed_windows":   failed_windows,
            "orphaned_bundles": orphaned_bundles,
            "missing_bundles":  missing_bundles,
            "errors":           errors,
        }

        print(f"\nvalidate_micro_tensors complete")
        print(f"  Total bundles      : {len(bundle_files):,}")
        print(f"  Total windows      : {total_windows:,}")
        print(f"  Passed windows     : {passed_windows:,}")
        print(f"  Failed windows     : {failed_windows:,}")
        print(f"  Orphaned bundles   : {len(orphaned_bundles):,}")
        print(f"  Missing bundles    : {len(missing_bundles):,}")

        if errors:
            print(f"\n  Files with errors:")
            for fname, errs in errors.items():
                for e in errs[:5]:  # show first 5 per file
                    print(f"    {fname}: {e}")
                if len(errs) > 5:
                    print(f"    ... and {len(errs) - 5} more")
        else:
            print("\n  All windows passed validation.")

        return result

    # ================================================================== #
    #  UPDATED: build_dataset_index — works with daily bundles           #
    # ================================================================== #
    def build_dataset_index(
        self,
        micro_tensor_dir: str,
        macro_tensor_dir: str,
        output_dir: str,
        split_name: str,
        date_stems: Optional[list[str]] = None,
    ) -> str:
        """
        Builds a master index that maps every micro window (inside a
        daily bundle) to its paired macro tensor by date.

        Updated for the bundle format — each pair now references the
        bundle file + the integer index within that bundle.
        """
        os.makedirs(output_dir, exist_ok=True)

        index_files = sorted([
            f for f in os.listdir(micro_tensor_dir)
            if f.endswith("_micro_index.json")
        ])

        if date_stems is not None:
            stem_set = set(date_stems)
            index_files = [
                f for f in index_files
                if f.replace("_micro_index.json", "") in stem_set
            ]

        if not index_files:
            print("No micro index files found.")
            return ""

        all_pairs: list[dict] = []

        for idx_fname in tqdm(index_files, desc="Building dataset index", unit="file"):
            idx_path = os.path.join(micro_tensor_dir, idx_fname)
            with open(idx_path, "r", encoding="utf-8") as fh:
                idx_data = json.load(fh)

            date = idx_data["date"]

            # ── verify macro tensor exists ───────────────────────────────
            macro_fname = f"{date}_macro.pt"
            macro_path  = os.path.abspath(
                os.path.join(macro_tensor_dir, macro_fname)
            )
            if not os.path.exists(macro_path):
                tqdm.write(
                    f"  WARNING: Macro tensor missing for {date} — "
                    f"skipping {len(idx_data.get('vessel_windows', []))} windows"
                )
                continue

            # ── verify micro bundle exists ────────────────────────────────
            bundle_fname = idx_data.get("micro_bundle", f"{date}_micro_bundle.pt")
            bundle_path  = os.path.abspath(
                os.path.join(micro_tensor_dir, bundle_fname)
            )
            if not os.path.exists(bundle_path):
                tqdm.write(
                    f"  WARNING: Micro bundle missing for {date} — skipping"
                )
                continue

            # ── build pairs ──────────────────────────────────────────────
            for entry in idx_data.get("vessel_windows", []):
                all_pairs.append({
                    "macro":        macro_path,
                    "micro_bundle": bundle_path,
                    "bundle_index": entry["bundle_index"],
                    "mmsi":         entry["mmsi"],
                    "date":         date,
                    "ping_count":   entry["ping_count"],
                })

        # ── save master index ────────────────────────────────────────────
        master_index = {
            "split":         split_name,
            "total_samples": len(all_pairs),
            "pairs":         all_pairs,
        }

        out_path = os.path.abspath(
            os.path.join(output_dir, f"{split_name}_dataset_index.json")
        )
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(master_index, fh, indent=2, ensure_ascii=False)

        print(f"\nbuild_dataset_index complete")
        print(f"  Split            : {split_name}")
        print(f"  Total samples    : {len(all_pairs):,}")
        print(f"  Index files read : {len(index_files):,}")
        print(f"  Saved            : {out_path}")

        return out_path

    # ================================================================== #
    #  EXTRA — plot_vessel_trajectory (unchanged)                        #
    # ================================================================== #
    def plot_vessel_trajectory(
        self,
        tensor_path: str,
        output_file_name: str,
        bundle_index: Optional[int] = None,
    ) -> None:
        """
        Plot a vessel trajectory from a micro tensor.

        For the new bundle format, pass bundle_index to select a specific
        window from a daily bundle file. If tensor_path points to a
        legacy individual .pt file, leave bundle_index=None.
        """
        if bundle_index is not None:
            # New bundle format
            bundle = torch.load(tensor_path, map_location="cpu", weights_only=False)
            tensor = bundle["windows"][bundle_index]
        else:
            tensor = torch.load(tensor_path, map_location="cpu", weights_only=True)

        lat_norm = tensor[:, 0].numpy()
        lon_norm = tensor[:, 1].numpy()

        lats = lat_norm * self.GULF_LAT_RANGE + self.GULF_LAT_MIN
        lons = lon_norm * self.GULF_LON_RANGE + self.GULF_LON_MIN

        world = gpd.read_file(geodatasets.get_path("naturalearth.land"))

        fig, ax = plt.subplots(figsize=(14, 9))
        world.plot(ax=ax, color="lightgrey", edgecolor="white", linewidth=0.5)

        if len(lats) >= 2:
            track_line = LineString(zip(lons, lats))
            gdf_line = gpd.GeoDataFrame(geometry=[track_line], crs="EPSG:4326")
            gdf_line.plot(ax=ax, color="steelblue", linewidth=1.8, alpha=0.7, zorder=2)

        scatter = ax.scatter(
            lons, lats,
            c=range(len(lats)), cmap="plasma", s=18, zorder=3, alpha=0.85, label="Pings",
        )
        plt.colorbar(scatter, ax=ax, label="Ping sequence (early → late)", shrink=0.5)

        ax.plot(
            lons[0], lats[0],
            marker="o", color="green", markersize=10,
            zorder=5, label="Start", markeredgecolor="white", markeredgewidth=1.2,
        )
        ax.plot(
            lons[-1], lats[-1],
            marker="s", color="red", markersize=10,
            zorder=5, label="End", markeredgecolor="white", markeredgewidth=1.2,
        )

        buffer = max((lons.max() - lons.min()) * 0.15, (lats.max() - lats.min()) * 0.15, 1.0)
        ax.set_xlim(lons.min() - buffer, lons.max() + buffer)
        ax.set_ylim(lats.min() - buffer, lats.max() + buffer)

        basename = os.path.basename(tensor_path)
        title_suffix = f" [window {bundle_index}]" if bundle_index is not None else ""
        ax.set_title(
            f"Micro Vessel Trajectory — {basename}{title_suffix}\n"
            f"Pings: {len(lats):,}",
            fontsize=13, pad=14,
        )
        ax.set_xlabel("Longitude", fontsize=11)
        ax.set_ylabel("Latitude", fontsize=11)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(loc="lower left", fontsize=10)

        plt.tight_layout()

        plots_dir = os.path.join(self.analysis_result_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        out_path = os.path.join(plots_dir, output_file_name)
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved → {out_path}")
        plt.close()

    # ================================================================== #
    #  EXTRA — plot_window_sog_profile (updated for bundles)             #
    # ================================================================== #
    def plot_window_sog_profile(
        self,
        tensor_path: str,
        output_file_name: str,
        bundle_index: Optional[int] = None,
    ) -> None:
        if bundle_index is not None:
            bundle = torch.load(tensor_path, map_location="cpu", weights_only=False)
            tensor = bundle["windows"][bundle_index]
        else:
            tensor = torch.load(tensor_path, map_location="cpu", weights_only=True)

        sog_norm  = tensor[:, 2].numpy()
        sog_knots = sog_norm * self.max_sog

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(
            range(len(sog_knots)), sog_knots,
            marker="o", markersize=4, color="steelblue",
            linewidth=1.5, alpha=0.85,
        )

        ax.set_xlabel("Ping Index", fontsize=11)
        ax.set_ylabel("SOG (knots)", fontsize=11)

        basename = os.path.basename(tensor_path)
        ax.set_title(
            f"SOG Profile — {basename}\n"
            f"Pings: {len(sog_knots):,}  |  Max SOG: {sog_knots.max():.1f} kn",
            fontsize=13, pad=14,
        )
        ax.grid(True, linestyle="--", alpha=0.4)

        plt.tight_layout()

        plots_dir = os.path.join(self.analysis_result_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        out_path = os.path.join(plots_dir, output_file_name)
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved → {out_path}")
        plt.close()

    # ================================================================== #
    #  EXTRA — clean_vessel_windows (unchanged)                          #
    # ================================================================== #
    def clean_vessel_windows(
        self,
        dataset_dir: str,
        output_dir: str,
        for_group: bool = False,
        create_new: bool = True,
    ) -> list[str]:
        os.makedirs(output_dir, exist_ok=True)

        if for_group:
            csv_files = sorted([
                os.path.join(self.datasets_location, f)
                for f in os.listdir(self.datasets_location)
                if f.endswith(".csv")
            ])
            if not csv_files:
                print("No CSV files found in datasets_location.")
                return []
        else:
            target = dataset_dir if dataset_dir is not None else self.dataset_dir
            csv_files = [target]

        created_files: list[str] = []

        for csv_path in tqdm(csv_files, desc="Cleaning vessel windows", unit="file"):
            df = pl.read_csv(
                csv_path,
                schema_overrides=self.SCHEMA_OVERRIDES,
                infer_schema_length=1000,
            )

            original_count = df.height

            filtered = (
                df
                .filter(pl.col("mmsi").is_not_null())
                .filter(pl.col("latitude").is_between(self.min_lat, self.max_lat))
                .filter(pl.col("longitude").is_between(self.min_lon, self.max_lon))
                .filter(
                    (pl.col("sog").is_null()) |
                    ((pl.col("sog") >= 0.0) & (pl.col("sog") <= self.max_sog))
                )
            )

            filtered_count = filtered.height

            basename = os.path.basename(csv_path)
            if create_new:
                base, ext = os.path.splitext(basename)
                out_path = os.path.join(output_dir, f"cleaned_{base}{ext}")
            else:
                out_path = csv_path

            filtered.write_csv(out_path)
            created_files.append(os.path.abspath(out_path))

            tqdm.write(
                f"  {basename}: {original_count:,} → {filtered_count:,} rows "
                f"({original_count - filtered_count:,} removed)"
            )

        print(f"\nclean_vessel_windows complete")
        print(f"  CSVs processed   : {len(csv_files):,}")
        print(f"  Output dir       : {output_dir}")

        return created_files

    # ================================================================== #
    #  PRIVATE HELPERS                                                     #
    # ================================================================== #
    def _extract_date_from_path(
        self,
        csv_path: str,
    ) -> Optional[str]:
        import re
        basename = os.path.basename(csv_path)
        match = re.search(r"(\d{4}-\d{2}-\d{2})", basename)
        if match:
            return match.group(1)
        match = re.search(r"(\d{4})_(\d{2})_(\d{2})", basename)
        if match:
            return f"{match.group(1)}-{match.group(2)}-{match.group(3)}"
        return None