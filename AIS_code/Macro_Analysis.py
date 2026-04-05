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
import shutil
import json
import math
import re
import ijson
from itertools import product
from .Analysis_system import Analysis


class MacroAnalysis(Analysis):

    # ── Polars schema overrides for raw AIS CSVs ─────────────────────────
    # Applying these upfront avoids costly infer_schema_length passes and
    # guarantees that columns like vessel_type are always Int32 (not Utf8).
    _CSV_SCHEMA = {
        "mmsi":          pl.Int64,
        "base_date_time": pl.Utf8,
        "longitude":     pl.Float64,
        "latitude":      pl.Float64,
        "sog":           pl.Float64,
        "cog":           pl.Float64,
        "heading":       pl.Float64,
        "vessel_name":   pl.Utf8,
        "imo":           pl.Utf8,
        "call_sign":     pl.Utf8,
        "vessel_type":   pl.Int32,
        "status":        pl.Utf8,
        "length":        pl.Float64,
        "width":         pl.Float64,
        "draft":         pl.Float64,
        "cargo":         pl.Utf8,
        "transceiver":   pl.Utf8,
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
        bin_size: float = 0.5,
        overlap_pct: float = 0.0
    ):
        super().__init__(datasets_location, dataset_dir, analysis_result_dir)
        self.min_lat = min_lat
        self.min_lon = min_lon
        self.max_lat = max_lat
        self.max_lon = max_lon
        self.bin_size = bin_size
        self.overlap_pct = overlap_pct
        self.bboxes_list = self.sort_bboxes(
            self.generate_coordinates(
                min_lat=self.min_lat,
                min_lon=self.min_lon,
                max_lat=self.max_lat,
                max_lon=self.max_lon,
                bin_size=self.bin_size,
                overlap_pct=self.overlap_pct
            )
        )
        print("The boxes are now sorted") if self.is_sorted_bboxes(self.bboxes_list) else print("Boxes are not sorted take care before moving forward")

        # ── pre-compute grid dimensions for the vectorized fast path ─────
        self._stride = bin_size * (1.0 - overlap_pct)
        self._lat_starts = self._get_starts(min_lat, max_lat)
        self._lon_starts = self._get_starts(min_lon, max_lon)
        self._n_lat_bins = len(self._lat_starts)
        self._n_lon_bins = len(self._lon_starts)

    # ── helper shared between __init__ and generate_coordinates ──────────
    def _get_starts(self, start: float, end: float) -> list[float]:
        stride = self._stride if hasattr(self, '_stride') else self.bin_size * (1.0 - self.overlap_pct)
        starts: list[float] = []
        current = start
        while current < end:
            starts.append(current)
            if current + self.bin_size >= end:
                break
            current += stride
        return starts

    # ── unchanged ─────────────────────────────────────────────────────────
    def generate_coordinates(
        self,
        min_lat: float,
        min_lon: float,
        max_lat: float,
        max_lon: float,
        bin_size: float,
        overlap_pct: float
    ) -> List[Tuple[float, float, float, float]]:
        stride = bin_size * (1.0 - overlap_pct)

        def get_starts(start, end):
            starts = []
            current = start
            while current < end:
                starts.append(current)
                if current + bin_size >= end:
                    break
                current += stride
            return starts

        lat_starts = get_starts(min_lat, max_lat)
        lon_starts = get_starts(min_lon, max_lon)

        bboxes = [
            (lat, lon, lat + bin_size, lon + bin_size)
            for lat, lon in product(lat_starts, lon_starts)
        ]
        return bboxes

    # ── unchanged ─────────────────────────────────────────────────────────
    def plot_bboxes(
        self,
        bboxes_list: List[Tuple[float, float, float, float]],
        title: str
    ):
        geometries = [
            box(lon_min, lat_min, lon_max, lat_max)
            for lat_min, lon_min, lat_max, lon_max in bboxes_list
        ]

        gdf_bboxes = gpd.GeoDataFrame(geometry=geometries, crs="EPSG:4326")
        world = gpd.read_file(geodatasets.get_path('naturalearth.land'))

        fig, ax = plt.subplots(figsize=(10, 8))
        world.plot(ax=ax, color='lightgrey', edgecolor='white')
        gdf_bboxes.boundary.plot(ax=ax, edgecolor='red', linewidth=0.5, alpha=0.6)

        buffer = 1
        minx, miny, maxx, maxy = gdf_bboxes.total_bounds
        ax.set_xlim(minx - buffer, maxx + buffer)
        ax.set_ylim(miny - buffer, maxy + buffer)

        plt.title(f"{title} ({len(bboxes_list)} cells)")
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.show()

    # ── unchanged ─────────────────────────────────────────────────────────
    def plot_bboxes_and_map(
        self,
        bboxes_list: List[Tuple[float, float, float, float]],
        title: str
    ):
        geometries = [
            box(lon_min, lat_min, lon_min, lat_max)
            for lat_min, lon_min, lat_max, lon_max in bboxes_list
        ]

        gdf_bboxes = gpd.GeoDataFrame(geometry=geometries, crs="EPSG:4326")
        world = gpd.read_file(geodatasets.get_path('naturalearth.land'))
        fig, ax = plt.subplots(figsize=(10, 8))
        world.plot(ax=ax, color='lightgrey', edgecolor='white')

        gdf_bboxes.boundary.plot(ax=ax, edgecolor='red', linewidth=0.5, alpha=0.6)

        for idx, geom in enumerate(gdf_bboxes.geometry):
            ax.text(
                x=geom.centroid.x, y=geom.centroid.y,
                s=str(idx), fontsize=8, ha='center', va='center',
                color='darkblue', weight='bold'
            )

        buffer = 1
        minx, miny, maxx, maxy = gdf_bboxes.total_bounds
        ax.set_xlim(minx - buffer, maxx + buffer)
        ax.set_ylim(miny - buffer, maxy + buffer)

        plt.title(f"{title} ({len(bboxes_list)} cells)")
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.show()

    # ── unchanged ─────────────────────────────────────────────────────────
    def plot_bboxes_intensity(
        self,
        bboxes_list: List[Tuple[float, float, float, float, float]],
        title: str
    ):
        geometries = []
        counts = []

        for lat_min, lon_min, lat_max, lon_max, v_count in bboxes_list:
            geometries.append(box(lon_min, lat_min, lon_max, lat_max))
            counts.append(v_count)

        gdf_bboxes = gpd.GeoDataFrame(
            {"vessel_count": counts, "geometry": geometries},
            crs="EPSG:4326"
        )

        world = gpd.read_file(geodatasets.get_path('naturalearth.land'))
        fig, ax = plt.subplots(figsize=(12, 8))
        world.plot(ax=ax, color='lightgrey', edgecolor='white')

        gdf_bboxes.plot(
            ax=ax, column='vessel_count', cmap='OrRd',
            edgecolor='black', linewidth=0.5, alpha=0.75,
            legend=True, legend_kwds={'label': "Vessel Count", 'shrink': 0.6}
        )

        buffer = 1
        minx, miny, maxx, maxy = gdf_bboxes.total_bounds
        ax.set_xlim(minx - buffer, maxx + buffer)
        ax.set_ylim(miny - buffer, maxy + buffer)

        plt.title(f"{title} ({len(bboxes_list)} cells)")
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.show()

    # ── unchanged ─────────────────────────────────────────────────────────
    def extract_bbox_and_counts(
        self,
        json_file_path: str
    ) -> List[Tuple[float, float, float, float, float]]:
        results = []
        with open(json_file_path, 'rb') as f:
            for record in ijson.items(f, 'item'):
                bbox_tuple = (
                    record["bbox_min_lat"],
                    record["bbox_min_lon"],
                    record["bbox_max_lat"],
                    record["bbox_max_lon"],
                    record["vessel_count"]
                )
                results.append(bbox_tuple)
        return results

    # ── unchanged ─────────────────────────────────────────────────────────
    def Get_patch_DataFile(
        self,
        min_lat: Optional[float] = None,
        max_lat: Optional[float] = None,
        min_lon: Optional[float] = None,
        max_lon: Optional[float] = None,
        location_name: str = "Gulf_of_Mexico",
        create_new: bool = True,
        dataset_dir: Optional[str] = None,
        for_group: bool = False,
        dataset_name: str = "Mexican_Gulf",
        dataset_location: str = r"D:\AIS_project\data\CSV_files\GulfMexico_AIS",
    ) -> list[str]:
        if not dataset_location:
            raise ValueError("dataset_location must be a non-empty path string.")

        os.makedirs(dataset_location, exist_ok=True)

        created_files: list[str] = self.filter_by_bbox_give_path(
            min_lat=min_lat if isinstance(min_lat, float) else self.min_lat,
            max_lat=max_lat if isinstance(max_lat, float) else self.max_lat,
            min_lon=min_lon if isinstance(min_lon, float) else self.min_lon,
            max_lon=max_lon if isinstance(max_lon, float) else self.max_lon,
            location_name=location_name,
            create_new=create_new,
            dataset_dir=dataset_dir,
            for_group=for_group,
        )

        if not created_files:
            print(f"No files were created for dataset '{dataset_name}'. Nothing to move.")
            return []

        moved_files: list[str] = []
        for src_path in created_files:
            fname    = os.path.basename(src_path)
            dst_path = os.path.abspath(os.path.join(dataset_location, fname))
            shutil.move(src_path, dst_path)
            moved_files.append(dst_path)
            print(f"  Moved: {src_path}\n      → {dst_path}")

        print(f"\nGet_patch_DataFile complete — '{dataset_name}'")
        print(f"  Files moved       : {len(moved_files):,}")
        print(f"  Destination dir   : {os.path.abspath(dataset_location)}")

        return moved_files

    # ── unchanged ─────────────────────────────────────────────────────────
    def Clean_MMSI_DataFrame(
        self,
        MMSI_CSV_FILE: str,
        Category_keep: List[str],
        create_new: bool,
        CSV_file_exist: bool = False,
        dataset_dir: Optional[str] = None,
        for_group: bool = True
    ):
        _, csv_path, _ = self.get_unique_mmsi(
            output_file_name=MMSI_CSV_FILE,
            dataset_dir=dataset_dir,
            for_group=for_group
        )
        if not os.path.exists(csv_path):
            raise ValueError("Csv file not exist")
        else:
            print("Csv file exist")
            df = pl.scan_csv(csv_path)
            row_count = df.select(pl.count()).collect().item()
            print(f"Total rows in MMSI file: {row_count}")

        dictionary = self.read_mmsi(path=csv_path, group=True)

        count = 0
        valid_mmsi_set: set[int] = set()

        for key in dictionary:
            num = len(dictionary[key])  # type: ignore
            print(f" {key}: Valid mmsi found {num}")
            count += num
            if key in Category_keep:
                valid_mmsi_set.update(dictionary[key])  # type: ignore

        print(f"\nTotal valid MMSIs across all categories: {count}")
        print(f"MMSIs to keep (categories={Category_keep}): {len(valid_mmsi_set)}")

        resolved_dir = dataset_dir if dataset_dir is not None else self.dataset_dir
        if os.path.isfile(resolved_dir):
            resolved_dir = os.path.dirname(resolved_dir)

        csv_files = [
            f for f in os.listdir(resolved_dir)
            if f.endswith(".csv") and os.path.isfile(os.path.join(resolved_dir, f))
        ]

        if not csv_files:
            print("No CSV files found in dataset directory.")
            return

        print(f"\nProcessing {len(csv_files)} CSV file(s) in: {resolved_dir}")
        valid_mmsi_list = list(valid_mmsi_set)

        for csv_file in csv_files:
            file_path = os.path.join(resolved_dir, csv_file)
            lf = pl.scan_csv(file_path)
            original_count = lf.select(pl.count()).collect().item()
            filtered_lf = (
                lf
                .filter(pl.col("mmsi").is_not_null())
                .with_columns(pl.col("mmsi").cast(pl.Int64))
                .filter(pl.col("mmsi").is_in(valid_mmsi_list))
            )
            filtered_df = filtered_lf.collect()
            filtered_count = len(filtered_df)
            print(f"  {csv_file}: {original_count} rows → {filtered_count} rows kept")

            if create_new:
                base, ext = os.path.splitext(csv_file)
                new_file_path = os.path.join(resolved_dir, f"filtered_{base}{ext}")
                filtered_df.write_csv(new_file_path)
                print(f"    Saved new file: filtered_{base}{ext}")
            else:
                filtered_df.write_csv(file_path)
                print(f"    Overwritten: {csv_file}")

        print("\nDone. All files processed.")

    # ── unchanged ─────────────────────────────────────────────────────────
    def Clean_MMSI_DataFrame_gpt(
        self,
        MMSI_CSV_FILE: str,
        Category_keep: List[str],
        create_new: bool,
        dataset_dir: Optional[str] = None,
        for_group: bool = True
    ):
        _, csv_path, _ = self.get_unique_mmsi(
            output_file_name=MMSI_CSV_FILE,
            dataset_dir=dataset_dir,
            for_group=for_group
        )

        if not os.path.exists(csv_path):
            raise ValueError("Csv file not exist")

        print("Csv file exist")
        df = pl.scan_csv(csv_path)
        row_count = df.select(pl.len()).collect().item()
        print(f"Total rows in MMSI file: {row_count}")

        dictionary = self.read_mmsi(path=csv_path, group=True)
        count = 0
        valid_mmsi_set: set[int] = set()

        for key in dictionary:
            num = len(dictionary[key])  # type: ignore
            print(f"{key}: Valid MMSI found {num}")
            count += num
            if key in Category_keep:
                valid_mmsi_set.update(dictionary[key])  # type: ignore

        print(f"\nTotal valid MMSIs across all categories: {count}")
        print(f"MMSIs to keep (categories={Category_keep}): {len(valid_mmsi_set)}")

        if len(valid_mmsi_set) == 0:
            print(" No valid MMSIs found. Exiting early.")
            return
        valid_df = pl.DataFrame({"mmsi": list(valid_mmsi_set)}).lazy()

        resolved_dir = dataset_dir if dataset_dir else self.dataset_dir
        if os.path.isfile(resolved_dir):
            resolved_dir = os.path.dirname(resolved_dir)
        csv_files = [
            f for f in os.listdir(resolved_dir)
            if f.endswith(".csv") and os.path.isfile(os.path.join(resolved_dir, f))
        ]
        if not csv_files:
            print("No CSV files found in dataset directory.")
            return
        print(f"\nProcessing {len(csv_files)} CSV file(s) in: {resolved_dir}")

        for csv_file in csv_files:
            file_path = os.path.join(resolved_dir, csv_file)
            print(f"\nProcessing: {csv_file}")
            lf = pl.scan_csv(file_path)
            original_count = lf.select(pl.len()).collect().item()
            filtered_lf = (
                lf
                .filter(pl.col("mmsi").is_not_null())
                .with_columns(pl.col("mmsi").cast(pl.Int64, strict=False))
                .join(valid_df, on="mmsi", how="inner")
            )
            filtered_count = filtered_lf.select(pl.len()).collect().item()
            print(f"  {original_count} rows → {filtered_count} rows kept")

            if create_new:
                base, ext = os.path.splitext(csv_file)
                output_path = os.path.join(resolved_dir, f"filtered_{base}{ext}")
                filtered_lf.sink_csv(output_path)
                print(f"    Saved new file: filtered_{base}{ext}")
            else:
                temp_path = file_path + ".tmp"
                filtered_lf.sink_csv(temp_path)
                os.replace(temp_path, file_path)
                print(f"    Overwritten safely: {csv_file}")
        print("\nDone. All files processed successfully.")

    # ── unchanged ─────────────────────────────────────────────────────────
    def sort_bboxes(
        self,
        bboxes: List[Tuple[float, float, float, float]]
    ) -> List[Tuple[float, float, float, float]]:
        return sorted(bboxes, key=lambda b: (-b[0], b[1]))

    # ── unchanged ─────────────────────────────────────────────────────────
    def is_sorted_bboxes(
        self,
        bboxes: List[Tuple[float, float, float, float]]
    ) -> bool:
        for i in range(1, len(bboxes)):
            prev_lat, prev_lon, _, _ = bboxes[i - 1]
            curr_lat, curr_lon, _, _ = bboxes[i]
            if curr_lat > prev_lat:
                return False
            if curr_lat == prev_lat and curr_lon < prev_lon:
                return False
        return True

    # ══════════════════════════════════════════════════════════════════════
    #  HELPER: extract clean YYYY-MM-DD date from any filename
    #  Handles: Gulf_of_Mexico_ais-2022-01-01_macro_raw.parquet
    #           AIS_2022_01_01_macro_raw.parquet
    #           2022-01-01_macro_raw.parquet
    #           ais-2022-01-01.csv
    # ══════════════════════════════════════════════════════════════════════
    @staticmethod
    def _date_from_filename(filename: str) -> Optional[str]:
        """Extract YYYY-MM-DD date from a filename, ignoring any prefix."""
        # Try YYYY-MM-DD (dash-separated)
        match = re.search(r"(\d{4}-\d{2}-\d{2})", filename)
        if match:
            return match.group(1)
        # Try YYYY_MM_DD (underscore-separated)
        match = re.search(r"(\d{4})_(\d{2})_(\d{2})", filename)
        if match:
            return f"{match.group(1)}-{match.group(2)}-{match.group(3)}"
        return None

    # ══════════════════════════════════════════════════════════════════════
    #  KEPT FOR BACKWARD COMPATIBILITY — not used in optimized hot-path
    # ══════════════════════════════════════════════════════════════════════
    def _each_block_mmsi_extract(
        self,
        bbox: Tuple[float, float, float, float],
        dataset_dir: str,
    ) -> pl.DataFrame:
        """Legacy method — kept for callers outside create_parquet_json.
        The optimized create_parquet_json no longer calls this."""
        if not isinstance(dataset_dir, str):
            raise ValueError("Cannot proceed: a CSV file path string is required.")

        min_lat, min_lon, max_lat, max_lon = bbox

        path_address = self.filter_by_bbox_give_path(
            min_lat=min_lat, max_lat=max_lat,
            min_lon=min_lon, max_lon=max_lon,
            location_name="patch_detection",
            dataset_dir=dataset_dir,
            create_new=True, for_group=False,
        )

        if not path_address:
            return pl.DataFrame()

        if len(path_address) > 1:
            raise ValueError(
                "Expected a single file path but received multiple. "
                "Check that for_group=False is set correctly."
            )

        data_path = path_address[0]
        if not os.path.exists(data_path):
            raise ValueError(f"Output file does not exist: {data_path}")

        return pl.read_csv(data_path, schema_overrides=self._CSV_SCHEMA)

    # ══════════════════════════════════════════════════════════════════════
    #  OPTIMIZED: _aggregate_bbox_features — vectorized sin/cos
    #  Replaces slow map_elements() Python UDF with native Polars ops.
    #  Still available for the overlap fallback path.
    # ══════════════════════════════════════════════════════════════════════
    def _aggregate_bbox_features(
        self,
        df: pl.DataFrame,
        bbox: Tuple[float, float, float, float],
    ) -> dict:
        min_lat, min_lon, max_lat, max_lon = bbox

        vessel_count = df["mmsi"].n_unique()
        ping_count   = len(df)

        sog_series = df["sog"].fill_null(0.0).fill_nan(0.0)
        mean_sog   = float(sog_series.mean())  # type:ignore

        # ── OPTIMIZED: vectorized COG sin/cos ────────────────────────────
        # Old code used map_elements(lambda v: ...) — row-by-row Python.
        # New code uses native Polars .radians().sin() — runs in Rust.
        cog_clean = df.select(
            pl.when(
                pl.col("cog").is_null() | pl.col("cog").is_nan() | (pl.col("cog") == 360.0)
            ).then(0.0).otherwise(pl.col("cog")).alias("cog_clean")
        )["cog_clean"]

        cog_rad    = cog_clean * (math.pi / 180.0)
        sin_series = cog_rad.sin()
        cos_series = cog_rad.cos()

        mean_cog_sin = float(sin_series.mean())  # type:ignore
        mean_cog_cos = float(cos_series.mean())  # type:ignore

        resultant_length = math.sqrt(mean_cog_sin ** 2 + mean_cog_cos ** 2)
        cog_circular_var = round(1.0 - resultant_length, 6)

        type_series    = df["vessel_type"].drop_nulls()
        type_diversity = type_series.n_unique() if len(type_series) > 0 else 0

        if len(type_series) > 0:
            vc = (
                type_series
                .value_counts()
                .sort("count", descending=True)
            )
            dominant_type = int(vc["vessel_type"][0])
        else:
            dominant_type = 0

        return {
            "bbox_min_lat":     min_lat,
            "bbox_min_lon":     min_lon,
            "bbox_max_lat":     max_lat,
            "bbox_max_lon":     max_lon,
            "vessel_count":     vessel_count,
            "ping_count":       ping_count,
            "mean_sog":         round(mean_sog, 6),
            "mean_cog_sin":     round(mean_cog_sin, 6),
            "mean_cog_cos":     round(mean_cog_cos, 6),
            "cog_circular_var": cog_circular_var,
            "type_diversity":   type_diversity,
            "dominant_type":    dominant_type,
        }

    # ══════════════════════════════════════════════════════════════════════
    #  FULLY REWRITTEN: create_parquet_json
    #
    #  OLD APPROACH (the 11-hour bottleneck):
    #    for each daily CSV:
    #      for each bbox (~800):
    #        _each_block_mmsi_extract()     ← full CSV scan from disk
    #          → filter_by_bbox_give_path() ← another CSV scan + disk write
    #          → pl.read_csv(temp_file)     ← disk read of the temp file
    #        _aggregate_bbox_features()     ← map_elements Python UDF
    #    Total: ~800 disk scans per day × 31 days = ~25,000 full scans
    #
    #  NEW APPROACH:
    #    for each daily CSV:
    #      1. Read CSV ONCE into memory
    #      2. Assign each row to its bbox using vectorized floor math
    #      3. Single group_by on (lat_idx, lon_idx) for all aggregations
    #      4. Compute dominant_type with a second lightweight group_by
    #    Total: 1 disk read per day × 31 days = 31 reads
    #
    #  Expected speedup: 100–800× for the no-overlap case.
    # ══════════════════════════════════════════════════════════════════════
    def create_parquet_json(
        self,
        output_dir: str,
        for_group: bool = False,
        dataset_dir: Optional[str] = None,
    ) -> List[str]:
        os.makedirs(output_dir, exist_ok=True)
        created_files: List[str] = []

        # ── resolve CSV files ────────────────────────────────────────────
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

        stride     = self._stride
        n_lat_bins = self._n_lat_bins
        n_lon_bins = self._n_lon_bins
        use_fast   = (self.overlap_pct == 0.0)

        # ── process each daily CSV ───────────────────────────────────────
        for csv_path in tqdm(csv_files, desc="Processing daily CSVs", unit="file"):

            date_stem = os.path.splitext(os.path.basename(csv_path))[0]

            # ── Step 1: READ CSV ONCE ────────────────────────────────────
            df = pl.read_csv(
                csv_path,
                schema_overrides=self._CSV_SCHEMA,
                infer_schema_length=1000,
            )

            # Filter to bounding region (rows outside the grid are useless)
            df = df.filter(
                pl.col("latitude").is_between(self.min_lat, self.max_lat) &
                pl.col("longitude").is_between(self.min_lon, self.max_lon)
            )

            if df.is_empty():
                tqdm.write(f"  [{date_stem}] No rows inside region — skipping.")
                continue

            # ==============================================================
            #  FAST PATH: no overlap → vectorized grid assignment + group_by
            # ==============================================================
            if use_fast:
                bbox_records = self._fast_aggregate_all_bboxes(df, date_stem)
            # ==============================================================
            #  FALLBACK: overlap > 0 → iterate bboxes on in-memory DataFrame
            #  Still 100×+ faster than old code (no disk I/O per bbox).
            # ==============================================================
            else:
                bbox_records = self._overlap_aggregate_all_bboxes(df, date_stem)

            if not bbox_records:
                tqdm.write(f"  [{date_stem}] No occupied bboxes found — skipping.")
                continue

            # ── sort north→south, west→east ──────────────────────────────
            bbox_records.sort(key=lambda r: (-r["bbox_min_lat"], r["bbox_min_lon"]))

            # ── save Parquet ─────────────────────────────────────────────
            parquet_path = os.path.abspath(
                os.path.join(output_dir, f"{date_stem}_macro_raw.parquet")
            )
            pl.DataFrame(bbox_records).write_parquet(parquet_path)
            created_files.append(parquet_path)

            # ── save JSON ────────────────────────────────────────────────
            json_path = os.path.abspath(
                os.path.join(output_dir, f"{date_stem}_macro_raw.json")
            )
            with open(json_path, "w", encoding="utf-8") as fh:
                json.dump(bbox_records, fh, indent=2, ensure_ascii=False)
            created_files.append(json_path)

            tqdm.write(
                f"  [{date_stem}] {len(bbox_records):>4} occupied cells → "
                f"{os.path.basename(parquet_path)}  +  {os.path.basename(json_path)}"
            )

        # ── summary ──────────────────────────────────────────────────────
        n_days = len(csv_files)
        n_parq = sum(1 for f in created_files if f.endswith(".parquet"))
        n_json = sum(1 for f in created_files if f.endswith(".json"))
        print(f"\ncreate_parquet_json complete")
        print(f"  CSVs processed : {n_days:,}")
        print(f"  Parquet files  : {n_parq:,}")
        print(f"  JSON files     : {n_json:,}")
        print(f"  Output dir     : {output_dir}")

        return created_files

    # ──────────────────────────────────────────────────────────────────────
    #  FAST PATH: vectorized bbox assignment + single group_by
    #  Only works when overlap_pct == 0 (no bin overlap).
    # ──────────────────────────────────────────────────────────────────────
    def _fast_aggregate_all_bboxes(
        self,
        df: pl.DataFrame,
        date_stem: str,
    ) -> List[dict]:

        stride     = self._stride
        n_lat_bins = self._n_lat_bins
        n_lon_bins = self._n_lon_bins

        # ── Step 2: assign each row to a grid cell ───────────────────────
        df = df.with_columns([
            ((pl.col("latitude")  - self.min_lat) / stride).floor().cast(pl.Int32).clip(0, n_lat_bins - 1).alias("_lat_idx"),
            ((pl.col("longitude") - self.min_lon) / stride).floor().cast(pl.Int32).clip(0, n_lon_bins - 1).alias("_lon_idx"),
        ])

        # ── Step 3: prepare COG column (clean 360 → 0, null → 0) ────────
        cog_clean_expr = (
            pl.when(
                pl.col("cog").is_null() | pl.col("cog").is_nan() | (pl.col("cog") == 360.0)
            ).then(0.0).otherwise(pl.col("cog"))
        )

        df = df.with_columns([
            (cog_clean_expr * (math.pi / 180.0)).sin().alias("_cog_sin"),
            (cog_clean_expr * (math.pi / 180.0)).cos().alias("_cog_cos"),
        ])

        # ── Step 3a: main group_by ──────────────────────────────────────
        agg_df = df.group_by(["_lat_idx", "_lon_idx"]).agg([
            pl.col("mmsi").n_unique().alias("vessel_count"),
            pl.len().alias("ping_count"),
            pl.col("sog").fill_null(0.0).fill_nan(0.0).mean().alias("mean_sog"),
            pl.col("_cog_sin").mean().alias("mean_cog_sin"),
            pl.col("_cog_cos").mean().alias("mean_cog_cos"),
            pl.col("vessel_type").drop_nulls().n_unique().alias("type_diversity"),
        ])

        # ── Step 3b: dominant type (mode per grid cell) ─────────────────
        #    Separate lightweight group_by to find the most frequent type.
        dominant_df = (
            df.filter(pl.col("vessel_type").is_not_null())
            .group_by(["_lat_idx", "_lon_idx", "vessel_type"])
            .agg(pl.len().alias("_type_count"))
            .sort(["_lat_idx", "_lon_idx", "_type_count"], descending=[False, False, True])
            .group_by(["_lat_idx", "_lon_idx"], maintain_order=True)
            .first()
            .select(["_lat_idx", "_lon_idx", pl.col("vessel_type").alias("dominant_type")])
        )

        agg_df = agg_df.join(dominant_df, on=["_lat_idx", "_lon_idx"], how="left")

        # ── Step 3c: compute derived columns ────────────────────────────
        agg_df = agg_df.with_columns([
            # bbox coordinates from grid indices
            (pl.col("_lat_idx").cast(pl.Float64) * stride + self.min_lat).alias("bbox_min_lat"),
            (pl.col("_lon_idx").cast(pl.Float64) * stride + self.min_lon).alias("bbox_min_lon"),
            # circular variance
            (1.0 - (pl.col("mean_cog_sin").pow(2) + pl.col("mean_cog_cos").pow(2)).sqrt())
            .clip(0.0, 1.0).alias("cog_circular_var"),
            # fill null dominant_type
            pl.col("dominant_type").fill_null(0),
        ]).with_columns([
            (pl.col("bbox_min_lat") + self.bin_size).alias("bbox_max_lat"),
            (pl.col("bbox_min_lon") + self.bin_size).alias("bbox_max_lon"),
        ])

        # ── Step 4: convert to list of dicts ─────────────────────────────
        output_cols = [
            "bbox_min_lat", "bbox_min_lon", "bbox_max_lat", "bbox_max_lon",
            "vessel_count", "ping_count", "mean_sog",
            "mean_cog_sin", "mean_cog_cos", "cog_circular_var",
            "type_diversity", "dominant_type",
        ]

        # Round floats for consistency with original output
        result_df = agg_df.select(output_cols).with_columns([
            pl.col("mean_sog").round(6),
            pl.col("mean_cog_sin").round(6),
            pl.col("mean_cog_cos").round(6),
            pl.col("cog_circular_var").round(6),
        ])

        return result_df.to_dicts()

    # ──────────────────────────────────────────────────────────────────────
    #  OVERLAP FALLBACK: iterate bboxes on in-memory DataFrame
    #  Still vastly faster than old code — no disk I/O per bbox.
    # ──────────────────────────────────────────────────────────────────────
    def _overlap_aggregate_all_bboxes(
        self,
        df: pl.DataFrame,
        date_stem: str,
    ) -> List[dict]:

        bbox_records: List[dict] = []

        for bbox in tqdm(
            self.bboxes_list,
            desc=f"  Bboxes [{date_stem}]",
            unit="cell",
            leave=False,
        ):
            min_lat, min_lon, max_lat, max_lon = bbox

            # In-memory filter — no disk I/O
            filtered = df.filter(
                pl.col("latitude").is_between(min_lat, max_lat) &
                pl.col("longitude").is_between(min_lon, max_lon)
            )

            if filtered.is_empty():
                continue

            record = self._aggregate_bbox_features(df=filtered, bbox=bbox)
            bbox_records.append(record)

        return bbox_records

    # ── unchanged ─────────────────────────────────────────────────────────
    def compute_norm_stats(
        self,
        raw_parquet_dir: str,
        output_dir: str,
        training_date_stems: Optional[List[str]] = None,
    ) -> dict:
        os.makedirs(output_dir, exist_ok=True)

        all_parquet = sorted([
            f for f in os.listdir(raw_parquet_dir)
            if f.endswith("_macro_raw.parquet")
        ])

        if not all_parquet:
            raise FileNotFoundError(
                f"No *_macro_raw.parquet files found in: {raw_parquet_dir}"
            )

        if training_date_stems is not None:
            training_set = set(training_date_stems)
            all_parquet = [
                f for f in all_parquet
                if self._date_from_filename(f) in training_set
            ]
            if not all_parquet:
                raise ValueError(
                    "No parquet files matched the provided training_date_stems. "
                    f"Stems expected: {list(training_set)[:5]}... "
                    f"Stems found in files: {[self._date_from_filename(f) for f in sorted(os.listdir(raw_parquet_dir)) if f.endswith('_macro_raw.parquet')][:5]}..."
                )

        print(f"Computing norm stats from {len(all_parquet)} training parquet file(s)...")

        max_vessel_count: int = 0
        max_ping_count:   int = 0
        all_type_codes:   set = set()

        for fname in tqdm(all_parquet, desc="Scanning training parquets", unit="file"):
            fpath = os.path.join(raw_parquet_dir, fname)
            df = pl.read_parquet(
                fpath,
                columns=["vessel_count", "ping_count", "dominant_type", "type_diversity"]
            )

            if df.is_empty():
                continue

            file_max_vc = int(df["vessel_count"].max() or 0)  # type:ignore
            if file_max_vc > max_vessel_count:
                max_vessel_count = file_max_vc

            file_max_pc = int(df["ping_count"].max() or 0)  # type:ignore
            if file_max_pc > max_ping_count:
                max_ping_count = file_max_pc

            type_vals = df["dominant_type"].drop_nulls().to_list()
            all_type_codes.update(int(v) for v in type_vals)

            tqdm.write(
                f"  {fname}: max_vc={file_max_vc}  max_pc={file_max_pc}  "
                f"types_seen={len(all_type_codes)}"
            )

        max_type_code      = int(max(all_type_codes)) if all_type_codes else 99
        total_unique_types = len(all_type_codes)

        stats = {
            "MAX_VESSEL_COUNT_PER_CELL": max_vessel_count,
            "MAX_PING_COUNT_PER_CELL":   max_ping_count,
            "TOTAL_UNIQUE_TYPES":        total_unique_types,
            "MAX_TYPE_CODE":             max_type_code,
            "GULF_LAT_MIN": self.min_lat,
            "GULF_LAT_MAX": self.max_lat,
            "GULF_LON_MIN": self.min_lon,
            "GULF_LON_MAX": self.max_lon,
            "MEAN_LENGTH":  None,
            "MEAN_DRAFT":   None,
        }

        stats_path = os.path.abspath(os.path.join(output_dir, "norm_stats.json"))
        with open(stats_path, "w", encoding="utf-8") as fh:
            json.dump(stats, fh, indent=2)

        print(f"\nNorm stats computed and saved → {stats_path}")
        print(f"  MAX_VESSEL_COUNT_PER_CELL : {max_vessel_count:,}")
        print(f"  MAX_PING_COUNT_PER_CELL   : {max_ping_count:,}")
        print(f"  TOTAL_UNIQUE_TYPES        : {total_unique_types}")
        print(f"  MAX_TYPE_CODE             : {max_type_code}")
        print(f"  Gulf bounds               : lat [{self.min_lat}, {self.max_lat}]"
              f"  lon [{self.min_lon}, {self.max_lon}]")

        return stats

    # ── unchanged ─────────────────────────────────────────────────────────
    def normalise_and_save_tensors(
        self,
        raw_parquet_dir: str,
        tensor_output_dir: str,
        norm_stats_path: str,
        date_stems: Optional[List[str]] = None,
    ) -> List[str]:
        os.makedirs(tensor_output_dir, exist_ok=True)

        if not os.path.exists(norm_stats_path):
            raise FileNotFoundError(
                f"norm_stats.json not found at: {norm_stats_path}\n"
                "Run compute_norm_stats() on training data first."
            )

        with open(norm_stats_path, "r", encoding="utf-8") as fh:
            stats = json.load(fh)

        MAX_VC    = stats["MAX_VESSEL_COUNT_PER_CELL"]
        MAX_PC    = stats["MAX_PING_COUNT_PER_CELL"]
        MAX_TYPE  = stats["MAX_TYPE_CODE"]
        TOT_TYPES = stats["TOTAL_UNIQUE_TYPES"]
        LAT_MIN   = stats["GULF_LAT_MIN"]
        LAT_MAX   = stats["GULF_LAT_MAX"]
        LON_MIN   = stats["GULF_LON_MIN"]
        LON_MAX   = stats["GULF_LON_MAX"]

        LAT_RANGE = LAT_MAX - LAT_MIN
        LON_RANGE = LON_MAX - LON_MIN

        if MAX_VC == 0 or MAX_PC == 0 or MAX_TYPE == 0 or TOT_TYPES == 0:
            raise ValueError(
                "One or more stats are zero — norm_stats.json may have been "
                "computed from empty data.  Re-run compute_norm_stats()."
            )

        all_parquet = sorted([
            f for f in os.listdir(raw_parquet_dir)
            if f.endswith("_macro_raw.parquet")
        ])

        if date_stems is not None:
            stem_set    = set(date_stems)
            all_parquet = [
                f for f in all_parquet
                if self._date_from_filename(f) in stem_set
            ]

        if not all_parquet:
            print("No matching parquet files found.")
            return []

        created_files: List[str] = []

        for fname in tqdm(all_parquet, desc="Normalising macro tensors", unit="file"):

            # Extract clean date for output naming — ensures macro .pt files
            # are named "2022-01-01_macro.pt" (matching what build_dataset_index expects)
            # regardless of the parquet's full filename prefix.
            date_stem = self._date_from_filename(fname) or fname.replace("_macro_raw.parquet", "")
            fpath     = os.path.join(raw_parquet_dir, fname)
            df        = pl.read_parquet(fpath)

            if df.is_empty():
                tqdm.write(f"  [{date_stem}] Empty parquet — skipping.")
                continue

            df = df.sort(
                by=["bbox_min_lat", "bbox_min_lon"],
                descending=[True, False]
            )

            rows = df.to_dicts()
            feature_rows: List[List[float]] = []

            for r in rows:
                lat_norm = (r["bbox_min_lat"] - LAT_MIN) / LAT_RANGE
                lat_norm = max(0.0, min(1.0, lat_norm))

                lon_norm = (r["bbox_min_lon"] - LON_MIN) / LON_RANGE
                lon_norm = max(0.0, min(1.0, lon_norm))

                vc = max(0, int(r["vessel_count"]))
                vessel_count_norm = math.log1p(vc) / math.log1p(MAX_VC)
                vessel_count_norm = max(0.0, min(1.0, vessel_count_norm))

                pc = max(0, int(r["ping_count"]))
                ping_density_norm = math.log1p(pc) / math.log1p(MAX_PC)
                ping_density_norm = max(0.0, min(1.0, ping_density_norm))

                mean_sog_norm = float(r["mean_sog"] or 0.0) / 30.0
                mean_sog_norm = max(0.0, min(1.0, mean_sog_norm))

                mean_cog_sin = float(r["mean_cog_sin"] or 0.0)
                mean_cog_cos = float(r["mean_cog_cos"] or 0.0)
                mean_cog_sin = max(-1.0, min(1.0, mean_cog_sin))
                mean_cog_cos = max(-1.0, min(1.0, mean_cog_cos))

                cog_circular_var = float(r["cog_circular_var"] or 0.0)
                cog_circular_var = max(0.0, min(1.0, cog_circular_var))

                td = int(r["type_diversity"] or 0)
                type_diversity_norm = td / TOT_TYPES
                type_diversity_norm = max(0.0, min(1.0, type_diversity_norm))

                dt = int(r["dominant_type"] or 0)
                dominant_type_norm = dt / MAX_TYPE
                dominant_type_norm = max(0.0, min(1.0, dominant_type_norm))

                feature_rows.append([
                    lat_norm, lon_norm, vessel_count_norm, ping_density_norm,
                    mean_sog_norm, mean_cog_sin, mean_cog_cos,
                    cog_circular_var, type_diversity_norm, dominant_type_norm,
                ])

            if not feature_rows:
                tqdm.write(f"  [{date_stem}] No features after normalisation — skipping.")
                continue

            tensor = torch.tensor(feature_rows, dtype=torch.float32)

            assert tensor.ndim == 2,              f"Expected 2D tensor, got {tensor.ndim}D"
            assert tensor.shape[1] == 10,         f"Expected 10 features, got {tensor.shape[1]}"
            assert not torch.isnan(tensor).any(), f"NaN detected in {date_stem}"

            out_path = os.path.abspath(
                os.path.join(tensor_output_dir, f"{date_stem}_macro.pt")
            )
            torch.save(tensor, out_path)
            created_files.append(out_path)

            tqdm.write(
                f"  [{date_stem}] shape {list(tensor.shape)} → "
                f"{os.path.basename(out_path)}"
            )

        print(f"\nnormalise_and_save_tensors complete")
        print(f"  Files processed : {len(all_parquet):,}")
        print(f"  Tensors saved   : {len(created_files):,}")
        print(f"  Output dir      : {tensor_output_dir}")

        return created_files

    # ── unchanged ─────────────────────────────────────────────────────────
    def validate_macro_tensors(
        self,
        tensor_dir: str,
        expected_date_stems: Optional[List[str]] = None,
        expected_feature_dim: int = 10,
        value_min: float = -1.1,
        value_max: float = 1.1,
    ) -> dict:
        pt_files = sorted([
            f for f in os.listdir(tensor_dir)
            if f.endswith("_macro.pt")
        ])

        missing_dates: List[str] = []
        if expected_date_stems is not None:
            found_stems = {self._date_from_filename(f) or f.replace("_macro.pt", "") for f in pt_files}
            missing_dates = [
                stem for stem in expected_date_stems
                if stem not in found_stems
            ]

        errors: dict = {}
        passed = 0
        failed = 0

        for fname in tqdm(pt_files, desc="Validating macro tensors", unit="file"):
            fpath       = os.path.join(tensor_dir, fname)
            file_errors: List[str] = []

            try:
                tensor = torch.load(fpath, map_location="cpu", weights_only=True)
            except Exception as e:
                errors[fname] = [f"Failed to load: {e}"]
                failed += 1
                continue

            if tensor.ndim != 2:
                file_errors.append(
                    f"Wrong number of dimensions: expected 2, got {tensor.ndim}"
                )
            elif tensor.shape[1] != expected_feature_dim:
                file_errors.append(
                    f"Wrong feature dim: expected {expected_feature_dim}, "
                    f"got {tensor.shape[1]}"
                )
            if torch.isnan(tensor).any():
                nan_count = int(torch.isnan(tensor).sum().item())
                file_errors.append(f"Contains {nan_count} NaN value(s)")
            if torch.isinf(tensor).any():
                inf_count = int(torch.isinf(tensor).sum().item())
                file_errors.append(f"Contains {inf_count} Inf value(s)")

            actual_min = float(tensor.min().item())
            actual_max = float(tensor.max().item())
            if actual_min < value_min:
                file_errors.append(
                    f"Value below expected min: {actual_min:.6f} < {value_min}"
                )
            if actual_max > value_max:
                file_errors.append(
                    f"Value above expected max: {actual_max:.6f} > {value_max}"
                )
            if tensor.shape[0] == 0:
                file_errors.append("Tensor has zero rows (no occupied cells)")

            if file_errors:
                errors[fname] = file_errors
                failed += 1
                tqdm.write(f"  FAIL [{fname}]: {' | '.join(file_errors)}")
            else:
                passed += 1

        result = {
            "total_files":   len(pt_files),
            "passed":        passed,
            "failed":        failed,
            "missing_dates": missing_dates,
            "errors":        errors,
        }

        print(f"\nvalidate_macro_tensors complete")
        print(f"  Total .pt files  : {len(pt_files):,}")
        print(f"  Passed           : {passed:,}")
        print(f"  Failed           : {failed:,}")
        print(f"  Missing dates    : {len(missing_dates):,}")

        if missing_dates:
            print(f"\n  Missing date stems:")
            for stem in missing_dates:
                print(f"    {stem}")

        if errors:
            print(f"\n  Files with errors:")
            for fname, errs in errors.items():
                for e in errs:
                    print(f"    {fname}: {e}")
        else:
            print("\n  All files passed validation.")

        return result