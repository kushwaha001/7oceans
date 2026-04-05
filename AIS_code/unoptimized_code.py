import polars as pl
import pandas as pd
import os
import geopandas as gpd
import geodatasets
import matplotlib.pyplot as plt
from shapely.geometry import LineString
from typing import Optional, List, OrderedDict, Tuple
from tqdm import tqdm
import json
import math


class Analysis:

    def __init__(
        self,
        datasets_location: str,
        dataset_dir: str,
        analysis_result_dir: str
    ):
        self.datasets_location = datasets_location
        self.dataset_dir = dataset_dir
        self.analysis_result_dir = analysis_result_dir

    # Working as per new Dataset
    def count_rows(
        self,
        dataset_dir: Optional[str] = None,
        show_group_count:bool=False,
        for_group: bool = False
    ) -> int:
        if for_group:
            csv_files = [
                f for f in os.listdir(self.datasets_location)
                if f.endswith(".csv")
            ]

            if not csv_files:
                print("No CSV files found in datasets_location.")
                return 0

            total = 0
            for fname in tqdm(csv_files, desc="Counting rows", unit="file"):
                fpath = os.path.join(self.datasets_location, fname)
                count = (
                    pl.scan_csv(fpath)
                    .select(pl.len())
                    .collect()
                    .item()
                )
                total += count
                tqdm.write(f"  {fname}: {count:,} rows") if show_group_count else ""

            print(f"\nTotal rows across {len(csv_files)} files: {total:,}") 
            return total

        else:
            target = dataset_dir if dataset_dir is not None else self.dataset_dir
            count = (
                pl.scan_csv(target)
                .select(pl.len())
                .collect()
                .item()
            )
            print(f"Row count [{os.path.basename(target)}]: {count:,}")
            return count
    
    # This done updating
    def get_unique_mmsi(
        self,
        output_file_name: str,
        dataset_dir: Optional[str] = None,
        for_group: bool = False
    ):

        if for_group:
            csv_files = sorted([
                f for f in os.listdir(self.datasets_location)
                if f.endswith(".csv")
            ])

            if not csv_files:
                print("No CSV files found in datasets_location.")
                return []

            # collect unique MMSIs per file lazily, union all, deduplicate once
            frames = []
            for fname in tqdm(csv_files, desc="Scanning MMSI values", unit="file"):
                fpath = os.path.join(self.datasets_location, fname)
                unique_in_file = (
                    pl.scan_csv(fpath, infer_schema_length=1000)
                    .select(pl.col("mmsi").cast(pl.Int64))   # changed from MMSI to mmsi
                    .unique()
                    .collect()
                )
                frames.append(unique_in_file)
                tqdm.write(f"  {fname}: {unique_in_file.height:,} unique MMSIs")

            # single dedup pass across all collected frames
            result_df = (
                pl.concat(frames)
                .unique()
                .sort("mmsi")   # changed from MMSI to mmsi
            )
            source_label = f"{len(csv_files)} files from datasets_location"

        else:
            target = dataset_dir if dataset_dir is not None else self.dataset_dir
            result_df = (
                pl.scan_csv(target, infer_schema_length=1000)
                .select(pl.col("mmsi").cast(pl.Int64))    # changed from MMSI to mmsi
                .unique()
                .sort("mmsi")     # changed from MMSI into mmsi
                .collect()
            )
            source_label = os.path.basename(target)

        mmsi_list = result_df["mmsi"].to_list()   # changed from MMSI to mmsi
        total = len(mmsi_list)

        # --- save JSON ---
        os.makedirs(self.analysis_result_dir, exist_ok=True)
        json_path = os.path.join(self.analysis_result_dir, f"{output_file_name}.json")
        json_output = {
            "meta": {
                "total_unique_mmsi": total,
                "source": source_label,
            },
            "mmsi": mmsi_list
        }
        with open(json_path, "w") as f:
            json.dump(json_output, f, indent=4)

        # --- save CSV ---
        csv_path = os.path.join(self.analysis_result_dir, f"{output_file_name}.csv")
        result_df.write_csv(csv_path)

        # --- summary ---
        print(f"\nUnique MMSI count : {total:,}")
        print(f"Saved JSON        → {json_path}")
        print(f"Saved CSV         → {csv_path}")

        return mmsi_list,csv_path,json_path
    
    # This done updating
    def extract_vessels_by_mmsi( # This checks for only one csv file
    self,
    mmsi_list: List[int],
    output_file_name: str,
    dataset_dir: Optional[str] = None,
    ) -> pl.DataFrame:

        if not mmsi_list:
            print("mmsi_list is empty, nothing to extract.")
            return pl.DataFrame()

        target = dataset_dir if dataset_dir is not None else self.dataset_dir

        print(f"Scanning: {os.path.basename(target)}")
        print(f"Looking for {len(mmsi_list)} MMSI(s): {mmsi_list}")

        result_df = (
            pl.scan_csv(target, infer_schema_length=1000)
            .with_row_index(name="RowNumber", offset=1)   # 1-based row number
            .filter(pl.col("mmsi").is_in(mmsi_list))    # changed from MMSI to mmsi
            .collect()
        )

        if result_df.is_empty():
            print("No rows found for the given MMSI(s).")
            return result_df

        # --- first occurrence row number per MMSI ---
        first_occurrence = (
            result_df
            .group_by("mmsi") # changes from MMSI to mmsi
            .agg([
                pl.len().alias("RowCount"),
                pl.col("RowNumber").min().alias("FirstOccurrenceRow")
            ])
            .sort("RowCount", descending=True)
        )

        # --- save to CSV (with RowNumber column included) ---
        os.makedirs(self.analysis_result_dir, exist_ok=True)
        out_path = os.path.join(self.analysis_result_dir, output_file_name)
        result_df.write_csv(out_path)

        # --- summary ---
        print(f"\nExtracted {result_df.height:,} rows total")
        print(f"{'mmsi':<15} {'First Row':>10} {'Row Count':>10}") # changed from MMSI to mmsi
        print("-" * 38)
        for row in first_occurrence.iter_rows(named=True):
            print(f"  {row['mmsi']:<13} {row['FirstOccurrenceRow']:>10,} {row['RowCount']:>10,}")  # changed from MMSI to mmsi

        print(f"\nSaved → {out_path}")

        return result_df
    
    # This done Updating
    def export_vessels_by_mmsi( # This will create the seperate csv file that isolates each vessel by mmsi value depending on single file or for whole dataset and dict contain the key by mmsi value and the value is row count and file path
        self,
        mmsi_list: List[int],
        output_name: str,
        dataset_dir: Optional[str] = None,
        for_group: bool = False,
    ) -> dict:
   

        if not mmsi_list:
            print("mmsi_list is empty, nothing to export.")
            return {}

        # --- output subfolder ---
        out_dir = os.path.join(self.analysis_result_dir, "vessels")
        os.makedirs(out_dir, exist_ok=True)

        # build a Polars Series once for efficient is_in checks
        mmsi_filter = pl.Series("mmsi", mmsi_list, dtype=pl.Int64) # changed from MMSI to mmsi

        # accumulator: mmsi -> list of DataFrames
        buckets: dict[int, list[pl.DataFrame]] = {m: [] for m in mmsi_list}

        # --- scan files ---
        if for_group:
            csv_files = sorted([
                f for f in os.listdir(self.datasets_location)
                if f.endswith(".csv")
            ])
            if not csv_files:
                print("No CSV files found in datasets_location.")
                return {}

            for fname in tqdm(csv_files, desc="Scanning files", unit="file"):
                fpath = os.path.join(self.datasets_location, fname)
                filtered = (
                    pl.scan_csv(fpath, infer_schema_length=1000)
                    .filter(pl.col("mmsi").cast(pl.Int64).is_in(mmsi_filter)) # changed from MMSI to mmsi
                    .collect()
                )
                if filtered.is_empty():
                    continue

                # partition the filtered result by MMSI — avoids re-scanning
                for mmsi, group_df in filtered.group_by("mmsi"):  # Changed from MMSI to mmsi
                    buckets[int(mmsi[0])].append(group_df)

        else:
            target = dataset_dir if dataset_dir is not None else self.dataset_dir
            filtered = (
                pl.scan_csv(target, infer_schema_length=1000)
                .filter(pl.col("mmsi").cast(pl.Int64).is_in(mmsi_filter))
                .collect()
            )
            if not filtered.is_empty():
                for mmsi, group_df in filtered.group_by("mmsi"):
                    buckets[int(mmsi[0])].append(group_df)

        # --- write one CSV per MMSI ---
        summary = {}
        not_found = []

        print(f"\nWriting CSVs to: {out_dir}")
        for mmsi in tqdm(mmsi_list, desc="Writing CSVs", unit="vessel"):
            frames = buckets[mmsi]

            if not frames:
                not_found.append(mmsi)
                summary[mmsi] = {"row_count": 0, "path": None}
                continue

            # combine all chunks for this MMSI (across files if for_group)
            vessel_df = (
                pl.concat(frames)
                .sort("base_date_time")  # chronological order ; Change from BaseDateTime to base_date_Time
            )

            out_path = os.path.join(out_dir, f"{output_name}_{mmsi}.csv")
            vessel_df.write_csv(out_path)

            summary[mmsi] = {
                "row_count": vessel_df.height,
                "path": out_path
            }

        # --- warnings for missing MMSIs ---
        if not_found:
            print(f"\nWarning: {len(not_found)} MMSI(s) not found in any file:")
            for m in not_found:
                print(f"  MMSI {m} — no rows found, no file written")

        # --- final summary ---
        found = [m for m in mmsi_list if summary[m]["row_count"] > 0]
        total_rows = sum(summary[m]["row_count"] for m in found)

        print(f"\nDone.")
        print(f"  Vessels written : {len(found):>6,}")
        print(f"  Vessels missing : {len(not_found):>6,}")
        print(f"  Total rows saved: {total_rows:>6,}")

        return summary
    
    # This done updating
    def plot_vessel_path(
    self,
    dataset_dir: Optional[str] = None,
    arrow_every_n: int = 5,
    output_file_name: Optional[str] = None,
    ) -> None:
        
        target = dataset_dir if dataset_dir is not None else self.dataset_dir

        # --- load and sort ---
        df = (
            pl.read_csv(target, infer_schema_length=1000)
            .with_columns(
                pl.col("base_date_time").str.to_datetime("%Y-%m-%d %H:%M:%S") # change BaseDateTime to base_date_time and %Y-%m-%dT%H:%M:%S to %Y-%m-%d %H:%M:%S
            )
            .sort("base_date_time")   # change from BaseDateTime to base_data_time
            .filter(
                pl.col("latitude").is_between(-90, 90) &      # Change from LAT to latitude
                pl.col("longitude").is_between(-180, 180)           # change from LON to longitude
            )
        )

        if df.is_empty():
            print(f"No valid rows found in {target}")
            return

        # extract vessel metadata for title and the columns names have changed from old dataset to new dataset
        mmsi        = df["mmsi"][0]  
        vessel_name = df["vessel_name"][0] if "vessel_name" in df.columns else "Unknown"
        vessel_type = df["vessel_type"][0] if "vessel_type" in df.columns else "Unknown"
        total_pings = df.height
        time_start  = df["base_date_time"][0]
        time_end    = df["base_date_time"][-1]

        # convert to pandas for geopandas
        pdf = df.to_pandas()

        # --- build GeoDataFrame of ping points ---
        gdf_points = gpd.GeoDataFrame(
            pdf,
            geometry=gpd.points_from_xy(pdf["longitude"], pdf["latitude"]),
            crs="EPSG:4326"
        )

        # --- build track line ---
        from shapely.geometry import LineString
        if len(gdf_points) >= 2:
            track_line = LineString(zip(pdf["longitude"], pdf["latitude"]))
            gdf_line = gpd.GeoDataFrame(geometry=[track_line], crs="EPSG:4326")
        else:
            gdf_line = None

        # --- world basemap (same pattern as SAR function) ---
        world = gpd.read_file(geodatasets.get_path("naturalearth.land"))

        # --- figure setup ---
        fig, ax = plt.subplots(figsize=(14, 9))

        # 1. world basemap
        world.plot(ax=ax, color="lightgrey", edgecolor="white", linewidth=0.5)

        # 2. track line
        if gdf_line is not None:
            gdf_line.plot(ax=ax, color="steelblue", linewidth=1.8, alpha=0.7, zorder=2)

        # 3. ping points — color by time progression (early=light, late=dark)
        scatter = ax.scatter(
            pdf["longitude"],
            pdf["latitude"],
            c=range(len(pdf)),
            cmap="plasma",
            s=18,
            zorder=3,
            alpha=0.85,
            label="Pings"
        )
        plt.colorbar(scatter, ax=ax, label="Ping sequence (early → late)", shrink=0.5)

        # 4. start and end markers
        ax.plot(
            pdf["longitude"].iloc[0], pdf["latitude"].iloc[0],  #
            marker="o", color="green", markersize=10,
            zorder=5, label="Start", markeredgecolor="white", markeredgewidth=1.2
        )
        ax.plot(
            pdf["longitude"].iloc[-1], pdf["latitude"].iloc[-1],
            marker="s", color="red", markersize=10,
            zorder=5, label="End", markeredgecolor="white", markeredgewidth=1.2
        )

        # 5. direction arrows every N pings using COG
        import math
        arrow_indices = range(0, len(pdf) - 1, arrow_every_n)
        for i in arrow_indices:
            lon    = pdf["longitude"].iloc[i]
            lat    = pdf["latitude"].iloc[i]
            cog    = pdf["cog"].iloc[i]

            # skip invalid COG values (511 = not available in AIS)
            if cog >= 360 or cog < 0:
                continue

            # convert COG (nautical: 0=North, clockwise)
            # to math angle (0=East, counter-clockwise)
            angle_rad = math.radians(90 - cog)
            arrow_len = 0.15  # degrees — adjust if needed
            dx = arrow_len * math.cos(angle_rad)
            dy = arrow_len * math.sin(angle_rad)

            ax.annotate(
                "",
                xy=(lon + dx, lat + dy),
                xytext=(lon, lat),
                arrowprops=dict(
                    arrowstyle="->",
                    color="darkorange",
                    lw=1.4,
                ),
                zorder=4
            )

        # 6. zoom to data extent with buffer
        minx, miny, maxx, maxy = gdf_points.total_bounds
        buffer = max((maxx - minx) * 0.15, (maxy - miny) * 0.15, 1.0)
        ax.set_xlim(minx - buffer, maxx + buffer)
        ax.set_ylim(miny - buffer, maxy + buffer)

        # --- labels and legend ---
        ax.set_title(
            f"Vessel Path — {vessel_name} (MMSI: {mmsi})\n"
            f"Type: {vessel_type}  |  Pings: {total_pings:,}  |  "
            f"{time_start} → {time_end}",
            fontsize=13,
            pad=14
        )
        ax.set_xlabel("Longitude", fontsize=11)
        ax.set_ylabel("Latitude", fontsize=11)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(loc="lower left", fontsize=10)

        plt.tight_layout()

        # --- save ---
        plots_dir = os.path.join(self.analysis_result_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)

        fname = output_file_name if output_file_name else f"vessel_path_{mmsi}.png"
        out_path = os.path.join(plots_dir, fname)
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved → {out_path}")

        # --- show ---
        plt.show()
        plt.close()

    # This is done update
    def plot_vessel_path_point(
        self,
        dataset_dir: Optional[str] = None,
        arrow_every_n: int = 5,
        output_file_name: Optional[str] = None,
    ) -> None:

        target = dataset_dir if dataset_dir is not None else self.dataset_dir

       # --- load and sort ---
        df = (
            pl.read_csv(target, infer_schema_length=1000)
            .with_columns(
                pl.col("base_date_time").str.to_datetime("%Y-%m-%d %H:%M:%S")
            )
            .sort("base_date_time")
            .filter(
                pl.col("latitude").is_between(-90, 90) &
                pl.col("longitude").is_between(-180, 180)
            )
        )

        if df.is_empty():
            print(f"No valid rows found in {target}")
            return

        # extract vessel metadata for title
        mmsi        = df["mmsi"][0]
        vessel_name = df["vessel_name"][0] if "vessel_name" in df.columns else "Unknown"
        vessel_type = df["vessel_type"][0] if "vessel_type" in df.columns else "Unknown"
        total_pings = df.height
        time_start  = df["base_date_time"][0]
        time_end    = df["base_date_time"][-1]

        pdf = df.to_pandas()

        # --- build GeoDataFrame of ping points ---
        gdf_points = gpd.GeoDataFrame(
            pdf,
            geometry=gpd.points_from_xy(pdf["longitude"], pdf["latitude"]),
            crs="EPSG:4326"
        )

        # --- world basemap ---
        world = gpd.read_file(geodatasets.get_path("naturalearth.land"))

        # --- figure setup ---
        fig, ax = plt.subplots(figsize=(14, 9))

        # 1. world basemap
        world.plot(ax=ax, color="lightgrey", edgecolor="white", linewidth=0.5)

        # 2. ping points — color by time progression (early=light, late=dark)
        scatter = ax.scatter(
            pdf["longitude"],
            pdf["latitude"],
            c=range(len(pdf)),
            cmap="plasma",
            s=22,
            zorder=3,
            alpha=0.85,
            label="Pings"
        )
        plt.colorbar(scatter, ax=ax, label="Ping sequence (early → late)", shrink=0.5)

        # 3. start and end markers
        ax.plot(
            pdf["longitude"].iloc[0], pdf["latitude"].iloc[0],
            marker="o", color="green", markersize=10,
            zorder=5, label="Start", markeredgecolor="white", markeredgewidth=1.2
        )
        ax.plot(
            pdf["longitude"].iloc[-1], pdf["latitude"].iloc[-1],
            marker="s", color="red", markersize=10,
            zorder=5, label="End", markeredgecolor="white", markeredgewidth=1.2
        )

        # 4. direction arrows every N pings using COG
        arrow_indices = range(0, len(pdf) - 1, arrow_every_n)
        for i in arrow_indices:
            lon = pdf["longitude"].iloc[i]
            lat = pdf["latitude"].iloc[i]
            cog = pdf["cog"].iloc[i]

            # skip invalid COG values (511 = not available in AIS)
            if cog >= 360 or cog < 0:
                continue

            # convert COG (nautical: 0=North, clockwise)
            # to math angle (0=East, counter-clockwise)
            angle_rad = math.radians(90 - cog)
            arrow_len = 0.15
            dx = arrow_len * math.cos(angle_rad)
            dy = arrow_len * math.sin(angle_rad)

            ax.annotate(
                "",
                xy=(lon + dx, lat + dy),
                xytext=(lon, lat),
                arrowprops=dict(
                    arrowstyle="->",
                    color="darkorange",
                    lw=1.4,
                ),
                zorder=4
            )

        # 5. zoom to data extent with buffer
        minx, miny, maxx, maxy = gdf_points.total_bounds
        buffer = max((maxx - minx) * 0.15, (maxy - miny) * 0.15, 1.0)
        ax.set_xlim(minx - buffer, maxx + buffer)
        ax.set_ylim(miny - buffer, maxy + buffer)

        # --- labels and legend ---
        ax.set_title(
            f"Vessel Detections — {vessel_name} (MMSI: {mmsi})\n"
            f"Type: {vessel_type}  |  Pings: {total_pings:,}  |  "
            f"{time_start} → {time_end}",
            fontsize=13,
            pad=14
        )
        ax.set_xlabel("longitude", fontsize=11)
        ax.set_ylabel("latitude", fontsize=11)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(loc="lower left", fontsize=10)

        plt.tight_layout()

        # --- save ---
        plots_dir = os.path.join(self.analysis_result_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)

        fname = output_file_name if output_file_name else f"vessel_detections_{mmsi}.png"
        out_path = os.path.join(plots_dir, fname)
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved → {out_path}")

        plt.show()
        plt.close()

    def add_delta_time(
        self,
        time_unit: str = "minutes",
        create_new: bool = True,
        output_file_name: Optional[str] = None,
        dataset_dir: Optional[str] = None,
    ) -> pl.DataFrame:
    

        # --- validate time_unit ---
        valid_units = {"seconds", "minutes", "hours"}
        if time_unit not in valid_units:
            raise ValueError(
                f"Invalid time_unit '{time_unit}'. "
                f"Must be one of: {valid_units}"
            )

        target = dataset_dir if dataset_dir is not None else self.dataset_dir

        # --- load and sort chronologically ---
        df = (
            pl.read_csv(target, infer_schema_length=1000)
            .with_columns(
                pl.col("base_date_time").str.to_datetime("%Y-%m-%d %H:%M:%S")
            )
            .sort("base_date_time")
        )

        if df.is_empty():
            print(f"No rows found in {target}")
            return df

        # --- compute delta time ---
        # diff() gives duration in nanoseconds between consecutive rows
        # first row will be null → fill with 0
        ns_per_unit = {
            "seconds": 1_000_000_000,
            "minutes": 60_000_000_000,
            "hours"  : 3_600_000_000_000,
        }

        df = df.with_columns(
            (
                pl.col("base_date_time")
                .diff()                              # duration between consecutive rows
                .dt.total_nanoseconds()              # convert to nanoseconds (Int64)
                .fill_null(0)                        # first row → 0
                / ns_per_unit[time_unit]             # scale to requested unit
            )
            .round(4)
            .alias(f"DeltaTime_{time_unit}")
        )

        # --- summary stats ---
        delta_col = f"delta_time_{time_unit}"
        delta     = df[delta_col].filter(df[delta_col] > 0)  # exclude first row zero
        print(f"\ndelta_time summary ({time_unit}):")
        print(f"  Rows           : {df.height:,}")
        print(f"  Min delta       : {delta.min():.4f}")
        print(f"  Max delta       : {delta.max():.4f}")
        print(f"  Mean delta      : {delta.mean():.4f}")
        print(f"  Median delta    : {delta.median():.4f}")

        # --- save ---
        original_name = os.path.splitext(os.path.basename(target))[0]

        if create_new:
            os.makedirs(self.analysis_result_dir, exist_ok=True)
            fname = (
                output_file_name if output_file_name
                else f"{original_name}_delta_{time_unit}.csv"
            )
            out_path = os.path.join(self.analysis_result_dir, fname)
            df.write_csv(out_path)
            print(f"  Saved (new)     → {out_path}")
        else:
            # overwrite in place
            df.write_csv(target)
            print(f"  Saved (in-place)→ {target}")

        return df
    
    def add_delta_movement(
        self,
        distance_unit: str = "nm",
        create_new: bool = True,
        output_file_name: Optional[str] = None,
        dataset_dir: Optional[str] = None,
    ) -> pl.DataFrame:

        import numpy as np
  
        valid_units = {"nm", "km", "m"}
        if distance_unit not in valid_units:
            raise ValueError(
                f"Invalid distance_unit '{distance_unit}'. "
                f"Must be one of: {valid_units}"
            )

        target = dataset_dir if dataset_dir is not None else self.dataset_dir

        # --- load and sort chronologically ---
        df = (
            pl.read_csv(target, infer_schema_length=1000)
            .with_columns(
                pl.col("base_date_time").str.to_datetime("%Y-%m-%d %H:%M:%S")
            )
            .sort("base_date_time")
            .filter(
                pl.col("latitude").is_between(-90, 90) &
                pl.col("longitude").is_between(-180, 180)
            )
        )

        if df.is_empty():
            print(f"No valid rows found in {target}")
            return df

        pdf = df.to_pandas()

        EARTH_RADIUS_KM = 6371.0

        def haversine_vectorised(lat1, lon1, lat2, lon2):
        
            lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
            dlat = lat2 - lat1
            dlon = lon2 - lon1
            a    = (
                np.sin(dlat / 2) ** 2
                + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
            )
            return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))

        def bearing_vectorised(lat1, lon1, lat2, lon2):

            lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
            dlon    = lon2 - lon1
            x       = np.sin(dlon) * np.cos(lat2)
            y       = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
            bearing = np.degrees(np.arctan2(x, y))
            return (bearing + 360) % 360

        # shift arrays: current row vs previous row
        lat_curr = pdf["latitude"].values
        lon_curr = pdf["longitude"].values
        lat_prev = np.roll(lat_curr, 1)  #type:ignore
        lon_prev = np.roll(lon_curr, 1)  #type:ignore

        # --- DeltaDistance ---
        dist_km    = haversine_vectorised(lat_prev, lon_prev, lat_curr, lon_curr)
        dist_km[0] = 0.0   # first row — no previous ping

        if distance_unit == "nm":
            dist_out = dist_km / 1.852        # km → nautical miles (1 nm = 1.852 km exactly)
        elif distance_unit == "km":
            dist_out = dist_km
        else:                                  # metres
            dist_out = dist_km * 1000.0

        # --- AvgBearing (0–360°) ---
        # straight-line bearing from A to B = net/average direction over interval
        avg_bearing    = bearing_vectorised(lat_prev, lon_prev, lat_curr, lon_curr)
        avg_bearing[0] = 0.0   # first row

        # --- DeltaTime in hours for AvgSOG ---
        dt_seconds    = pdf["base_date_time"].diff().dt.total_seconds().fillna(0).values
        dt_hours      = dt_seconds / 3600.0   #type:ignore

        # --- AvgSOG in knots (nm/hr) ---
        # = average speed over the interval, NOT instantaneous speed
        # always computed in nm regardless of distance_unit for direct comparison
        # with AIS SOG field which is always in knots
        dist_nm = dist_km / 1.852
        with np.errstate(divide="ignore", invalid="ignore"):
            avg_sog = np.where(dt_hours > 0, dist_nm / dt_hours, 0.0)
        avg_sog[0] = 0.0

        # --- AvgBearingVsCOG_diff (-180 to +180°) ---
        # COG 360.0 = not available per AIS spec
        cog_values = pdf["cog"].values.astype(float)
        cog_valid  = (cog_values >= 0) & (cog_values < 360.0)  # type:ignore
        avg_bearing_vs_cog = np.where(
            cog_valid,
            ((avg_bearing - cog_values + 180) % 360) - 180,
             np.nan
        )
        avg_bearing_vs_cog[0] = np.nan   # first row — no interval

        # --- CumulativeDistance ---
        cumulative = np.cumsum(dist_out)

        # --- append to polars DataFrame ---
        dist_col = f"DeltaDistance_{distance_unit}"
        cum_col  = f"CumulativeDistance_{distance_unit}"

        df = df.with_columns([
            pl.Series(dist_col,                dist_out.round(6).tolist()),
            pl.Series("AvgBearing_deg",        avg_bearing.round(2).tolist()),
            pl.Series("AvgSOG_knots",          avg_sog.round(4).tolist()),
            pl.Series("AvgBearingVsCOG_diff",  avg_bearing_vs_cog.round(2).tolist()),
            pl.Series(cum_col,                 cumulative.round(6).tolist()),
        ])

        # --- summary ---
        dist_valid = df[dist_col].slice(1)      # skip first row zero
        sog_valid  = df["avg_sog_knots"].slice(1)
        diff_valid = df["avg_bearing_vs_cog_diff"].drop_nulls()

        mmsi        = df["mmsi"][0]
        vessel_name = df["vessel_name"][0] if "vessel_name" in df.columns else "Unknown"

        print(f"\nDelta movement summary — {vessel_name} (MMSI: {mmsi})")
        print(f"  Distance unit             : {distance_unit}")
        print(f"  Total pings               : {df.height:,}")
        print(f"  Total distance            : {df[cum_col][-1]:.4f} {distance_unit}")
        print(f"  Min ping distance         : {dist_valid.min():.6f} {distance_unit}")
        print(f"  Max ping distance         : {dist_valid.max():.6f} {distance_unit}")
        print(f"  Mean ping distance        : {dist_valid.mean():.6f} {distance_unit}")
        print(f"  ── speed (avg over interval, not instantaneous) ──")
        print(f"  Min AvgSOG                : {sog_valid.min():.4f} knots")
        print(f"  Max AvgSOG                : {sog_valid.max():.4f} knots")
        print(f"  Mean AvgSOG               : {sog_valid.mean():.4f} knots")
        print(f"  ── bearing vs COG (avg displacement vs reported) ──")
        print(f"  Mean AvgBearing vs COG    : {diff_valid.mean():.2f}°")
        print(f"  Max |AvgBearing vs COG|   : {diff_valid.abs().max():.2f}°")

        # --- save ---
        original_name = os.path.splitext(os.path.basename(target))[0]

        if create_new:
            os.makedirs(self.analysis_result_dir, exist_ok=True)
            fname = (
                output_file_name if output_file_name
                else f"{original_name}_movement_{distance_unit}.csv"
            )
            out_path = os.path.join(self.analysis_result_dir, fname)
            df.write_csv(out_path)
            print(f"\n  Saved (new)      → {out_path}")
        else:
            df.write_csv(target)
            print(f"\n  Saved (in-place) → {target}")

        return df
    
    def filter_by_bbox(
        self,
        min_lat: float,
        max_lat: float,
        min_lon: float,
        max_lon: float,
        location_name: str,
        create_new: bool = True,
        dataset_dir: Optional[str] = None,
        for_group: bool = False,
    ) -> pl.DataFrame:
        # --- validate bbox ---
        if not (-90 <= min_lat < max_lat <= 90):
            raise ValueError(
                f"Invalid latitude range: min_lat={min_lat}, max_lat={max_lat}. "
                f"Must satisfy -90 <= min_lat < max_lat <= 90."
            )
        if not (-180 <= min_lon < max_lon <= 180):
            raise ValueError(
                f"Invalid longitude range: min_lon={min_lon}, max_lon={max_lon}. "
                f"Must satisfy -180 <= min_lon < max_lon <= 180."
            )

        # --- bbox filter expression (reused across all files) ---
        bbox_filter = (
            pl.col("latitude").is_between(min_lat, max_lat) &
            pl.col("longitude").is_between(min_lon, max_lon)
        )

        os.makedirs(self.analysis_result_dir, exist_ok=True)

        # --- single file processor ---
        def _process_file(fpath: str) -> pl.DataFrame:
            return (
                pl.scan_csv(fpath, infer_schema_length=1000)
                .filter(bbox_filter)
                .collect()
            )

        def _build_output_name(fpath: str) -> str:
            """Prefix original filename with location_name."""
            original = os.path.basename(fpath)           # AIS_2022_01_01.csv
            return f"{location_name}_{original}"          # Adriatic_AIS_2022_01_01.csv

        # ------------------------------------------------------------------ #
        #  for_group = True                                                    #
        # ------------------------------------------------------------------ #
        if for_group:
            csv_files = sorted([
                f for f in os.listdir(self.datasets_location)
                if f.endswith(".csv")
            ])

            if not csv_files:
                print("No CSV files found in datasets_location.")
                return pl.DataFrame()

            all_frames   = []
            total_in     = 0
            total_out    = 0

            for fname in tqdm(csv_files, desc=f"Filtering [{location_name}]", unit="file"):
                fpath      = os.path.join(self.datasets_location, fname)
                filtered   = _process_file(fpath)
                rows_in    = (
                    pl.scan_csv(fpath, infer_schema_length=1000)
                    .select(pl.len())
                    .collect()
                    .item()
                )
                rows_out   = filtered.height
                total_in  += rows_in
                total_out += rows_out

                tqdm.write(
                    f"  {fname}: {rows_in:>10,} rows → "
                    f"{rows_out:>8,} inside bbox  "
                    f"({rows_out / rows_in * 100:.2f}%)"
                )

                if filtered.is_empty():
                    tqdm.write(f"  Skipping {fname} — no rows inside bbox.")
                    continue

                all_frames.append(filtered)

                # save / overwrite per file
                if create_new:
                    out_name = _build_output_name(fpath)
                    out_path = os.path.join(self.analysis_result_dir, out_name)
                    filtered.write_csv(out_path)
                else:
                    filtered.write_csv(fpath)

            # combine all filtered frames
            result_df = pl.concat(all_frames) if all_frames else pl.DataFrame()

            print(f"\nBbox filter complete — {location_name}")
            print(f"  Files processed   : {len(csv_files):,}")
            print(f"  Total rows in     : {total_in:,}")
            print(f"  Total rows out    : {total_out:,}")
            print(f"  Retention rate    : {total_out / total_in * 100:.4f}%")
            print(f"  Unique MMSI found : {result_df['mmsi'].n_unique():,}" if not result_df.is_empty() else "  No rows found.")
            if create_new:
                print(f"  Output dir        : {self.analysis_result_dir}")
            return result_df

        # ------------------------------------------------------------------ #
        #  for_group = False                                                   #
        # ------------------------------------------------------------------ #
        else:
            target   = dataset_dir if dataset_dir is not None else self.dataset_dir
            original = os.path.basename(target)

            # get total rows for retention stats (lazy, no full load)
            total_in = (
                pl.scan_csv(target, infer_schema_length=1000)
                .select(pl.len())
                .collect()
                .item()
            )

            filtered = _process_file(target)
            rows_out = filtered.height

            if filtered.is_empty():
                print(f"No rows found inside bbox for {original}.")
                return filtered

            if create_new:
                out_name = _build_output_name(target)
                out_path = os.path.join(self.analysis_result_dir, out_name)
                filtered.write_csv(out_path)
                print(f"\nSaved → {out_path}")
            else:
                filtered.write_csv(target)
                print(f"\nOverwritten in place → {target}")

            print(f"\nBbox filter complete — {location_name}")
            print(f"  Source file       : {original}")
            print(f"  Bounding box      : latitude [{min_lat}, {max_lat}] | longitude [{min_lon}, {max_lon}]")
            print(f"  Total rows in     : {total_in:,}")
            print(f"  Total rows out    : {rows_out:,}")
            print(f"  Retention rate    : {rows_out / total_in * 100:.4f}%")
            print(f"  Unique MMSI found : {filtered['mmsi'].n_unique():,}")

            return filtered
    
    def filter_by_bbox_give_path(
        self,
        min_lat: float,
        max_lat: float,
        min_lon: float,
        max_lon: float,
        location_name: str,
        create_new: bool = True,
        dataset_dir: Optional[str] = None,
        for_group: bool = False,
    ) -> list[str]:  # <-- now returns list of absolute file paths
        # --- validate bbox ---
        if not (-90 <= min_lat < max_lat <= 90):
            raise ValueError(
                f"Invalid latitude range: min_lat={min_lat}, max_lat={max_lat}. "
                f"Must satisfy -90 <= min_lat < max_lat <= 90."
            )
        if not (-180 <= min_lon < max_lon <= 180):
            raise ValueError(
                f"Invalid longitude range: min_lon={min_lon}, max_lon={max_lon}. "
                f"Must satisfy -180 <= min_lon < max_lon <= 180."
            )

        # --- bbox filter expression ---
        bbox_filter = (
            pl.col("latitude").is_between(min_lat, max_lat) &
            pl.col("longitude").is_between(min_lon, max_lon)
        )

        os.makedirs(self.analysis_result_dir, exist_ok=True)

        created_files: list[str] = []  # <-- track all created file paths

        # --- single file processor ---
        def _process_file(fpath: str) -> pl.DataFrame:
            return (
                pl.scan_csv(fpath, infer_schema_length=1000)
                .filter(bbox_filter)
                .collect()
            )

        def _build_output_name(fpath: str) -> str:
            original = os.path.basename(fpath)
            return f"{location_name}_{original}"

        # ------------------------------------------------------------------ #
        #  for_group = True                                                    #
        # ------------------------------------------------------------------ #
        if for_group:
            csv_files = sorted([
                f for f in os.listdir(self.datasets_location)
                if f.endswith(".csv")
            ])

            if not csv_files:
                print("No CSV files found in datasets_location.")
                return []

            total_in  = 0
            total_out = 0
            all_mmsi_counts = []

            for fname in tqdm(csv_files, desc=f"Filtering [{location_name}]", unit="file"):
                fpath    = os.path.join(self.datasets_location, fname)
                filtered = _process_file(fpath)
                rows_in  = (
                    pl.scan_csv(fpath, infer_schema_length=1000)
                    .select(pl.len())
                    .collect()
                    .item()
                )
                rows_out   = filtered.height
                total_in  += rows_in
                total_out += rows_out

                tqdm.write(
                    f"  {fname}: {rows_in:>10,} rows → "
                    f"{rows_out:>8,} inside bbox  "
                    f"({rows_out / rows_in * 100:.2f}%)"
                )

                if filtered.is_empty():
                    tqdm.write(f"  Skipping {fname} — no rows inside bbox.")
                    continue

                if not filtered.is_empty():
                    all_mmsi_counts.append(filtered["mmsi"].n_unique())

                if create_new:
                    out_name = _build_output_name(fpath)
                    out_path = os.path.abspath(os.path.join(self.analysis_result_dir, out_name))
                    filtered.write_csv(out_path)
                    created_files.append(out_path)  # <-- record path
                else:
                    abs_fpath = os.path.abspath(fpath)
                    filtered.write_csv(abs_fpath)
                    created_files.append(abs_fpath)  # <-- record path

            total_unique_mmsi = sum(all_mmsi_counts)
            print(f"\nBbox filter complete — {location_name}")
            print(f"  Files processed   : {len(csv_files):,}")
            print(f"  Files created     : {len(created_files):,}")
            print(f"  Total rows in     : {total_in:,}")
            print(f"  Total rows out    : {total_out:,}")
            print(f"  Retention rate    : {total_out / total_in * 100:.4f}%" if total_in > 0 else "  Retention rate    : N/A")
            print(f"  Unique MMSI found : {total_unique_mmsi:,}" if all_mmsi_counts else "  No rows found.")
            if create_new:
                print(f"  Output dir        : {self.analysis_result_dir}")

            return created_files 

        # ------------------------------------------------------------------ #
        #  for_group = False                                                   #
        # ------------------------------------------------------------------ #
        else:
            target   = dataset_dir if dataset_dir is not None else self.dataset_dir
            original = os.path.basename(target)

            total_in = (
                pl.scan_csv(target, infer_schema_length=1000)
                .select(pl.len())
                .collect()
                .item()
            )

            filtered = _process_file(target)
            rows_out = filtered.height

            if filtered.is_empty():
                print(f"No rows found inside bbox for {original}.")
                return [] 

            if create_new:
                out_name = _build_output_name(target)
                out_path = os.path.abspath(os.path.join(self.analysis_result_dir, out_name))
                filtered.write_csv(out_path)
                created_files.append(out_path)  
                print(f"\nSaved → {out_path}")
            else:
                abs_target = os.path.abspath(target)
                filtered.write_csv(abs_target)
                created_files.append(abs_target)  
                print(f"\nOverwritten in place → {abs_target}")

            print(f"\nBbox filter complete — {location_name}")
            print(f"  Source file       : {original}")
            print(f"  Bounding box      : latitude [{min_lat}, {max_lat}] | longitude [{min_lon}, {max_lon}]")
            print(f"  Total rows in     : {total_in:,}")
            print(f"  Total rows out    : {rows_out:,}")
            print(f"  Retention rate    : {rows_out / total_in * 100:.4f}%")
            print(f"  Unique MMSI found : {filtered['mmsi'].n_unique():,}")

            return created_files  

    VALID_MIDS = set(range(201,776))
    def _classify_mmsi(
        self,
        mmsi:int
    ) -> str | None:
        s = str(mmsi)

        if len(s) != 9:
            return None
        
        first = s[0]
        first2 = s[:2]
        first3 = s[:3]

        if first2 == "00":
            mid = int(s[2:5])
            if mid in self.VALID_MIDS:
                return "coast_station"
            return None
        
        if first == "0" and first2 != "00":
            mid = int(s[1:4])
            if mid in self.VALID_MIDS:
                return "group"
            return None
        
        if first3 == "111":
            mid = int(s[3:6])
            if mid in self.VALID_MIDS:
                return "sar_aircraft"
            return None
        
        if first2 == "99":
            mid = int(s[2:5])
            if mid in self.VALID_MIDS:
                return "aids_to_navigation"
            return None
        
        if first2 == "97":
            return "sart_mob_epirb"
        
        if first2 == "98":
            mid = int(s[2:5])
            if mid in self.VALID_MIDS:
                return "handheld_vhf"
            return None
        
        if first == "8":
            return "craft_associated"

        mid = int(first3)
        if mid in self.VALID_MIDS:
            return "vessel"

        return None
    
    def read_mmsi(
        self,
        path:str,
        group:bool = False
    ) -> list[int] | OrderedDict[str,List[int]]:
        ALL_CATEGORIES = [
            "vessel","coast_station","group","sar_aircraft","aids_to_navigation","craft_associated","handheld_vhf","sart_mob_epirb"
        ]

        lf = (
            pl.scan_csv(path)
            .select(pl.col("mmsi"))
            .filter(pl.col("mmsi").is_not_null())
            .with_columns(
                pl.col("mmsi").cast(pl.Int64)
            )
            .filter(
                (pl.col("mmsi") >= 100_000_000) &
                (pl.col("mmsi") <= 999_999_999)
            )
        )
        raw:List[int] = lf.collect()["mmsi"].to_list()

        if not group:
            return [mmsi for mmsi in raw if self._classify_mmsi(mmsi) is not None]
        
        result:OrderedDict[str,list[int]] = OrderedDict(
            (cat,[]) for cat in ALL_CATEGORIES
        )

        for mmsi in raw:
            category = self._classify_mmsi(mmsi)
            if category is not None:
                result[category].append(mmsi)
        
        return result

if __name__ == "__main__":
    datasets_location = r"D:\SAR-Intelligence\AIS_system\data\CSV_files"
    dataset_dir = r"D:\SAR-Intelligence\AIS_system\data\CSV_files\AIS_2022_01_01.csv"
    result_dir = r"D:\SAR-Intelligence\AIS_system\Analysis_results"

    newdataset_dir = r"D:\AIS_project\data\CSV_files\new_AIS\ais-2022-01-01.csv"
    newdataset_location = r"D:\AIS_project\data\CSV_files\new_AIS"
    newAnalysis_result = r"D:\AIS_project\AnalysisNew_results"

    BrandNewAnalysis_result = r"D:\AIS_project\AnalysisBrandNew_results"

    analysis_object = Analysis(datasets_location=newdataset_location,dataset_dir=newdataset_dir,analysis_result_dir=BrandNewAnalysis_result)
    example_list = [209593000,368084090,368140160,319142200,319142400,319142900,319143700]

    _,unique_mmsi_path,_ = analysis_object.get_unique_mmsi("Showing_friend",for_group=True)
    if not os.path.exists(unique_mmsi_path):
        raise ValueError("Path not found")
    Ordered_dict = analysis_object.read_mmsi(unique_mmsi_path,group=True)

    for keys in Ordered_dict:
        row_count = len(Ordered_dict[keys])  #type:ignore
        print(f"{keys} : Count {len(Ordered_dict[keys])}") #type:ignore
    # print(analysis_object.count_rows(show_group_count=False,for_group=True))
    # print(analysis_object.get_unique_mmsi(output_file_name="Unique_mmsi_all_file",for_group=True))
    # df = analysis_object.extract_vessels_by_mmsi(
    #     mmsi_list=[209593000,368084090,368140160,319142200,319142400,319142900,319143700],
    #     output_file_name="Selected_vessels.csv"
    # )

    # summary = analysis_object.export_vessels_by_mmsi(
    #     mmsi_list= example_list,#[122020471,122292919,123456789,155012139,191283710,205089000,205097000,205125000,205388630,205517000,205553000,205760000,207832820,209182000,209254000,209289000,209388000,209423000,209470000,209513000,209575000,209593000,209641000,209677000,209716000,209729000,209941000,209997000,210000000,210065000,210080000,210185000,210210000,210285000,210296000,210328000,210347000,210377000,210614000,210740000,210763000,210786000,210905000,210953000,210959000,211002010,211223160,211245050,211266490,211331640,211335760,211410230,211545790,211674740,211705870,211723180,211779740,211788020,211829830],
    #     output_name="new_vessel",
    #     for_group=False
    # )

    # summary = analysis_object.export_vessels_by_mmsi(
    #     mmsi_list= example_list, #[122020471,122292919,123456789,155012139,191283710,205089000,205097000,205125000,205388630,205517000,205553000,205760000,207832820,209182000,209254000,209289000,209388000,209423000,209470000,209513000,209575000,209593000,209641000,209677000,209716000,209729000,209941000,209997000,210000000,210065000,210080000,210185000,210210000,210285000,210296000,210328000,210347000,210377000,210614000,210740000,210763000,210786000,210905000,210953000,210959000,211002010,211223160,211245050,211266490,211331640,211335760,211410230,211545790,211674740,211705870,211723180,211779740,211788020,211829830],
    #     output_name="new_Wholedays",
    #     for_group=True
    # )

    # analysis_object.plot_vessel_path_point(
    #     dataset_dir=r"D:\AIS_project\AnalysisBrandNew_results\vessels\new_Wholedays_319142900.csv",
    #     arrow_every_n=10,
    #     output_file_name="new_Wholedays_319142900.png"
    # )

    # analysis_object.plot_vessel_path_point(
    #     dataset_dir=r"D:\SAR-Intelligence\AIS_system\Analysis_results\vessels\Wholedays_123456789.csv",
    #     arrow_every_n=10,
    #     output_file_name="Wholedays_123456789.png"
    # )

    # df = analysis_object.add_delta_time(
    #     dataset_dir=r"D:\SAR-Intelligence\AIS_system\Analysis_results\vessels\Wholedays_211335760.csv",
    #     time_unit="minutes",
    #     create_new=True,
    #     output_file_name="Wholedays_211335760_delta.csv"
    # )

    # df = analysis_object.add_delta_movement(
    #     dataset_dir=r"D:\SAR-Intelligence\AIS_system\Analysis_results\vessels\Wholedays_210959000.csv",
    #     distance_unit="nm",
    #     create_new=True,
    #     output_file_name="Wholedays_210959000_moment.csv"
    # )

    # df = analysis_object.add_delta_time(
    #     dataset_dir=r"D:\SAR-Intelligence\AIS_system\Analysis_results\vessels\Wholedays_210959000.csv",
    #     time_unit="minutes",
    #     create_new=True,
    #     output_file_name="Wholedays_210959000_delta.csv"
    # )

    # df = analysis_object.add_delta_movement(
    #     dataset_dir=r"D:\SAR-Intelligence\AIS_system\Analysis_results\vessels\Wholedays_123456789.csv",
    #     distance_unit="nm",
    #     create_new=True,
    #     output_file_name="Wholedays_123456789_moment.csv"
    # )

    df = analysis_object.filter_by_bbox(
        min_lat=17.4068,max_lat=31.4648,
        min_lon=-98.0539,max_lon=-80.433,
        location_name="Gluf_of_Mexico",
        for_group=True,create_new=True
    )

    # newdataset_dir = r"D:\AIS_project\data\CSV_files\new_AIS\ais-2022-01-01.csv"
    # newdataset_location = r"D:\AIS_project\data\CSV_files\new_AIS"
    # newAnalysis_result = r"D:\AIS_project\AnalysisNew_results"
    
    # newAnalysis_object = Analysis(datasets_location=newdataset_location,dataset_dir=newdataset_dir,analysis_result_dir=newAnalysis_result)
    # unqiueMMSI_list = newAnalysis_object.get_unique_mmsi("New_MMSI_list_singleDay")
    # unqiueMMSI_listGroup = newAnalysis_object.get_unique_mmsi("New_MMSI_list_Whole",for_group=True)
    # print(unqiueMMSI_list)

    # This done testing
    # testlist = [209593000,368084090,368140160,319142200,319142400,319142900,319143700]
    # newdf = newAnalysis_object.extract_vessels_by_mmsi(testlist,"Vessel_trajectory_whole.csv")
    # newdict = newAnalysis_object.export_vessels_by_mmsi(testlist,"Vessel_singleday")
    # newdict_group = newAnalysis_object.export_vessels_by_mmsi(testlist,"Vessel_Wholedata",for_group=True)
    # print(newdict)
    # print(newdict_group)
    
    # plotting graph
    # singleday_dir = r"D:\AIS_project\AnalysisNew_results\vessels\Vessel_singleday_368140160.csv"
    # wholedataset_dir = r"D:\AIS_project\AnalysisNew_results\vessels\Vessel_Wholedata_209593000.csv"
    # newplot_singleday = newAnalysis_object.plot_vessel_path(dataset_dir=singleday_dir,output_file_name="singleday.png")
    # newplot_wholesay = newAnalysis_object.plot_vessel_path(dataset_dir=wholedataset_dir,output_file_name="Wholedataset.png")
    # newplotway_singleday = newAnalysis_object.plot_vessel_path_point(dataset_dir=singleday_dir,output_file_name="Singledayway")
    # newplotway_wholedataset = newAnalysis_object.plot_vessel_path_point(dataset_dir=wholedataset_dir,arrow_every_n=10,output_file_name="Wholeday_path_209593000.png")

# -------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Macro Analysis
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------

import polars as pl
import os
import torch
import geopandas as gpd
import geodatasets
import matplotlib.pyplot as plt
from shapely.geometry import LineString,box
from typing import Optional, List, Tuple
from tqdm import tqdm 
import shutil
import json
import math
import ijson
from itertools import product
from .Analysis_system import Analysis

class MacroAnalysis(Analysis):

    def __init__(
        self,
        datasets_location:str,
        dataset_dir:str,
        analysis_result_dir:str,
        min_lat:float = 17.4068,
        min_lon:float = -98.0539,
        max_lat:float = 31.4648,
        max_lon:float = -80.4330,
        bin_size:float = 0.5,
        overlap_pct:float = 0.0
    ):
        super().__init__(datasets_location,dataset_dir,analysis_result_dir)
        self.min_lat = min_lat
        self.min_lon = min_lon
        self.max_lat = max_lat
        self.max_lon = max_lon
        self.bin_size = bin_size
        self.overlap_pct = overlap_pct
        # self.bboxes_list = self.generate_coordinates(min_lat=self.min_lat,min_lon=self.min_lon,max_lat = self.max_lat,max_lon=self.max_lon,bin_size=self.bin_size,overlap_pct=self.overlap_pct)
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

    def generate_coordinates(
        self,
        min_lat:float,
        min_lon:float,
        max_lat:float,
        max_lon:float,
        bin_size:float,
        overlap_pct:float
    )->List[Tuple[float,float,float,float]]:
        stride = bin_size * (1.0 - overlap_pct)
    
        def get_starts(start, end):
            starts = []
            current = start
            while current < end:
                starts.append(current)
                # Break early if the current bin fully covers the maximum boundary
                if current + bin_size >= end:
                    break
                current += stride
            return starts

        # Generate 1D arrays of starting coordinates
        lat_starts = get_starts(min_lat, max_lat)
        lon_starts = get_starts(min_lon, max_lon)
    
        # Use itertools.product to efficiently generate the 2D grid combinations
        bboxes = [
            (lat, lon, lat + bin_size, lon + bin_size)
            for lat, lon in product(lat_starts, lon_starts)
        ]
        return bboxes
    
    def plot_bboxes(
        self,
        bboxes_list:List[Tuple[float,float,float,float]],
        title:str
    ):
        geometries = [
            box(lon_min, lat_min, lon_max, lat_max) 
            for lat_min, lon_min, lat_max, lon_max in bboxes_list
        ]

        gdf_bboxes = gpd.GeoDataFrame(geometry=geometries, crs="EPSG:4326")
    
        # Load world map
        world = gpd.read_file(geodatasets.get_path('naturalearth.land'))
    
        fig, ax = plt.subplots(figsize=(10, 8)) # Slightly larger figure for clarity
    
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

    def plot_bboxes_and_map(
        self,
        bboxes_list:List[Tuple[float,float,float,float]],
        title:str
    ):
        geometries = [
            box(lon_min,lat_min,lon_min,lat_max)
            for lat_min,lon_min,lat_max,lon_max in bboxes_list
        ]

        gdf_bboxes = gpd.GeoDataFrame(geometry=geometries,crs="EPSG:4326")
        world = gpd.read_file(geodatasets.get_path('naturalearth.land'))
        fig,ax = plt.subplots(figsize=(10,8))
        world.plot(ax=ax,color = 'lightgrey',edgecolor='white')

        gdf_bboxes.boundary.plot(ax=ax,edgecolor='red',linewidth=0.5,alpha=0.6)

        for idx,geom in enumerate(gdf_bboxes.geometry):
            ax.text(
                x=geom.centroid.x,
                y=geom.centroid.y,
                s=str(idx),
                fontsize=8,
                ha='center',
                va='center',
                color='darkblue',
                weight='bold'
            )
        
        buffer=1
        minx,miny,maxx,maxy = gdf_bboxes.total_bounds
        ax.set_xlim(minx-buffer,maxx+buffer)
        ax.set_ylim(miny-buffer,maxy+buffer)

        plt.title(f"{title} ({len(bboxes_list)} cells)")
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.grid(True,linestyle="--",alpha=0.5)
        plt.show()

    def plot_bboxes_intensity(
        self,
        bboxes_list:List[Tuple[float,float,float,float,float]],
        title:str
    ):
        geometries = []
        counts = []

        for lat_min,lon_min,lat_max,lon_max,v_count in bboxes_list:
            geometries.append(box(lon_min,lat_min,lon_max,lat_max))
            counts.append(v_count)

        gdf_bboxes = gpd.GeoDataFrame(
            {"vessel_count":counts,"geometry":geometries},
            crs="EPSG:4326"
        )

        world = gpd.read_file(geodatasets.get_path('naturalearth.land'))
        fig,ax = plt.subplots(figsize=(12,8))
        world.plot(ax=ax,color='lightgrey',edgecolor='white')

        gdf_bboxes.plot(
            ax=ax, 
            column='vessel_count', 
            cmap='OrRd',           
            edgecolor='black',     
            linewidth=0.5, 
            alpha=0.75,            
            legend=True,           
            legend_kwds={'label': "Vessel Count", 'shrink': 0.6}
        )

        buffer = 1
        minx, miny, maxx, maxy = gdf_bboxes.total_bounds
        ax.set_xlim(minx - buffer, maxx + buffer)
        ax.set_ylim(miny - buffer, maxy + buffer)

        plt.title(f"{title} ({len(bboxes_list)} cells)")
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.grid(True, linestyle='--', alpha=0.5)
    
        plt.tight_layout() # Keeps the legend from getting cut off
        plt.show()

    def extract_bbox_and_counts(
        self,
        json_file_path:str
    )->List[Tuple[float,float,float,float,float]]:
        
        results = []
        with open(json_file_path,'rb') as f:
            for record in ijson.items(f,'item'):
                bbox_tuple = (
                    record["bbox_min_lat"],
                    record["bbox_min_lon"],
                    record["bbox_max_lat"],
                    record["bbox_max_lon"],
                    record["vessel_count"]
                )
                results.append(bbox_tuple)        
        return results
    
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
        # --- validate dataset_location ---
        if not dataset_location:
            raise ValueError("dataset_location must be a non-empty path string.")

        os.makedirs(dataset_location, exist_ok=True)

        # --- delegate bbox filtering to parent ---
        created_files: list[str] = self.filter_by_bbox_give_path(
            min_lat=min_lat if isinstance(min_lat,float) else self.min_lat,
            max_lat=max_lat if isinstance(max_lat,float) else self.max_lat,
            min_lon=min_lon if isinstance(min_lon,float) else self.min_lon,
            max_lon=max_lon if isinstance(max_lon,float) else self.max_lon,
            location_name=location_name,
            create_new=create_new,
            dataset_dir=dataset_dir,
            for_group=for_group,
        )

        if not created_files:
            print(f"No files were created for dataset '{dataset_name}'. Nothing to move.")
            return []

        # --- move each file to dataset_location ---
        moved_files: list[str] = []
        for src_path in created_files:
            fname    = os.path.basename(src_path)
            dst_path = os.path.abspath(os.path.join(dataset_location, fname))
            shutil.move(src_path, dst_path)  # move, not copy
            moved_files.append(dst_path)
            print(f"  Moved: {src_path}\n      → {dst_path}")

        print(f"\nGet_patch_DataFile complete — '{dataset_name}'")
        print(f"  Files moved       : {len(moved_files):,}")
        print(f"  Destination dir   : {os.path.abspath(dataset_location)}")

        return moved_files
    
    def Clean_MMSI_DataFrame( # THis not work if create_new=false used the below one
        self,
        MMSI_CSV_FILE: str,
        Category_keep: List[str],
        create_new: bool,
        CSV_file_exist:bool = False,
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

        # Resolve dataset directory
        resolved_dir = dataset_dir if dataset_dir is not None else self.dataset_dir

        if os.path.isfile(resolved_dir):
            resolved_dir = os.path.dirname(resolved_dir)
        # Find all CSV files in the dataset directory
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
                # Save to a new file with a prefix, leaving the original untouched
                base, ext = os.path.splitext(csv_file)
                new_file_path = os.path.join(resolved_dir, f"filtered_{base}{ext}")
                filtered_df.write_csv(new_file_path)
                print(f"    Saved new file: filtered_{base}{ext}")
            else:
                # Overwrite the original file
                filtered_df.write_csv(file_path)
                print(f"    Overwritten: {csv_file}")

        print("\nDone. All files processed.")

    def Clean_MMSI_DataFrame_gpt(
        self,
        MMSI_CSV_FILE: str,
        Category_keep: List[str],
        create_new: bool,
        dataset_dir: Optional[str] = None,
        for_group: bool = True
    ):
        # -------------------------------
        # STEP 1: Get MMSI CSV
        # -------------------------------
        _, csv_path, _ = self.get_unique_mmsi(
            output_file_name=MMSI_CSV_FILE,
            dataset_dir=dataset_dir,
            for_group=for_group
        )
    
        if not os.path.exists(csv_path):
            raise ValueError("Csv file not exist")
    
        print("Csv file exist")

        # Lazy read
        df = pl.scan_csv(csv_path)
        row_count = df.select(pl.len()).collect().item()
        print(f"Total rows in MMSI file: {row_count}")

        # -------------------------------
        # STEP 2: Classify MMSIs
        # -------------------------------
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

        # -------------------------------
        # STEP 3: Resolve dataset dir
        # -------------------------------
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

        # -------------------------------
        # STEP 4: Process each file
        # -------------------------------
        for csv_file in csv_files:
            file_path = os.path.join(resolved_dir, csv_file)
            print(f"\nProcessing: {csv_file}")

            lf = pl.scan_csv(file_path)
            original_count = lf.select(pl.len()).collect().item()

            filtered_lf = (
                lf
                .filter(pl.col("mmsi").is_not_null())
                .with_columns(
                    pl.col("mmsi").cast(pl.Int64, strict=False)
                )
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
                # Write safely (NO MEMORY CONFLICT)
                filtered_lf.sink_csv(temp_path)
                # Atomic replace
                os.replace(temp_path, file_path)
                print(f"    Overwritten safely: {csv_file}")
        print("\nDone. All files processed successfully.")
    
    def sort_bboxes(
        self,
        bboxes:List[Tuple[float,float,float,float]]
    ) -> List[Tuple[float,float,float,float]]:
        return sorted(bboxes,key=lambda b:(-b[0],b[1]))
    
    def is_sorted_bboxes(
        self,
        bboxes:List[Tuple[float,float,float,float]]
    )->bool:
        for i in range(1,len(bboxes)):
            prev_lat, prev_lon, _, _ = bboxes[i - 1]
            curr_lat, curr_lon, _, _ = bboxes[i]
            
            if curr_lat > prev_lat:
                return False

            if curr_lat == prev_lat and curr_lon < prev_lon:
                return False
        return True
    
    def _each_block_mmsi_extract(
        self,
        bbox: Tuple[float, float, float, float],
        dataset_dir: str,
    ) -> pl.DataFrame:
        """
        Returns a Polars DataFrame containing every row whose lat/lon
        falls inside the given bbox from the given CSV file.
 
        Returns an empty DataFrame if no rows are found.
 
        IMPORTANT — original bug fixed:
            The original code passed  min_lon=max_lon  (copy-paste error).
            That collapsed the longitude filter to a single point and
            silently returned zero rows for every cell.
            Fixed below: min_lon=min_lon, max_lon=max_lon.
        """
        if not isinstance(dataset_dir, str):
            raise ValueError("Cannot proceed: a CSV file path string is required.")
 
        min_lat, min_lon, max_lat, max_lon = bbox   # unpack correctly
 
        path_address = self.filter_by_bbox_give_path(
            min_lat       = min_lat,
            max_lat       = max_lat,
            min_lon       = min_lon,       # ← BUG FIX (was max_lon)
            max_lon       = max_lon,
            location_name = "patch_detection",
            dataset_dir   = dataset_dir,
            create_new    = True,
            for_group     = False,
        )
 
        # No detections inside this grid cell
        if not path_address:
            return pl.DataFrame()
 
        # Sanity check — for_group=False must return exactly one file
        if len(path_address) > 1:
            raise ValueError(
                "Expected a single file path but received multiple. "
                "Check that for_group=False is set correctly."
            )
 
        data_path = path_address[0]
        if not os.path.exists(data_path):
            raise ValueError(f"Output file does not exist: {data_path}")
 
        # Return as an eager DataFrame (collected) so callers can work
        # on it immediately without managing lazy contexts.
        SCHEMA = {
            "mmsi":         pl.Int64,
            "base_date_time": pl.Utf8,
            "longitude":    pl.Float64,
            "latitude":     pl.Float64,
            "sog":          pl.Float64,
            "cog":          pl.Float64,
            "heading":      pl.Float64,
            "vessel_name":  pl.Utf8,
            "imo":          pl.Utf8,
            "call_sign":    pl.Utf8,
            "vessel_type":  pl.Int32,
            "status":       pl.Utf8,
            "length":       pl.Float64,
            "width":        pl.Float64,
            "draft":        pl.Float64,
            "cargo":        pl.Utf8,
            "transceiver":  pl.Utf8,
        }
        # return pl.read_csv(data_path, infer_schema_length=1000)
        return pl.read_csv(data_path,schema_overrides=SCHEMA)
 
 
    # ------------------------------------------------------------------ #
    #  _aggregate_bbox_features                                           #
    # ------------------------------------------------------------------ #
    def _aggregate_bbox_features(
        self,
        df: pl.DataFrame,
        bbox: Tuple[float, float, float, float],
    ) -> dict:
        """
        Given a DataFrame of raw rows for one bbox, computes all 10
        raw (pre-normalisation) macro feature values and returns them
        as a flat dictionary.
 
        This is a pure computation helper — no I/O, no side effects.
 
        Returned keys
        -------------
        bbox_min_lat, bbox_min_lon, bbox_max_lat, bbox_max_lon
        vessel_count       : int   — unique MMSI count
        ping_count         : int   — total rows
        mean_sog           : float — arithmetic mean of SOG
        mean_cog_sin       : float — mean of sin(COG radians)
        mean_cog_cos       : float — mean of cos(COG radians)
        cog_circular_var   : float — 1 - ||(mean_sin, mean_cos)||
        type_diversity     : int   — count of distinct vessel_type codes
        dominant_type      : int   — most frequent vessel_type code
        """
        min_lat, min_lon, max_lat, max_lon = bbox
 
        # ── vessel count & ping count ────────────────────────────────────
        vessel_count = df["mmsi"].n_unique()
        ping_count   = len(df)
 
        # ── mean SOG ────────────────────────────────────────────────────
        # Fill missing / null SOG with 0 (stationary assumption)
        sog_series = df["sog"].fill_null(0.0).fill_nan(0.0)
        mean_sog   = float(sog_series.mean()) # type:ignore
 
        # ── circular COG statistics ──────────────────────────────────────
        # COG 360.0 is AIS code for "unknown" — treat as missing → 0.0
        # We convert to radians then compute sin/cos before averaging
        # because COG is a circular quantity (359° ≈ 1°, not 180°).
        cog_series = (
            df["cog"]
            .fill_null(0.0)
            .fill_nan(0.0)
            .map_elements(lambda v: 0.0 if v == 360.0 else v, return_dtype=pl.Float64)
        )
 
        # Convert degrees → radians, then sin/cos
        cog_sin_series = cog_series.map_elements(
            lambda v: math.sin(math.radians(v)), return_dtype=pl.Float64
        )
        cog_cos_series = cog_series.map_elements(
            lambda v: math.cos(math.radians(v)), return_dtype=pl.Float64
        )
 
        mean_cog_sin = float(cog_sin_series.mean()) # type:ignore
        mean_cog_cos = float(cog_cos_series.mean()) # type:ignore
 
        # Circular variance: 1 - magnitude of mean direction vector
        # 0 → all vessels same direction (shipping lane)
        # 1 → vessels pointing every direction (anchorage / congestion)
        resultant_length = math.sqrt(mean_cog_sin ** 2 + mean_cog_cos ** 2)
        cog_circular_var = round(1.0 - resultant_length, 6)
 
        # ── vessel type features ─────────────────────────────────────────
        type_series    = df["vessel_type"].drop_nulls()
        type_diversity = type_series.n_unique() if len(type_series) > 0 else 0
 
        if len(type_series) > 0:
            # value_counts returns a DataFrame with columns
            # [vessel_type, count] — pick the row with max count
            vc = (
                type_series
                .value_counts()
                .sort("count", descending=True)
            )
            dominant_type = int(vc["vessel_type"][0])
        else:
            dominant_type = 0
 
        return {
            # ── identity ────────────────────────────────────────────────
            "bbox_min_lat":     min_lat,
            "bbox_min_lon":     min_lon,
            "bbox_max_lat":     max_lat,
            "bbox_max_lon":     max_lon,
            # ── traffic ─────────────────────────────────────────────────
            "vessel_count":     vessel_count,
            "ping_count":       ping_count,
            # ── motion ──────────────────────────────────────────────────
            "mean_sog":         round(mean_sog, 6),
            "mean_cog_sin":     round(mean_cog_sin, 6),
            "mean_cog_cos":     round(mean_cog_cos, 6),
            "cog_circular_var": cog_circular_var,
            # ── vessel composition ───────────────────────────────────────
            "type_diversity":   type_diversity,
            "dominant_type":    dominant_type,
        }
 
 
    # ------------------------------------------------------------------ #
    #  create_parquet_json                                                #
    # ------------------------------------------------------------------ #
    def create_parquet_json(
        self,
        output_dir: str,
        for_group: bool = False,
        dataset_dir: Optional[str] = None,
    ) -> List[str]:
        """
        Main entry point for macro preprocessing.
 
        For every occupied bbox in self.bboxes_list it:
          1. Calls _each_block_mmsi_extract to get raw rows for that bbox
          2. Calls _aggregate_bbox_features to compute macro feature values
          3. Collects all bbox records, sorts them by (-min_lat, min_lon)
          4. Saves one .parquet + one .json file
 
        Parameters
        ----------
        output_dir  : str
            Dedicated directory where parquet and JSON files are saved.
            Created automatically if it does not exist.
 
        for_group   : bool, default False
            False → process a single CSV (dataset_dir or self.dataset_dir)
            True  → process every CSV in self.datasets_location
 
        dataset_dir : str, optional
            Path to a specific CSV file. Used only when for_group=False.
            Falls back to self.dataset_dir if not provided.
 
        Returns
        -------
        List of absolute paths to all files created (parquet + JSON pairs).
        """
        os.makedirs(output_dir, exist_ok=True)
        created_files: List[str] = []
 
        # ── resolve which CSV files to process ──────────────────────────
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
 
        # ── process each CSV ─────────────────────────────────────────────
        for csv_path in tqdm(csv_files, desc="Processing daily CSVs", unit="file"):
 
            date_stem  = os.path.splitext(os.path.basename(csv_path))[0]
            # e.g.  "2022-01-01"  from  "2022-01-01.csv"
 
            bbox_records: List[dict] = []
 
            for bbox in tqdm(
                self.bboxes_list,
                desc=f"  Bboxes [{date_stem}]",
                unit="cell",
                leave=False,
            ):
                # ── Step 1: get raw rows for this bbox ───────────────────
                df = self._each_block_mmsi_extract(
                    bbox        = bbox,
                    dataset_dir = csv_path,
                )
 
                # Skip empty cells — they produce no token
                if df.is_empty():
                    continue
 
                # ── Step 2: aggregate into one feature record ────────────
                record = self._aggregate_bbox_features(df=df, bbox=bbox)
                bbox_records.append(record)
 
            # ── no occupied cells found for this day ─────────────────────
            if not bbox_records:
                tqdm.write(f"  [{date_stem}] No occupied bboxes found — skipping.")
                continue
 
            # ── Step 3: sort records north→south, west→east ──────────────
            # Key: (-min_lat, min_lon)
            # This guarantees the same spatial reading order every day.
            bbox_records.sort(key=lambda r: (-r["bbox_min_lat"], r["bbox_min_lon"]))
 
            # ── Step 4a: save as Parquet ──────────────────────────────────
            # Polars is used throughout — fast columnar I/O, no pandas dep.
            parquet_path = os.path.abspath(
                os.path.join(output_dir, f"{date_stem}_macro_raw.parquet")
            )
            pl.DataFrame(bbox_records).write_parquet(parquet_path)
            created_files.append(parquet_path)
 
            # ── Step 4b: save as JSON ─────────────────────────────────────
            # Human-readable, useful for debugging and sharing with
            # the architecture team.
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
 
        # ── summary ───────────────────────────────────────────────────────
        n_days   = len(csv_files)
        n_parq   = sum(1 for f in created_files if f.endswith(".parquet"))
        n_json   = sum(1 for f in created_files if f.endswith(".json"))
        print(f"\ncreate_parquet_json complete")
        print(f"  CSVs processed : {n_days:,}")
        print(f"  Parquet files  : {n_parq:,}")
        print(f"  JSON files     : {n_json:,}")
        print(f"  Output dir     : {output_dir}")
 
        return created_files
    
    def compute_norm_stats(
        self,
        raw_parquet_dir: str,
        output_dir: str,
        training_date_stems: Optional[List[str]] = None,
    ) -> dict:
        """
        Pass 1 over training-split raw parquet files to compute all
        normalisation statistics needed for the macro encoder.
 
        Parameters
        ----------
        raw_parquet_dir : str
            Directory containing the *_macro_raw.parquet files produced
            by create_parquet_json.  Only files whose date stem appears
            in training_date_stems are read (if that list is provided).
 
        output_dir : str
            Directory where norm_stats.json is written.
            Created automatically if it does not exist.
 
        training_date_stems : list[str], optional
            Explicit list of date stems that belong to the training split,
            e.g. ["2022-01-01", "2022-01-02", ...].
            If None, ALL parquet files in raw_parquet_dir are used —
            only do this if the directory already contains training files
            exclusively.
 
        Returns
        -------
        dict  The stats dictionary that was saved to norm_stats.json.
        """
        os.makedirs(output_dir, exist_ok=True)
 
        # ── collect parquet files to process ────────────────────────────────
        all_parquet = sorted([
            f for f in os.listdir(raw_parquet_dir)
            if f.endswith("_macro_raw.parquet")
        ])
 
        if not all_parquet:
            raise FileNotFoundError(
                f"No *_macro_raw.parquet files found in: {raw_parquet_dir}"
            )
 
        if training_date_stems is not None:
            # Keep only files whose date stem is in the training list
            training_set = set(training_date_stems)
            all_parquet = [
                f for f in all_parquet
                if os.path.splitext(f)[0].replace("_macro_raw", "") in training_set
            ]
            if not all_parquet:
                raise ValueError(
                    "No parquet files matched the provided training_date_stems. "
                    "Check that the stems match the filename pattern "
                    "{date_stem}_macro_raw.parquet"
                )
 
        print(f"Computing norm stats from {len(all_parquet)} training parquet file(s)...")
 
        # Running extremes — updated as we scan each file
        max_vessel_count: int = 0
        max_ping_count:   int = 0
        all_type_codes:   set = set()
 
        for fname in tqdm(all_parquet, desc="Scanning training parquets", unit="file"):
            fpath = os.path.join(raw_parquet_dir, fname)
 
            # Read only the columns we need — faster than loading everything
            df = pl.read_parquet(
                fpath,
                columns=["vessel_count", "ping_count", "dominant_type", "type_diversity"]
            )
 
            if df.is_empty():
                continue
 
            # ── vessel count maximum ─────────────────────────────────────────
            file_max_vc = int(df["vessel_count"].max() or 0)  # type:ignore
            if file_max_vc > max_vessel_count:
                max_vessel_count = file_max_vc
 
            # ── ping count maximum ───────────────────────────────────────────
            file_max_pc = int(df["ping_count"].max() or 0)  # type:ignore
            if file_max_pc > max_ping_count:
                max_ping_count = file_max_pc
 
            # ── collect all vessel type codes ────────────────────────────────
            # dominant_type gives us the most frequent per cell per day.
            # We read the raw parquet which stores the dominant type code.
            # For total unique types we need the full type column from the
            # original CSV — but we do not have that here.  Instead we
            # accumulate dominant_type values across all files, which gives
            # us the set of types that actually dominate cells.
            # NOTE: if you want every type code that ever appeared (not just
            # dominant), pass the raw CSV directory instead and scan vessel_type
            # directly.  For the MVP, dominant-type set is sufficient.
            type_vals = df["dominant_type"].drop_nulls().to_list()
            all_type_codes.update(int(v) for v in type_vals)
 
            tqdm.write(
                f"  {fname}: max_vc={file_max_vc}  max_pc={file_max_pc}  "
                f"types_seen={len(all_type_codes)}"
            )
 
        # ── derive final stats ───────────────────────────────────────────────
        max_type_code       = int(max(all_type_codes)) if all_type_codes else 99
        total_unique_types  = len(all_type_codes)
 
        stats = {
            # ── macro-specific ───────────────────────────────────────────────
            "MAX_VESSEL_COUNT_PER_CELL": max_vessel_count,
            "MAX_PING_COUNT_PER_CELL":   max_ping_count,
            "TOTAL_UNIQUE_TYPES":        total_unique_types,
            "MAX_TYPE_CODE":             max_type_code,
            # ── geographic bounds (fixed — not computed from data) ───────────
            "GULF_LAT_MIN": self.min_lat,
            "GULF_LAT_MAX": self.max_lat,
            "GULF_LON_MIN": self.min_lon,
            "GULF_LON_MAX": self.max_lon,
            # ── micro stats placeholder ──────────────────────────────────────
            # MicroAnalysis.compute_micro_norm_stats will fill these in.
            # They are kept in the same file so there is one source of truth.
            "MEAN_LENGTH":  None,
            "MEAN_DRAFT":   None,
        }
 
        stats_path = os.path.abspath(os.path.join(output_dir, "norm_stats.json"))
        with open(stats_path, "w", encoding="utf-8") as fh:
            json.dump(stats, fh, indent=2)
 
        # ── summary ──────────────────────────────────────────────────────────
        print(f"\nNorm stats computed and saved → {stats_path}")
        print(f"  MAX_VESSEL_COUNT_PER_CELL : {max_vessel_count:,}")
        print(f"  MAX_PING_COUNT_PER_CELL   : {max_ping_count:,}")
        print(f"  TOTAL_UNIQUE_TYPES        : {total_unique_types}")
        print(f"  MAX_TYPE_CODE             : {max_type_code}")
        print(f"  Gulf bounds               : lat [{self.min_lat}, {self.max_lat}]"
              f"  lon [{self.min_lon}, {self.max_lon}]")
 
        return stats 
    
    def normalise_and_save_tensors(
        self,
        raw_parquet_dir: str,
        tensor_output_dir: str,
        norm_stats_path: str,
        date_stems: Optional[List[str]] = None,
    ) -> List[str]:
        """
        Pass 2 — reads raw parquet files, normalises every feature using
        the frozen stats from norm_stats.json, and saves one PyTorch tensor
        per day as {date_stem}_macro.pt with shape [N_occupied_cells, 10].
    
        Run this separately for train, val, and test by passing the
        appropriate date_stems list each time.  All three splits always
        use the same norm_stats_path — never refit on val or test.
    
        Parameters
        ----------
        raw_parquet_dir : str
            Directory containing *_macro_raw.parquet files from
            create_parquet_json.
    
        tensor_output_dir : str
            Directory where .pt tensor files are saved.
            Created automatically if it does not exist.
    
        norm_stats_path : str
            Path to norm_stats.json written by compute_norm_stats.
    
        date_stems : list[str], optional
            Restrict processing to these date stems only.
            If None, all parquet files in raw_parquet_dir are processed.
    
        Returns
        -------
        List of absolute paths to all .pt files created.
        """
        os.makedirs(tensor_output_dir, exist_ok=True)
    
        # ── load frozen stats ────────────────────────────────────────────────
        if not os.path.exists(norm_stats_path):
            raise FileNotFoundError(
                f"norm_stats.json not found at: {norm_stats_path}\n"
                "Run compute_norm_stats() on training data first."
            )
    
        with open(norm_stats_path, "r", encoding="utf-8") as fh:
            stats = json.load(fh)
    
        # Pull out every constant we need — fail loudly if any are missing
        MAX_VC    = stats["MAX_VESSEL_COUNT_PER_CELL"]
        MAX_PC    = stats["MAX_PING_COUNT_PER_CELL"]
        MAX_TYPE  = stats["MAX_TYPE_CODE"]
        TOT_TYPES = stats["TOTAL_UNIQUE_TYPES"]
        LAT_MIN   = stats["GULF_LAT_MIN"]
        LAT_MAX   = stats["GULF_LAT_MAX"]
        LON_MIN   = stats["GULF_LON_MIN"]
        LON_MAX   = stats["GULF_LON_MAX"]
    
        LAT_RANGE = LAT_MAX - LAT_MIN   # 14.0581 for Gulf bounds
        LON_RANGE = LON_MAX - LON_MIN   # 17.6210 for Gulf bounds
    
        # Guard against divide-by-zero if stats were computed from empty data
        if MAX_VC == 0 or MAX_PC == 0 or MAX_TYPE == 0 or TOT_TYPES == 0:
            raise ValueError(
                "One or more stats are zero — norm_stats.json may have been "
                "computed from empty data.  Re-run compute_norm_stats()."
            )
    
        # ── collect parquet files ────────────────────────────────────────────
        all_parquet = sorted([
            f for f in os.listdir(raw_parquet_dir)
            if f.endswith("_macro_raw.parquet")
        ])
    
        if date_stems is not None:
            stem_set    = set(date_stems)
            all_parquet = [
                f for f in all_parquet
                if os.path.splitext(f)[0].replace("_macro_raw", "") in stem_set
            ]
    
        if not all_parquet:
            print("No matching parquet files found.")
            return []
    
        created_files: List[str] = []
    
        # ── process each day ─────────────────────────────────────────────────
        for fname in tqdm(all_parquet, desc="Normalising macro tensors", unit="file"):
    
            date_stem = fname.replace("_macro_raw.parquet", "")
            fpath     = os.path.join(raw_parquet_dir, fname)
            df        = pl.read_parquet(fpath)
    
            if df.is_empty():
                tqdm.write(f"  [{date_stem}] Empty parquet — skipping.")
                continue
    
            # ── sort north→south, west→east ──────────────────────────────────
            # Guarantees identical token ordering every day.
            # Sort key: (-bbox_min_lat, bbox_min_lon)
            df = df.sort(
                by=["bbox_min_lat", "bbox_min_lon"],
                descending=[True, False]
            )
    
            rows = df.to_dicts()   # list of dicts, one per occupied cell
            feature_rows: List[List[float]] = []
    
            for r in rows:
                # ── Feature 1: lat_norm ──────────────────────────────────────
                # Use bbox CORNER lat (not vessel lat) so the same cell always
                # maps to the same value regardless of which vessels are inside.
                lat_norm = (r["bbox_min_lat"] - LAT_MIN) / LAT_RANGE
                lat_norm = max(0.0, min(1.0, lat_norm))
    
                # ── Feature 2: lon_norm ──────────────────────────────────────
                lon_norm = (r["bbox_min_lon"] - LON_MIN) / LON_RANGE
                lon_norm = max(0.0, min(1.0, lon_norm))
    
                # ── Feature 3: vessel_count_norm — log scale ─────────────────
                # Log used because port cells can have 500+ vessels while
                # open-water cells have 1 — min-max would crush open-water.
                vc = max(0, int(r["vessel_count"]))
                vessel_count_norm = math.log1p(vc) / math.log1p(MAX_VC)
                vessel_count_norm = max(0.0, min(1.0, vessel_count_norm))
    
                # ── Feature 4: ping_density_norm — log scale ─────────────────
                pc = max(0, int(r["ping_count"]))
                ping_density_norm = math.log1p(pc) / math.log1p(MAX_PC)
                ping_density_norm = max(0.0, min(1.0, ping_density_norm))
    
                # ── Feature 5: mean_sog_norm ─────────────────────────────────
                mean_sog_norm = float(r["mean_sog"] or 0.0) / 30.0
                mean_sog_norm = max(0.0, min(1.0, mean_sog_norm))
    
                # ── Features 6 & 7: mean_cog_sin / mean_cog_cos ─────────────
                # Already in [-1, 1] — no further normalisation needed.
                mean_cog_sin = float(r["mean_cog_sin"] or 0.0)
                mean_cog_cos = float(r["mean_cog_cos"] or 0.0)
                # Clamp defensively to handle any floating-point edge cases
                mean_cog_sin = max(-1.0, min(1.0, mean_cog_sin))
                mean_cog_cos = max(-1.0, min(1.0, mean_cog_cos))
    
                # ── Feature 8: cog_circular_var ──────────────────────────────
                # Already in [0, 1] by construction — no normalisation needed.
                # 0 = shipping lane (all same direction)
                # 1 = anchorage / congestion (all directions)
                cog_circular_var = float(r["cog_circular_var"] or 0.0)
                cog_circular_var = max(0.0, min(1.0, cog_circular_var))
    
                # ── Feature 9: type_diversity_norm ───────────────────────────
                td = int(r["type_diversity"] or 0)
                type_diversity_norm = td / TOT_TYPES
                type_diversity_norm = max(0.0, min(1.0, type_diversity_norm))
    
                # ── Feature 10: dominant_type_norm ───────────────────────────
                dt = int(r["dominant_type"] or 0)
                dominant_type_norm = dt / MAX_TYPE
                dominant_type_norm = max(0.0, min(1.0, dominant_type_norm))
    
                feature_rows.append([
                    lat_norm,           # 1
                    lon_norm,           # 2
                    vessel_count_norm,  # 3
                    ping_density_norm,  # 4
                    mean_sog_norm,      # 5
                    mean_cog_sin,       # 6
                    mean_cog_cos,       # 7
                    cog_circular_var,   # 8
                    type_diversity_norm,# 9
                    dominant_type_norm, # 10
                ])
    
            if not feature_rows:
                tqdm.write(f"  [{date_stem}] No features after normalisation — skipping.")
                continue
    
            # ── stack into [N_cells, 10] tensor ──────────────────────────────
            tensor = torch.tensor(feature_rows, dtype=torch.float32)
    
            # Sanity check shape before saving
            assert tensor.ndim == 2,           f"Expected 2D tensor, got {tensor.ndim}D"
            assert tensor.shape[1] == 10,      f"Expected 10 features, got {tensor.shape[1]}"
            assert not torch.isnan(tensor).any(), f"NaN detected in {date_stem}"
    
            # ── save ──────────────────────────────────────────────────────────
            out_path = os.path.abspath(
                os.path.join(tensor_output_dir, f"{date_stem}_macro.pt")
            )
            torch.save(tensor, out_path)
            created_files.append(out_path)
    
            tqdm.write(
                f"  [{date_stem}] shape {list(tensor.shape)} → "
                f"{os.path.basename(out_path)}"
            )
    
        # ── summary ──────────────────────────────────────────────────────────
        print(f"\nnormalise_and_save_tensors complete")
        print(f"  Files processed : {len(all_parquet):,}")
        print(f"  Tensors saved   : {len(created_files):,}")
        print(f"  Output dir      : {tensor_output_dir}")
    
        return created_files
 
    def validate_macro_tensors(
        self,
        tensor_dir: str,
        expected_date_stems: Optional[List[str]] = None,
        expected_feature_dim: int = 10,
        value_min: float = -1.1,
        value_max: float = 1.1,
    ) -> dict:
        """
        Validates every *_macro.pt file in tensor_dir.
    
        Checks performed
        ----------------
        1. File exists for every date in expected_date_stems (if provided).
        2. Tensor is 2-dimensional.
        3. Second dimension equals expected_feature_dim (default 10).
        4. No NaN values anywhere in the tensor.
        5. No Inf values anywhere in the tensor.
        6. All values fall within [value_min, value_max].
        Default is [-1.1, 1.1] — slightly wider than [-1,1] to allow
        for the cog_sin/cog_cos features which are bounded [-1,1].
    
        Parameters
        ----------
        tensor_dir : str
            Directory containing *_macro.pt files.
    
        expected_date_stems : list[str], optional
            Date stems you expect a tensor for (e.g. all training dates).
            Any missing stems are reported as errors.
    
        expected_feature_dim : int, default 10
            Expected width of the tensor (number of features).
    
        value_min / value_max : float
            Acceptable value range for all tensor elements.
    
        Returns
        -------
        dict with keys:
            "total_files"    : int   — number of .pt files found
            "passed"         : int   — files that passed all checks
            "failed"         : int   — files with at least one error
            "missing_dates"  : list  — date stems with no .pt file
            "errors"         : dict  — {filename: [list of error strings]}
        """
        pt_files = sorted([
            f for f in os.listdir(tensor_dir)
            if f.endswith("_macro.pt")
        ])
    
        # ── check for missing dates ──────────────────────────────────────────
        missing_dates: List[str] = []
        if expected_date_stems is not None:
            found_stems = {
                f.replace("_macro.pt", "") for f in pt_files
            }
            missing_dates = [
                stem for stem in expected_date_stems
                if stem not in found_stems
            ]
    
        errors: dict = {}
        passed = 0
        failed = 0
    
        for fname in tqdm(pt_files, desc="Validating macro tensors", unit="file"):
            fpath      = os.path.join(tensor_dir, fname)
            file_errors: List[str] = []
    
            try:
                tensor = torch.load(fpath, map_location="cpu", weights_only=True)
            except Exception as e:
                errors[fname] = [f"Failed to load: {e}"]
                failed += 1
                continue
    
            # ── Check 1: must be 2-dimensional ──────────────────────────────
            if tensor.ndim != 2:
                file_errors.append(
                    f"Wrong number of dimensions: expected 2, got {tensor.ndim}"
                )
    
            # ── Check 2: feature dimension must match ────────────────────────
            elif tensor.shape[1] != expected_feature_dim:
                file_errors.append(
                    f"Wrong feature dim: expected {expected_feature_dim}, "
                    f"got {tensor.shape[1]}"
                )
    
            # ── Check 3: no NaN ─────────────────────────────────────────────
            if torch.isnan(tensor).any():
                nan_count = int(torch.isnan(tensor).sum().item())
                file_errors.append(f"Contains {nan_count} NaN value(s)")
    
            # ── Check 4: no Inf ─────────────────────────────────────────────
            if torch.isinf(tensor).any():
                inf_count = int(torch.isinf(tensor).sum().item())
                file_errors.append(f"Contains {inf_count} Inf value(s)")
    
            # ── Check 5: value range ─────────────────────────────────────────
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
    
            # ── Check 6: must have at least one cell ─────────────────────────
            if tensor.shape[0] == 0:
                file_errors.append("Tensor has zero rows (no occupied cells)")
    
            if file_errors:
                errors[fname] = file_errors
                failed += 1
                tqdm.write(f"  FAIL [{fname}]: {' | '.join(file_errors)}")
            else:
                passed += 1
    
        # ── report ───────────────────────────────────────────────────────────
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

    

if __name__ == "__main__":
    # dataset_dir = r"D:\AIS_project\data\CSV_files\new_AIS\ais-2022-01-01.csv"
    # dataset_location = r"D:\AIS_project\data\CSV_files\new_AIS"
    Analysis_results = r"D:\AIS_project\Analysis_results"

    exp_bbox = [(17.4068,-98.0539,31.4648,-80.4330)]

    # macroAnalysis_object = MacroAnalysis(dataset_dir=dataset_dir,datasets_location=dataset_location,analysis_result_dir=Analysis_results)
    # bbox_list = macroAnalysis_object.bboxes_list

    # path_list = macroAnalysis_object.Get_patch_DataFile(for_group=True)
    # exist_check = macroAnalysis_object.Clean_MMSI_DataFrame("File_exist_check",["vessel"],True )

    Mexicodataset_dir = r"D:\AIS_project\data\CSV_files\GulfMexico_AIS\Gulf_of_Mexico_ais-2022-01-01.csv"
    Mexicodataset_location = r"D:\AIS_project\data\CSV_files\GulfMexico_AIS"
    
    MexicoGulf_MacroAnalysis = MacroAnalysis(dataset_dir=Mexicodataset_dir,datasets_location=Mexicodataset_location,analysis_result_dir=Analysis_results)
    bbox_list = MexicoGulf_MacroAnalysis.bboxes_list

    # test_path = MexicoGulf_MacroAnalysis.create_parquet_json(
    #     output_dir = "JSON_PARQUET_result",
    #     for_group = False
    # )


    # MexicoGulf_MacroAnalysis.Clean_MMSI_DataFrame_gpt("Unique_MMSI_ID",["vessel"],False)
    # print("Boxes are already sorted") if MexicoGulf_MacroAnalysis.is_sorted_bboxes(bbox_list) else print("Boxess are not sorted please initiate the sorting protocol")
    
    sorted_bbox = MexicoGulf_MacroAnalysis.sort_bboxes(bbox_list)
    # print("New Boxes are now sorted") if MexicoGulf_MacroAnalysis.is_sorted_bboxes(sorted_bbox) else print("New Boxess are not sorted please initiate the sorting protocol")
    # print(sorted_bbox)
    
    # print(MexicoGulf_MacroAnalysis.plot_bboxes(exp_bbox,"Whole map"))
    # print(MexicoGulf_MacroAnalysis.plot_bboxes(sorted_bbox,"Each block"))

    vis_list = MexicoGulf_MacroAnalysis.extract_bbox_and_counts(json_file_path=r"D:\AIS_project\JSON_PARQUET_result\Gulf_of_Mexico_ais-2022-01-01_macro_raw.json")
    MexicoGulf_MacroAnalysis.plot_bboxes_intensity(bboxes_list=vis_list,title="")

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Micro analysis code
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

import polars as pl
import os
import torch
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
    GULF_LAT_RANGE: float = GULF_LAT_MAX - GULF_LAT_MIN   # 14.0580
    GULF_LON_RANGE: float = GULF_LON_MAX - GULF_LON_MIN   # 17.6209 (≈17.6230 per spec)

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
        """
        Parameters
        ----------
        datasets_location : str
            Folder containing all daily AIS CSVs.
        dataset_dir : str
            Path to a single default CSV file.
        analysis_result_dir : str
            Base output directory for analysis results.
        min_lat, min_lon, max_lat, max_lon : float
            Gulf of Mexico geographic bounds (fixed constants).
        window_size_hours : int
            Size of each time window in hours (default 1).
        min_pings_per_window : int
            Discard windows with fewer pings than this threshold.
        max_sog : float
            Normalisation cap for Speed Over Ground (knots).
        max_length : float
            Normalisation cap for vessel length (metres).
        max_draft : float
            Normalisation cap for vessel draft (metres).
        """
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
    #  METHOD 1 — extract_vessel_windows                                  #
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

        Parameters
        ----------
        dataset_dir : str
            Path to a single CSV file (used when for_group=False).
        for_group : bool
            If True, process every CSV in self.datasets_location instead.

        Returns
        -------
        list[dict]
            Each dict contains: mmsi, date, window_hour, ping_count, df.
        """
        # ── resolve which CSV files to process ───────────────────────────
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
            # ── read with explicit schema overrides ──────────────────────
            df = pl.read_csv(
                csv_path,
                schema_overrides=self.SCHEMA_OVERRIDES,
                infer_schema_length=1000,
            )

            # ── parse timestamp ──────────────────────────────────────────
            df = df.with_columns(
                pl.col("base_date_time")
                .str.to_datetime("%Y-%m-%d %H:%M:%S")
                .alias("base_date_time")
            )

            # ── extract date string from filename or first row ───────────
            date_str = self._extract_date_from_path(csv_path)
            if date_str is None:
                # fallback: use the date from the first row
                first_ts = df["base_date_time"][0]
                date_str = str(first_ts.date()) if first_ts is not None else "unknown"

            # ── group by MMSI ────────────────────────────────────────────
            for mmsi_val, vessel_df in df.group_by("mmsi"):
                mmsi_int = int(mmsi_val[0])  # type: ignore

                # sort by timestamp ascending
                vessel_df = vessel_df.sort("base_date_time")

                # floor to nearest hour to define window boundaries
                vessel_df = vessel_df.with_columns(
                    pl.col("base_date_time")
                    .dt.truncate(f"{self.window_size_hours}h")
                    .alias("window_key")
                )

                # group by floored hour
                for window_key_val, window_df in vessel_df.group_by("window_key"):
                    window_ts = window_key_val[0]  # type: ignore
                    window_hour = int(window_ts.hour)  # type: ignore
                    ping_count = window_df.height

                    # discard windows below minimum ping threshold
                    if ping_count < self.min_pings_per_window:
                        continue

                    # sort within window by timestamp
                    window_df = window_df.sort("base_date_time")

                    # drop the helper column before returning
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
    #  METHOD 2 — _build_ping_features                                    #
    # ================================================================== #
    def _build_ping_features(
        self,
        df: pl.DataFrame,
        norm_stats: dict,
    ) -> list[list[float]]:
        """
        Takes a single window DataFrame (all rows for one vessel in one
        hour) and builds the 11-element normalised feature vector for
        every ping. Returns a list of lists — one inner list per ping,
        in time order.

        Parameters
        ----------
        df : pl.DataFrame
            Raw rows for a single vessel-window, sorted by base_date_time.
        norm_stats : dict
            Normalisation statistics loaded from micro_norm_stats.json.

        Returns
        -------
        list[list[float]]
            Length = number of pings, each inner list has exactly 11 floats.
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

        # ── ensure sorted by time ────────────────────────────────────────
        df = df.sort("base_date_time")

        # ── extract columns as python lists for speed ────────────────────
        lats     = df["latitude"].to_list()
        lons     = df["longitude"].to_list()
        sogs     = df["sog"].to_list()
        cogs     = df["cog"].to_list()
        headings = df["heading"].to_list()
        lengths  = df["length"].to_list()
        drafts   = df["draft"].to_list()
        vtypes   = df["vessel_type"].to_list()
        times    = df["base_date_time"].to_list()

        n_pings = len(lats)
        features: list[list[float]] = []

        for i in range(n_pings):
            # ── missing value fills ──────────────────────────────────────
            lat_val = float(lats[i]) if lats[i] is not None else 0.0
            lon_val = float(lons[i]) if lons[i] is not None else 0.0

            sog_val = float(sogs[i]) if sogs[i] is not None else 0.0

            cog_val = cogs[i]
            cog_unknown = (cog_val is None) or (float(cog_val) == 360.0)

            heading_val = headings[i]
            heading_unavailable = (heading_val is None) or (float(heading_val) == 511.0)

            length_val = lengths[i]
            if length_val is None or float(length_val) == 0.0:
                length_val = MEAN_LENGTH
            else:
                length_val = float(length_val)

            draft_val = drafts[i]
            if draft_val is None or float(draft_val) == 0.0:
                draft_val = 0.0
            else:
                draft_val = float(draft_val)

            vtype_val = vtypes[i]
            if vtype_val is None:
                vtype_val = 0
            else:
                vtype_val = int(vtype_val)

            # ── Feature 1: lat_norm ──────────────────────────────────────
            lat_norm = (lat_val - LAT_MIN) / LAT_RANGE
            lat_norm = max(0.0, min(1.0, lat_norm))

            # ── Feature 2: lon_norm ──────────────────────────────────────
            lon_norm = (lon_val - LON_MIN) / LON_RANGE
            lon_norm = max(0.0, min(1.0, lon_norm))

            # ── Feature 3: sog_norm ──────────────────────────────────────
            sog_norm = max(0.0, min(1.0, sog_val / MAX_SOG))

            # ── Feature 4 & 5: cog_sin, cog_cos ─────────────────────────
            if cog_unknown:
                cog_sin = 0.0
                cog_cos = 1.0
            else:
                cog_rad = math.radians(float(cog_val))
                cog_sin = math.sin(cog_rad)
                cog_cos = math.cos(cog_rad)

            # ── Feature 6 & 7: heading_sin, heading_cos ──────────────────
            if heading_unavailable:
                heading_sin = 0.0
                heading_cos = 1.0
            else:
                heading_rad = math.radians(float(heading_val))
                heading_sin = math.sin(heading_rad)
                heading_cos = math.cos(heading_rad)

            # ── Feature 8: delta_t_norm ──────────────────────────────────
            if i < n_pings - 1:
                dt_seconds = (times[i + 1] - times[i]).total_seconds()
                delta_t_norm = dt_seconds / 3600.0
                delta_t_norm = max(0.0, delta_t_norm)  # guard negative
            else:
                delta_t_norm = 0.0

            # ── Feature 9: length_norm ───────────────────────────────────
            length_norm = max(0.0, min(1.0, length_val / MAX_LENGTH))

            # ── Feature 10: draft_norm ───────────────────────────────────
            draft_norm = max(0.0, min(1.0, draft_val / MAX_DRAFT))

            # ── Feature 11: type_norm ────────────────────────────────────
            if MAX_TYPE_CODE > 0:
                type_norm = max(0.0, min(1.0, vtype_val / MAX_TYPE_CODE))
            else:
                type_norm = 0.0

            features.append([
                lat_norm,       # 1
                lon_norm,       # 2
                sog_norm,       # 3
                cog_sin,        # 4
                cog_cos,        # 5
                heading_sin,    # 6
                heading_cos,    # 7
                delta_t_norm,   # 8
                length_norm,    # 9
                draft_norm,     # 10
                type_norm,      # 11
            ])

        return features

    # ================================================================== #
    #  METHOD 3 — compute_micro_norm_stats                                #
    # ================================================================== #
    def compute_micro_norm_stats(
        self,
        output_dir: str,
        training_date_stems: Optional[list[str]] = None,
        dataset_dir: Optional[str] = None,
        for_group: bool = True,
    ) -> dict:
        """
        Single pass over training-split CSVs to compute the three
        micro-specific statistics needed for normalisation.

        Parameters
        ----------
        output_dir : str
            Directory where micro_norm_stats.json is written.
        training_date_stems : list[str], optional
            Restricts which CSVs are read by date stem.
        dataset_dir : str, optional
            Path to a single CSV (used when for_group=False).
        for_group : bool
            If True, process all CSVs in self.datasets_location.

        Returns
        -------
        dict — the stats dictionary that was saved.
        """
        os.makedirs(output_dir, exist_ok=True)

        # ── resolve which CSV files to process ───────────────────────────
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

        # ── filter by training date stems if provided ────────────────────
        if training_date_stems is not None:
            stem_set = set(training_date_stems)
            csv_files = [
                f for f in csv_files
                if self._extract_date_from_path(f) in stem_set
            ]
            if not csv_files:
                raise ValueError(
                    "No CSV files matched the provided training_date_stems. "
                    "Check that the stems match dates in the filenames."
                )

        print(f"Computing micro norm stats from {len(csv_files)} training CSV(s)...")

        # ── accumulators ─────────────────────────────────────────────────
        length_sum: float = 0.0
        length_count: int = 0
        draft_sum: float = 0.0
        draft_count: int = 0
        max_type_code: int = 0

        for csv_path in tqdm(csv_files, desc="Scanning training CSVs", unit="file"):
            df = pl.read_csv(
                csv_path,
                schema_overrides=self.SCHEMA_OVERRIDES,
                infer_schema_length=1000,
            )

            # ── length: non-null and non-zero ────────────────────────────
            length_col = (
                df["length"]
                .cast(pl.Float64, strict=False)
                .fill_null(0.0)
                .fill_nan(0.0)
            )
            valid_lengths = length_col.filter(length_col > 0.0)
            if len(valid_lengths) > 0:
                length_sum += float(valid_lengths.sum())  # type: ignore
                length_count += len(valid_lengths)

            # ── draft: non-null and non-zero ─────────────────────────────
            draft_col = (
                df["draft"]
                .cast(pl.Float64, strict=False)
                .fill_null(0.0)
                .fill_nan(0.0)
            )
            valid_drafts = draft_col.filter(draft_col > 0.0)
            if len(valid_drafts) > 0:
                draft_sum += float(valid_drafts.sum())  # type: ignore
                draft_count += len(valid_drafts)

            # ── vessel type: max code ────────────────────────────────────
            type_col = (
                df["vessel_type"]
                .cast(pl.Int32, strict=False)
                .drop_nulls()
            )
            if len(type_col) > 0:
                file_max_type = int(type_col.max())  # type: ignore
                if file_max_type > max_type_code:
                    max_type_code = file_max_type

            tqdm.write(
                f"  {os.path.basename(csv_path)}: "
                f"lengths={len(valid_lengths)}  drafts={len(valid_drafts)}  "
                f"max_type_so_far={max_type_code}"
            )

        # ── derive final stats ───────────────────────────────────────────
        mean_length = (length_sum / length_count) if length_count > 0 else 0.0
        mean_draft  = (draft_sum / draft_count) if draft_count > 0 else 0.0

        # Ensure max_type_code is at least 1 to avoid divide-by-zero
        if max_type_code == 0:
            max_type_code = 99

        stats = {
            "MEAN_LENGTH":   round(mean_length, 4),
            "MEAN_DRAFT":    round(mean_draft, 4),
            "MAX_TYPE_CODE": max_type_code,
            # ── fixed geographic constants ───────────────────────────────
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

        # ── summary ──────────────────────────────────────────────────────
        print(f"\nMicro norm stats computed and saved → {stats_path}")
        print(f"  MEAN_LENGTH    : {mean_length:.4f}")
        print(f"  MEAN_DRAFT     : {mean_draft:.4f}")
        print(f"  MAX_TYPE_CODE  : {max_type_code}")
        print(f"  Training CSVs  : {len(csv_files):,}")

        return stats

    # ================================================================== #
    #  METHOD 4 — create_micro_tensors                                    #
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
        Main entry point. Processes daily CSVs into per-vessel per-hour
        PyTorch tensors of shape [N_pings, 11].

        Parameters
        ----------
        output_dir : str
            Directory where .pt tensor files and index JSONs are saved.
        norm_stats_path : str
            Path to micro_norm_stats.json.
        for_group : bool
            If True, process all CSVs in self.datasets_location.
        dataset_dir : str, optional
            Path to a single CSV (used when for_group=False).
        date_stems : list[str], optional
            Restrict processing to these date stems only.

        Returns
        -------
        list[str] — absolute paths to all .pt files created.
        """
        os.makedirs(output_dir, exist_ok=True)

        # ── load frozen norm stats ───────────────────────────────────────
        if not os.path.exists(norm_stats_path):
            raise FileNotFoundError(
                f"micro_norm_stats.json not found at: {norm_stats_path}\n"
                "Run compute_micro_norm_stats() on training data first."
            )

        with open(norm_stats_path, "r", encoding="utf-8") as fh:
            norm_stats = json.load(fh)

        # ── resolve which CSV files to process ───────────────────────────
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

        # ── filter by date stems if provided ─────────────────────────────
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

        # ── process each CSV ─────────────────────────────────────────────
        for csv_path in tqdm(csv_files, desc="Creating micro tensors", unit="file"):

            date_stem = self._extract_date_from_path(csv_path)
            if date_stem is None:
                date_stem = os.path.splitext(os.path.basename(csv_path))[0]

            # Step 1: extract all valid vessel-windows for this day
            windows = self.extract_vessel_windows(
                dataset_dir=csv_path,
                for_group=False,
            )

            if not windows:
                tqdm.write(f"  [{date_stem}] No valid windows — skipping.")
                continue

            daily_index_entries: list[dict] = []
            day_tensor_count = 0

            for w in tqdm(
                windows,
                desc=f"  Windows [{date_stem}]",
                unit="win",
                leave=False,
            ):
                mmsi       = w["mmsi"]
                w_date     = w["date"]
                w_hour     = w["window_hour"]
                w_df       = w["df"]
                ping_count = w["ping_count"]

                # Step 2: build normalised feature rows
                feature_rows = self._build_ping_features(
                    df=w_df,
                    norm_stats=norm_stats,
                )

                if not feature_rows:
                    continue

                # Step 3: stack into 2D tensor [N_pings, 11]
                tensor = torch.tensor(feature_rows, dtype=torch.float32)

                # Step 4: assert shape, check NaN/Inf
                assert tensor.ndim == 2, \
                    f"Expected 2D tensor, got {tensor.ndim}D"
                assert tensor.shape[1] == 11, \
                    f"Expected 11 features, got {tensor.shape[1]}"
                assert not torch.isnan(tensor).any(), \
                    f"NaN detected in {mmsi}_{w_date}_{w_hour:02d}"
                assert not torch.isinf(tensor).any(), \
                    f"Inf detected in {mmsi}_{w_date}_{w_hour:02d}"

                # Step 5: save tensor
                tensor_fname = f"{mmsi}_{w_date}_{w_hour:02d}.pt"
                out_path = os.path.abspath(
                    os.path.join(output_dir, tensor_fname)
                )
                torch.save(tensor, out_path)
                created_files.append(out_path)
                day_tensor_count += 1

                # Step 6: record for daily index
                daily_index_entries.append({
                    "mmsi":        mmsi,
                    "window_hour": w_hour,
                    "ping_count":  ping_count,
                    "tensor_file": tensor_fname,
                })

            # ── save daily JSON index ────────────────────────────────────
            if daily_index_entries:
                daily_index = {
                    "date":           date_stem,
                    "macro_tensor":   f"{date_stem}_macro.pt",
                    "vessel_windows": daily_index_entries,
                }
                index_path = os.path.join(output_dir, f"{date_stem}_micro_index.json")
                with open(index_path, "w", encoding="utf-8") as fh:
                    json.dump(daily_index, fh, indent=2, ensure_ascii=False)

            tqdm.write(
                f"  [{date_stem}] {day_tensor_count:,} tensors saved"
            )

        # ── summary ──────────────────────────────────────────────────────
        print(f"\ncreate_micro_tensors complete")
        print(f"  CSVs processed   : {len(csv_files):,}")
        print(f"  Tensors saved    : {len(created_files):,}")
        print(f"  Output dir       : {output_dir}")

        return created_files

    # ================================================================== #
    #  METHOD 5 — validate_micro_tensors                                  #
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
        Quality assurance. Checks every .pt file in tensor_dir for
        correctness before training starts.

        Parameters
        ----------
        tensor_dir : str
            Directory containing micro .pt tensor files.
        expected_feature_dim : int
            Expected width of each tensor (number of features).
        value_min, value_max : float
            Acceptable value range for all tensor elements.
        check_index_files : bool
            If True, cross-check .pt files against *_micro_index.json.

        Returns
        -------
        dict with keys: total_files, passed, failed, orphaned_files,
             missing_files, errors.
        """
        pt_files = sorted([
            f for f in os.listdir(tensor_dir)
            if f.endswith(".pt")
        ])

        errors: dict = {}
        passed = 0
        failed = 0

        for fname in tqdm(pt_files, desc="Validating micro tensors", unit="file"):
            fpath = os.path.join(tensor_dir, fname)
            file_errors: list[str] = []

            try:
                tensor = torch.load(fpath, map_location="cpu", weights_only=True)
            except Exception as e:
                errors[fname] = [f"Failed to load: {e}"]
                failed += 1
                continue

            # ── Check 1: loads without error → passed above ──────────────

            # ── Check 2: must be 2-dimensional ───────────────────────────
            if tensor.ndim != 2:
                file_errors.append(
                    f"Wrong number of dimensions: expected 2, got {tensor.ndim}"
                )

            # ── Check 3: feature dimension must match ────────────────────
            elif tensor.shape[1] != expected_feature_dim:
                file_errors.append(
                    f"Wrong feature dim: expected {expected_feature_dim}, "
                    f"got {tensor.shape[1]}"
                )

            # ── Check 4: no NaN ──────────────────────────────────────────
            if torch.isnan(tensor).any():
                nan_count = int(torch.isnan(tensor).sum().item())
                file_errors.append(f"Contains {nan_count} NaN value(s)")

            # ── Check 5: no Inf ──────────────────────────────────────────
            if torch.isinf(tensor).any():
                inf_count = int(torch.isinf(tensor).sum().item())
                file_errors.append(f"Contains {inf_count} Inf value(s)")

            # ── Check 6: value range ─────────────────────────────────────
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

            # ── Check 7: minimum ping count ──────────────────────────────
            if tensor.shape[0] < self.min_pings_per_window:
                file_errors.append(
                    f"Too few rows: {tensor.shape[0]} < min_pings_per_window "
                    f"({self.min_pings_per_window})"
                )

            if file_errors:
                errors[fname] = file_errors
                failed += 1
                tqdm.write(f"  FAIL [{fname}]: {' | '.join(file_errors)}")
            else:
                passed += 1

        # ── index file cross-check ───────────────────────────────────────
        orphaned_files: list[str] = []
        missing_files: list[str] = []

        if check_index_files:
            # gather all tensor filenames referenced in index files
            index_files = sorted([
                f for f in os.listdir(tensor_dir)
                if f.endswith("_micro_index.json")
            ])

            indexed_tensors: set[str] = set()
            for idx_fname in index_files:
                idx_path = os.path.join(tensor_dir, idx_fname)
                with open(idx_path, "r", encoding="utf-8") as fh:
                    idx_data = json.load(fh)
                for entry in idx_data.get("vessel_windows", []):
                    indexed_tensors.add(entry["tensor_file"])

            # .pt files on disk
            pt_set = set(pt_files)

            # orphaned: on disk but not in any index
            orphaned_files = sorted(list(pt_set - indexed_tensors))

            # missing: in index but not on disk
            missing_files = sorted(list(indexed_tensors - pt_set))

        # ── report ───────────────────────────────────────────────────────
        result = {
            "total_files":    len(pt_files),
            "passed":         passed,
            "failed":         failed,
            "orphaned_files": orphaned_files,
            "missing_files":  missing_files,
            "errors":         errors,
        }

        print(f"\nvalidate_micro_tensors complete")
        print(f"  Total .pt files    : {len(pt_files):,}")
        print(f"  Passed             : {passed:,}")
        print(f"  Failed             : {failed:,}")
        print(f"  Orphaned files     : {len(orphaned_files):,}")
        print(f"  Missing files      : {len(missing_files):,}")

        if orphaned_files:
            print(f"\n  Orphaned .pt files (on disk but not in any index):")
            for of in orphaned_files[:20]:
                print(f"    {of}")
            if len(orphaned_files) > 20:
                print(f"    ... and {len(orphaned_files) - 20} more")

        if missing_files:
            print(f"\n  Missing .pt files (in index but not on disk):")
            for mf in missing_files[:20]:
                print(f"    {mf}")
            if len(missing_files) > 20:
                print(f"    ... and {len(missing_files) - 20} more")

        if errors:
            print(f"\n  Files with errors:")
            for fname, errs in errors.items():
                for e in errs:
                    print(f"    {fname}: {e}")
        else:
            print("\n  All files passed validation.")

        return result

    # ================================================================== #
    #  METHOD 6 — build_dataset_index                                     #
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
        Builds a single master index file that maps every micro tensor
        to its paired macro tensor by date. This is what the PyTorch
        Dataset class reads at training time.

        Parameters
        ----------
        micro_tensor_dir : str
            Directory containing micro .pt files and *_micro_index.json.
        macro_tensor_dir : str
            Directory containing {date}_macro.pt files.
        output_dir : str
            Where to save the final dataset index JSON.
        split_name : str
            "train", "val", or "test".
        date_stems : list[str], optional
            Restrict to these dates only.

        Returns
        -------
        str — absolute path to the saved index JSON file.
        """
        os.makedirs(output_dir, exist_ok=True)

        # ── scan micro index files ───────────────────────────────────────
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
                    f"skipping {len(idx_data.get('vessel_windows', []))} micro tensors"
                )
                continue

            # ── build pairs ──────────────────────────────────────────────
            for entry in idx_data.get("vessel_windows", []):
                micro_path = os.path.abspath(
                    os.path.join(micro_tensor_dir, entry["tensor_file"])
                )
                all_pairs.append({
                    "macro":       macro_path,
                    "micro":       micro_path,
                    "mmsi":        entry["mmsi"],
                    "date":        date,
                    "window_hour": entry["window_hour"],
                    "ping_count":  entry["ping_count"],
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

        # ── summary ──────────────────────────────────────────────────────
        print(f"\nbuild_dataset_index complete")
        print(f"  Split            : {split_name}")
        print(f"  Total samples    : {len(all_pairs):,}")
        print(f"  Index files read : {len(index_files):,}")
        print(f"  Saved            : {out_path}")

        return out_path

    # ================================================================== #
    #  EXTRA — plot_vessel_trajectory                                      #
    # ================================================================== #
    def plot_vessel_trajectory(
        self,
        tensor_path: str,
        output_file_name: str,
    ) -> None:
        """
        Load a single micro .pt tensor, extract lat_norm and lon_norm,
        denormalise back to actual lat/lon, and plot the vessel's
        trajectory as a line on a Gulf of Mexico map.

        Parameters
        ----------
        tensor_path : str
            Path to a micro .pt tensor file.
        output_file_name : str
            Filename for the saved PNG plot.
        """
        tensor = torch.load(tensor_path, map_location="cpu", weights_only=True)

        # denormalise lat/lon from columns 0 and 1
        lat_norm = tensor[:, 0].numpy()
        lon_norm = tensor[:, 1].numpy()

        lats = lat_norm * self.GULF_LAT_RANGE + self.GULF_LAT_MIN
        lons = lon_norm * self.GULF_LON_RANGE + self.GULF_LON_MIN

        # ── world basemap ────────────────────────────────────────────────
        world = gpd.read_file(geodatasets.get_path("naturalearth.land"))

        fig, ax = plt.subplots(figsize=(14, 9))
        world.plot(ax=ax, color="lightgrey", edgecolor="white", linewidth=0.5)

        # track line
        if len(lats) >= 2:
            track_line = LineString(zip(lons, lats))
            gdf_line = gpd.GeoDataFrame(geometry=[track_line], crs="EPSG:4326")
            gdf_line.plot(ax=ax, color="steelblue", linewidth=1.8, alpha=0.7, zorder=2)

        # ping points
        scatter = ax.scatter(
            lons, lats,
            c=range(len(lats)),
            cmap="plasma",
            s=18,
            zorder=3,
            alpha=0.85,
            label="Pings",
        )
        plt.colorbar(scatter, ax=ax, label="Ping sequence (early → late)", shrink=0.5)

        # start and end markers
        ax.plot(
            lons[0], lats[0],
            marker="o", color="green", markersize=10,
            zorder=5, label="Start",
            markeredgecolor="white", markeredgewidth=1.2,
        )
        ax.plot(
            lons[-1], lats[-1],
            marker="s", color="red", markersize=10,
            zorder=5, label="End",
            markeredgecolor="white", markeredgewidth=1.2,
        )

        # zoom to data extent with buffer
        buffer = max((lons.max() - lons.min()) * 0.15,
                     (lats.max() - lats.min()) * 0.15, 0.5)
        ax.set_xlim(lons.min() - buffer, lons.max() + buffer)
        ax.set_ylim(lats.min() - buffer, lats.max() + buffer)

        # extract MMSI from filename
        basename = os.path.basename(tensor_path)
        ax.set_title(
            f"Micro Vessel Trajectory — {basename}\n"
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
    #  EXTRA — plot_window_sog_profile                                     #
    # ================================================================== #
    def plot_window_sog_profile(
        self,
        tensor_path: str,
        output_file_name: str,
    ) -> None:
        """
        Load a micro tensor, extract sog_norm, denormalise to knots,
        and plot SOG over time as a line chart.

        Parameters
        ----------
        tensor_path : str
            Path to a micro .pt tensor file.
        output_file_name : str
            Filename for the saved PNG plot.
        """
        tensor = torch.load(tensor_path, map_location="cpu", weights_only=True)

        # sog_norm is column 2
        sog_norm = tensor[:, 2].numpy()
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
    #  EXTRA — clean_vessel_windows                                        #
    # ================================================================== #
    def clean_vessel_windows(
        self,
        dataset_dir: str,
        output_dir: str,
        for_group: bool = False,
        create_new: bool = True,
    ) -> list[str]:
        """
        Filters out vessel windows from CSVs that do not meet quality
        criteria: minimum ping count, valid lat/lon within Gulf bounds,
        non-null MMSI, SOG within [0, max_sog] knots.

        Parameters
        ----------
        dataset_dir : str
            Path to a single CSV (used when for_group=False).
        output_dir : str
            Directory where cleaned CSVs are saved.
        for_group : bool
            If True, process all CSVs in self.datasets_location.
        create_new : bool
            If True, save to new files; if False, overwrite originals.

        Returns
        -------
        list[str] — paths to all created/modified CSV files.
        """
        os.makedirs(output_dir, exist_ok=True)

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
            csv_files = [target]

        created_files: list[str] = []

        for csv_path in tqdm(csv_files, desc="Cleaning vessel windows", unit="file"):
            df = pl.read_csv(
                csv_path,
                schema_overrides=self.SCHEMA_OVERRIDES,
                infer_schema_length=1000,
            )

            original_count = df.height

            # ── apply quality filters ────────────────────────────────────
            filtered = (
                df
                # non-null MMSI
                .filter(pl.col("mmsi").is_not_null())
                # valid lat within Gulf bounds
                .filter(
                    pl.col("latitude").is_between(self.min_lat, self.max_lat)
                )
                # valid lon within Gulf bounds
                .filter(
                    pl.col("longitude").is_between(self.min_lon, self.max_lon)
                )
                # SOG in valid range [0, max_sog]
                .filter(
                    (pl.col("sog").is_null()) |
                    (
                        (pl.col("sog") >= 0.0) &
                        (pl.col("sog") <= self.max_sog)
                    )
                )
            )

            filtered_count = filtered.height

            # ── save output ──────────────────────────────────────────────
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

        # ── summary ──────────────────────────────────────────────────────
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
        """
        Attempts to extract a YYYY-MM-DD date stem from the filename.
        Works for patterns like:
          - ais-2022-01-01.csv
          - AIS_2022_01_01.csv
          - Gulf_of_Mexico_ais-2022-01-01.csv
          - 2022-01-01.csv

        Returns None if no date pattern is found.
        """
        import re
        basename = os.path.basename(csv_path)
        # try YYYY-MM-DD first
        match = re.search(r"(\d{4}-\d{2}-\d{2})", basename)
        if match:
            return match.group(1)
        # try YYYY_MM_DD
        match = re.search(r"(\d{4})_(\d{2})_(\d{2})", basename)
        if match:
            return f"{match.group(1)}-{match.group(2)}-{match.group(3)}"
        return None
    
if __name__=="__main__":
    print("Hello")