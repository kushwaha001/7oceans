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

    # ── unchanged ─────────────────────────────────────────────────────────
    def count_rows(
        self,
        dataset_dir: Optional[str] = None,
        show_group_count: bool = False,
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

    # ── unchanged ─────────────────────────────────────────────────────────
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

            frames = []
            for fname in tqdm(csv_files, desc="Scanning MMSI values", unit="file"):
                fpath = os.path.join(self.datasets_location, fname)
                unique_in_file = (
                    pl.scan_csv(fpath, infer_schema_length=1000)
                    .select(pl.col("mmsi").cast(pl.Int64))
                    .unique()
                    .collect()
                )
                frames.append(unique_in_file)
                tqdm.write(f"  {fname}: {unique_in_file.height:,} unique MMSIs")

            result_df = (
                pl.concat(frames)
                .unique()
                .sort("mmsi")
            )
            source_label = f"{len(csv_files)} files from datasets_location"

        else:
            target = dataset_dir if dataset_dir is not None else self.dataset_dir
            result_df = (
                pl.scan_csv(target, infer_schema_length=1000)
                .select(pl.col("mmsi").cast(pl.Int64))
                .unique()
                .sort("mmsi")
                .collect()
            )
            source_label = os.path.basename(target)

        mmsi_list = result_df["mmsi"].to_list()
        total = len(mmsi_list)

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

        csv_path = os.path.join(self.analysis_result_dir, f"{output_file_name}.csv")
        result_df.write_csv(csv_path)

        print(f"\nUnique MMSI count : {total:,}")
        print(f"Saved JSON        → {json_path}")
        print(f"Saved CSV         → {csv_path}")

        return mmsi_list, csv_path, json_path

    # ── unchanged ─────────────────────────────────────────────────────────
    def extract_vessels_by_mmsi(
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
            .with_row_index(name="RowNumber", offset=1)
            .filter(pl.col("mmsi").is_in(mmsi_list))
            .collect()
        )

        if result_df.is_empty():
            print("No rows found for the given MMSI(s).")
            return result_df

        first_occurrence = (
            result_df
            .group_by("mmsi")
            .agg([
                pl.len().alias("RowCount"),
                pl.col("RowNumber").min().alias("FirstOccurrenceRow")
            ])
            .sort("RowCount", descending=True)
        )

        os.makedirs(self.analysis_result_dir, exist_ok=True)
        out_path = os.path.join(self.analysis_result_dir, output_file_name)
        result_df.write_csv(out_path)

        print(f"\nExtracted {result_df.height:,} rows total")
        print(f"{'mmsi':<15} {'First Row':>10} {'Row Count':>10}")
        print("-" * 38)
        for row in first_occurrence.iter_rows(named=True):
            print(f"  {row['mmsi']:<13} {row['FirstOccurrenceRow']:>10,} {row['RowCount']:>10,}")

        print(f"\nSaved → {out_path}")

        return result_df

    # ── unchanged ─────────────────────────────────────────────────────────
    def export_vessels_by_mmsi(
        self,
        mmsi_list: List[int],
        output_name: str,
        dataset_dir: Optional[str] = None,
        for_group: bool = False,
    ) -> dict:

        if not mmsi_list:
            print("mmsi_list is empty, nothing to export.")
            return {}

        out_dir = os.path.join(self.analysis_result_dir, "vessels")
        os.makedirs(out_dir, exist_ok=True)

        mmsi_filter = pl.Series("mmsi", mmsi_list, dtype=pl.Int64)

        buckets: dict[int, list[pl.DataFrame]] = {m: [] for m in mmsi_list}

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
                    .filter(pl.col("mmsi").cast(pl.Int64).is_in(mmsi_filter))
                    .collect()
                )
                if filtered.is_empty():
                    continue

                for mmsi, group_df in filtered.group_by("mmsi"):
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

        summary = {}
        not_found = []

        print(f"\nWriting CSVs to: {out_dir}")
        for mmsi in tqdm(mmsi_list, desc="Writing CSVs", unit="vessel"):
            frames = buckets[mmsi]

            if not frames:
                not_found.append(mmsi)
                summary[mmsi] = {"row_count": 0, "path": None}
                continue

            vessel_df = (
                pl.concat(frames)
                .sort("base_date_time")
            )

            out_path = os.path.join(out_dir, f"{output_name}_{mmsi}.csv")
            vessel_df.write_csv(out_path)

            summary[mmsi] = {
                "row_count": vessel_df.height,
                "path": out_path
            }

        if not_found:
            print(f"\nWarning: {len(not_found)} MMSI(s) not found in any file:")
            for m in not_found:
                print(f"  MMSI {m} — no rows found, no file written")

        found = [m for m in mmsi_list if summary[m]["row_count"] > 0]
        total_rows = sum(summary[m]["row_count"] for m in found)

        print(f"\nDone.")
        print(f"  Vessels written : {len(found):>6,}")
        print(f"  Vessels missing : {len(not_found):>6,}")
        print(f"  Total rows saved: {total_rows:>6,}")

        return summary

    # ── unchanged ─────────────────────────────────────────────────────────
    def plot_vessel_path(
        self,
        dataset_dir: Optional[str] = None,
        arrow_every_n: int = 5,
        output_file_name: Optional[str] = None,
    ) -> None:

        target = dataset_dir if dataset_dir is not None else self.dataset_dir

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

        mmsi        = df["mmsi"][0]
        vessel_name = df["vessel_name"][0] if "vessel_name" in df.columns else "Unknown"
        vessel_type = df["vessel_type"][0] if "vessel_type" in df.columns else "Unknown"
        total_pings = df.height
        time_start  = df["base_date_time"][0]
        time_end    = df["base_date_time"][-1]

        pdf = df.to_pandas()

        gdf_points = gpd.GeoDataFrame(
            pdf,
            geometry=gpd.points_from_xy(pdf["longitude"], pdf["latitude"]),
            crs="EPSG:4326"
        )

        if len(gdf_points) >= 2:
            track_line = LineString(zip(pdf["longitude"], pdf["latitude"]))
            gdf_line = gpd.GeoDataFrame(geometry=[track_line], crs="EPSG:4326")
        else:
            gdf_line = None

        world = gpd.read_file(geodatasets.get_path("naturalearth.land"))

        fig, ax = plt.subplots(figsize=(14, 9))
        world.plot(ax=ax, color="lightgrey", edgecolor="white", linewidth=0.5)

        if gdf_line is not None:
            gdf_line.plot(ax=ax, color="steelblue", linewidth=1.8, alpha=0.7, zorder=2)

        scatter = ax.scatter(
            pdf["longitude"], pdf["latitude"],
            c=range(len(pdf)), cmap="plasma", s=18, zorder=3, alpha=0.85, label="Pings"
        )
        plt.colorbar(scatter, ax=ax, label="Ping sequence (early → late)", shrink=0.5)

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

        arrow_indices = range(0, len(pdf) - 1, arrow_every_n)
        for i in arrow_indices:
            lon = pdf["longitude"].iloc[i]
            lat = pdf["latitude"].iloc[i]
            cog = pdf["cog"].iloc[i]
            if cog >= 360 or cog < 0:
                continue
            angle_rad = math.radians(90 - cog)
            arrow_len = 0.15
            dx = arrow_len * math.cos(angle_rad)
            dy = arrow_len * math.sin(angle_rad)
            ax.annotate(
                "", xy=(lon + dx, lat + dy), xytext=(lon, lat),
                arrowprops=dict(arrowstyle="->", color="darkorange", lw=1.4),
                zorder=4
            )

        minx, miny, maxx, maxy = gdf_points.total_bounds
        buffer = max((maxx - minx) * 0.15, (maxy - miny) * 0.15, 1.0)
        ax.set_xlim(minx - buffer, maxx + buffer)
        ax.set_ylim(miny - buffer, maxy + buffer)

        ax.set_title(
            f"Vessel Path — {vessel_name} (MMSI: {mmsi})\n"
            f"Type: {vessel_type}  |  Pings: {total_pings:,}  |  "
            f"{time_start} → {time_end}",
            fontsize=13, pad=14
        )
        ax.set_xlabel("Longitude", fontsize=11)
        ax.set_ylabel("Latitude", fontsize=11)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(loc="lower left", fontsize=10)

        plt.tight_layout()

        plots_dir = os.path.join(self.analysis_result_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)

        fname = output_file_name if output_file_name else f"vessel_path_{mmsi}.png"
        out_path = os.path.join(plots_dir, fname)
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved → {out_path}")
        plt.show()
        plt.close()

    # ── unchanged ─────────────────────────────────────────────────────────
    def plot_vessel_path_point(
        self,
        dataset_dir: Optional[str] = None,
        arrow_every_n: int = 5,
        output_file_name: Optional[str] = None,
    ) -> None:

        target = dataset_dir if dataset_dir is not None else self.dataset_dir

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

        mmsi        = df["mmsi"][0]
        vessel_name = df["vessel_name"][0] if "vessel_name" in df.columns else "Unknown"
        vessel_type = df["vessel_type"][0] if "vessel_type" in df.columns else "Unknown"
        total_pings = df.height
        time_start  = df["base_date_time"][0]
        time_end    = df["base_date_time"][-1]

        pdf = df.to_pandas()

        gdf_points = gpd.GeoDataFrame(
            pdf,
            geometry=gpd.points_from_xy(pdf["longitude"], pdf["latitude"]),
            crs="EPSG:4326"
        )

        world = gpd.read_file(geodatasets.get_path("naturalearth.land"))

        fig, ax = plt.subplots(figsize=(14, 9))
        world.plot(ax=ax, color="lightgrey", edgecolor="white", linewidth=0.5)

        scatter = ax.scatter(
            pdf["longitude"], pdf["latitude"],
            c=range(len(pdf)), cmap="plasma", s=22, zorder=3, alpha=0.85, label="Pings"
        )
        plt.colorbar(scatter, ax=ax, label="Ping sequence (early → late)", shrink=0.5)

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

        arrow_indices = range(0, len(pdf) - 1, arrow_every_n)
        for i in arrow_indices:
            lon = pdf["longitude"].iloc[i]
            lat = pdf["latitude"].iloc[i]
            cog = pdf["cog"].iloc[i]
            if cog >= 360 or cog < 0:
                continue
            angle_rad = math.radians(90 - cog)
            arrow_len = 0.15
            dx = arrow_len * math.cos(angle_rad)
            dy = arrow_len * math.sin(angle_rad)
            ax.annotate(
                "", xy=(lon + dx, lat + dy), xytext=(lon, lat),
                arrowprops=dict(arrowstyle="->", color="darkorange", lw=1.4),
                zorder=4
            )

        minx, miny, maxx, maxy = gdf_points.total_bounds
        buffer = max((maxx - minx) * 0.15, (maxy - miny) * 0.15, 1.0)
        ax.set_xlim(minx - buffer, maxx + buffer)
        ax.set_ylim(miny - buffer, maxy + buffer)

        ax.set_title(
            f"Vessel Detections — {vessel_name} (MMSI: {mmsi})\n"
            f"Type: {vessel_type}  |  Pings: {total_pings:,}  |  "
            f"{time_start} → {time_end}",
            fontsize=13, pad=14
        )
        ax.set_xlabel("longitude", fontsize=11)
        ax.set_ylabel("latitude", fontsize=11)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(loc="lower left", fontsize=10)

        plt.tight_layout()

        plots_dir = os.path.join(self.analysis_result_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)

        fname = output_file_name if output_file_name else f"vessel_detections_{mmsi}.png"
        out_path = os.path.join(plots_dir, fname)
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved → {out_path}")
        plt.show()
        plt.close()

    # ── unchanged ─────────────────────────────────────────────────────────
    def add_delta_time(
        self,
        time_unit: str = "minutes",
        create_new: bool = True,
        output_file_name: Optional[str] = None,
        dataset_dir: Optional[str] = None,
    ) -> pl.DataFrame:

        valid_units = {"seconds", "minutes", "hours"}
        if time_unit not in valid_units:
            raise ValueError(
                f"Invalid time_unit '{time_unit}'. "
                f"Must be one of: {valid_units}"
            )

        target = dataset_dir if dataset_dir is not None else self.dataset_dir

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

        ns_per_unit = {
            "seconds": 1_000_000_000,
            "minutes": 60_000_000_000,
            "hours":   3_600_000_000_000,
        }

        df = df.with_columns(
            (
                pl.col("base_date_time")
                .diff()
                .dt.total_nanoseconds()
                .fill_null(0)
                / ns_per_unit[time_unit]
            )
            .round(4)
            .alias(f"DeltaTime_{time_unit}")
        )

        delta_col = f"delta_time_{time_unit}"
        delta     = df[delta_col].filter(df[delta_col] > 0)
        print(f"\ndelta_time summary ({time_unit}):")
        print(f"  Rows           : {df.height:,}")
        print(f"  Min delta       : {delta.min():.4f}")
        print(f"  Max delta       : {delta.max():.4f}")
        print(f"  Mean delta      : {delta.mean():.4f}")
        print(f"  Median delta    : {delta.median():.4f}")

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
            df.write_csv(target)
            print(f"  Saved (in-place)→ {target}")

        return df

    # ── unchanged ─────────────────────────────────────────────────────────
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
            a = (
                np.sin(dlat / 2) ** 2
                + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
            )
            return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))

        def bearing_vectorised(lat1, lon1, lat2, lon2):
            lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
            dlon = lon2 - lon1
            x = np.sin(dlon) * np.cos(lat2)
            y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
            bearing = np.degrees(np.arctan2(x, y))
            return (bearing + 360) % 360

        lat_curr = pdf["latitude"].values
        lon_curr = pdf["longitude"].values
        lat_prev = np.roll(lat_curr, 1)    # type:ignore
        lon_prev = np.roll(lon_curr, 1)    # type:ignore

        dist_km    = haversine_vectorised(lat_prev, lon_prev, lat_curr, lon_curr)
        dist_km[0] = 0.0

        if distance_unit == "nm":
            dist_out = dist_km / 1.852
        elif distance_unit == "km":
            dist_out = dist_km
        else:
            dist_out = dist_km * 1000.0

        avg_bearing    = bearing_vectorised(lat_prev, lon_prev, lat_curr, lon_curr)
        avg_bearing[0] = 0.0

        dt_seconds    = pdf["base_date_time"].diff().dt.total_seconds().fillna(0).values
        dt_hours      = dt_seconds / 3600.0  # type:ignore

        dist_nm = dist_km / 1.852
        with np.errstate(divide="ignore", invalid="ignore"):
            avg_sog = np.where(dt_hours > 0, dist_nm / dt_hours, 0.0)
        avg_sog[0] = 0.0

        cog_values = pdf["cog"].values.astype(float)
        cog_valid  = (cog_values >= 0) & (cog_values < 360.0) # type:ignore
        avg_bearing_vs_cog = np.where(
            cog_valid,
            ((avg_bearing - cog_values + 180) % 360) - 180,
            np.nan
        )
        avg_bearing_vs_cog[0] = np.nan

        cumulative = np.cumsum(dist_out)

        dist_col = f"DeltaDistance_{distance_unit}"
        cum_col  = f"CumulativeDistance_{distance_unit}"

        df = df.with_columns([
            pl.Series(dist_col,                dist_out.round(6).tolist()),
            pl.Series("AvgBearing_deg",        avg_bearing.round(2).tolist()),
            pl.Series("AvgSOG_knots",          avg_sog.round(4).tolist()),
            pl.Series("AvgBearingVsCOG_diff",  avg_bearing_vs_cog.round(2).tolist()),
            pl.Series(cum_col,                 cumulative.round(6).tolist()),
        ])

        dist_valid = df[dist_col].slice(1)
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

    # ══════════════════════════════════════════════════════════════════════
    #  OPTIMIZED: filter_by_bbox — single read per file instead of double
    # ══════════════════════════════════════════════════════════════════════
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

        bbox_filter = (
            pl.col("latitude").is_between(min_lat, max_lat) &
            pl.col("longitude").is_between(min_lon, max_lon)
        )

        os.makedirs(self.analysis_result_dir, exist_ok=True)

        def _build_output_name(fpath: str) -> str:
            original = os.path.basename(fpath)
            return f"{location_name}_{original}"

        # ── for_group = True ──────────────────────────────────────────────
        if for_group:
            csv_files = sorted([
                f for f in os.listdir(self.datasets_location)
                if f.endswith(".csv")
            ])

            if not csv_files:
                print("No CSV files found in datasets_location.")
                return pl.DataFrame()

            all_frames = []
            total_in   = 0
            total_out  = 0

            for fname in tqdm(csv_files, desc=f"Filtering [{location_name}]", unit="file"):
                fpath = os.path.join(self.datasets_location, fname)

                # ── OPTIMIZATION: single scan instead of two ─────────────
                #    Old code scanned once for filter, once for row count.
                #    Now we use scan_csv once and collect the full frame,
                #    then filter in-memory.
                df_full  = pl.scan_csv(fpath, infer_schema_length=1000).collect()
                rows_in  = df_full.height
                filtered = df_full.filter(bbox_filter)
                rows_out = filtered.height
                del df_full                              # free memory immediately

                total_in  += rows_in
                total_out += rows_out

                tqdm.write(
                    f"  {fname}: {rows_in:>10,} rows → "
                    f"{rows_out:>8,} inside bbox  "
                    f"({rows_out / rows_in * 100:.2f}%)" if rows_in > 0 else
                    f"  {fname}: 0 rows"
                )

                if filtered.is_empty():
                    continue

                all_frames.append(filtered)

                if create_new:
                    out_name = _build_output_name(fpath)
                    out_path = os.path.join(self.analysis_result_dir, out_name)
                    filtered.write_csv(out_path)
                else:
                    filtered.write_csv(fpath)

            result_df = pl.concat(all_frames) if all_frames else pl.DataFrame()

            print(f"\nBbox filter complete — {location_name}")
            print(f"  Files processed   : {len(csv_files):,}")
            print(f"  Total rows in     : {total_in:,}")
            print(f"  Total rows out    : {total_out:,}")
            if total_in > 0:
                print(f"  Retention rate    : {total_out / total_in * 100:.4f}%")
            print(f"  Unique MMSI found : {result_df['mmsi'].n_unique():,}" if not result_df.is_empty() else "  No rows found.")
            if create_new:
                print(f"  Output dir        : {self.analysis_result_dir}")
            return result_df

        # ── for_group = False ─────────────────────────────────────────────
        else:
            target   = dataset_dir if dataset_dir is not None else self.dataset_dir
            original = os.path.basename(target)

            # ── OPTIMIZATION: single read ────────────────────────────────
            df_full  = pl.scan_csv(target, infer_schema_length=1000).collect()
            total_in = df_full.height
            filtered = df_full.filter(bbox_filter)
            rows_out = filtered.height
            del df_full

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
            if total_in > 0:
                print(f"  Retention rate    : {rows_out / total_in * 100:.4f}%")
            print(f"  Unique MMSI found : {filtered['mmsi'].n_unique():,}")

            return filtered

    # ══════════════════════════════════════════════════════════════════════
    #  OPTIMIZED: filter_by_bbox_give_path — single read per file
    # ══════════════════════════════════════════════════════════════════════
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
    ) -> list[str]:
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

        bbox_filter = (
            pl.col("latitude").is_between(min_lat, max_lat) &
            pl.col("longitude").is_between(min_lon, max_lon)
        )

        os.makedirs(self.analysis_result_dir, exist_ok=True)

        created_files: list[str] = []

        def _build_output_name(fpath: str) -> str:
            original = os.path.basename(fpath)
            return f"{location_name}_{original}"

        # ── for_group = True ──────────────────────────────────────────────
        if for_group:
            csv_files = sorted([
                f for f in os.listdir(self.datasets_location)
                if f.endswith(".csv")
            ])

            if not csv_files:
                print("No CSV files found in datasets_location.")
                return []

            total_in       = 0
            total_out      = 0
            all_mmsi_counts = []

            for fname in tqdm(csv_files, desc=f"Filtering [{location_name}]", unit="file"):
                fpath = os.path.join(self.datasets_location, fname)

                # ── OPTIMIZATION: single read ────────────────────────────
                df_full  = pl.scan_csv(fpath, infer_schema_length=1000).collect()
                rows_in  = df_full.height
                filtered = df_full.filter(bbox_filter)
                rows_out = filtered.height
                del df_full

                total_in  += rows_in
                total_out += rows_out

                tqdm.write(
                    f"  {fname}: {rows_in:>10,} rows → "
                    f"{rows_out:>8,} inside bbox  "
                    f"({rows_out / rows_in * 100:.2f}%)" if rows_in > 0 else
                    f"  {fname}: 0 rows"
                )

                if filtered.is_empty():
                    continue

                all_mmsi_counts.append(filtered["mmsi"].n_unique())

                if create_new:
                    out_name = _build_output_name(fpath)
                    out_path = os.path.abspath(os.path.join(self.analysis_result_dir, out_name))
                    filtered.write_csv(out_path)
                    created_files.append(out_path)
                else:
                    abs_fpath = os.path.abspath(fpath)
                    filtered.write_csv(abs_fpath)
                    created_files.append(abs_fpath)

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

        # ── for_group = False ─────────────────────────────────────────────
        else:
            target   = dataset_dir if dataset_dir is not None else self.dataset_dir
            original = os.path.basename(target)

            # ── OPTIMIZATION: single read ────────────────────────────────
            df_full  = pl.scan_csv(target, infer_schema_length=1000).collect()
            total_in = df_full.height
            filtered = df_full.filter(bbox_filter)
            rows_out = filtered.height
            del df_full

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
            if total_in > 0:
                print(f"  Retention rate    : {rows_out / total_in * 100:.4f}%")
            print(f"  Unique MMSI found : {filtered['mmsi'].n_unique():,}")

            return created_files

    # ── unchanged ─────────────────────────────────────────────────────────
    VALID_MIDS = set(range(201, 776))

    def _classify_mmsi(
        self,
        mmsi: int
    ) -> str | None:
        s = str(mmsi)

        if len(s) != 9:
            return None

        first  = s[0]
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

    # ── unchanged ─────────────────────────────────────────────────────────
    def read_mmsi(
        self,
        path: str,
        group: bool = False
    ) -> list[int] | OrderedDict[str, List[int]]:
        ALL_CATEGORIES = [
            "vessel", "coast_station", "group", "sar_aircraft",
            "aids_to_navigation", "craft_associated", "handheld_vhf",
            "sart_mob_epirb"
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
        raw: List[int] = lf.collect()["mmsi"].to_list()

        if not group:
            return [mmsi for mmsi in raw if self._classify_mmsi(mmsi) is not None]

        result: OrderedDict[str, list[int]] = OrderedDict(
            (cat, []) for cat in ALL_CATEGORIES
        )

        for mmsi in raw:
            category = self._classify_mmsi(mmsi)
            if category is not None:
                result[category].append(mmsi)

        return result