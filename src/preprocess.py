import os
from typing import Dict, Tuple

import pandas as pd


def load_data(data_dir: str = "data") -> Dict[str, pd.DataFrame]:
    """
    Load dataset CSVs from data_dir.

    Expects the following files to exist inside data_dir:
      - train.csv, test.csv, dev.csv
      - level_seq.csv, level_meta.csv

    Returns a dictionary of DataFrames keyed by file stem.
    """
    file_names = [
        "train.csv",
        "test.csv",
        "dev.csv",
        "level_seq.csv",
        "level_meta.csv",
    ]

    data_frames: Dict[str, pd.DataFrame] = {}
    for file_name in file_names:
        file_path = os.path.join(data_dir, file_name)
        if not os.path.exists(file_path):
            raise FileNotFoundError(
                f"Expected file not found: {file_path}. Please place the Kaggle CSVs under '{data_dir}/'."
            )
        key = file_name.replace(".csv", "")
        # Auto-detect delimiter to handle comma or tab-separated files
        df = pd.read_csv(file_path, sep=None, engine="python")
        # Drop stray unnamed index columns
        df = df.loc[:, [c for c in df.columns if not str(c).startswith("Unnamed")]]
        data_frames[key] = df

    return data_frames


def _aggregate_level_features(level_seq: pd.DataFrame, level_meta: pd.DataFrame) -> pd.DataFrame:
    """
    Join level_seq with level_meta and aggregate per player_id.
    Produces features like levels_attempted, levels_cleared, avg_level_difficulty, last_level_index.
    """
    # Best-effort joins; tolerate missing columns by filling if absent
    seq = level_seq.copy()
    meta = level_meta.copy()

    # Normalize expected columns
    if "player_id" not in seq.columns:
        # try common alternatives
        for alt in ["user_id", "uid", "player", "ser_id"]:
            if alt in seq.columns:
                seq = seq.rename(columns={alt: "player_id"})
                break
    if "level_id" not in seq.columns:
        for alt in ["level", "stage_id", "stage"]:
            if alt in seq.columns:
                seq = seq.rename(columns={alt: "level_id"})
                break
    if "timestamp" not in seq.columns:
        for alt in ["event_time", "ts", "time"]:
            if alt in seq.columns:
                seq = seq.rename(columns={alt: "timestamp"})
                break

    if "level_id" not in meta.columns:
        for alt in ["level", "stage_id", "stage"]:
            if alt in meta.columns:
                meta = meta.rename(columns={alt: "level_id"})
                break

    # Merge level metadata if possible
    if "level_id" in seq.columns and "level_id" in meta.columns:
        seq_meta = seq.merge(meta, on="level_id", how="left")
    else:
        seq_meta = seq

    # Create helper columns
    if "cleared" not in seq_meta.columns:
        # infer cleared if result/status exists
        cleared = None
        for c in ["result", "status", "pass", "is_cleared"]:
            if c in seq_meta.columns:
                cleared = (seq_meta[c].astype(str).str.lower().isin(["pass", "cleared", "true", "1"]))
                break
        if cleared is None:
            cleared = pd.Series(False, index=seq_meta.index)
        seq_meta["cleared"] = cleared.astype(int)

    # Aggregate per player
    group_cols = [c for c in ["player_id"] if c in seq_meta.columns]
    if not group_cols:
        return pd.DataFrame()

    agg_dict = {
        "level_id": "nunique" if "level_id" in seq_meta.columns else "size",
        "cleared": "sum",
    }

    # difficulty-like column if present
    difficulty_col = None
    for c in ["difficulty", "level_difficulty", "stars", "rating", "f_avg_passrate"]:
        if c in seq_meta.columns:
            difficulty_col = c
            break
    if difficulty_col:
        agg_dict[difficulty_col] = "mean"

    # timestamp recency
    if "timestamp" in seq_meta.columns:
        try:
            seq_meta["timestamp"] = pd.to_datetime(seq_meta["timestamp"], errors="coerce")
            agg_df = seq_meta.groupby(group_cols).agg({
                **agg_dict,
                "timestamp": ["min", "max", "count"],
            })
            agg_df.columns = ["_".join(col).rstrip("_") for col in agg_df.columns.values]
        except Exception:
            agg_df = seq_meta.groupby(group_cols).agg(agg_dict)
            agg_df = agg_df.rename(columns={"level_id": "levels_attempted", "cleared": "levels_cleared"})
    else:
        agg_df = seq_meta.groupby(group_cols).agg(agg_dict)

    # Rename columns to stable feature names
    rename_map = {}
    if "level_id_nunique" in agg_df.columns:
        rename_map["level_id_nunique"] = "levels_attempted"
    if "cleared_sum" in agg_df.columns:
        rename_map["cleared_sum"] = "levels_cleared"
    if difficulty_col and f"{difficulty_col}_mean" in agg_df.columns:
        rename_map[f"{difficulty_col}_mean"] = "avg_level_difficulty"
    if "timestamp_min" in agg_df.columns:
        rename_map["timestamp_min"] = "first_play_ts"
    if "timestamp_max" in agg_df.columns:
        rename_map["timestamp_max"] = "last_play_ts"
    if "timestamp_count" in agg_df.columns:
        rename_map["timestamp_count"] = "play_events_count"

    agg_df = agg_df.rename(columns=rename_map)

    # Derived metrics
    if set(["levels_attempted", "levels_cleared"]).issubset(agg_df.columns):
        agg_df["level_clear_rate"] = (
            (agg_df["levels_cleared"].astype(float))
            .div(agg_df["levels_attempted"].replace(0, pd.NA))
            .fillna(0.0)
        )

    return agg_df.reset_index()


def create_features(dfs: Dict[str, pd.DataFrame]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Create feature tables for train/dev/test by joining player-level files with
    aggregated level progression features.

    Returns: (train_df, dev_df, test_df)
    """
    required_keys = ["train", "dev", "test", "level_seq", "level_meta"]
    for k in required_keys:
        if k not in dfs:
            raise KeyError(f"Missing key in loaded data: {k}")

    train = dfs["train"].copy()
    dev = dfs["dev"].copy()
    test = dfs["test"].copy()
    level_seq = dfs["level_seq"].copy()
    level_meta = dfs["level_meta"].copy()

    # Normalize player_id column across splits
    def normalize_player_id(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        if "player_id" not in out.columns:
            for alt in ["user_id", "uid", "player", "ser_id"]:
                if alt in out.columns:
                    out.rename(columns={alt: "player_id"}, inplace=True)
                    break
        return out

    train = normalize_player_id(train)
    dev = normalize_player_id(dev)
    test = normalize_player_id(test)

    # Basic activity features if present
    def basic_activity_features(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        # session length
        if "avg_session_length" not in out.columns:
            if set(["session_length", "player_id"]).issubset(out.columns):
                agg = out.groupby("player_id")["session_length"].mean().rename("avg_session_length")
                out = out.merge(agg, on="player_id", how="left")
            else:
                out["avg_session_length"] = pd.NA
        # purchase count
        if "purchase_count" not in out.columns:
            purchase_cols = [c for c in out.columns if c.lower().startswith("purchase")]
            if purchase_cols:
                out["purchase_count"] = out[purchase_cols].sum(axis=1, numeric_only=True)
            else:
                out["purchase_count"] = 0
        # recency if last_login exists
        if "last_login" in out.columns:
            try:
                out["last_login"] = pd.to_datetime(out["last_login"], errors="coerce")
                max_date = out["last_login"].max()
                out["recency_days"] = (max_date - out["last_login"]).dt.days
            except Exception:
                out["recency_days"] = pd.NA
        elif "last_active" in out.columns:
            try:
                out["last_active"] = pd.to_datetime(out["last_active"], errors="coerce")
                max_date = out["last_active"].max()
                out["recency_days"] = (max_date - out["last_active"]).dt.days
            except Exception:
                out["recency_days"] = pd.NA
        else:
            out["recency_days"] = pd.NA
        return out

    train = basic_activity_features(train)
    dev = basic_activity_features(dev)
    test = basic_activity_features(test)

    # Aggregate level features and join (only when player_id exists)
    level_agg = _aggregate_level_features(level_seq, level_meta)
    if not level_agg.empty and "player_id" in level_agg.columns:
        def safe_merge(df: pd.DataFrame) -> pd.DataFrame:
            if "player_id" in df.columns:
                return df.merge(level_agg, on="player_id", how="left")
            return df
        train = safe_merge(train)
        dev = safe_merge(dev)
        test = safe_merge(test)

    # Fill simple NaNs for model readiness
    for df in [train, dev, test]:
        for col in df.select_dtypes(include=["number"]).columns:
            df[col] = df[col].fillna(0)

    return train, dev, test


__all__ = [
    "load_data",
    "create_features",
]


