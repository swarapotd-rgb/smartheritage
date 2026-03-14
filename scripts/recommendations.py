"""
Recommendation utilities for SmartHeritage.

This module focuses on two product features:
1. Hidden gem discovery
2. Best time to visit suggestions

The functions are intentionally lightweight so they can be imported
directly into the Streamlit app during integration.
"""

from __future__ import annotations

from pathlib import Path
import re
from typing import Iterable

import pandas as pd


CROWD_TIME_MAP = {
    "high": "Early morning",
    "medium": "Morning",
    "low": "Anytime",
}
HIGH_CROWD_THRESHOLD = 2_000_000
MEDIUM_CROWD_THRESHOLD = 500_000


def load_tourism_data(csv_path: str | Path) -> pd.DataFrame:
    """Load a CSV file into a DataFrame."""
    return pd.read_csv(csv_path)


def _normalize_identifier(value: str) -> str:
    """Convert labels into consistent snake_case identifiers."""
    normalized = re.sub(r"[^0-9a-zA-Z]+", "_", str(value).strip().lower())
    normalized = re.sub(r"_+", "_", normalized)
    return normalized.strip("_")


def _normalize_columns(df: pd.DataFrame) -> dict[str, str]:
    """Create a normalized name lookup for flexible column matching."""
    return {_normalize_identifier(str(column)): column for column in df.columns}


def _find_first_column(df: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    """Return the first matching column name from a list of aliases."""
    normalized = _normalize_columns(df)
    for candidate in candidates:
        key = _normalize_identifier(candidate)
        if key in normalized:
            return normalized[key]
    return None


def _coerce_numeric(series: pd.Series) -> pd.Series:
    """Convert a series to numeric while safely handling invalid values."""
    return pd.to_numeric(series, errors="coerce").fillna(0)


def _min_max_scale(series: pd.Series) -> pd.Series:
    """Scale a numeric series to 0-1 while handling constant values."""
    numeric = _coerce_numeric(series)
    minimum = float(numeric.min()) if not numeric.empty else 0.0
    maximum = float(numeric.max()) if not numeric.empty else 0.0
    if maximum - minimum <= 0:
        return pd.Series(0.5, index=numeric.index, dtype="float64")
    return (numeric - minimum) / (maximum - minimum)


def _resolve_requested_column(df: pd.DataFrame, column_name: str | None) -> str | None:
    """Match a caller-supplied column name against the normalized dataset."""
    if not column_name:
        return None
    if column_name in df.columns:
        return column_name
    return _find_first_column(df, [column_name])


def _standardize_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names and align raw/processed CSV schemas."""
    result = df.copy()
    result.columns = [_normalize_identifier(column) for column in result.columns]

    alias_map = {
        "name_of_the_monument": "monument",
        "domestic_2019_20": "domestic_2019",
        "foreign_2019_20": "foreign_2019",
        "domestic_2020_21": "domestic_2020",
        "foreign_2020_21": "foreign_2020",
        "growth_2021_21_2019_20_domestic": "domestic_growth_pct",
        "growth_2021_21_2019_20_foreign": "foreign_growth_pct",
    }
    rename_map = {
        source: target
        for source, target in alias_map.items()
        if source in result.columns and target not in result.columns
    }
    if rename_map:
        result = result.rename(columns=rename_map)

    return result


def _filter_monument_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Drop aggregate total rows so recommendations only use real monuments."""
    result = df.copy()
    name_col = _find_first_column(
        result,
        [
            "monument",
            "monument_name",
            "site_name",
            "destination",
            "place",
            "name",
        ],
    )
    if name_col:
        monument_names = result[name_col].astype(str).str.strip().str.lower()
        result = result.loc[~monument_names.isin(["total", "grand total"])].copy()

    circle_col = _find_first_column(result, ["circle", "region", "district"])
    if circle_col:
        circles = result[circle_col].astype(str).str.strip().str.lower()
        result = result.loc[~circles.isin(["total", "grand total"])].copy()

    return result.reset_index(drop=True)


def add_total_visitors_column(
    df: pd.DataFrame,
    total_visitors_col: str = "total_visitors",
) -> pd.DataFrame:
    """
    Ensure a total visitors column exists.

    The function first uses an existing total visitors column if available.
    Otherwise it tries to derive one from the 2019 domestic and foreign
    visitor columns used in the project dataset.
    """
    result = df.copy()

    existing_total = _find_first_column(
        result,
        [
            total_visitors_col,
            "total_visitors_2019",
            "visitor_count",
            "visitors",
            "total_visitor",
            "total_visitors_count",
        ],
    )
    if existing_total and existing_total != total_visitors_col:
        result[total_visitors_col] = _coerce_numeric(result[existing_total])
        return result
    if existing_total == total_visitors_col:
        result[total_visitors_col] = _coerce_numeric(result[total_visitors_col])
        return result

    domestic_col = _find_first_column(
        result,
        [
            "domestic_2019",
            "domestic_visitors",
            "domestic",
            "indian_visitors",
            "local_visitors",
        ],
    )
    foreign_col = _find_first_column(
        result,
        [
            "foreign_2019",
            "foreign_visitors",
            "foreign",
            "international_visitors",
            "tourist_visitors",
        ],
    )

    if not domestic_col and not foreign_col:
        raise ValueError(
            "Could not find a total visitor column or domestic/foreign visitor columns."
        )

    domestic_values = (
        _coerce_numeric(result[domestic_col])
        if domestic_col
        else pd.Series(0, index=result.index)
    )
    foreign_values = (
        _coerce_numeric(result[foreign_col])
        if foreign_col
        else pd.Series(0, index=result.index)
    )

    result[total_visitors_col] = domestic_values + foreign_values
    return result


def add_crowd_index_column(
    df: pd.DataFrame,
    visitor_col: str = "total_visitors",
    crowd_index_col: str = "crowd_index",
) -> pd.DataFrame:
    """Normalize visitor volume into a 0-1 crowd index."""
    result = df.copy()
    result[visitor_col] = _coerce_numeric(result[visitor_col])

    max_visitors = float(result[visitor_col].max()) if not result.empty else 0.0
    if max_visitors <= 0:
        result[crowd_index_col] = 0.0
        return result

    result[crowd_index_col] = result[visitor_col] / max_visitors
    return result


def add_crowd_level_column(
    df: pd.DataFrame,
    visitor_col: str = "total_visitors",
    crowd_level_col: str = "crowd_level",
) -> pd.DataFrame:
    """
    Add crowd level buckets aligned with the preprocessing script.

    Low: <= 500,000 visitors
    Medium: <= 2,000,000 visitors
    High: above 2,000,000 visitors
    """
    result = df.copy()
    result[visitor_col] = _coerce_numeric(result[visitor_col])

    def classify_crowd(visitor_count: float) -> str:
        if visitor_count > HIGH_CROWD_THRESHOLD:
            return "High"
        if visitor_count > MEDIUM_CROWD_THRESHOLD:
            return "Medium"
        return "Low"

    result[crowd_level_col] = result[visitor_col].apply(classify_crowd)
    return result


def best_time_to_visit(crowd_level: str) -> str:
    """Map crowd level to a user-friendly visit window."""
    if crowd_level is None:
        return "Morning"
    return CROWD_TIME_MAP.get(str(crowd_level).strip().lower(), "Morning")


def add_best_time_column(
    df: pd.DataFrame,
    crowd_level_col: str = "crowd_level",
    best_time_col: str = "best_time_to_visit",
) -> pd.DataFrame:
    """Append best-time recommendations for each row."""
    result = df.copy()
    result[best_time_col] = result[crowd_level_col].apply(best_time_to_visit)
    return result


def add_heritage_score_column(
    df: pd.DataFrame,
    heritage_score_col: str = "heritage_score",
) -> pd.DataFrame:
    """
    Build a simple cultural importance proxy from available tourism signals.

    The current dataset does not contain an explicit heritage rating, so the
    score blends:
    - normalized total visitors
    - normalized international appeal
    - normalized growth resilience
    """
    result = df.copy()
    total_visitors = _coerce_numeric(result["total_visitors"])
    foreign_2019 = _coerce_numeric(result.get("foreign_2019", pd.Series(0, index=result.index)))
    domestic_growth = _coerce_numeric(
        result.get("domestic_growth_pct", pd.Series(0, index=result.index))
    )
    foreign_growth = _coerce_numeric(
        result.get("foreign_growth_pct", pd.Series(0, index=result.index))
    )

    foreign_ratio = foreign_2019 / (total_visitors + 1)
    growth_signal = (domestic_growth + foreign_growth) / 2

    visitor_score = _min_max_scale(total_visitors)
    foreign_ratio_score = _min_max_scale(foreign_ratio)
    growth_score = _min_max_scale(growth_signal)

    result[heritage_score_col] = (
        0.55 * visitor_score + 0.30 * foreign_ratio_score + 0.15 * growth_score
    ).round(4)
    return result


def add_hidden_gem_score_column(
    df: pd.DataFrame,
    heritage_score_col: str = "heritage_score",
    crowd_index_col: str = "crowd_index",
    hidden_gem_score_col: str = "hidden_gem_score",
) -> pd.DataFrame:
    """
    Assign hidden gem score using only crowd level.

    Per the current product rule:
    - Low crowd = hidden gem
    - Medium / High crowd = not a hidden gem
    """
    result = df.copy()
    is_hidden_gem = (
        result["crowd_level"].astype(str).str.strip().str.lower() == "low"
    )
    result[hidden_gem_score_col] = is_hidden_gem.astype(float)
    return result


def add_hidden_gem_category_column(
    df: pd.DataFrame,
    hidden_gem_score_col: str = "hidden_gem_score",
    hidden_gem_category_col: str = "hidden_gem_category",
) -> pd.DataFrame:
    """
    Label hidden gems using the current low-crowd rule.
    """
    result = df.copy()
    result[hidden_gem_category_col] = "Hidden Gem"
    return result


def recommend_hidden_gems(
    df: pd.DataFrame,
    top_n: int = 10,
    name_col: str | None = None,
    visitor_col: str = "total_visitors",
    relevance_col: str | None = None,
    circle_name: str | None = None,
) -> pd.DataFrame:
    """
    Return hidden gems using the hackathon-friendly heuristic.

    Primary rule:
    - any monument with `crowd_level = Low` is a hidden gem
    """
    result = build_recommendation_dataset(df)

    resolved_name_col = _resolve_requested_column(result, name_col) or _find_first_column(
        result,
        [
            "monument",
            "monument_name",
            "site_name",
            "destination",
            "place",
            "name",
        ],
    )
    if not resolved_name_col:
        raise ValueError("Could not find a monument/site name column.")

    circle_col = _find_first_column(result, ["circle", "region", "district"])
    if circle_name and circle_col:
        result = result[
            result[circle_col].astype(str).str.strip().str.lower()
            == str(circle_name).strip().lower()
        ].copy()
    if result.empty:
        return pd.DataFrame(
            columns=[
                "monument_name",
                visitor_col,
                "circle",
                "crowd_level",
                "best_time_to_visit",
                "heritage_score",
                "hidden_gem_score",
                "hidden_gem_category",
            ]
        )

    resolved_relevance_col = _resolve_requested_column(result, relevance_col) or _find_first_column(
        result,
        [
            "historical_relevance",
            "historical_significance",
            "relevance_score",
            "heritage_score",
            "importance_score",
            "rating",
        ],
    )

    result[visitor_col] = _coerce_numeric(result[visitor_col])
    hidden_gems = result[
        result["crowd_level"].astype(str).str.strip().str.lower() == "low"
    ].copy()

    sort_columns = [visitor_col]
    ascending = [True]
    if resolved_relevance_col:
        hidden_gems[resolved_relevance_col] = _coerce_numeric(hidden_gems[resolved_relevance_col])
        sort_columns = [visitor_col]
        ascending = [True]
    hidden_gems = hidden_gems.sort_values(by=sort_columns, ascending=ascending)
    hidden_gems = add_hidden_gem_category_column(hidden_gems)

    selected_columns = [resolved_name_col, visitor_col]
    circle_col = _find_first_column(hidden_gems, ["circle", "region", "district"])
    if circle_col:
        selected_columns.append(circle_col)
    if "crowd_level" in hidden_gems.columns:
        selected_columns.append("crowd_level")
    if "best_time_to_visit" in hidden_gems.columns:
        selected_columns.append("best_time_to_visit")
    if "heritage_score" in hidden_gems.columns:
        selected_columns.append("heritage_score")
    if "hidden_gem_score" in hidden_gems.columns:
        selected_columns.append("hidden_gem_score")
    if "hidden_gem_category" in hidden_gems.columns:
        selected_columns.append("hidden_gem_category")
    if resolved_relevance_col:
        selected_columns.append(resolved_relevance_col)

    selected_columns = list(dict.fromkeys(selected_columns))
    hidden_gems = hidden_gems[selected_columns].head(top_n).reset_index(drop=True)
    hidden_gems.rename(columns={resolved_name_col: "monument_name"}, inplace=True)
    return hidden_gems


def get_monument_recommendation(
    df: pd.DataFrame,
    monument_name: str,
    name_col: str | None = None,
) -> dict[str, object]:
    """
    Return a single monument summary for the app's crowd checker card.

    Expected output keys:
    - monument_name
    - crowd_level
    - best_time_to_visit
    - total_visitors
    """
    prepared = build_recommendation_dataset(df)
    resolved_name_col = _resolve_requested_column(prepared, name_col) or _find_first_column(
        prepared,
        [
            "monument",
            "monument_name",
            "site_name",
            "destination",
            "place",
            "name",
        ],
    )
    if not resolved_name_col:
        raise ValueError("Could not find a monument/site name column.")

    monument_rows = prepared[
        prepared[resolved_name_col].astype(str).str.strip().str.lower()
        == str(monument_name).strip().lower()
    ]

    if monument_rows.empty:
        raise ValueError(f"Monument '{monument_name}' was not found in the dataset.")

    match = monument_rows.iloc[0]
    response = {
        "monument_name": match[resolved_name_col],
        "crowd_level": match["crowd_level"],
        "best_time_to_visit": match["best_time_to_visit"],
        "total_visitors": int(match["total_visitors"]),
    }

    circle_col = _find_first_column(prepared, ["circle", "region", "district"])
    if circle_col:
        response["circle"] = match[circle_col]

    return response


def list_monument_names(df: pd.DataFrame, name_col: str | None = None) -> list[str]:
    """Return sorted monument names for a Streamlit select box."""
    prepared = build_recommendation_dataset(df)
    resolved_name_col = _resolve_requested_column(prepared, name_col) or _find_first_column(
        prepared,
        [
            "monument",
            "monument_name",
            "site_name",
            "destination",
            "place",
            "name",
        ],
    )
    if not resolved_name_col:
        raise ValueError("Could not find a monument/site name column.")

    names = (
        prepared[resolved_name_col]
        .dropna()
        .astype(str)
        .str.strip()
        .sort_values()
        .unique()
        .tolist()
    )
    return names


def get_hidden_gems_cards(
    df: pd.DataFrame,
    top_n: int = 10,
) -> list[dict[str, object]]:
    """
    Return hidden gems as a list of dictionaries for easy UI rendering.
    """
    hidden_gems_df = recommend_hidden_gems(df, top_n=top_n)
    return hidden_gems_df.to_dict(orient="records")


def get_quick_insights(df: pd.DataFrame) -> dict[str, object]:
    """
    Build small summary metrics for the dashboard header.
    """
    prepared = build_recommendation_dataset(df)
    crowd_counts = prepared["crowd_level"].value_counts().to_dict()
    total_sites = int(len(prepared))
    avg_visitors = float(prepared["total_visitors"].mean()) if total_sites else 0.0

    return {
        "total_sites": total_sites,
        "average_visitors": round(avg_visitors, 2),
        "high_crowd_sites": int(crowd_counts.get("High", 0)),
        "medium_crowd_sites": int(crowd_counts.get("Medium", 0)),
        "low_crowd_sites": int(crowd_counts.get("Low", 0)),
    }


def build_recommendation_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare a dataset that the app can directly use.

    Steps:
    - normalize processed dataset columns
    - remove aggregate total rows
    - ensure total visitors exist
    - compute crowd index
    - compute crowd level
    - compute best visit time
    - compute heritage score
    - compute hidden gem score
    """
    prepared = _standardize_dataset(df)
    prepared = _filter_monument_rows(prepared)
    prepared = add_total_visitors_column(prepared)
    prepared = add_crowd_index_column(prepared)
    prepared = add_crowd_level_column(prepared)
    prepared = add_best_time_column(prepared)
    prepared = add_heritage_score_column(prepared)
    prepared = add_hidden_gem_score_column(prepared)
    return prepared


def generate_recommendation_outputs(
    csv_path: str | Path,
    top_n: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load data and return:
    1. Prepared tourism dataset with crowd and timing fields
    2. Hidden gems table
    """
    source_df = load_tourism_data(csv_path)
    prepared_df = build_recommendation_dataset(source_df)
    hidden_gems_df = recommend_hidden_gems(prepared_df, top_n=top_n)
    return prepared_df, hidden_gems_df


if __name__ == "__main__":
    data_dir = Path(__file__).resolve().parents[1] / "data"
    sample_path = data_dir / "processed_tourism.csv"

    if not sample_path.exists():
        raise FileNotFoundError(
            f"Processed dataset not found at {sample_path}. Add the CSV and rerun."
        )

    prepared_data, hidden_gems = generate_recommendation_outputs(sample_path)

    print("Prepared tourism data preview:")
    print(prepared_data.head())
    print("\nTop hidden gems:")
    print(hidden_gems)
