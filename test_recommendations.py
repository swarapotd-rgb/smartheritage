from __future__ import annotations

from pathlib import Path

from scripts.recommendations import (
    build_recommendation_dataset,
    get_monument_recommendation,
    list_monument_names,
    load_tourism_data,
    recommend_hidden_gems,
)


DATA_DIR = Path(__file__).resolve().parent / "data"
PROCESSED_CSV = DATA_DIR / "processed_tourism.csv"
REQUIRED_COLUMNS = {
    "monument",
    "total_visitors",
    "crowd_index",
    "crowd_level",
    "best_time_to_visit",
}


def assert_condition(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def validate_dataset(csv_path: Path) -> None:
    df = load_tourism_data(csv_path)
    prepared = build_recommendation_dataset(df)
    hidden_gems = recommend_hidden_gems(df, top_n=10)
    monument_names = list_monument_names(df)
    taj_mahal = get_monument_recommendation(df, "Taj Mahal")

    assert_condition(not prepared.empty, f"{csv_path.name}: prepared dataset is empty")
    assert_condition(
        REQUIRED_COLUMNS.issubset(prepared.columns),
        f"{csv_path.name}: missing required columns {REQUIRED_COLUMNS - set(prepared.columns)}",
    )
    assert_condition(
        "Taj Mahal" in monument_names,
        f"{csv_path.name}: Taj Mahal missing from monument names",
    )
    assert_condition(
        "Total" not in monument_names,
        f"{csv_path.name}: aggregate Total row still appears in monument names",
    )
    assert_condition(
        not prepared["monument"].astype(str).str.strip().str.lower().eq("total").any(),
        f"{csv_path.name}: aggregate Total row still appears in prepared dataset",
    )
    assert_condition(
        not hidden_gems["monument_name"].astype(str).str.strip().str.lower().eq("total").any(),
        f"{csv_path.name}: aggregate Total row still appears in hidden gems",
    )
    assert_condition(
        taj_mahal["crowd_level"] == "High",
        f"{csv_path.name}: Taj Mahal crowd level expected High, got {taj_mahal['crowd_level']}",
    )
    assert_condition(
        taj_mahal["best_time_to_visit"] == "Early morning",
        (
            f"{csv_path.name}: Taj Mahal best time expected Early morning, "
            f"got {taj_mahal['best_time_to_visit']}"
        ),
    )
    assert_condition(
        taj_mahal["total_visitors"] == 5_075_125,
        (
            f"{csv_path.name}: Taj Mahal total visitors expected 5075125, "
            f"got {taj_mahal['total_visitors']}"
        ),
    )
    assert_condition(
        hidden_gems.shape[0] == 10,
        f"{csv_path.name}: expected 10 hidden gems, got {hidden_gems.shape[0]}",
    )

    print(f"[PASS] {csv_path.name}")


def main() -> None:
    assert_condition(PROCESSED_CSV.exists(), f"Missing file: {PROCESSED_CSV}")

    validate_dataset(PROCESSED_CSV)
    print("All recommendation checks passed for processed_tourism.csv.")


if __name__ == "__main__":
    main()
