import importlib.util

import pandas as pd
import pytest

from app.core.data_loader import load_dataset, summarize_dataframe


def sample_dataframe():
    return pd.DataFrame(
        {
            "id": [1, 2, 2],
            "score": [10.5, None, None],
            "label": ["a", "b", "b"],
            "event_date": ["2026-01-01", "2026-01-02", "2026-01-02"],
        }
    )


def test_load_dataset_csv_with_latin1_fallback(tmp_path):
    csv_path = tmp_path / "sample.csv"
    csv_path.write_bytes("name,city\nAndr\xe9,Montr\xe9al\n".encode("latin1"))

    df = load_dataset(csv_path)

    assert df.to_dict(orient="records") == [{"name": "André", "city": "Montréal"}]


def test_load_dataset_normalizes_common_null_markers(tmp_path):
    csv_path = tmp_path / "nulls.csv"
    csv_path.write_text(
        'a,b,c\nnull,N/A,""\nnan, ,value\nNone,NA,text\n',
        encoding="utf-8",
    )

    df = load_dataset(csv_path)
    summary = summarize_dataframe(df)

    assert df["a"].isna().sum() == 3
    assert df["b"].isna().sum() == 3
    assert df["c"].isna().sum() == 1
    assert summary["total_missing_values"] == 7


def test_load_dataset_excel(tmp_path):
    df = sample_dataframe()
    excel_path = tmp_path / "sample.xlsx"
    df.to_excel(excel_path, index=False)

    loaded = load_dataset(excel_path)

    assert loaded.shape == df.shape
    assert loaded["id"].tolist() == [1, 2, 2]


@pytest.mark.skipif(
    importlib.util.find_spec("pyarrow") is None and importlib.util.find_spec("fastparquet") is None,
    reason="Parquet support requires pyarrow or fastparquet.",
)
def test_load_dataset_parquet(tmp_path):
    df = sample_dataframe()
    parquet_path = tmp_path / "sample.parquet"
    df.to_parquet(parquet_path, index=False)

    loaded = load_dataset(parquet_path)

    assert loaded.shape == df.shape
    assert loaded["label"].tolist() == ["a", "b", "b"]


def test_summarize_dataframe():
    df = sample_dataframe()

    summary = summarize_dataframe(df)

    assert summary["rows"] == 3
    assert summary["columns"] == 4
    assert summary["column_names"] == ["id", "score", "label", "event_date"]
    assert summary["missing_counts"]["score"] == 2
    assert summary["missing_percent"]["score"] == pytest.approx(66.6666667)
    assert summary["column_missing_percent"]["score"] == pytest.approx(66.6666667)
    assert summary["total_missing_values"] == 2
    assert summary["total_missing_percent"] == pytest.approx(16.6666667)
    assert summary["duplicate_rows"] == 1
    assert summary["duplicate_row_percent"] == pytest.approx(33.3333333)
    assert "id" in summary["numeric_columns"]
    assert "score" in summary["numeric_columns"]
    assert "label" in summary["categorical_columns"]
    assert "event_date" in summary["datetime_like_columns"]
    assert summary["numeric_percent"] == pytest.approx(50.0)
    assert summary["categorical_percent"] == pytest.approx(25.0)
    assert summary["simplified_column_types"]["id"] == "Numeric"
    assert summary["simplified_column_types"]["label"] == "Categorical"
    assert summary["simplified_column_types"]["event_date"] == "Date/Time"
    assert summary["memory_usage_mb"] > 0


def test_load_dataset_rejects_unsupported_format(tmp_path):
    file_path = tmp_path / "sample.txt"
    file_path.write_text("hello", encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported dataset format"):
        load_dataset(file_path)
