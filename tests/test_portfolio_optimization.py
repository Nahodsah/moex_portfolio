import json

import numpy as np
import pandas as pd
import pytest

from moex_portfolio.data_source import MoexCSVDataSource
from moex_portfolio.exceptions import EmptyDataError, MissingDateError
from moex_portfolio.exporter import ReportExporter
from moex_portfolio.optimizer import (
    MinimumVarianceOptimizer,
    PortfolioOptimizationError,
    portfolio_variance,
    portfolio_variance_gradient,
)
from moex_portfolio.processor import MarketDataProcessor


@pytest.fixture
def clean_csv_dir(tmp_path):
    """Папка с корректными CSV-файлами без пропущенных дат."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    sber = pd.DataFrame(
        {
            "datetime": pd.date_range("2023-01-01", periods=5, freq="D"),
            "open": [99, 100, 101, 102, 103],
            "high": [101, 102, 103, 104, 105],
            "low": [98, 99, 100, 101, 102],
            "close": [100, 101, 102, 103, 104],
        }
    )
    sber.to_csv(data_dir / "SBER_2020.csv", index=False)

    gazp = pd.DataFrame(
        {
            "datetime": pd.date_range("2023-01-01", periods=5, freq="D"),
            "open": [149, 151, 150, 152, 154],
            "high": [151, 153, 152, 154, 156],
            "low": [148, 150, 149, 151, 153],
            "close": [150, 152, 151, 153, 155],
        }
    )
    gazp.to_csv(data_dir / "GAZP_2020.csv", index=False)

    return data_dir


@pytest.fixture
def processor(clean_csv_dir):
    """Процессор, работающий с корректными тестовыми CSV-файлами."""
    data_source = MoexCSVDataSource(str(clean_csv_dir))
    return MarketDataProcessor(data_source)


@pytest.fixture
def dir_with_bad_file(tmp_path):
    """Папка с одним корректным CSV и одним CSV с пропущенной датой."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    good = pd.DataFrame(
        {
            "datetime": pd.date_range("2023-01-01", periods=3, freq="D"),
            "open": [99, 100, 101],
            "high": [101, 102, 103],
            "low": [98, 99, 100],
            "close": [100, 101, 102],
        }
    )
    good.to_csv(data_dir / "GOOD_2020.csv", index=False)

    bad = pd.DataFrame(
        {
            "datetime": ["2023-01-01", None, "2023-01-03"],
            "open": [9, 10, 11],
            "high": [11, 12, 13],
            "low": [8, 9, 10],
            "close": [10, 11, 12],
        }
    )
    bad.to_csv(data_dir / "BAD_dates.csv", index=False)

    return data_dir


# Тесты для MoexCSVDataSource

def test_iter_csv_files_returns_only_csv_paths(clean_csv_dir):
    data_source = MoexCSVDataSource(str(clean_csv_dir))

    files = list(data_source.iter_csv_files())

    assert len(files) == 2
    assert all(file_path.endswith(".csv") for file_path in files)


@pytest.mark.parametrize(
    "filename, expected_ticker",
    [
        ("SBER_2020.csv", "SBER"),
        ("GAZP_data.csv", "GAZP"),
        ("VTBR.csv", "VTBR.csv"),
    ],
)
def test_extract_ticker(filename, expected_ticker):
    assert MoexCSVDataSource.extract_ticker(filename) == expected_ticker


def test_iter_price_rows_reads_only_required_fields(clean_csv_dir):
    data_source = MoexCSVDataSource(str(clean_csv_dir))
    file_path = clean_csv_dir / "SBER_2020.csv"

    rows = list(data_source.iter_price_rows(str(file_path)))

    assert len(rows) == 5
    assert rows[0] == {
        "datetime": "2023-01-01",
        "ticker": "SBER",
        "close": "100",
    }
    assert set(rows[0].keys()) == {"datetime", "ticker", "close"}


def test_iter_all_price_rows_reads_all_files_lazily(clean_csv_dir):
    data_source = MoexCSVDataSource(str(clean_csv_dir))

    rows = list(data_source.iter_all_price_rows())
    tickers = {row["ticker"] for row in rows}

    assert len(rows) == 10
    assert tickers == {"SBER", "GAZP"}
    assert all(set(row.keys()) == {"datetime", "ticker", "close"} for row in rows)


# Тесты для MarketDataProcessor

def test_load_tables_raises_on_missing_date(dir_with_bad_file):
    data_source = MoexCSVDataSource(str(dir_with_bad_file))
    processor = MarketDataProcessor(data_source)

    with pytest.raises(MissingDateError, match="BAD_dates.csv"):
        processor.load_tables()


def test_concat_price_tables_works(processor):
    prices = processor.concat_price_tables()

    assert prices.shape == (5, 2)
    assert set(prices.columns) == {"SBER", "GAZP"}
    assert not prices.isna().any().any()


def test_concat_price_tables_empty_dir(tmp_path):
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()

    data_source = MoexCSVDataSource(str(empty_dir))
    processor = MarketDataProcessor(data_source)

    with pytest.raises(EmptyDataError, match="не найдено CSV"):
        processor.concat_price_tables()


@pytest.mark.parametrize(
    "input_series, expected_pct",
    [
        (pd.Series([100, 101, 102]), pd.Series([0.01, 0.00990099])),
        (pd.Series([50, 50, 50]), pd.Series([0.0, 0.0])),
    ],
)
def test_calc_returns(input_series, expected_pct):
    prices = pd.DataFrame({"A": input_series})

    returns = MarketDataProcessor.calc_returns(prices)

    pd.testing.assert_series_equal(
        returns["A"],
        expected_pct,
        check_names=False,
        check_index=False,
    )


def test_calc_cov_matrix():
    returns = pd.DataFrame(
        {
            "A": [0.01, -0.02, 0.03],
            "B": [0.02, 0.01, -0.01],
        }
    )

    covariance = MarketDataProcessor.calc_cov_matrix(returns)

    assert covariance.shape == (2, 2)
    assert list(covariance.columns) == ["A", "B"]
    assert list(covariance.index) == ["A", "B"]
    assert covariance.loc["A", "A"] > 0
    assert covariance.loc["B", "B"] > 0


# Тесты для функций оптимизации

def test_portfolio_variance():
    weights = np.array([0.5, 0.5])
    covariance = np.array([[0.04, 0.01], [0.01, 0.09]])

    variance = portfolio_variance(weights, covariance)

    expected = 0.5**2 * 0.04 + 2 * 0.5 * 0.5 * 0.01 + 0.5**2 * 0.09
    assert np.isclose(variance, expected)


def test_portfolio_variance_gradient():
    weights = np.array([0.5, 0.5])
    covariance = np.array([[0.04, 0.01], [0.01, 0.09]])

    gradient = portfolio_variance_gradient(weights, covariance)

    expected = 2 * covariance @ weights
    np.testing.assert_array_almost_equal(gradient, expected)


# Тесты для MinimumVarianceOptimizer

@pytest.mark.optimization
def test_optimizer_returns_valid_weights():
    returns = pd.DataFrame(
        {
            "A": [0.01, 0.02, 0.015, 0.017],
            "B": [0.02, 0.025, 0.018, 0.021],
        }
    )
    optimizer = MinimumVarianceOptimizer()

    weights = optimizer.optimize(returns)

    assert np.isclose(weights.sum(), 1.0)
    assert (weights >= 0).all()
    assert set(weights.index) == {"A", "B"}


def test_optimizer_raises_on_empty_returns():
    optimizer = MinimumVarianceOptimizer()

    with pytest.raises(PortfolioOptimizationError, match="пуста"):
        optimizer.optimize(pd.DataFrame())


def test_optimizer_raises_on_nan_returns():
    returns = pd.DataFrame(
        {
            "A": [0.01, np.nan, 0.03],
            "B": [0.02, 0.01, 0.04],
        }
    )
    optimizer = MinimumVarianceOptimizer()

    with pytest.raises(PortfolioOptimizationError, match="пропуски"):
        optimizer.optimize(returns)


# Тесты для ReportExporter

def test_exporter_creates_directory(tmp_path):
    output_dir = tmp_path / "results"

    ReportExporter(str(output_dir))

    assert output_dir.exists()


def test_exporter_save_close_prices(tmp_path):
    exporter = ReportExporter(str(tmp_path))
    prices = pd.DataFrame({"A": [100, 101], "B": [200, 201]})

    exporter.save_close_prices(prices)

    saved = pd.read_csv(tmp_path / "close_prices.csv", index_col=0)
    pd.testing.assert_frame_equal(saved, prices)


def test_exporter_save_returns(tmp_path):
    exporter = ReportExporter(str(tmp_path))
    returns = pd.DataFrame({"A": [0.01, 0.02], "B": [0.03, 0.04]})

    exporter.save_returns(returns)

    saved = pd.read_csv(tmp_path / "weekly_returns.csv", index_col=0)
    pd.testing.assert_frame_equal(saved, returns)


def test_exporter_save_covariance_matrix(tmp_path):
    exporter = ReportExporter(str(tmp_path))
    covariance = pd.DataFrame(
        [[0.01, 0.002], [0.002, 0.03]],
        index=["A", "B"],
        columns=["A", "B"],
    )

    exporter.save_covariance_matrix(covariance)

    saved = pd.read_csv(tmp_path / "covariance_matrix.csv", index_col=0)
    pd.testing.assert_frame_equal(saved, covariance)


def test_exporter_save_weights(tmp_path):
    exporter = ReportExporter(str(tmp_path))
    weights = pd.Series([0.6, 0.4], index=["A", "B"], name="weight")

    exporter.save_weights(weights)

    saved = pd.read_csv(tmp_path / "portfolio_weights.csv", index_col=0)
    assert saved.columns[0] == "weight"
    assert saved.loc["A", "weight"] == 0.6
    assert saved.loc["B", "weight"] == 0.4


def test_exporter_save_summary(tmp_path):
    exporter = ReportExporter(str(tmp_path))
    summary = {"weekly_variance": 0.01, "annual_volatility": 0.15}

    exporter.save_summary(summary)

    with open(tmp_path / "summary.json", encoding="utf-8") as file:
        loaded = json.load(file)

    assert loaded == summary
