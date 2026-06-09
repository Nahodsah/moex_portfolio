import json
import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from moex_portfolio.data_source import MoexCSVDataSource
from moex_portfolio.exceptions import MissingDateError, EmptyDataError, PortfolioOptimizationError
from moex_portfolio.processor import MarketDataProcessor
from moex_portfolio.optimizer import MinimumVarianceOptimizer, portfolio_variance, portfolio_variance_gradient
from moex_portfolio.exporter import ReportExporter


# Фикстуры
@pytest.fixture
def clean_csv_dir(tmp_path):
    """
    Фикстура с корректными CSV (без пропусков дат).
    """
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    df_sber = pd.DataFrame({
        "datetime": pd.date_range("2023-01-01", periods=5, freq="D"),
        "dummy1": 0,
        "dummy2": 0,
        "dummy3": 0,
        "close": [100, 101, 102, 103, 104]
    })
    df_sber.to_csv(data_dir / "SBER_2020.csv", index=False)

    df_gazp = pd.DataFrame({
        "datetime": pd.date_range("2023-01-01", periods=5, freq="D"),
        "dummy1": 0,
        "dummy2": 0,
        "dummy3": 0,
        "close": [150, 152, 151, 153, 155]
    })
    df_gazp.to_csv(data_dir / "GAZP_2020.csv", index=False)

    return data_dir


@pytest.fixture
def processor(clean_csv_dir):
    """Фикстура с процессором, использующим только корректные файлы."""
    ds = MoexCSVDataSource(str(clean_csv_dir))
    return MarketDataProcessor(ds)


@pytest.fixture
def dir_with_bad_file(tmp_path):
    """
    Фикстура с одним корректным и одним битым файлом (пропуск даты).
    """
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    df_good = pd.DataFrame({
        "datetime": pd.date_range("2023-01-01", periods=3, freq="D"),
        "dummy1": 0,
        "dummy2": 0,
        "dummy3": 0,
        "close": [100, 101, 102]
    })
    df_good.to_csv(data_dir / "GOOD.csv", index=False)

    df_bad = pd.DataFrame({
        "datetime": ["2023-01-01", None, "2023-01-03"],
        "dummy1": 0,
        "dummy2": 0,
        "dummy3": 0,
        "close": [10, 11, 12]
    })
    df_bad.to_csv(data_dir / "BAD_dates.csv", index=False)

    return data_dir


# Тесты для MoexCSVDataSource
def test_iter_csv_files_returns_paths(clean_csv_dir):
    ds = MoexCSVDataSource(str(clean_csv_dir))
    files = list(ds.iter_csv_files())
    assert len(files) == 2  # только SBER и GAZP
    assert all(f.endswith(".csv") for f in files)


@pytest.mark.parametrize("filename,expected_ticker", [
    ("SBER_2020.csv", "SBER"),
    ("GAZP_data.csv", "GAZP"),
    ("VTBR.csv", "VTBR.csv"),
])
def test_extract_ticker(filename, expected_ticker):
    assert MoexCSVDataSource.extract_ticker(filename) == expected_ticker


# Тесты для MarketDataProcessor
def test_load_tables_raises_on_missing_date(dir_with_bad_file):
    ds = MoexCSVDataSource(str(dir_with_bad_file))
    proc = MarketDataProcessor(ds)
    with pytest.raises(MissingDateError, match="BAD_dates.csv"):
        proc.load_tables()

def test_concat_price_tables_works(processor):
    prices = processor.concat_price_tables()
    assert prices.shape == (5, 2)
    assert set(prices.columns) == {"SBER", "GAZP"}
    assert not prices.isna().any().any()

def test_concat_price_tables_empty_dir(tmp_path):
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    ds = MoexCSVDataSource(str(empty_dir))
    proc = MarketDataProcessor(ds)
    with pytest.raises(EmptyDataError, match="не найдено CSV"):
        proc.concat_price_tables()


@pytest.mark.parametrize("input_series,expected_pct", [
    (pd.Series([100, 101, 102]), pd.Series([0.01, 0.00990099])),
    (pd.Series([50, 50, 50]), pd.Series([0.0, 0.0])),
])
def test_calc_returns(input_series, expected_pct):
    df = pd.DataFrame({"A": input_series})
    returns = MarketDataProcessor.calc_returns(df)
    # Сравниваем значения, игнорируя индекс (после dropna() он начинается с 1)
    pd.testing.assert_series_equal(
        returns["A"], expected_pct, 
        check_names=False, 
        check_index=False
    )


def test_calc_cov_matrix():
    returns = pd.DataFrame({
        "A": [0.01, -0.02, 0.03],
        "B": [0.02, 0.01, -0.01]
    })
    cov = MarketDataProcessor.calc_cov_matrix(returns)
    assert cov.shape == (2, 2)
    assert cov.iloc[0, 0] > 0
    assert cov.iloc[1, 1] > 0


# Тесты для оптимизации
def test_portfolio_variance():
    w = np.array([0.5, 0.5])
    Sigma = np.array([[0.04, 0.01], [0.01, 0.09]])
    var = portfolio_variance(w, Sigma)
    expected = 0.5**2 * 0.04 + 2 * 0.5 * 0.5 * 0.01 + 0.5**2 * 0.09
    assert np.isclose(var, expected)


def test_portfolio_variance_gradient():
    w = np.array([0.5, 0.5])
    Sigma = np.array([[0.04, 0.01], [0.01, 0.09]])
    grad = portfolio_variance_gradient(w, Sigma)
    expected = 2 * Sigma @ w
    np.testing.assert_array_almost_equal(grad, expected)


@pytest.mark.optimization
def test_optimizer_returns_correct_weights():
    returns = pd.DataFrame({
        "A": [0.01, 0.02, 0.015],
        "B": [0.02, 0.025, 0.018]
    })
    opt = MinimumVarianceOptimizer()
    weights = opt.optimize(returns)
    assert np.isclose(weights.sum(), 1.0)
    assert (weights >= 0).all()

    cov = returns.cov()
    var_A = cov.loc["A", "A"]
    var_B = cov.loc["B", "B"]

    if var_A > var_B:
        assert weights["A"] <= weights["B"] + 1e-8
    elif var_B > var_A:
        assert weights["B"] <= weights["A"] + 1e-8
    else:
        assert abs(weights["A"] - weights["B"]) < 1e-8


def test_optimizer_raises_on_empty_returns():
    opt = MinimumVarianceOptimizer()
    empty = pd.DataFrame()
    with pytest.raises(PortfolioOptimizationError, match="пуста"):
        opt.optimize(empty)


def test_optimizer_raises_on_nan_returns():
    returns = pd.DataFrame({
        "A": [0.01, np.nan, 0.03],
        "B": [0.02, 0.01, 0.04]
    })
    opt = MinimumVarianceOptimizer()
    with pytest.raises(PortfolioOptimizationError, match="пропуски"):
        opt.optimize(returns)


# Тесты для ReportExporter
def test_exporter_creates_directory(tmp_path):
    exporter = ReportExporter(str(tmp_path))
    assert tmp_path.exists()


def test_exporter_save_weights(tmp_path):
    exporter = ReportExporter(str(tmp_path))
    weights = pd.Series([0.6, 0.4], index=["A", "B"], name="weight")
    exporter.save_weights(weights)
    saved = pd.read_csv(tmp_path / "portfolio_weights.csv", index_col=0)
    assert saved.columns[0] == "weight"
    assert saved.loc["A", "weight"] == 0.6


def test_exporter_save_summary(tmp_path):
    exporter = ReportExporter(str(tmp_path))
    summary = {"sharpe": 1.23, "risk": 0.05}
    exporter.save_summary(summary)
    with open(tmp_path / "summary.json", "r", encoding="utf-8") as f:
        loaded = json.load(f)
    assert loaded == summary