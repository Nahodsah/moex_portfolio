import json
from pathlib import Path

import pandas as pd


class ReportExporter:
    """Сохраняет полученные данные в указанную папку."""

    def __init__(self, output_dir: str = "results") -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def save_close_prices(self, prices: pd.DataFrame) -> None:
        prices.to_csv(self.output_dir / "close_prices.csv")

    def save_returns(self, returns: pd.DataFrame) -> None:
        returns.to_csv(self.output_dir / "weekly_returns.csv")

    def save_cov_matrix(self, covariance: pd.DataFrame) -> None:
        covariance.to_csv(self.output_dir / "covariance_matrix.csv")

    def save_weights(self, weights: pd.Series) -> None:
        weights.to_csv(
            self.output_dir / "portfolio_weights.csv",
            header=True,
        )

    def save_summary(self, summary: dict) -> None:
        with open(self.output_dir / "summary.json", "w", encoding="utf-8") as file:
            json.dump(summary, file, ensure_ascii=False, indent=4)
