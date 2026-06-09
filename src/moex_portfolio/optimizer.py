import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, minimize

from .exceptions import PortfolioOptimizationError


def portfolio_variance(weights: np.ndarray, covariance: np.ndarray) -> float:
    """Вычисляет дисперсию портфеля: w^T Sigma w."""
    return float(weights.T @ covariance @ weights)


def portfolio_variance_gradient(
    weights: np.ndarray,
    covariance: np.ndarray,
) -> np.ndarray:
    """Вычисляет градиент функции w^T Sigma w."""
    return 2 * covariance @ weights


class MinimumVarianceOptimizer:
    """Находит портфель минимальной дисперсии."""

    def optimize(self, returns: pd.DataFrame) -> pd.Series:
        if returns.empty:
            raise PortfolioOptimizationError("Таблица доходностей пуста")

        if returns.isna().any().any():
            raise PortfolioOptimizationError(
                "В таблице доходностей есть пропуски"
            )

        covariance = returns.cov()
        sigma = covariance.to_numpy()
        tickers = covariance.columns
        n = len(tickers)

        initial_weights = np.full(n, 1 / n)

        bounds = Bounds(
            lb=np.zeros(n),
            ub=np.ones(n),
        )

        budget_constraint = LinearConstraint(
            A=np.ones((1, n)),
            lb=1.0,
            ub=1.0,
        )

        result = minimize(
            fun=portfolio_variance,
            x0=initial_weights,
            args=(sigma,),
            jac=portfolio_variance_gradient,
            method="SLSQP",
            bounds=bounds,
            constraints=[budget_constraint],
        )

        if not result.success:
            raise PortfolioOptimizationError(result.message)

        weights = pd.Series(
            result.x,
            index=tickers,
            name="weight",
        )

        if not np.isclose(weights.sum(), 1.0):
            raise PortfolioOptimizationError("Сумма весов не равна 1")

        if (weights < -1e-8).any():
            raise PortfolioOptimizationError("Получены отрицательные веса")

        return weights
