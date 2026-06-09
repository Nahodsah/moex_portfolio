import pandas as pd

from .data_source import MoexCSVDataSource
from .exceptions import EmptyDataError, MissingDateError
from .decorators import execution_time


class MarketDataProcessor:
    """Загружает csv-таблички из папки и загружает из pandas.Dataframe"""

    def __init__(self, data_source: MoexCSVDataSource) -> None:
        self.data_source = data_source

    def load_tables(self) -> list[pd.DataFrame]:
        """Загружает таблицы в pandas.Dataframe и делает первичную обработку:
        1. проверяет на пропуски
        2. из столбца datetime делает индексы для дальнейшего объединения
        """
        tables: list[pd.DataFrame] = []

        for table_name in self.data_source.iter_csv_files():
            ticker = self.data_source.extract_ticker(table_name)

            data = pd.read_csv(
                table_name,
                usecols=[0, 4],
                parse_dates=[0],
                dtype={"close": float},
            )

            data = data.rename(columns={"close": ticker})

            if data["datetime"].isna().sum():
                raise MissingDateError(
                    f"В таблице {table_name} есть пропуск даты"
                )

            data = data.set_index("datetime")
            tables.append(data)

        return tables

    @execution_time
    def concat_price_tables(self) -> pd.DataFrame:
        """Объединяет таблицы цен закрытия по общим датам."""
        tables = self.load_tables()

        if not tables:
            raise EmptyDataError("В папке с данными не найдено CSV-файлов")

        return pd.concat(tables, axis=1, join="inner").sort_index()

    @staticmethod
    @execution_time
    def calc_returns(prices: pd.DataFrame) -> pd.DataFrame:
        """Возвращает недельную доходность."""
        return prices.pct_change().dropna()

    @staticmethod
    @execution_time
    def calc_cov_matrix(returns: pd.DataFrame) -> pd.DataFrame:
        """Вычисляет ковариационную матрицу."""
        return returns.cov()
