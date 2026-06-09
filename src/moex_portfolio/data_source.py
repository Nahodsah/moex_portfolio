import csv
from collections.abc import Iterator
from glob import iglob
from pathlib import Path


class MoexCSVDataSource:
    """Лениво читает CSV-файлы с котировками и отдаёт только дату, тикер и цену закрытия."""

    def __init__(self, data_dir: str = "data") -> None:
        self.data_dir = data_dir

    def iter_csv_files(self) -> Iterator[str]:
        """Лениво возвращает пути к CSV-файлам из папки с данными."""
        yield from iglob(f"{self.data_dir}/*.csv")

    @staticmethod
    def extract_ticker(file_path: str) -> str:
        """Получает тикер из названия CSV-файла."""
        return Path(file_path).name.split("_")[0]

    def iter_price_rows(self, file_path: str) -> Iterator[dict[str, str]]:
        """Лениво читает один CSV-файл и возвращает только дату, тикер и цену закрытия."""
        ticker = self.extract_ticker(file_path)

        with open(file_path, encoding="utf-8", newline="") as file:
            reader = csv.DictReader(file)

            for row in reader:
                yield {
                    "datetime": row["datetime"],
                    "ticker": ticker,
                    "close": row["close"],
                }

    def iter_all_price_rows(self) -> Iterator[dict[str, str]]:
        """Лениво читает все CSV-файлы и возвращает строки с датой, тикером и ценой закрытия."""
        for file_path in self.iter_csv_files():
            yield from self.iter_price_rows(file_path)
