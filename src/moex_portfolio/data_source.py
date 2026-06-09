from collections.abc import Iterator
from glob import iglob
from pathlib import Path


class MoexCSVDataSource:
    """Класс-итератор, который лениво достаёт csv-таблички из папки с данными."""

    def __init__(self, data_dir: str = "data") -> None:
        self.data_dir = data_dir

    def iter_csv_files(self) -> Iterator[str]:
        """Метод-итератор, возвращающий название файла."""
        for file_path in iglob(f"{self.data_dir}/*.csv"):
            yield file_path

    @staticmethod
    def extract_ticker(file_path: str) -> str:
        """Статик-метод, получающий тикер из названия csv-таблички."""
        return Path(file_path).name.split("_")[0]
