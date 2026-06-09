class MarketDataError(Exception):
    """Базовый класс для ошибок, связанных биржевыми данными."""


class MissingDateError(MarketDataError):
    """Выскакивает, когда в строке с датой стоит пропуск."""


class EmptyDataError(MarketDataError):
    """Выскакивает, когда python не смог собрать данные из папки."""


class PortfolioOptimizationError(Exception):
    """Ошибка при оптимизации портфеля."""
