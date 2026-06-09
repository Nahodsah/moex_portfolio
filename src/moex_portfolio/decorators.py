from time import time
from functools import wraps


def execution_time(func):

    @wraps
    def wrapper(*args, **kwargs):
        start = time()
        result = func(*args, **kwargs)
        end = time()

        print(f"{func.__name__} отрабатывает за {end - start:.4f} секунд.")

        return result

    return wrapper