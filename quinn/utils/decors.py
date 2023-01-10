#!/usr/bin/env python
"""Module for various decorators."""

import time
from functools import wraps


def timer(func):
    """Prints the runtime of the decorated function.

    Args:
        func (callable): Original function.

    Returns:
        callable: Decorated function.
    """
    @wraps(func)
    def logger(*args, **kwargs):
        start = time.time()
        output_value = func(*args, **kwargs)
        print(f"Finished {func.__name__} in {time.time() - start:.5f}")
        return output_value
    return logger

def show_start_end(func):
    """Highlights start and end of the function.

    Args:
        func (callable): Original function.

    Returns:
        callable: Decorated function.
    """
    @wraps(func)
    def inner_func(*args, **kwargs):
        print(f"Before calling func {func.__name__}")
        func(*args, **kwargs)
        print(f"After calling func {func.__name__}")

    return inner_func

