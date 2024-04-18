import time
from typing import Any, Optional

from functools import lru_cache, wraps
from typing import Callable, TypeVar


def timer(func):
    """
    Decorator function to time the execution of another function.

    Parameters:
        func: The function to be timed

    Returns:
        wrapper: A wrapped version of the input function with timing functionality added.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        """
        Wrapper function that calculates the time the function takes to execute.
        """
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        time_took = end_time - start_time
        print("")

        # Format time to more readable units
        if time_took < 1:
            print("*******************")
            print(f"Processing took: {time_took * 1000:.2f} milliseconds")
            print("*******************")
        elif time_took >= 1:
            print("*******************")
            print(f"Processing took: {end_time - start_time:.4f} seconds")
            print("*******************")

        return result
    return wrapper



# Create a type variable that can be any callable
F = TypeVar('F', bound=Callable[..., Any])

def cached_func(func: F) -> F:
    """
    A decorator to cache the output of a function using lru_cache. 
    Called by adding @cached_func above a function to decorate it.
    
    Parameters:
        func: The function to be cached.
    
    Returns:
        wrapper(func): A wrapped version of the input function with caching enabled.
    """
    # Use wraps to preserve the name, docstring, etc., of the original function
    @wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        return func(*args, **kwargs)
    
    # Apply lru_cache with no limit on the cache size
    cached_wrapper = lru_cache(maxsize=None)(wrapper)
    
    return cached_wrapper 


class AttrDict(dict):
    """
    AttrDict extends the standard Python dictionary to allow attribute-like access to its items.

    Parameters:
    -----------
    dict : *args, **kwargs
        The constructor for AttrDict accepts the same arguments as the standard dict, allowing 
        for flexible dictionary initialization with key-value pairs, iterable of key-value pairs, 
        or keyword arguments.
    """
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(f"Attribute {key} not found.")
        
