import os
import pickle
import logging
from typing import Any, Optional

from functools import lru_cache, wraps
from typing import Callable, TypeVar

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
        
