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


def load_pv_model(name: str = "saved_models/Solar_Model_Results.wwm") -> Optional[Any]:
    """
    Load a serialized PV model from a file using pickle.

    Parameters:
        name (str): The path and name of the file to load the model from.
                    Defaults to "saved_models/Solar_Model_Results.wwm".
                    The file should have a '.wwm' extension.

    Returns:
        Optional[Any]: The deserialized model object if successful, None otherwise.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        Exception: For other errors that occur during model loading.
    """
    # Simple check if file is in the correct format
    if not name.endswith(".wwm"):
        logging.info(f"File being loaded is not in the correct format (.wwm): {name}")
        logging.info("*******************")
        return None

    # Open model if possible, return error if not
    try:
        if not os.path.exists(name):
            raise FileNotFoundError(f"The file '{name}' does not exist.")
        
        with open(name, 'rb') as filehandler_open:
            model = pickle.load(filehandler_open)
            logging.info(f"Model successfully loaded: {name}")
            logging.info("*******************")
            return model
            
    except FileNotFoundError as e:
        logging.info(f"Error: {e}")
        logging.info("*******************")
        return None

    except Exception as e:
        logging.info(f"Error loading model from '{name}': {e}")
        logging.info("*******************")
        return None
