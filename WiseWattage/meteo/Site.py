from dataclasses import dataclass, field
from functools import lru_cache
import pandas as pd
import logging

from meteo.jrc_tmy import get_jrc_tmy

@dataclass
class Site:
    """
    A class representing a site.

    Attributes:
        name (str): Name of the site.
        address (str): Address of the site.
        client (str): Client associated with the site.
        latitude (float): Latitude of the site.
        longitude (float): Longitude of the site.
        tmz_hrs_east (int): Timezone hours east of GMT.
        tmy_data (pd.DataFrame): DataFrame containing TMY data for the site.
    """
    name: str = ""  # Name of the site
    address: str = ""  # Address of the site
    client: str = ""  # Client associated with the site
    latitude: float = 54.60452  # Latitude of the site
    longitude: float = -5.92860  # Longitude of the site
    tmz_hrs_east: int = 0  # Timezone hours east of GMT
    tmy_data: pd.DataFrame = field(default=None, init=False)  # DataFrame containing TMY data for the site

    def __post_init__(self):
        """
        Post-initialization method.
        Fetches TMY data for the site and logs a message.
        """
        self.tmy_data = get_jrc_tmy_cached(self.latitude, self.longitude)  # Fetch TMY data
        logging.info(f'TMY data obtained for: {self.latitude}, longitude: {self.longitude}')  # Log message
        logging.info("*******************")


@lru_cache(maxsize=None)  # Cache TMY results
def get_jrc_tmy_cached(latitude, longitude):
    """
    Fetch TMY data for a given latitude and longitude, with caching.

    Args:
        latitude (float): Latitude of the location.
        longitude (float): Longitude of the location.

    Returns:
        pd.DataFrame: DataFrame containing TMY data.
    """
    logging.info(f'Fetching TMY data for latitude: {latitude}, longitude: {longitude}')  # Log message
    return get_jrc_tmy(latitude, longitude)  # Call function to fetch TMY data