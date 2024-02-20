from dataclasses import dataclass, field
import pandas as pd
import logging

from meteo.jrc_tmy import get_jrc_tmy

@dataclass
class Site:
    """
    A class representing a site.

    Attributes:
        latitude (float): Latitude of the site.
        longitude (float): Longitude of the site.
        name (str): Name of the site.
        address (str): Address of the site.
        client (str): Client associated with the site.
        size: float = None  # Size of the site in metres squared(m2)
        tmz_hrs_east (int): Timezone hours east of GMT, UTC=0.
        timestep (int): Timestep of weather data in minutes
        tmy_data (pd.DataFrame): DataFrame containing TMY data for the site.
    """
    latitude: float = 54.60452
    longitude: float = -5.92860
    name: str = ""
    address: str = ""
    client: str = ""
    size: float = None
    tmz_hrs_east: int = 0
    timestep: int = 60
    tmy_data: pd.DataFrame = field(default=None, init=False)

    def __post_init__(self):
        """
        Post-initialization method.
        Fetches TMY data for the site and logs a message.
        """
        logging.info(f'Fetching TMY data for latitude: {self.latitude}, longitude: {self.longitude}') 
        # Fetch TMY data
        self.tmy_data = get_jrc_tmy(self.latitude, self.longitude)
        logging.info(f'TMY data obtained for: {self.latitude}, longitude: {self.longitude}')
        logging.info("*******************")


