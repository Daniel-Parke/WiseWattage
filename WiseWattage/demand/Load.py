from dataclasses import dataclass, field
import logging
import pandas as pd

from demand.load_profile import calc_load_profile

@dataclass
class Load:
    daily_electric: float = 9.91
    daily_variablity: float = 0.15
    timestep_variability: float = 0.1
    profile: str = "Domestic"
    country: str = "UK"
    annual_electric: float = None
    load_profile: pd.DataFrame = field(default=None, init=False)

    def __post_init__(self):
        """
        Post-initialization method.
        Calculate annual electric use if None input
        """
        # Check for annual electric and calculate if None
        if self.annual_electric == None:
            self.annual_electric = self.daily_electric * 365


        self.load_profile = calc_load_profile(daily_demand = self.daily_electric,
                                              daily_std = self.daily_variablity,
                                              hourly_std = self.timestep_variability,
                                              profile = self.profile,
                                              country = self.country)
        
        logging.info(
            f"Load Profile Generated: Daily Electricity Use: {self.daily_electric}kWh, "
            f"Annual Electricity Use: {self.annual_electric}kWh, Daily Variability: {self.daily_variablity*100}%, "
            f"Hourly Variability: {self.timestep_variability*100}%"
        )