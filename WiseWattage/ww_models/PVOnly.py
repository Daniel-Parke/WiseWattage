from dataclasses import dataclass
from typing import List, Union, Dict

from solar.SolarPVModel import Site, SolarPVArray, SolarPVPanel, SolarPVModel
from misc.util import AttrDict

@dataclass
class PV_Only:
    site: Union[Site, List[Site]]
    arrays: Union[SolarPVArray, List[SolarPVArray]]
    name: str = ""
    pv_models: Dict[str, SolarPVModel] = None

    def __post_init__(self):
        """
        Post-initialization method.
        Normalizes site and arrays to lists if they are not already.
        Runs SolarPVModel for each site with the entire list of arrays,
        storing the results in pv_models with keys matching name of site modelled.
        """
        self.site = [self.site] if not isinstance(self.site, list) else self.site
        self.arrays = [self.arrays] if not isinstance(self.arrays, list) else self.arrays
        
        self.pv_models = AttrDict()  # Initialize pv_models as an empty AttrDict
        site_name_counter = {}  # Helper dictionary to track occurrences of site names

        # Iterate over each site
        for site in self.site:
            # Generate a unique key for each site name
            base_key = site.name
            key = base_key
            if base_key in site_name_counter:
                site_name_counter[base_key] += 1
                key = f"{base_key}_{site_name_counter[base_key]}"
            else:
                site_name_counter[base_key] = 1

            # Create a SolarPVModel for the site with the entire list of arrays
            pv_model = SolarPVModel(site, self.arrays)

            # Store the model in the pv_models AttrDict with the unique key
            self.pv_models[key] = pv_model
