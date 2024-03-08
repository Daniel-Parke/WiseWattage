from dataclasses import dataclass
from typing import List, Union, Dict

from solar.SolarPVModel import Site, SolarPVArray, SolarPVPanel, SolarPVModel
from misc.util import AttrDict

@dataclass
class PV_Only:
    sites: Union[Site, List[Site]]
    arrays: Union[SolarPVArray, List[SolarPVArray], List[List[SolarPVArray]]]  # Updated type hint to include nested lists
    name: str = ""
    pv_models: Dict[str, Dict[str, SolarPVModel]] = None  # Updated type hint for nested dictionaries

    def __post_init__(self):
        self.site = [self.site] if not isinstance(self.site, list) else self.site
        self.arrays = [self.arrays] if not isinstance(self.arrays, list) else self.arrays
        
        self.pv_models = AttrDict()

        # Function to check if any element in the list is a list itself
        def is_nested(arrays):
            return any(isinstance(i, list) for i in arrays)

        for site in self.site:
            base_key = site.name

            # Initialize the nested dictionary for this site if not already present
            if base_key not in self.pv_models:
                self.pv_models[base_key] = AttrDict()

            # Check if arrays is nested and iterate accordingly
            if is_nested(self.arrays):
                # Ensure we start indexing from 1 for nested lists
                for idx, array_set in enumerate(self.arrays, start=1):
                    # Adjust key generation to include the index for all nested lists
                    key = f"{base_key}_{idx}"
                    # Create SolarPVModel for each nested list
                    pv_model = SolarPVModel(site, array_set)
                    self.pv_models[base_key][key] = pv_model  
            else:
                # Handle non-nested (flat) list of arrays
                key = f"{base_key}_1"  # Default to 1 if non-nested
                pv_model = SolarPVModel(site, self.arrays)
                self.pv_models[base_key][key] = pv_model 
