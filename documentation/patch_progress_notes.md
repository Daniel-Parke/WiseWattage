# Wattsun and Homes - Energy Modelling Application
#### *** NOTE: These patch notes will not have data removed unless to improve readability and  serve as a time capsule of sorts to record development progress and program improvements over time. ***

*** As such certain class methods or code snippets may become defunct as the project progress. For specific and up to date functionality please see specific readme files and support documentation for update methodology on package usage. ***

## Patch/Progress Notes:

**17/02/2024:**
- [x] Implemented further vectorisation of code, reducing pandas dataframes to numpy arrays before 
completing model.
- [x] Refactored code structure for jrc_tmy meteo function to improve performance and remove redundant code.
- [x] Solar PV Panel, Wind Turbines, Inverter/Converter, & Battery Energy Storage module data added.
- [x] Added documentation, docstrings and function descriptions. Updated readme files and created documentation folder to further document and detail code structure.
- [x] 


**16/02/2024:**
- [x] Removed graphing functionality to allow for restructuring of dataframes, will possibly add back later.
- [x] Updated Exception handling to further address API return issues. JRC TMY data is identified as the main
performance and reliability issue with the program.
- [x] Ability to store class objects by pickling. Will adapt to JSON data structure at a later date.


**15/02/2024:**
- [x] Removed graphing functionality to allow for restructuring of dataframes, will possibly add back later.
- [x] Updated Exception handling to further address API return issues. JRC TMY data is identified as the main
performance and reliability issue with the program.
- [x] Ability to store class objects by pickling. Will adapt to JSON data structure at a later date.


**14/02/2024:**
- [x] Class folders and files restructured to allow for easier editing.
- [x] Removal of Flask specific functionality. 
- [x] Updated and improved logging system, general tidying of codebase.


**14/02/2024:**
- [x] Class folders and files restructured to allow for easier editing.
- [x] Removal of Flask specific functionality. 
- [x] Updated and improved logging system, general tidying of codebase.


**PRIOR TO RECORDS:**
- [x] Creation of `get_TMY` function to retrieve TMY meteorological data for a given location
- [x] Creation of solar radiation calculation codebase and integration with TMY data
- [x] Creation of fundamental solar PV power output calculations
- [x] Integration of sky model, clamping of beam/diffuse values & timestep adjustment
- [x] Adjustment of formulas to remove potential erroneous generation calc when AOI/Zenith close to 90 degrees
- [x] Inclusion of base physical processes affecting solar PV performance (Thermals, derating, etc.)
- [x] Inclusion of IAM losses to reduce beam insolation
- [x] Vectorised functionality added to radiation and modelling calculations to improve performance
- [x] Data aggregation functionality and time series grouping
- [x] Data logging implemented to assist with debugging and awareness of timescales for large models
- [x] Creation of `Site` class to allow for storing of TMY and project data
- [x] Function caching of TMY data enabled for faster reiteration
- [x] Creation of `SolarPVArray` class to allow for multiple scenarios to be modelled at the same location
- [x] Creation of `SolarPVModel` class to simulate solar PV performance at location
- [x] Data aggregation functions added to initial PVModel class that enable easier statistical analysis
- [x] Storing of individual full model results as well as aggregated data to be accessed separately
- [x] Enable saving of produced models as CSV file
- [x] Basic exception handling implemented
- [x] Ability to save model objects built in to class methods
- [x] Vectorised model calculation transforming from dataframes to arrays to utilise numpy speed
- [x] Further optimisations completed that bring solar model time to around 15.8ms/model
- [x] Added cProfile and pstats benchmarking to identify bottlenecks.
<br>


## Model classes and methods already created prior to record keeping.

**Site**
```python
site = Site()
site.latitude
site.longitude
site.name
site.address
site.client
site.size
site.tmz_hrs_east
site.tmy_data
```

**SolarPVArray**
```python
array_1 = SolarPVArray()
array_1.pv_kwp
array_1.surface_pitch
array_1.surface_azimuth
array_1.lifespan
array_1.pv_eol_derating
array_1.cost_per_kwp
array_1.electrical_eff
array_1.cell_temp_coeff
array_1.transmittance_absorptance
array_1.refraction_index
array_1.cell_NOCT
array_1.ambient_NOCT
array_1.e_poa_NOCT
array_1.e_poa_STC
array_1.cell_temp_STC
```

**SolarPVModel**
```python
pv_model = SolarPVModel()
pv_model.site
pv_model.arrays
pv_model.models
pv_model.all_models
pv_model.combined_model
pv_model.summary
pv_model.summary_grouped
pv_model.albedo
pv_model.timestep

pv_model.array_model(n)
pv_model.save_model_csv()
pv_model.model_summary_html_export(freq, grouped)
```