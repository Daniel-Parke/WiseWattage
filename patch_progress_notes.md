# Wattsun and Homes - Energy Modelling Application

### **NOTE:** These patch notes will not have data removed unless to improve readability and serve as a time capsule of sorts to record development progress and program improvements over time.

### As such, certain class methods or code snippets may become defunct as the project progresses. For specific and up-to-date functionality, please see specific readme files and support documentation for updated methodology on package usage.

## Patch/Progress Notes:

### **17/02/2024:**
- Implemented further vectorization of code, reducing pandas DataFrames to numpy arrays before completing model.
- Refactored code structure for jrc_tmy meteo function to improve performance and remove redundant code.
- Solar PV Panel, Wind Turbines, Inverter/Converter, & Battery Energy Storage module data added.
- Added documentation, docstrings, and function descriptions. Updated readme files and created a documentation folder to further document and detail code structure.
- Further optimizations completed that bring solar model time to around 15.8ms/model.
- Added cProfile and pstats benchmarking to identify bottlenecks.

### **16/02/2024:**
- Removed graphing functionality to allow for restructuring of DataFrames; may possibly add back later.
- Updated Exception handling to further address API return issues. JRC TMY data is identified as the main performance and reliability issue with the program.
- Ability to store class objects by pickling. Will adapt to JSON data structure at a later date.
- Function caching of TMY data enabled for faster reiteration

### **14/02/2024:**
- Class folders and files restructured to allow for easier editing.
- Removal of Flask specific functionality.
- Updated and improved logging system, general tidying of codebase.
- Vectorized model calculation transforming from DataFrames to arrays to utilize numpy speed.

### **PRIOR TO RECORDS:**
#### **Functional Programmes:**
- Creation of `get_TMY` function to retrieve TMY meteorological data for a given location.
- Creation of solar radiation calculation codebase and integration with TMY data.
- Creation of fundamental solar PV power output calculations.
- Integration of sky model, clamping of beam/diffuse values & timestep adjustment.
- Adjustment of formulas to remove potential erroneous generation calc when AOI/Zenith close to 90 degrees.
- Inclusion of base physical processes affecting solar PV performance (Thermals, derating, etc.).
- Inclusion of IAM losses to reduce beam insolation.
- Vectorized compatiability added to radiation and modeling calculations to improve performance.
- Data aggregation functionality and time series grouping.
- Data logging implemented to assist with debugging and awareness of timescales for large models.

#### **Class related Programmes:**
- Creation of `Site` class to allow for storing of TMY and project data..
- Creation of `SolarPVArray` class to allow for multiple scenarios to be modeled at the same location.
- Creation of `SolarPVModel` class to simulate solar PV performance at location.
- Data aggregation functions added to initial PVModel class that enable easier statistical analysis.
- Storing of individual full model results as well as aggregated data to be accessed separately.
- Enable saving of produced models as CSV file.
- Basic exception handling implemented.
- Ability to save model objects built into class methods.
- Data summaries grouped by timeperiod implemented into returned dataframes

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