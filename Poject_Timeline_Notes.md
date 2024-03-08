# Wattsun and Homes - Energy Modelling Application

### **NOTE:** These notes will not have any data entries removed unless it is to improve readability. These are to serve as a time capsule of sorts to record development progress and program improvements over time. 

#### As such, certain class methods or code snippets may become defunct as the project progresses. For specific and up-to-date functionality, please see specific readme files and support documentation for updated methodology on package usage.

###### **Author Note (17/02/2024 @ 23:54 GMT+00:00):** <br><br> At the time of this project's inception, I have just completed CS50x and have been "regularly" programming following online tutorials for around 6 months. I thought it would be a good idea to record my progress and (hopefully) improvements as the project develops. The idea for this project came from my own personal work as a student/energy researcher, and my final project for CS50x where I made a Solar PV calculator. The original code that contains those functions can be seen [here](/old_files/original_energy_modelling.py) if you are interested. <br><br>                                                                                                      If you have any feedback, or there are any features that you would like implemented please let me know. The more challenging and/or interesting it is, the more likely it will be that I will give it a try! 

## Patch/Progress Notes:
### **07/03/2024:**
- Updated PV_only methods to save site models within nested dictionary

### **25/02/2024:**
- Implemented overarching project class, PV_Only implemented
- Enabled modelling of multiple sites, for multiple arrays
- Updated logging to store more information and record modelling progress more accurately
- Updated PV_Only modelling approach to allow sensitivity analysis of nested list of array list, as well as sites.

### **24/02/2024:**
- Added project model to build template for hybrid system
- Removed ability to save pickle object of models, also removed export to HTML
- Added attributes to models stored in dictionaries for ".xxx" retrieval

### **23/02/2024:**
- Updated codebase, combined functionality where possible.
- Added column for combined total PV losses, made edits to existing functions to work with new columns

### **23/02/2024:**
- Updated docstrings, added documentation notes throughout functions, added typehints to all functions.
- Further function caching, vectorisation and optimisation done to reduce modelling time by 20%. Model of 5 arrays went from original time of 100ms, to 80ms. Timed on local machine, user results may vary

### **22/02/2024:**
- Updated base units to calculate radiation values in W, with only PV Gen, Thermal & Low light losses
being returned as kWh.
- Add SolarPVPanel class to further differentiate performance, updated model structure to incorporate this
- Removed redundant variables no longer being used
- Finalised Sandia array temperature model, adjusted module dataset to align with parameters required
- Added low light loss to solar PV model calculations
- Updated all loss calculations to return kWh values, and aggregated these during result summaries

### **21/02/2024:**
- Refined array temperature calculations, implemented Sandia array temperature model to incorporate
wind speeds into array temperature calculations.
- Added low light losses utilising sigmoid function
- Further added reference documentation and added static images for forumulae.

### **20/02/2024:**
- Rearranged class structure to store variables at most relevant stage (I.e. Timestep moved to site so TMY data can be matched accordingly, albedo moved to array to enable sensitivity analysis of variable at same location)
- Added range of Array temperature calculations to include models from Sandia, Homer, PVSyst and Faiman.
- Added to documentation and static images, mainly a range of supporting documentation for formulae used.


### **17/02/2024:**
- Changed caching process so that `get_jrc_tmy()` is now cached by wrapper function `@cached_func` which can be used elsewhere. This was previously done during Site initialisation.

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


### **System Design Schematic Concepts already created:**
#### **Design 1:**
![alt text](<static/Solar_Model_Technical_Diagram_2.png>)

<br>

#### **Design 2:**
![alt text](<static/Solar_Model_Technical_Diagram_1.png>)