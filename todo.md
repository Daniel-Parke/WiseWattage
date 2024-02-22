# WiseWattage - Energy Modelling Application

## **NOTE:** These entries are not guarantees for features, for more details in that area see the readme documentation. This is more of a running diary to track what ideas I want to test, or implementations that need finished/added to later. 


#### **TO DO LIST:**
###### **HIGH PRIOIRTY**
- Add wind speed into data returned to dataframes
- Align SolarPV class variables with minimum details on the worst PV panel technical sheet, analyse data truly required for range of modelling.
- Implement low irradiance losses

###### **MEDIUM PRIOIRTY**
- Complete and finalise wind turbine modelling & datasets
- Update documentation for all formulae and methods used
- Formula variable type declarations to be added to functions

###### **LOW PRIOIRTY**.
- Implement one-diode modelling method and compare to current methodology.
- Consider spectral losses.
- Implement full physical solar PV model, more to benchmark performance than model timeseries results.


<br><br>

#### **CURRENTLY BROKEN, NEEDS FIXED:**
###### **HIGH PRIOIRTY**
- Update consistency in column names across different areas data is returned
- Update Radiation returned values to reflect m2 value, currently it is sum of all array area


###### **MEDIUM PRIOIRTY**


###### **LOW PRIOIRTY**
- Rename columns and class variables to simplify and make easier to read/interpret
