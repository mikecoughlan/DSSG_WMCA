# Pull Data


## Getting proxies
The following creates the data required to predicting EPC ratings, estimate solar PV output and determine heat pump capacity. The final output from the process outlined in this document will be a series of .geojson files while another set of files will be encoded and saved as .csv for model training. For more details see [here]().

### Data
1. [OS MasterMap Topography Layer](https://www.ordnancesurvey.co.uk/business-government/products/mastermap-topography): building footprints (format: `5882272-{tilename}.gml`)
2. [OS MasterMap Building Height Attribute](https://www.ordnancesurvey.co.uk/business-government/products/mastermap-building): heights of each building (format: `{tilename}.csv`)
3. [AddressBase Premium](https://www.ordnancesurvey.co.uk/business-government/products/addressbase-premium): address data for each house (format: `{tilename}.gml`)
4. [ONS UPRN Directory](https://geoportal.statistics.gov.uk/datasets/ons-uprn-directory-august-2022/about) (August 2022 West Midlands)
5. [Sub-regional fuel poverty data 2022](https://www.gov.uk/government/statistics/sub-regional-fuel-poverty-data-2022)
6. [Lower and Middle Super Output Areas electricity consumption](https://www.gov.uk/government/statistics/lower-and-middle-super-output-areas-electricity-consumption)


## Folder structure
```bash
data
└── external   
    ├── building_height	        
    ├── landbaseprem	            
    ├── topology	                  
    ├── ONSUD_AUG_2022_WM.csv    
    ├── LSOA_domestic_elect_2010-20.xlsx
    └── sub-regional-fuel-poverty-2022-tables.xlsx   
pull_data
    ├── get_EPC.py	                  
    └── get_proxies.py	
    
```

## Installation
- [geopandas](https://geopandas.org/en/stable/getting_started/install.html) Python library to read geospatial data
- [openpyxl](https://openpyxl.readthedocs.io/en/stable/) Python library to read Excel worksheets

## Setup
1. Set `ROOT_DIR` in `getting_proxies.py` to the main folder with all the OS Master Map data.
2. The folder with the topology files don’t have an extension so add .gml to them, using
```python
for path in glob.glob(TOPOLOGY_DIR+’*’):
  os.rename(path, path + ".gml")
```
3. Run the Python script.
