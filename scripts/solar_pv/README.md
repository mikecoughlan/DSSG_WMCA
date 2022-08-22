# Estimating Solar PV Output
Solar PV output is determined by the amount of solar radiation a module receives, influenced by the weather (cloud cover, temperature, shade etc.), and the efficiency of the module. The [Microgeneration Certification Scheme Service](https://mcscertified.com/) (MCS) creates and maintains the standards for low-carbon products and installations used to produce electricity and heat from renewable sources in the UK. It has data on all the solar PV arrays installed in the UK with their models and estimated yearly solar PV output, calculated from a formula using the (1) shading on the roof, (2) roof slope, (3) roof azimuth or orientation towards the Sun and (4) number of solar panels that can be installed.

The problem is that determining these four input variables requires a site visit. Instead, the following script derives these values from other data source and compute the estimated solar PV output. We will focus our scope on residential buildings in the West Midlands. For more details, see our in-depth documentation.

### Data
- Ordinance Survey Data (Building Height Attribute, OS Master Map Topography, AddressBase Premium)
- [LIDAR Composite DSM 1m](https://environment.data.gov.uk/DefraDataDownload/?Mode=survey)
- `02_calc_pv_output`: [MCS Irradiance Dataset](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&ved=2ahUKEwi2upKmosv5AhWTiFwKHRy2CSAQFnoECBIQAQ&url=https%3A%2F%2Fmcscertified.com%2Fwp-content%2Fuploads%2F2019%2F08%2FIrradiance-Datasets.xlsx&usg=AOvVaw27Q48eb99hbZqKVtBAbKzr)
- `03_test_pv_output`: MCS Baseline data on solar PV installations and estimated output

### Installations
- `01_calc_shadow`: [QGIS 3.26](https://www.qgis.org/en/site/forusers/download.html)
- `01_calc_shadow`: [UMEP](https://umep-docs.readthedocs.io/en/latest/) plugin on QGIS (Plugins > Manage and Install Plugins… and search for UMEP for Processing)
- `01_calc_shadow`: UMEP has certain dependencies on Python so if you run into any issues check [here](https://umep-docs.readthedocs.io/projects/tutorial/en/latest/Tutorials/PythonProcessing1.html?highlight=dependencies).
- `02_calc_pv_output`: [pvlib](https://pvlib-python.readthedocs.io/en/stable/user_guide/package_overview.html) Python library
- `02_calc_pv_output`: [geopandas](https://geopandas.org/en/stable/getting_started/install.html) Python library

### Folder structure
solar pv
├── 01_calc_shadow              
│   ├── temp	                  # Auto-created to store temp files
│   ├── output	                # Auto-created to store outputs
│   │   ├── roof_segments	    
│   │   ├── roof_segments_unfiltered
│   │   └── no_DSM
│   ├── shading_with_DSM.py	    # Roof segmentation & shading
│   └── shading_without_DSM.py	# Pseudo-DSM & shading
├── 02_calc_pv_output           # PV output estimates
│   ├── output                  # Stores csv outputs
│   ├── MCS_output.py	
│   └── pvlib_output.py
└── 03_test_pv_output					
    └── pv_test_set.ipynb      

### Setup
`01_calc_shadow`
1. Follow the [instructions](https://www.qgistutorials.com/en/docs/running_qgis_jobs.html) from step 14-17 to set up the paths.
2. Change `ROOT_DIR` in `CONFIG` to the folder which holds the three main folders with the OS data
3. Run `launch.bat` from the OSGeo Shell (for Windows, other OS might need different setups)

`02_calc_pv_output`
1. Run Python script

`03_test_pv_output`
1. Run to compare estimations for solar PV output
