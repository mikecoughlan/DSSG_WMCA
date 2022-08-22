# Estimating Solar PV Output

### Data
- Merged OSMap data
- (LIDAR Composite DSM 1m)[https://environment.data.gov.uk/DefraDataDownload/?Mode=survey]
- `02_calc_pv_output`: (MCS Irradiance Dataset)[https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&ved=2ahUKEwi2upKmosv5AhWTiFwKHRy2CSAQFnoECBIQAQ&url=https%3A%2F%2Fmcscertified.com%2Fwp-content%2Fuploads%2F2019%2F08%2FIrradiance-Datasets.xlsx&usg=AOvVaw27Q48eb99hbZqKVtBAbKzr]
- `03_test_pv_output`: MCS Baseline data on solar PV installations and estimated output

### Installations
- (QGIS 3.26)[https://www.qgis.org/en/site/forusers/download.html]
- (UMEP)[https://umep-docs.readthedocs.io/en/latest/] plugin on QGIS (Plugins > Manage and Install Plugins… and search for UMEP for Processing)
-

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
├── 03_test_pv_output					
│   └── pv_test_set.ipynb      
└── launch.bat

