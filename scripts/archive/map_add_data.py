import pandas as pd
import geopandas as gpd
import pickle
from glob import glob
from pathlib import Path

# Post code to LSOA to MSOA converting data
# Retrieved from https://geoportal.statistics.gov.uk/datasets/ons-uprn-directory-august-2022/about
PCD_LSOA_MSOA_PATH = "data\external\ONSUD_AUG_2022_WM.csv"
pcd_lsoa_msoa_df = pd.read_csv(PCD_LSOA_MSOA_PATH, low_memory=False, encoding='latin-1')

# Filter for local authorities in WMCA
with open("data\\raw\WMCA_council_code.pkl", 'rb') as f:
    WMCA_code = pickle.load(f)
pcd_lsoa_msoa_df = pcd_lsoa_msoa_df[pcd_lsoa_msoa_df['LAD21CD'].isin(WMCA_code)]

# Rename and select columns to keep
keep_col = ['UPRN', 'PCDS', 'lsoa11cd', 'msoa11cd', 'LAD21CD']
col_names = ['uprn', 'postcode', 'lsoa_code', 'msoa_code', 'local-authority']
pcd_lsoa_msoa_df = pcd_lsoa_msoa_df[keep_col]
pcd_lsoa_msoa_df = pcd_lsoa_msoa_df.rename(columns=dict(zip(keep_col,col_names)))

# Load fuel poverty data
# Retrieved from https://www.gov.uk/government/statistics/sub-regional-fuel-poverty-data-2022 
FUEL_POVERTY_PATH = "data\external\sub-regional-fuel-poverty-2022-tables.xlsx"
fuel_poverty_df = pd.read_excel(FUEL_POVERTY_PATH, sheet_name="Table 3", header=2)
fuel_poverty_df.drop(columns=["LSOA Name", "LA Code", "LA Name", "Region"], inplace=True)
fuel_poverty_df.columns = ["lsoa_code", "num_households", "num_households_fuel_poverty", "prop_households_fuel_poor"]

OUTPUT_DIR = "data\\processed\\proxy\\"
PROXY_DATA_DIR = "C:\Users\lilia\Documents\GitHub\WMCA\scripts\output_osmp\SJ9000_results.gml"
PROXY_DATA_FILES = glob(PROXY_DATA_DIR + '*.gml')

for path in PROXY_DATA_FILES:
    filename = Path(path).stem
    proxy_df = pd.DataFrame(gpd.read_file(path, driver='GML'))

    # Map LSOA, MSOA and LA to postcode
    for col in col_names[1:]:
        mapping = dict(zip(pcd_lsoa_msoa_df['postcode'], pcd_lsoa_msoa_df[col]))
        proxy_df[col] = proxy_df['postcode'].map(mapping)

    # Merge data to get postcodes associated with each LSOA code
    proxy_df = pd.merge(proxy_df, fuel_poverty_df, on="lsoa_code", how="left")
    
    print(f"{OUTPUT_DIR}{filename}.csv")
    proxy_df.to_csv(f"{OUTPUT_DIR}{filename}.csv", index=False)