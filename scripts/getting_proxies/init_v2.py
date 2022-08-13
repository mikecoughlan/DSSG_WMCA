import pandas as pd
import geopandas as gpd
from glob import glob
from pathlib import Path
import time
import os
import numpy as np
from sklearn.preprocessing import LabelEncoder
    
def merge_os_files(addressbase_path, topology_path, building_path):
    """
    Merge all AddressBase Premium, OS Master Map Topology and OS Master Map Building Height Attribute

    Input
    addressbase_path(str): Path to AddressBase tile
    topology_path(str): Path to Topology tile
    building_path(str): Path to building height csv

    Output
    merged_gdf(GeoDataFrame): Merged dataframe with building polygons
    """
    print("Merging files...")
    start = time.time()

    addressbase_gdf = gpd.read_file(addressbase_path, layer='BasicLandPropertyUnit', driver="GML")
    addressbase_gdf = addressbase_gdf[['uprn','postcode', 'geometry']]
    # Original coordinates given in European Terrestrial Reference System 89 (ETRS89)
    addressbase_gdf = addressbase_gdf.set_crs('epsg:4258', allow_override=True)
    addressbase_gdf = addressbase_gdf.to_crs('epsg:27700')

    topology_gdf = gpd.read_file(topology_path, layer='TopographicArea', driver="GML")
    topology_gdf = topology_gdf[topology_gdf['featureCode'] == 10021]
    topology_gdf = topology_gdf[['fid', 'geometry', 'calculatedAreaValue']]

    building_df = pd.read_csv(
        building_path, 
        header=None, 
        names=['fid','OS_TOPO_TOID_VERSION','BHA_ProcessDate','TileRef', 'AbsHMin', 'AbsH2','AbsHMax','RelH2','RelHMax','BHA_Conf']
        )
    building_df = building_df[['fid', 'RelH2','RelHMax', 'AbsHMin', 'AbsHMax', 'AbsH2']]

    topology_gdf = topology_gdf.merge(building_df, on='fid')

    merged_gdf = gpd.sjoin(addressbase_gdf, topology_gdf, how="left")
    merged_gdf = merged_gdf[merged_gdf['index_right'].notna()]
    merged_gdf['geometry'] = merged_gdf['index_right'].apply(lambda col: topology_gdf['geometry'][col])
    merged_gdf.drop(columns=['index_right'], inplace=True)

    end = time.time()
    print(f"Completed merging data in {end-start}s")
    return merged_gdf


def load_info():
    """
    Load all fuel poverty and mapping data.

    Output
    pcd_lsoa_msoa_df(DataFrame): Mapping data for postcodes in the West Midlands
    fuel_poverty_df(DataFrame): Fuel poverty data in the West Midlands
    
    """
    # Post code to LSOA to MSOA converting data
    # Retrieved from https://geoportal.statistics.gov.uk/datasets/ons-uprn-directory-august-2022/about
    PCD_LSOA_MSOA_PATH = "data\external\ONSUD_AUG_2022_WM.csv"
    pcd_lsoa_msoa_df = pd.read_csv(PCD_LSOA_MSOA_PATH, low_memory=False, encoding='latin-1')

    # Filter for local authorities in WMCA
    WMCA_code = ['E08000025', 'E08000031', 'E08000026', 'E08000027', 'E08000028', 'E08000029', 'E08000030', 'E07000192', 'E07000218', 'E07000219', 'E07000236', 'E07000220', 'E06000051', 'E07000221', 'E07000199', 'E06000020', 'E07000222']
    pcd_lsoa_msoa_df = pcd_lsoa_msoa_df[pcd_lsoa_msoa_df['LAD21CD'].isin(WMCA_code)]

    # Rename and select columns to keep
    keep_col = ['UPRN', 'PCDS', 'lsoa11cd', 'msoa11cd', 'LAD21CD']
    col_names = ['uprn', 'postcode', 'lsoa_code', 'msoa_code', 'local-authority', 'constituency']
    pcd_lsoa_msoa_df = pcd_lsoa_msoa_df[keep_col]
    pcd_lsoa_msoa_df = pcd_lsoa_msoa_df.rename(columns=dict(zip(keep_col,col_names)))

    # Load fuel poverty data
    # Retrieved from https://www.gov.uk/government/statistics/sub-regional-fuel-poverty-data-2022 
    FUEL_POVERTY_PATH = "data\external\sub-regional-fuel-poverty-2022-tables.xlsx"
    fuel_poverty_df = pd.read_excel(FUEL_POVERTY_PATH, sheet_name="Table 3", header=2)
    fuel_poverty_df.drop(columns=["LSOA Name", "LA Code", "LA Name", "Region"], inplace=True)
    fuel_poverty_df.columns = ["lsoa_code", "num_households", "num_households_fuel_poverty", "prop_households_fuel_poor"]
    
    # Remove bottom text rows
    crop_idx = np.where(fuel_poverty_df.isna().sum(axis=1) == len(fuel_poverty_df.columns))[0][0]
    fuel_poverty_df = fuel_poverty_df[:crop_idx-1]

    # Load energy consumption data
    ENERGY_CONSUMP_PATH = "data\external\sub-regional-fuel-poverty-2022-tables.xlsx"
    energy_consump_df = pd.read_excel(ENERGY_CONSUMP_PATH, sheet_name="2020", header=4)
    energy_consump_df.columns = [
        'local-authority', 'la', 'msoa_code', 'msoa', 'lsoa_code', 'lsoa', 'num_meter', 'total_consumption', 'mean_counsumption', 'median_consumption'
        ]
    energy_consump_df = energy_consump_df[['la_code','msoa_code','lsoa_code', 'total_consumption', 'mean_counsumption', 'median_consumption']]
    energy_consump_df = energy_consump_df[energy_consump_df['la_code'].isin(WMCA_code)]

    return pcd_lsoa_msoa_df, fuel_poverty_df, energy_consump_df

def map_add_info(gdf, filename, pcd_lsoa_msoa_df, fuel_poverty_df, energy_consump_df, ROOT_DIR):
    """
    Add LSOA, MSOA and local authority code, and fuel poverty data.

    Input
    gdf(GeoDataFrame): Merged OS dataframe
    filename(str): Name of saved output file
    """
    print("Adding info...")
    start = time.time()

    OUTPUT_DIR = ROOT_DIR + 'output\\'
    if not os.path.isdir(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    gdf.drop(columns=['fid'], inplace=True)

    # Map LSOA, MSOA and LA to postcode
    for col in list(pcd_lsoa_msoa_df.columns)[1:]:
        mapping = dict(zip(pcd_lsoa_msoa_df['uprn'], pcd_lsoa_msoa_df[col]))
        gdf[col] = gdf['uprn'].map(mapping)

    # Merge with fuel poverty on LSOA code
    gdf = gdf.merge(fuel_poverty_df, on="lsoa_code", how="left")

    # Merge with energy consumption on LSOA code
    energy_consump_df = energy_consump_df[["lsoa_code",'total_consumption', 'mean_counsumption', 'median_consumption']]
    gdf = gdf.merge(energy_consump_df, on="lsoa_code", how="left")

    gdf = gdf.to_crs("epsg:4326")
    gdf.to_file(f"{OUTPUT_DIR}{filename}.geojson", driver='GeoJSON')
    print(f"{OUTPUT_DIR}{filename}.geojson")
    end = time.time()
    print(f"Completed adding info in {end-start}s")
    
    return gdf

def encode_var(gdf, filename, fuel_poverty_avg, energy_consump_df, ROOT_DIR, nunique_limit=20):
    """
    Encode non-numeric variables for model training and fill numeric na with mean. Exported as csv.

    Input
    gdf(GeoDataFrame): Merged dataframe
    filename(str): Name of output file
    ROOT_DIR(str): Current directory
    nunique_limit(int): Threshold of unique values for one-hot encoding
    """
    print("Encoding variables...")
    OUTPUT_DIR = ROOT_DIR + 'encoded_proxy\\'
    if not os.path.isdir(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    df = pd.DataFrame(gdf.drop(columns=['geometry']))

    fuel_poverty_col = ["num_households", "num_households_fuel_poverty", "prop_households_fuel_poor"]
    for col in fuel_poverty_col:
        df[col] = df[col].fillna(fuel_poverty_avg[col])

    energy_consump_col = ['total_consumption', 'mean_counsumption', 'median_consumption']
    hierarchy = ["msoa_code", "local-authority"]

    for col in energy_consump_col:
        for grouped_var in hierarchy:
            if df[col].isna().sum() > 0:
                grouped_df = energy_consump_df.groupby(grouped_var).mean()
                mapping = dict(zip(energy_consump_df[grouped_var], grouped_df[col]))
                idx = energy_consump_df[energy_consump_df[col].isna() == True].index
                df.loc[idx, col] = df.loc[idx, grouped_var].map(mapping)

        if df[col].isna().sum() > 0:
            df[col] = df[col].fillna(energy_consump_df[col].mean())
    
    non_numeric_col = ['postcode', 'lsoa_code', 'msoa_code', 'local-authority']
    
    for col in non_numeric_col:
        if len(gdf[col].unique())<nunique_limit:
            one_hot_encoded = pd.get_dummies(gdf[col])
            one_hot_encoded.columns = [col+"_"+colname for colname in one_hot_encoded.columns]
            df = pd.concat([df, one_hot_encoded], axis=1, index=False)
        else:
            label_encoder = LabelEncoder()
            df[col] = label_encoder.fit_transform(df[col])
    
    df.to_csv(f"{OUTPUT_DIR}{filename}.csv", index=False)
    
    print("Completed encoding variables.")
    return df

def main():
    """
    Assign all homes in AddressBasePremium to its building footprint from OSMap Topography and building height from OSMap Building Height Attribute. Building shapefiles saved in 'output' as .gml
    """
    ROOT_DIR = "C:\\Users\\lilia\\Downloads\\wmca_download_2022-07-29_10-07-36\\files\\wmca_prj\\project\\unzip_files\\"
    building_height_dir = ROOT_DIR + "building_height\\"
    building_height_files = glob(building_height_dir + '*.csv')

    landbaseprem_dir = ROOT_DIR + "landbaseprem\\"
    landbaseprem_files = glob(landbaseprem_dir+"*.gml")

    topology_dir = ROOT_DIR + "topology\\"
    topology_files = glob(topology_dir+"*.gml")

    pcd_lsoa_msoa_df, fuel_poverty_df = load_info()
    fuel_poverty_avg = fuel_poverty_df.mean(axis=0) # impute with national average

    for i in range(len(landbaseprem_files)):
        merged_gdf = merge_os_files(landbaseprem_files[i], topology_files[i], building_height_files[i])
        merged_gdf.head()
        filename = Path(landbaseprem_files[i]).stem
        final_gdf = map_add_info(merged_gdf, filename, pcd_lsoa_msoa_df, fuel_poverty_df, ROOT_DIR)
        
        encode_var(final_gdf, filename, fuel_poverty_avg, ROOT_DIR)

if __name__ == "__main__":
    main()


