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
    col_names = ['uprn', 'postcode', 'lsoa_code', 'msoa_code', 'local-authority']
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

    return pcd_lsoa_msoa_df, fuel_poverty_df

def map_add_info(gdf, filename, pcd_lsoa_msoa_df, fuel_poverty_df, ROOT_DIR):
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

    # Merge data to get postcodes associated with each LSOA code
    gdf = gdf.merge(fuel_poverty_df, on="lsoa_code", how="left")

    gdf.to_file(f"{OUTPUT_DIR}{filename}.gml", driver='GML')
    print(f"{OUTPUT_DIR}{filename}.gml")
    end = time.time()
    print(f"Completed adding info in {end-start}s")
    
    return gdf

def encode_var(gdf, filename, ROOT_DIR, nunique_limit=20):
    """
    Encode non-numeric variables for model training. Exported as csv.

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
    
    non_numeric_col = ['postcode', 'lsoa_code', 'msoa_code', 'local-authority']
    
    for col in non_numeric_col:
        if len(gdf[col].unique())<nunique_limit:
            one_hot_encoded = pd.get_dummies(gdf[col])
            gdf[col] = one_hot_encoded.to_numpy().tolist()
        else:
            label_encoder = LabelEncoder()
            gdf[col] = label_encoder.fit_transform(gdf[col])

    df = pd.DataFrame(gdf.drop(columns=['geometry']))
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

    for i in range(len(landbaseprem_files)):
        merged_gdf = merge_os_files(landbaseprem_files[i], topology_files[i], building_height_files[i])
        merged_gdf.head()
        filename = Path(landbaseprem_files[i]).stem
        final_gdf = map_add_info(merged_gdf, filename, pcd_lsoa_msoa_df, fuel_poverty_df, ROOT_DIR)
        
        encode_var(final_gdf, filename, ROOT_DIR)

if __name__ == "__main__":
    main()


