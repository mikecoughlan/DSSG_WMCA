# Run on OSGeo Shell
from qgis.core import *
import sys
import pandas as pd
import geopandas as gpd
import pickle
import shutil

# Initialize QGIS Application
QgsApplication.setPrefixPath("C:\\OSGeo4W64\\apps\\qgis", True)
app = QgsApplication([], True)
app.initQgis()

# Add the path to Processing framework
sys.path.append(
    'C:\\Program Files\\QGIS 3.24.3\\apps\\qgis\\python\\plugins')

# Import and initialize Processing framework
from processing.core.Processing import Processing
Processing.initialize()

from glob import glob
import processing
import os
import time
from pathlib import Path

class MergeOSMaps():
    def __init__(self, ROOT_DIR):
        self.ROOT_DIR = ROOT_DIR
        if not os.path.isdir(self.ROOT_DIR + "temp\\"):
            os.makedirs(self.ROOT_DIR + "temp\\")
        self.TEMP_PATH = self.ROOT_DIR + "temp\\" 

        self.OUTPUT_DIR = 'data\\output\\proxy_data\\'

        building_height_dir = self.ROOT_DIR + "building_height\\"
        self.building_height_files = glob(building_height_dir + '*.csv')

        landbaseprem_dir = self.ROOT_DIR + "landbaseprem\\"
        self.landbaseprem_files = glob(landbaseprem_dir+"*.gml")

        topology_dir = self.ROOT_DIR + "topology\\"
        self.topology_files = glob(topology_dir+"*")
    
    def clear_temp_folder(self):
        "Clear temp folder"
        print("Clearing temp folder...")
        shutil.rmtree(self.TEMP_PATH)
        os.makedirs(self.TEMP_PATH)
        print("Temp folder cleared.")

    def filter_topography(self, layer):
        """
        Remove columns and filter topography layer for buildings.

        Input:
        layer(str): Path to Topographic Area layer

        Output:
        (str): Path to refactored layer
        """
        print('Filtering layer...')
        start = time.time()

        params = {
            'INPUT': layer,
            'FIELD':'featureCode',
            'OPERATOR':0,
            'VALUE':'10021',
            'OUTPUT': 'TEMPORARY_OUTPUT'
            }
        
        output = processing.run("native:extractbyattribute", params, None) 

        end = time.time()
        print(f"Completed filtering in {end-start}s")
        return output['OUTPUT']

    def retain_columns(self, layer, fields):
        """
        Select fields in vector layer to retain.

        Input
        layer(str): Path to vector layer
        fields(list): List of fields to retain

        Output
        (str): Path to output vector layer
        
        """
        print('Retaining columns...')
        start = time.time()
        params = {
            'INPUT':layer,
            'FIELDS':fields,
            'OUTPUT':'TEMPORARY_OUTPUT'
            }

        output = processing.run("native:retainfields", params)     

        end = time.time()
        print(f"Completed retaining columns in {end-start}s")
        return output['OUTPUT']
    
    def set_projection(self, layer):
        """
        Remove columns and set projection for landbaseprem to EPSG:4326.

        Input:
        layer(str): Path to landbaseprem 'BasicLandPropertyUnit' layer.

        Output:
        (str): Path to layer with set projection.
        """
        print('Setting projection...')
        start = time.time()
    
        projected_params = {
            'INPUT': output['OUTPUT'],
            'CRS':QgsCoordinateReferenceSystem('EPSG:4326'),
            'OUTPUT': 'TEMPORARY_OUTPUT',
            }
        
        output = processing.run("native:assignprojection", projected_params)

        end = time.time()
        print(f'Completed setting projection in {end-start}s')
        
        return output['OUTPUT']

    def join_table(self, polygon_layer, point_layer, building_table):
        """
        Join attribute table of point layer to polygon layer.

        Input:
        polygon_layer(str): Path to layer with polygons of houses
        point_layer(str): Path to layer with points of houses

        Output:
        (str): Path to polygon layer with joined attributes from point layer
        """
        print('Joining parameters...')
        start = time.time()

        tile = Path(building_table).stem

        if not os.path.isdir(self.BASE_DIR + "output\\"):
            os.makedirs(self.BASE_DIR + "output\\")

        # Add headers
        building_table_file = pd.read_csv(
            building_table, 
            names= ['OS_TOPO_TOID','OS_TOPO_TOID_VERSION','BHA_ProcessDate','TileRef', 'AbsHMin', 'AbsH2','AbsHMax','RelH2','RelHMax','BHA_Conf']
            )
        building_table_file.to_csv(building_table, index=False)

        params = {
            'INPUT':building_table,
            'FIELDS_MAPPING':[{'expression': '"OS_TOPO_TOID"','length': 0,'name': 'OS_TOPO_TOID','precision': 0,'sub_type': 0,'type': 10,'type_name': 'text'},{'expression': '"OS_TOPO_TOID_VERSION"','length': 0,'name': 'OS_TOPO_TOID_VERSION','precision': 0,'sub_type': 0,'type': 10,'type_name': 'text'},{'expression': '"BHA_ProcessDate"','length': 0,'name': 'BHA_ProcessDate','precision': 0,'sub_type': 0,'type': 10,'type_name': 'text'},{'expression': '"TileRef"','length': 0,'name': 'TileRef','precision': 0,'sub_type': 0,'type': 10,'type_name': 'text'},{'expression': '"AbsHMin"','length': 0,'name': 'AbsHMin','precision': 0,'sub_type': 0,'type': 6,'type_name': 'double precision'},{'expression': '"AbsH2"','length': 0,'name': 'AbsH2','precision': 0,'sub_type': 0,'type': 6,'type_name': 'double precision'},{'expression': '"AbsHMax"','length': 0,'name': 'AbsHMax','precision': 0,'sub_type': 0,'type': 6,'type_name': 'double precision'},{'expression': '"RelH2"','length': 0,'name': 'RelH2','precision': 0,'sub_type': 0,'type': 6,'type_name': 'double precision'},{'expression': '"RelHMax"','length': 0,'name': 'RelHMax','precision': 0,'sub_type': 0,'type': 6,'type_name': 'double precision'},{'expression': '"BHA_Conf"','length': 0,'name': 'BHA_Conf','precision': 0,'sub_type': 0,'type': 10,'type_name': 'text'}],
            'OUTPUT':'TEMPORARY_OUTPUT'
            }
        
        output = processing.run("native:refactorfields", params)

        params = {
            'INPUT':polygon_layer,
            'FIELD':'fid',
            'INPUT_2':output['OUTPUT'],
            'FIELD_2':'OS_TOPO_TOID',
            'FIELDS_TO_COPY':['RelH2','RelHMax', 'AbsHMin', 'AbsHMax', 'AbsH2'],
            'METHOD':1,
            'DISCARD_NONMATCHING':True,
            'PREFIX':'',
            'OUTPUT':'TEMPORARY_OUTPUT'
            }
        
        output = processing.run("native:joinattributestable", params)
        
        # Set time
        join_params = {
            'INPUT':output['OUTPUT'],
            'PREDICATE':[1],
            'JOIN':point_layer,
            'JOIN_FIELDS' : [],
            'METHOD':0,
            'DISCARD_NONMATCHING':True,
            'PREFIX':'',
            'OUTPUT': self.TEMP_DIR + "output\\" + tile + ".gml" 
            }
        
        layer = processing.run("native:joinattributesbylocation", join_params)
        
        end = time.time()
        print(f'Completed joining tables in {end-start}s')

        return layer['OUTPUT']

    def load_info(self):
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
        self.col_names = ['uprn', 'postcode', 'lsoa_code', 'msoa_code', 'local-authority']
        pcd_lsoa_msoa_df = pcd_lsoa_msoa_df[keep_col]
        pcd_lsoa_msoa_df = pcd_lsoa_msoa_df.rename(columns=dict(zip(keep_col,self.col_names)))

        # Load fuel poverty data
        # Retrieved from https://www.gov.uk/government/statistics/sub-regional-fuel-poverty-data-2022 
        FUEL_POVERTY_PATH = "data\external\sub-regional-fuel-poverty-2022-tables.xlsx"
        fuel_poverty_df = pd.read_excel(FUEL_POVERTY_PATH, sheet_name="Table 3", header=2)
        fuel_poverty_df.drop(columns=["LSOA Name", "LA Code", "LA Name", "Region"], inplace=True)
        fuel_poverty_df.columns = ["lsoa_code", "num_households", "num_households_fuel_poverty", "prop_households_fuel_poor"]

        return pcd_lsoa_msoa_df, fuel_poverty_df
    
    def map_add_info(self, path):
        """
        Add LSOA, MSOA and local authority code, and fuel poverty data.

        Input
        path(str): Path to joined OS layer
        """
        pcd_lsoa_msoa_df, fuel_poverty_df = self.load_info()

        filename = Path(path).stem
        proxy_df = pd.DataFrame(gpd.read_file(path, driver='GML'))
        proxy_df.drop(columns=['gml_id', 'fid'], inplace=True)

        # Map LSOA, MSOA and LA to postcode
        for col in list(pcd_lsoa_msoa_df.columns)[1:]:
            mapping = dict(zip(pcd_lsoa_msoa_df['postcode'], pcd_lsoa_msoa_df[col]))
            proxy_df[col] = proxy_df['postcode'].map(mapping)

        # Merge data to get postcodes associated with each LSOA code
        proxy_df = pd.merge(proxy_df, fuel_poverty_df, on="lsoa_code", how="left")
        proxy_df = gpd.GeoDataFrame(proxy_df, geometry='geometry')

        print(f"{self.OUTPUT_DIR}{filename}.gml")
        proxy_df.to_file(f"{self.OUTPUT_DIR}{filename}.gml", driver='GML')


def main():
    """
    Assign all homes in AddressBasePremium to its building footprint from OSMap Topography and building height from OSMap Building Height Attribute. Building shapefiles saved in 'output' as .gml
    """
    ROOT_DIR = "C:\\Users\\lilia\\Downloads\\wmca_download_2022-07-29_10-07-36\\files\\wmca_prj\\project\\unzip_files\\"
    program = MergeOSMaps(ROOT_DIR)

    for i in range(len(program.landbaseprem_files)):
        program.clear_temp_folder()

        filtered_topology = program.filter_topography(program.topology_files[i] + '|layername=TopographicArea')
        topology_fields = ['fid','calculatedAreaValue']
        topology_layer = program.retain_columns(filtered_topology, topology_fields)

        addressbase_fields = ['gml_id','uprn','thoroughfare','postTown','postcode','level']
        addressbase_layer = program.retain_columns(program.landbaseprem_files[i]+ '|layername=BasicLandPropertyUnit', addressbase_fields)
        projected_addressbase = program.set_projection(addressbase_layer)
        
        joined_layer = program.join_table(topology_layer, projected_addressbase, program.building_height_files[i])

        # program.map_add_info(joined_layer)

if __name__ == "__main__":
    main()


