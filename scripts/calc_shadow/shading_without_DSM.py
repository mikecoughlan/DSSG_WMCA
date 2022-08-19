# Run on OSGeo Shell
from qgis.core import *
from PyQt5.QtCore import QDate, QTime
import sys

# Initialize QGIS Application
QgsApplication.setPrefixPath("C://OSGeo4W64//apps//qgis", True)
app = QgsApplication([], True)
app.initQgis()

# Add the path to Processing framework
sys.path.append('C://Program Files//QGIS 3.24.3//apps//qgis//python//plugins')
sys.path.append('C://Users//lilia//AppData//Roaming//QGIS//QGIS3//profiles//default//python//plugins')

# Import UMEP
from processing_umep.processing_umep_provider import ProcessingUMEPProvider
umep_provider = ProcessingUMEPProvider()
QgsApplication.processingRegistry().addProvider(umep_provider)

# Import and initialize Processing framework
from processing.core.Processing import Processing
Processing.initialize()

import processing
from osgeo import gdal
from osgeo.gdalconst import *

import pandas as pd
from glob import glob
import shutil
import os
from pathlib import Path
import time

from shading_with_DSM import CalculateShading

class ApproximateShading(CalculateShading):
    def __init__(self, HOUSE_SHP_PATH, crs='EPSG:27700'):
        self.PROJECT_CRS = QgsCoordinateReferenceSystem(crs)
        self.ROOT_DIR = os.getcwd() + "//"
        
        self.TEMP_PATH = self.ROOT_DIR + "temp//"
        if not os.path.isdir(self.TEMP_PATH):
            os.makedirs(self.TEMP_PATH)
        self.clear_temp_folder()
          
        self.HOUSE_SHP_PATH = HOUSE_SHP_PATH
        self.extent = self.extract_extent(self.HOUSE_SHP_PATH)
        self.tile_name = Path(HOUSE_SHP_PATH).stem
        print(self.tile_name)     
        
    def build_pseudo_DSM(self, building_height, topology_area):
        print("Building pseudo DSM...")
        start = time.time()

        building_df = pd.read_csv(
            building_height, 
            header=None,
            names= ['fid','OS_TOPO_TOID_VERSION','BHA_ProcessDate','TileRef', 'AbsHMin', 'AbsH2','AbsHMax','RelH2','RelHMax','BHA_Conf'],
            on_bad_lines='skip'
            )
        building_df.head()
        building_df = building_df[['fid', 'AbsHMax']]

        building_df.to_csv(self.TEMP_PATH + 'building_height.csv', index=False)

        params = {
            'INPUT':self.TEMP_PATH + 'building_height.csv',
            'FIELDS_MAPPING':[{'expression': '"fid"','length': 0,'name': 'fid','precision': 0,'sub_type': 0,'type': 10,'type_name': 'text'},{'expression': '"AbsHMax"','length': 0,'name': 'AbsHMax','precision': 0,'sub_type': 0,'type': 6,'type_name': 'double precision'}],
            'OUTPUT':'TEMPORARY_OUTPUT'
            }
        building_refactored = processing.run("native:refactorfields", params)

        params = {
            'INPUT': topology_area + '|layername=TopographicArea',
            'FIELD':'fid',
            'INPUT_2':building_refactored['OUTPUT'],
            'FIELD_2':'fid',
            'FIELDS_TO_COPY':['AbsHMax'],
            'METHOD':1,
            'DISCARD_NONMATCHING':True,
            'PREFIX':'',
            'OUTPUT':self.TEMP_PATH + 'pseudo_DSM.geojson'
            }
        
        output = processing.run("native:joinattributestable", params)

        self.pseudo_DSM = self.rasterize(output['OUTPUT']) 

        end = time.time()
        print(f"Completed building pseudo DSM in {end-start}s")

        return self.pseudo_DSM
    
    def extract_extent(self, layer):
        ext = QgsVectorLayer(layer, '', 'ogr' ).extent()
        ext = f"{ext.xMinimum()},{ext.xMaximum()},{ext.yMinimum()},{ext.yMaximum()} [EPSG:27700]"
        return ext

    def rasterize(self, layer):
        """
        Convert vector layer to pseudo DSM raster layer.

        Input
        layer(str): Path to vector layer

        Output
        (str): Path to raster layer
        """
        print('Rasterising vector layer...')
        start = time.time()

        params = {
            'INPUT':layer,
            'FIELD':'AbsHMax',
            'BURN':0,
            'USE_Z':False,
            'UNITS':1,
            'WIDTH':0.5,
            'HEIGHT':0.5,
            'EXTENT':self.extent,
            'NODATA':0,
            'OPTIONS':'',
            'DATA_TYPE':5,
            'INIT':0,
            'INVERT':False,
            'EXTRA':'',
            'OUTPUT':'TEMPORARY_OUTPUT'
            }

        output = processing.run("gdal:rasterize", params)

        end = time.time()
        print(f"Completed rasterising in {end-start}s")

        return output['OUTPUT']

    def filter_houses(self, filename):
        """
        Calculate shading on rooftops and export as vector layer.

        Input:
        layer(str): Path to vector layer

        Output:
        (str): Path to merged vector layer
        
        """
        output_path = self.ROOT_DIR + 'output//no_DSM_output//'
        if not os.path.isdir(output_path):
            os.makedirs(output_path)

        shading_stats = self.calculate_shading(self.pseudo_DSM, self.HOUSE_SHP_PATH)
        merged = self.merge_vector_layers(
            self.HOUSE_SHP_PATH, 
            shading_stats, 
            ['shading_mean'],
            output_path + f"{filename}.geojson"
            )

        return merged


def main(): 
    TOPOLOGY_DIR = ""
    BUILDING_HEIGHT_DIR = ""
    BUILDING_FOOTPRINT_DIR = ""

    # footprint_files = glob(BUILDING_FOOTPRINT_DIR + "*.gml")
    footprint_files = ["C://Users//lilia//Documents//GitHub//WMCA//DSSG_WMCA//data//external//output//SJ9000.gml"]

    for path in footprint_files:
        filename = Path(path).stem
        # topology_path = TOPOLOGY_DIR + f"5882272-{filename}.gml"
        # building_path = BUILDING_HEIGHT_DIR + f"{filename}.csv"
        topology_path = "C://Users//lilia//Documents//GitHub//WMCA//DSSG_WMCA//data//external//topology//5882272-SJ9000.gml"
        building_path = "C://Users//lilia//Documents//GitHub//WMCA//DSSG_WMCA//data//external//building_height//SJ9000.csv"
        program = ApproximateShading(path)
        program.build_pseudo_DSM(building_path, topology_path)
        program.filter_houses(filename)

if __name__ == "__main__":
    main()