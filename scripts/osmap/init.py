# Run on OSGeo Shell
from cgi import test
from lib2to3 import refactor
from multiprocessing.dummy import Process
from qgis.core import *
import sys

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
import pandas as pd

class MergeOSMaps():
    def __init__(self, BASE_DIR):
        self.BASE_DIR = BASE_DIR

        building_height_dir = self.BASE_DIR + "building_height\\"
        self.building_height_files = glob(building_height_dir + '*.csv')

        landbaseprem_dir = self.BASE_DIR + "landbaseprem\\"
        self.landbaseprem_files = glob(landbaseprem_dir+"*.gml")

        topology_dir = self.BASE_DIR + "topology\\"
        self.topology_files = glob(topology_dir+"*.gml")
    
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
            'INPUT': layer + '|layername=TopographicArea',
            'FIELD':'featureCode',
            'OPERATOR':0,
            'VALUE':'10021',
            'OUTPUT': 'TEMPORARY_OUTPUT'
            }
        
        output = processing.run("native:extractbyattribute", params, None) 

        params = {
            'INPUT':output['OUTPUT'],
            'COLUMN':['version','theme','descriptiveGroup','make','physicalLevel','descriptiveTerm','physicalPresence'],
            'OUTPUT':'TEMPORARY_OUTPUT'
            }
        layer = processing.run("native:deletecolumn", params)      

        end = time.time()
        print(f"Completed filtering in {end-start}s")
        return layer['OUTPUT']

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

        params = {
            'INPUT':layer+'|layername=BasicLandPropertyUnit',
            'COLUMN':['changeType','startDate','logicalStatus','blpuState','blpuStateDate','rpc','localCustodianCode','addressbasePostal','postcodeLocator','deliveryPointAddressMember|DeliveryPointAddress|startDate','deliveryPointAddressMember|DeliveryPointAddress|lastUpdateDate','deliveryPointAddressMember|DeliveryPointAddress|entryDate','udprn','postcodeType','deliveryPointSuffix','processDate','landPropertyIdentifierMember|LandPropertyIdentifier|startDate','landPropertyIdentifierMember|LandPropertyIdentifier|lastUpdateDate','landPropertyIdentifierMember|LandPropertyIdentifier|entryDate','lpiKey','landPropertyIdentifierMember|LandPropertyIdentifier|logicalStatus','paoStartNumber','usrn','usrnMatchIndicator','officialFlag','applicationCrossReferenceMember|ApplicationCrossReference|startDate','applicationCrossReferenceMember|ApplicationCrossReference|lastUpdateDate','applicationCrossReferenceMember|ApplicationCrossReference|entryDate','xRefKey','crossReference','version','source','classificationMember|Classification|startDate','classificationMember|Classification|lastUpdateDate','classificationMember|Classification|entryDate','classKey','classificationCode','classScheme','schemeVersion','paoText','dependentLocality','endDate','landPropertyIdentifierMember|LandPropertyIdentifier|endDate','saoStartNumber','applicationCrossReferenceMember|ApplicationCrossReference|endDate','classificationMember|Classification|endDate','parentUPRN','subBuildingName','buildingName','saoText','paoStartSuffix','organisationName','poBoxNumber','organisationMember|Organisation|startDate','organisationMember|Organisation|lastUpdateDate','organisationMember|Organisation|entryDate','orgKey','organisation','dependentThoroughfare','paoEndNumber','organisationMember|Organisation|endDate','saoStartSuffix','paoEndSuffix','departmentName','saoEndNumber','saoEndSuffix'],
            'OUTPUT':'TEMPORARY_OUTPUT'
            }
        output = processing.run("native:deletecolumn", params)
    
        projected_params = {
            'INPUT': output['OUTPUT'],
            'CRS':QgsCoordinateReferenceSystem('EPSG:4326'),
            'OUTPUT': 'TEMPORARY_OUTPUT',
            }
        
        layer = processing.run("native:assignprojection", projected_params)

        end = time.time()
        print(f'Completed setting projection in {end-start}s')
        
        return layer['OUTPUT']

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
            'OUTPUT': self.BASE_DIR + "output\\" + tile + ".gml" 
            }
        
        layer = processing.run("native:joinattributesbylocation", join_params)
        
        end = time.time()
        print(f'Completed joining tables in {end-start}s')

        return layer['OUTPUT']

def main():
    """
    Assign all homes in AddressBasePremium to its building footprint from OSMap Topography and building height from OSMap Building Height Attribute. Building shapefiles saved in 'output' as .gml
    """
    BASE_DIR = "C:\\Users\\lilia\\Downloads\\wmca_download_2022-07-29_10-07-36\\files\\wmca_prj\\project\\unzip_files\\"
    program = MergeOSMaps(BASE_DIR)

    for i in range(len(program.landbaseprem_files)):
        refactored_layer = program.filter_topography(program.topology_files[i])
    
        projected_layer = program.set_projection(program.landbaseprem_files[i])
        joined_layer = program.join_table(refactored_layer, projected_layer, program.building_height_files[i])

if __name__ == "__main__":
    main()


