import math
from os import rename
from scipy.stats import skew
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from shapely.geometry import Point  #Polygon
import geopandas

class PopulateNonEPC():
    def __init__(self, EPC_df, DATA_PATH, OUTPUT_PATH):
        self.EPC_df = EPC_df
        self.OUTPUT_PATH = OUTPUT_PATH
        self.DATA_PATH = DATA_PATH

        self.EPC_df['postcode_first'] = self.EPC_df['postcode'].str.split(' ', 1, expand=True)[0]

    def pull_non_epc(self):
        # Get uprn, lat and lng
        # Retrieved from https://osdatahub.os.uk/downloads/open/OpenUPRN 
        uprn_df = pd.read_csv(self.DATA_PATH + 'osopenuprn_202205.csv')
        gdf = geopandas.GeoDataFrame(
            uprn_df[['UPRN', 'LATITUDE', 'LONGITUDE']],
            geometry=geopandas.points_from_xy(uprn_df.LONGITUDE, uprn_df.LATITUDE),
            crs='epsg:4326')

        uk_shp = geopandas.read_file(self.DATA_PATH + 'WMCA Shapefile/LAD_DEC_2021_GB_BFC.shp')
        # filter shapefiles
        counties = ['Birmingham','Wolverhampton','Coventry','Dudley','Sandwell','Solihull','Walsall']
        westmidlands_shp = uk_shp[uk_shp.LAD21NM.isin(counties)] 
        # convert to lat/lng
        westmidlands_shp = westmidlands_shp.to_crs('epsg:4326') 
        # combine shapefiles 
        westmidlands_shp_boundary = westmidlands_shp.dissolve() 
        # Filter for houses within boundary
        houseInWestMidlands = geopandas.tools.sjoin(gdf,westmidlands_shp_boundary, how='right')
        # Convert back to pandas
        houseInWestMidlands = pd.DataFrame(houseInWestMidlands)

        # Get difference between datasets
        self.non_epc_df = houseInWestMidlands.merge(self.df.drop_duplicates(), left_on='UPRN', right_on='uprn', 
                        how='left', indicator=True)
        mask = self.non_epc_df['_merge'] == 'left_only'
        self.non_epc_df = self.non_epc_df[mask]

        # Retrieved from https://geoportal.statistics.gov.uk/datasets/ons-uprn-directory-may-2022/about 
        feature_df = pd.read_csv(self.DATA_PATH, 'ONSUD_MAY_2022_WM.csv')
        rename_col = {
            'UPRN': 'UPRN',
            'PCDS': 'postcode',
            'LAD21CD': 'local-authority',
            'LSOA11CD': 'lsoa_code',
            'MSOA11CD': 'msoa_code'
        }
        feature_df = feature_df[list(rename_col.keys())]
        feature_df.rename(columns=rename_col, inplace=True)

        self.non_epc_df = self.non_epc_df.merge(feature_df, on='UPRN', how='left')
        self.non_epc_df['postcode_first'] = self.non_epc_df['postcode'].str.split(' ', 1, expand=True)[0]
        

    def baseline_pred(self):
        self.non_EPC_df['naive-current-energy-efficiency'] = np.nan
        self.non_EPC_df['naive-current-energy-rating'] = np.nan

        hierarchy = ['postcode', 'postcode_first', 'lsoa_code', 'msoa_code', 'local-authortiy']

        for pred in ['naive-current-energy-efficiency', 'naive-current-energy-rating']:
            hierarchy_counter = 0

            while self.non_EPC_df[pred].isna().sum() > 0:
                group_var = hierarchy[hierarchy_counter]
                subset_EPC_df = self.EPC_df[[group_var, pred]]
                grouped_train = subset_EPC_df.groupby(group_var).mean()

                mapping = dict(zip(grouped_train.index, grouped_train[pred]))
                self.non_EPC_df[pred] = self.X_test[group_var].map(mapping)
                self.non_EPC_df[pred][np.isfinite(self.non_EPC_df[pred]) == False] = np.nan
                
                hierarchy_counter +=1


def main(EPC_df, DATA_PATH, OUTPUT_PATH):
    model = PopulateNonEPC(EPC_df, DATA_PATH, OUTPUT_PATH)
    model.pull_non_epc()
    model.baseline_pred()

    ## FILL THIS ##
    model.non_epc_df.to_csv()


if __name__ == "__main__":
    ## FILL THIS ##
    main()