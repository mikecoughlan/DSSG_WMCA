######################################################################################
#
#	model/multiclass_randomforest.py
#	
#	File for running a random forest model. Gives the option for the target variable
#	to be either the EPC rating, in which cace the output will be multi-class, or
#	the electric/non-electric heating type, which will have a binary output. File 
#	saves the model and resulting predictions. Takes in the already seperated 
#	training and testing data. Run from the main project directory
#	
#
######################################################################################


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob
import pickle
import argparse
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import geopandas as gpd


DATA_PATH = 'data/processed/'		# path to the data from the project directory
TILE_PATH = '../project/unzip_files/encoded_proxy/'

'''CONFIG dict storing global information. 
	random_int: integer setting the reandom seed for reproducibility.
	input_features: list of features to be segmented from the larger training and testing datasets
					for input to the models.
'''
CONFIG = {
		'random_int': 123,
		# 'input_features': ['postcode', 'constituency', 'msoa_code', 'lsoa_code',
		# 					'local-authority_E06000023', 'local-authority_E06000040', 'local-authority_E07000192',
		# 					'local-authority_E07000194', 'local-authority_E07000196', 'local-authority_E07000218',
		# 					'local-authority_E07000219', 'local-authority_E07000220', 'local-authority_E07000221',
		# 					'local-authority_E07000222', 'local-authority_E07000234', 'local-authority_E08000025',
		# 					'local-authority_E08000026', 'local-authority_E08000027', 'local-authority_E08000028',
		# 					'local-authority_E08000029', 'local-authority_E08000030', 'local-authority_E08000031',
		# 					'local-authority_E09000026', 'RelHMax', 'calculatedAreaValue',
		# 					'total_consumption', 'mean_counsumption', 'median_consumption',
		# 					'prop_households_fuel_poor', 'LATITUDE', 'LONGITUDE'],
		'input_features': ['postcode', 'lsoa_code',
							'local-authority_E06000023', 'local-authority_E06000040', 'local-authority_E07000192',
							'local-authority_E07000194', 'local-authority_E07000196', 'local-authority_E07000218',
							'local-authority_E07000219', 'local-authority_E07000220', 'local-authority_E07000221',
							'local-authority_E07000222', 'local-authority_E07000234', 'local-authority_E08000025',
							'local-authority_E08000026', 'local-authority_E08000027', 'local-authority_E08000028',
							'local-authority_E08000029', 'local-authority_E08000030', 'local-authority_E08000031',
							'local-authority_E09000026', 'msoa_code_E02002149', 'msoa_code_E02002150', 
							'msoa_code_E02002151', 'msoa_code_E02002152', 'msoa_code_E02002154', 
							'msoa_code_E02002155', 'msoa_code_E02002156', 'msoa_code_E02002157', 
							'msoa_code_E02002158', 'msoa_code_E02002159', 'msoa_code_E02002160', 
							'msoa_code_E02002161', 'msoa_code_E02002163', 'msoa_code_E02002164', 
							'msoa_code_E02002168', 'local-authority_E08000031', 'constituency_E14000945', 
							'constituency_E14001011', 'constituency_E14001049', 'constituency_E14001050', 
							'constituency_E14001051', 'RelHMax', 'calculatedAreaValue',
							'total_consumption', 'mean_counsumption', 'median_consumption',
							'prop_households_fuel_poor', 'LATITUDE', 'LONGITUDE']}

# setting random seed for reporducibility. SK learn uses numpy random seed.
np.random.seed(CONFIG['random_int'])

def adding_zeros_columns(df):
	'''function for adding zeros columns to fill in the missing local-authority, constituency,
		and msoa level columns. This occurs when the processed proxy file does not contain all of the 
		possible values and so a column is not created in the one-hot-encoding.

		INPUTS:
			df (pd.dataframe): df of the just loaded proxy information

		RETURNS:
			df (pd.dataframe): df with the additional columns'''


	for feat in CONFIG['input_features']: 		# looping through the list of all the columns that should be in the df
		if feat not in df.columns.tolist():		# checking the features that are supposed to be presetna gainst the columns that are in the df
			df[feat] = 0						# filling that column with zeros if it does not exist

	return df


def extract_and_concat():
	'''Function to load and combine the seperate proxy files into
		one dataframe. 

		RETURNS:
			tiles_df (pd.dataframe): df containing the data for all the homes in the target area (west midlands)'''

	tiles = glob.glob(DATA_PATH+'S*.csv', recursive=True)		# getting all the file names for the proxy data

	tiles_df = pd.DataFrame()		# initilizing the dataframe for storing the proxy data

	for tile in tiles:
		# file = gpd.read_file(tile)
		# file['calculatedAreaValue'] = file['geometry'].area
		# df = pd.DataFrame(file)
		df = pd.read_csv(tile) 		# loading the individual csv files
		df - adding_zeros_columns(df) 	# checkiong for missing columns and filling them with zeros
		tiles_df = pd.concat([tiles_df, df], axis=0)	# concatenating the individual dataframes together

	return tiles_df


def loading_epc():
	'''Function for loading the EPC data.

		RETURNS:
			epc_df (pd.dataframe): df containing the epc data'''

	epc_df = pd.read_csv(DATA_PATH+'numerical_individual_columns_data.csv')

	return epc_df


def merging_epc_and_proxies(epc_df, tiles_df):
	'''Merging the EPC data with all of the data from the target area (west midlands).
		The data is then seperated into two dataframes. One for the homes that are in 
		the epc database and one for those without. This is done because there is a
		difference between the variables in the EPC database, and those available for 
		all of the homes in the West Midlands. Combining and then seperating them in 
		this way allows for the matching of the epc ratings with the varaibles that
		are not available in the epc database.

		INPUTS: 
			epc_df (pd.Dataframe): df of the epc data
			tiles_df (pd.dataframe): df of the proxy data from all the homes in the WM

		RETURNS:
			proxies (pd.dataframe): df of all the homes without an EPC rating
			epc (pd.dataframe): df of all of the homes with an epc rating'''

	combined = tiles_df.merge(epc_df, on='uprn', how='left', suffixes=(None, '_epc'))		# merging the two dfs on the Unique Property Reference Number (UPRN)

	proxies = combined[combined['current-energy-rating'].isnull()].reset_index(drop=True) 	# homes not in teh EPC database will not have a EPC rating. Use this to seperate
	epc = combined[combined['current-energy-rating'].notna()].reset_index(drop=True)		# homes in the database will have a rating

	# creating a column that indicates whether the eventaul results will be predicted or are a ground truth.
	epc['predicted'] = 0
	proxies['predicted'] = 1

	return proxies, epc		



def main():
	'''Main function for preparing and seperating the epc and proxy data.

		RETURNS:
			proxies (pd.dataframe): df of all the homes without an EPC rating
			epc (pd.dataframe): df of all of the homes with an epc rating'''

	tiles_df = extract_and_concat()

	epc_df = loading_epc()

	proxies, epc = merging_epc_and_proxies(epc_df, tiles_df)

	# saving the data frames 
	proxies.to_csv(DATA_PATH+'homes_with_proxies.csv', index=False)
	epc.to_csv(DATA_PATH+'homes_with_epc_ratings.csv', index=False)

	return proxies, epc




if __name__ == '__main__':

	proxies, epc = main()		# calling the main function

	print('It ran. Good job!')

