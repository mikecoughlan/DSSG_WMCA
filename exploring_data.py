############################################################################################
#
#
# exploring_data.py
# 
# Loads a CSV file from projectDir/data/ and creates a pandas dataframe. The dataframe then is examined using 
# pandas profiling to do exploritory data anaylsis. Saves the exploritory analysis as an 
# html file in projectDir/outputs/
#
#
############################################################################################


import pandas as pd
import matplotlib.pyplot as plt
import argparse
from pandas_profiling import ProfileReport



def profiling(df):
	'''Function exploring all of the EPC data. Does pandas profiling on the imported data frame
		and saves it as an interactive html file in projectDir/outputs/.

	INPUTS:
		df(pd dataframe): dataframe containing all the EPC data.

	SAVES:
		profile: exploritory data information for the dataframe. File name will label how many
				homes are being examined.
	'''

	profile = ProfileReport(df, title="name", explorative=True)

	profile.to_file('outputs/pandas_profiling_{0}_homes.html'.format(len(df)))



if __name__ == '__main__':
	'''Here is where all the arg parsing is done for inputting features into the main code.
		The only arg will be the name of the CSV file containing the data '''

	parser = argparse.ArgumentParser() 		# initiates the argParser 
	parser.add_argument('--file_name',			# defines the name of the input parameter
						action='store',		# will be stored as avariable
						type=str,			# input variable type
						required=True,
						help='Name of the CSV file containing the data. Must be in the projectDir/data/folder.')		# help message
	


	args=parser.parse_args()	# arg parse that parses the args

	df = pd.read_csv('data/{0}.csv'.format(args.file_name))  # loads in the file to a pd.dataframe
	

	profiling(df)		# calling the data processing function function.

	print('It ran. Good job!')										# if we get here we're doing alright.





