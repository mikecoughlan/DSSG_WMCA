##############################################################################
#
#
#	DSSG_WMCA/scripts/models/plotting_metrics.py
#
#	function for plotting all of the metrics for the different models
#	so we cn evaluate the best model. Wil break up the models into 
# 	the different classes to test the scores on each class because of the
# 	dataset imbalance.
#
#
##############################################################################


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from pandas_profiling import ProfileReport
import json
import pickle
import requests
import argparse
from scipy.stats import pearsonr
from tqdm import tqdm
import scipy.stats as stats
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import KNNImputer, SimpleImputer, IterativeImputer
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split, cross_val_score
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score, roc_auc_score, make_scorer
from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score, r2_score
from sklearn.metrics import precision_recall_curve, f1_score, auc, roc_curve, confusion_matrix

from tensorflow.keras.utils import to_categorical
import modelfunction as mf

import inspect
import gc



def plotting_mainheat_metrics(metrics_df):
	'''Plotting the different metrics that will be used to 
		evaluate the mainheat-description models.

		INPUTS:
			metrics_df: pandas df containing the metric results.

		RETURNS:
			saves plots to plots/model_eval/ folder'''

	names = metrics_df.index.tolist()
	x = [i+1 for i in range(len(metrics_df))]
	print(names)

	fig = plt.figure(figsize=(60,55))													# establishing the figure
	plt.subplots_adjust(bottom=0.1, top=0.9, left=0.1, right=0.9, hspace=0.3)			# trimming the whitespace in the subplots

	ax1 = fig.add_subplot(311)					# adding the subplot
	plt.title('Accuracy', fontsize='70')		# titling the plot
	ax1.bar([n-0.1 for n in x], metrics_df['non_elec_accuracy'], width=0.2, color='xkcd:blue', align='center', label='Negetive Node')
	ax1.bar([n+0.1 for n in x], metrics_df['elec_accuracy'], width=0.2, color='xkcd:orange', align='center', label='Positive Node')
	plt.ylabel('Accuracy', fontsize='60')				# adding teh y axis label
	plt.xticks(x, names, fontsize='58')		# adding ticks to the points on the x axis
	# plt.legend(fontsize='58', bbox_to_anchor = (0.7, 1.0))
	plt.yticks(fontsize='58')						# making the y ticks a bit bigger. They're a bit more important

	ax2 = fig.add_subplot(312)					# adding the subplot
	plt.title('RMSE', fontsize='70')		# titling the plot
	ax2.bar([n-0.1 for n in x], metrics_df['non_elec_rmse'], width=0.2, color='xkcd:blue', align='center', label='Negetive Node')
	ax2.bar([n+0.1 for n in x], metrics_df['elec_rmse'], width=0.2, color='xkcd:orange', align='center', label='Positive Node')
	plt.ylabel('RMSE', fontsize='60')				# adding teh y axis label
	plt.xticks(x, names, fontsize='58')		# adding ticks to the points on the x axis
	plt.legend(fontsize='58')
	plt.yticks(fontsize='58')						# making the y ticks a bit bigger. They're a bit more important

	ax3 = fig.add_subplot(313)					# adding the subplot
	plt.title('Precision-Recall, ROC, Macro F1', fontsize='70')		# titling the plot
	ax3.bar([n-0.2 for n in x], metrics_df['PR'], width=0.2, color='xkcd:blue', align='center', label='PR')
	ax3.bar(x, metrics_df['roc_score'], width=0.2, color='xkcd:orange', align='center', label='ROC')
	ax3.bar([n+0.2 for n in x], metrics_df['F1_macro'], width=0.2, color='xkcd:green', align='center', label='M-F1')
	plt.xticks(x, names, fontsize='58')		# adding ticks to the points on the x axis
	plt.legend(fontsize='58', bbox_to_anchor = (1.0, 0.9))
	plt.yticks(fontsize='58')						# making the y ticks a bit bigger. They're a bit more important

	plt.savefig('plots/model_eval/smote_epc_metrics.png')





def plotting_epc_metrics(metrics_df):
	'''Plotting the different metrics that will be used to 
		evaluate the epc models.

		INPUTS:
			metrics_df: pandas df containing the metric results.

		RETURNS:
			saves plots to plots/model_eval/ folder'''
	
	names = metrics_df.index.tolist()
	x = [i+1 for i in range(len(names))]
	print(names)

	fig = plt.figure(figsize=(60,55))													# establishing the figure
	plt.subplots_adjust(bottom=0.1, top=0.9, left=0.1, right=0.9, hspace=0.3)			# trimming the whitespace in the subplots

	ax1 = fig.add_subplot(311)					# adding the subplot
	plt.title('Accuracy', fontsize='70')		# titling the plot
	ax1.bar([n-0.15 for n in x], metrics_df['total_accuracy'], width=0.05, color='xkcd:blue', align='center', label='Total')
	ax1.bar([n-0.1 for n in x], metrics_df['A_accuracy'], width=0.05, color='xkcd:orange', align='center', label='A')
	ax1.bar([n-0.05 for n in x], metrics_df['B_accuracy'], width=0.05, color='xkcd:green', align='center', label='B')
	ax1.bar([n for n in x], metrics_df['C_accuracy'], width=0.05, color='xkcd:red', align='center', label='C')
	ax1.bar([n+0.05 for n in x], metrics_df['D_accuracy'], width=0.05, color='xkcd:cyan', align='center', label='D')
	ax1.bar([n+0.1 for n in x], metrics_df['E_accuracy'], width=0.05, color='xkcd:magenta', align='center', label='E')
	ax1.bar([n+0.15 for n in x], metrics_df['F_accuracy'], width=0.05, color='xkcd:black', align='center', label='F')
	ax1.bar([n+0.2 for n in x], metrics_df['G_accuracy'], width=0.05, color='xkcd:pink', align='center', label='G')
	plt.ylabel('Accuracy', fontsize='60')				# adding teh y axis label
	plt.xticks(x, names, fontsize='58')		# adding ticks to the points on the x axis
	plt.legend(fontsize='58', bbox_to_anchor = (1.0, 0.9))
	plt.yticks(fontsize='58')						# making the y ticks a bit bigger. They're a bit more important

	ax2 = fig.add_subplot(312)					# adding the subplot
	plt.title('RMSE', fontsize='70')		# titling the plot
	ax2.bar([n-0.15 for n in x], metrics_df['A_rmse'], width=0.05, color='xkcd:orange', align='center', label='A')
	ax2.bar([n-0.1 for n in x], metrics_df['B_rmse'], width=0.05, color='xkcd:green', align='center', label='B')
	ax2.bar([n-0.05 for n in x], metrics_df['C_rmse'], width=0.05, color='xkcd:red', align='center', label='C')
	ax2.bar([n for n in x], metrics_df['D_rmse'], width=0.05, color='xkcd:cyan', align='center', label='D')
	ax2.bar([n+0.05 for n in x], metrics_df['E_rmse'], width=0.05, color='xkcd:magenta', align='center', label='E')
	ax2.bar([n+0.1 for n in x], metrics_df['F_rmse'], width=0.05, color='xkcd:black', align='center', label='F')
	ax2.bar([n+0.15 for n in x], metrics_df['G_rmse'], width=0.05, color='xkcd:pink', align='center', label='G')
	plt.ylabel('RMSE', fontsize='60')				# adding teh y axis label
	plt.xticks(x, names, fontsize='58')		# adding ticks to the points on the x axis
	plt.legend(fontsize='58', bbox_to_anchor = (1.0, 0.9))
	plt.yticks(fontsize='58')	

	ax3 = fig.add_subplot(313)					# adding the subplot
	plt.title('ROC, Macro F1', fontsize='70')		# titling the plot
	ax3.bar([n-0.1 for n in x], metrics_df['roc_score'], width=0.2, color='xkcd:orange', align='center', label='ROC')
	ax3.set_ylabel('ROC Score', color='xkcd:orange', fontsize=58)
	# plt.ylim([0.9,1])
	plt.xticks(x, names, fontsize='58')		# adding ticks to the points on the x axis
	plt.yticks(fontsize='58')				
	ax4 = ax3.twinx()
	ax4.bar([n+0.1 for n in x], metrics_df['F1_macro'], width=0.2, color='xkcd:green', align='center', label='Macro F1')
	ax4.set_ylabel('Macro F1 Score', color='xkcd:green', fontsize='58')
	plt.xticks(x, names, fontsize='58')		# adding ticks to the points on the x axis
	# plt.legend(fontsize='58')
	plt.yticks(fontsize='58')				

	plt.savefig('plots/model_eval/initial_random_forest_metrics.png')









