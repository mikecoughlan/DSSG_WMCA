import os
from dotenv import load_dotenv, find_dotenv
import requests
import pandas as pd 
import json

# find .env automagically by walking up directories until it's found
dotenv_path = find_dotenv()

# load up the entries as environment variables
load_dotenv(dotenv_path)

# Get auth token for pulling EPC data
AUTH_TOKEN = os.environ.get("EPC_AUTH_TOKEN")

def get_epc_data(postcode, num_rows=5000):
    """
    Pull data from Domestic Energy Performance Certificates API.

    Input:
    postcode(str): (1) Postcode 
    num_rows(int): Number of rows to pull. Max 5000 allowed at one time

    Output:
    (str): Pulled data from API

    """
    headers = {
        'Authorization': f'Basic {AUTH_TOKEN}',
        'Accept': 'application/json'
    }
    params = {
        'postcode': postcode,
        'size': num_rows
    }
    url = f'https://epc.opendatacommunities.org/api/v1/domestic/search'
    res = requests.get(url, headers=headers, params=params)
    return res.text

# Get postcodes in WMCA
# Retrieved from https://geoportal.statistics.gov.uk/datasets/06938ffe68de49de98709b0c2ea7c21a/about 
PCD_LSOA_MSOA_PATH = "data\\external\\ONSUD_AUG_2022_WM.csv"
postcode_df = pd.read_csv(PCD_LSOA_MSOA_PATH, low_memory=False, encoding='latin-1')

# Filter for local authorities in WMCA
WMCA_code = ['E08000025', 'E08000031', 'E08000026', 'E08000027', 'E08000028', 'E08000029', 'E08000030', 'E07000192', 'E07000218', 'E07000219', 'E07000236', 'E07000220', 'E06000051', 'E07000221', 'E07000199', 'E06000020', 'E07000222']
postcode_df = postcode_df[postcode_df['ladcd'].isin(WMCA_code)]
postcode_list = list(postcode_df['pcds'].unique())

# Pull WMCA postcode data and save as CSV
result = list()

for code in postcode_list:
    requested_data = get_epc_data(code)
    if len(requested_data)!=0:
        result.extend(json.loads(requested_data)['rows'])

EPC_data = pd.DataFrame(result)
EPC_data.to_csv('../data/processed/EPC_data.csv', index=False)
