import streamlit as st
import pandas as pd

header = st.container()
dataset = st.container()
features = st.container()
model_training = st.container()

pathname = '/Users/meghna_mac2/PycharmProjects/WMCA/wmca_app/'  # your pathname

st.markdown(
    """
    <style>
    .main {
        background-color: F5F5F
    }
    </style>
    """,
    unsafe_allow_html = True
)

@st.cache
def get_data(filename):
    data = pd.read_csv(filename)
    return data

with header:
    st.title('WMCA - Pure LeapFrog Demo')
    st.text('')

with dataset:
    st.header('EPC data')
    epc_data = get_data(pathname+'data/numerical_individual_columns_data.csv')
    st.write(epc_data.head())
    constituency = pd.DataFrame(epc_data['constituency'].value_counts())
    st.bar_chart(constituency)



with features:
    st.header('feature')


with model_training:            
    st.header('model')