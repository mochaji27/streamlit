import streamlit as st
import pandas as pd
from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
import utils.utils as utils
#from tpot import TPOTClassifier, TPOTRegressor
import numpy as np
#import pycaret.classification as pycc
#import h2o 
#from h2o.automl import H2OAutoML
import pycaret.classification as pycc
import pickle

list_schema = {"Pollution" : ["Temperature","Humidity","PM2.5","PM10","NO2","SO2","CO","Proximity_to_Industrial_Areas","Population_Density","Air Quality"],
                    "Test" : ["asdas", "zxczxc", "123123"]}

@st.fragment
def fragment_upload_data():
    file_upload = st.file_uploader(key = "file_upload", label="Upload a XLSX / CSV file", type=["xlsx", "csv"])
    if file_upload is None:
        # if 'run_model' in st.session_state:
        #     del st.session_state['run_model']
        # if 'data_profiling' in st.session_state:
        #     del st.session_state['data_profiling']
        # if 'result_predict' in st.session_state:
        #     del st.session_state['result_predict']
        return
    df = utils.check_file(file_upload)
    selected_schema = st.selectbox("Select Table Schema", ["Pollution", "Test"])
    try:
        df.columns = list_schema[selected_schema]
    except:
        st.error("Wrong schema, Total schema on file : {}, Total schema on table : {}".format(len(df.columns), len(selected_schema)))
        return
    st.dataframe(df)

@st.fragment
def fragment_create_schema():
    file_upload = st.file_uploader(key = 'file_create_schema',label="Upload a XLSX / CSV file", type=["xlsx", "csv"])
    if file_upload is None:
        return
    df = utils.check_file(file_upload)
    selected_column = st.selectbox("Select row", [i for i in range(df.shape[0])])
    df_schema = df.copy()
    df_schema.columns = df.columns if selected_column == 0 else df.iloc[selected_column]
    st.write("List Column")
    st.write(df_schema.columns.to_list())

def main_page():
    st.title("GeLo (General Loader)")
    tab_upload, tab_create_schema, tab_mod_schema = st.tabs(["Upload Data", "Create Schema", "Modify Schema"])

    with tab_create_schema:
        fragment_create_schema()

    with tab_mod_schema:
        print("asd")

    with tab_upload:
        fragment_upload_data()
        