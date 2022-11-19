import streamlit as st
import pandas as pd
import pandas_profiling 
from streamlit_pandas_profiling import st_profile_report
import utils
from tpot import TPOTClassifier
    

def main_page():
    st.title("Generate Machine Learning")
    st.write("Upload your data")
    file_upload = st.file_uploader("Choose a XLSX file", type=["xlsx", "csv"])
    status_profile = False
    if file_upload is None:
        status_profile = False
    if file_upload is not None:
        df = utils.check_file(file_upload)
        tab_view, tab_profile, tab_predict = st.tabs(["View Data", "Data Profiling", "Predict"])
        with tab_view:
            st.dataframe(df)
        with tab_profile:
            if status_profile is False:
                pr = df.profile_report()
            status_profile = True
            st_profile_report(pr)
        with tab_predict:
            target = st.selectbox("Select Your Target", df.columns)
        


    



if __name__=="__main__":
    main_page()
