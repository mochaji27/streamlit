import streamlit as st
import pandas as pd
import pandas_profiling 
from streamlit_pandas_profiling import st_profile_report

def check_file(file_upload):
    if file_upload is None:
        return
    if ".csv" in file_upload.name:
        return pd.read_csv(file_upload, sep=",")
    if ".xlsx" in file_upload.name:
        return pd.read_excel(file_upload)
    return
    

def main_page():
    st.title("Generate Machine Learning")
    st.write("1. Upload your data")
    file_upload = st.file_uploader("Choose a XLSX file", type=["xlsx", "csv"])
    df = check_file(file_upload)
    if df is not None:
        pr = df.profile_report()
        st.title("Data Profiling")
        st_profile_report(pr)
    



if __name__=="__main__":
    with st.sidebar:
        st.title("""
            Option
        """)
        add_radio = st.radio(
            "Choose a Shipping method",
            ("Standard", "Express")
        )
    if add_radio == "Standard":
        main_page()
    else:
        st.title("Test")
