import pandas as pd
import streamlit as st
import pickle

def check_file(file_upload):
    if file_upload is None:
        return
    if ".csv" in file_upload.name:
        return pd.read_csv(file_upload, sep=",")
    if ".xlsx" in file_upload.name:
        return pd.read_excel(file_upload)
    return

def download_model(model):
    return pickle.dumps(model)