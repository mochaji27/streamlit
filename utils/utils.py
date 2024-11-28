import pandas as pd
import streamlit as st
import pickle
import pycaret.classification as pycc
import os

def check_file(file_upload):
    if file_upload is None:
        return
    if ".csv" in file_upload.name:
        return pd.read_csv(file_upload, sep=",")
    if ".xlsx" in file_upload.name:
        return pd.read_excel(file_upload)
    return

def download_model(model, model_name):
    return pycc.save_model(model, model_name)

def get_list_model_result():
    # folder path
    dir_path = r'model_result'

    # list file and directories
    res = os.listdir(dir_path)
    return res
