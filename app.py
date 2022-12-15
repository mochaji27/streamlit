import streamlit as st
import classification as cla
import regression as reg
import clustering as clu
    


if __name__=="__main__":
    type = st.sidebar.radio(
        "Choose Machine Learning type",
        ("Classification", "Regression", "Clustering")
        )
    if type == "Classification":
        cla.main_page()

    if type == "Regression":
        reg.main_page()

    if type == "Clustering":
        clu.main_page()
