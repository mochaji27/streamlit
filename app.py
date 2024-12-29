import streamlit as st
import page.classification as cla
import page.regression as reg
import page.clustering as clu
import page.gelo as gelo    


if __name__=="__main__":
    type = st.sidebar.radio(
        "Choose Machine Learning type",
        ("GeLo", "Classification", "Regression", "Clustering")
        )
    if type == "Classification":
        cla.main_page()

    if type == "Regression":
       reg.main_page()

    if type == "Clustering":
       clu.main_page()

    if type == "GeLo":
       gelo.main_page()
