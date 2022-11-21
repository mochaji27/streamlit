import streamlit as st
import pandas as pd
import pandas_profiling 
from streamlit_pandas_profiling import st_profile_report
import utils
#from tpot import TPOTClassifier, TPOTRegressor
import numpy as np
#import pycaret.classification as pycc
from sklearn.model_selection import train_test_split
#import h2o 
#from h2o.automl import H2OAutoML
import pycaret.classification as pycc
    

def main_page(type):
    st.title("Simple AutoML using PyCaret")
    st.write("Upload your data")
    file_upload = st.file_uploader("Choose a XLSX file", type=["xlsx", "csv"])
    if file_upload is None:
        return

    df = utils.check_file(file_upload)
    tab_view, tab_profile, tab_predict = st.tabs(["View Data", "Data Profiling", "ML"])
    with tab_view:
        st.dataframe(df)
    #with tab_profile:
    #    st_profile_report(data_profiling(df))
    with tab_predict:
        #init_h2o()
        if type == "Classification":
            list_col_numeric = df.select_dtypes(include=[np.float64, np.int8, np.int16, np.int32, np.int64]).columns
            list_col_category = df.select_dtypes(exclude=[np.float64, np.int8, np.int16, np.int32, np.int64]).columns
            col1, col2 = st.columns(2)
            selected_col_exc_cat = col1.multiselect('Select your column that exclude on ML by Category',
                list_col_category,
                []
                )
            selected_col_exc_num = col2.multiselect('Select your column that exclude on ML by Numeric',
                list_col_numeric,
                []
                )
                
            target = st.selectbox("Select Your Target", df.columns)
            df = df.drop(columns=selected_col_exc_cat).drop(columns=selected_col_exc_num)
            y = df[target].values
            X = df.drop(columns=target)
            #x = pd.get_dummies(data=x, columns=x.select_dtypes(exclude=[np.float64, np.int8, np.int16, np.int32, np.int64]).columns)
            X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75, test_size=0.25, random_state=2022)
            if st.button('Go', key="key_classification_btn"):
                #pipeline_optimizer = TPOTClassifier(generations=5, population_size=20, cv=5,
                #                    random_state=42, verbosity=2, max_eval_time_mins=60)
                #pipeline_optimizer.fit(X_train, y_train)
                #score = pipeline_optimizer.score(X_test, y_test)
                #st.success(score)
                #csf = pycc.set
                st.info("This is Experimental Machine Learning")
                pycc.setup(df, target = target, silent=True)
                setup_df = pycc.pull()
                best_clf = pycc.compare_models()
                compare_df = pycc.pull()
                st.dataframe(compare_df)
                best_clf
                pycc.evaluate_model(best_clf)
                #pycc.plot_model(best_clf, plot = 'auc')


        else:
            list_col_numeric = df.select_dtypes(include=[np.float64, np.int8, np.int16, np.int32, np.int64]).columns
            list_col_category = df.select_dtypes(exclude=[np.float64, np.int8, np.int16, np.int32, np.int64]).columns
            col1, col2 = st.columns(2)
            selected_col_exc_cat = col1.multiselect('Select your column that exclude on ML by Category',
                list_col_category,
                []
                )
            selected_col_exc_num = col2.multiselect('Select your column that exclude on ML by Numeric',
                list_col_numeric,
                []
                )
                

            target = st.selectbox("Select Your Target", list_col_numeric)
            y = df[target].values
            x = df.drop(columns=target).drop(columns=selected_col_exc_cat).drop(columns=selected_col_exc_num)
            #x = pd.get_dummies(data=x, columns=x.select_dtypes(exclude=[np.float64, np.int8, np.int16, np.int32, np.int64]).columns)
            X_train, X_test, y_train, y_test = train_test_split(x, y, train_size=0.75, test_size=0.25, random_state=2022)
            #if st.button('Go', key="key_regression_btn"):
                #pipeline_optimizer = TPOTRegressor(generations=5, population_size=20, cv=5,
                #                    random_state=42, verbosity=2)
                #pipeline_optimizer.fit(X_train, y_train)
                #score = pipeline_optimizer.score(X_test, y_test)
                #st.success(score)

    
#@st.cache()
#def init_h2o():
#    h2o.connect()

@st.cache
def data_profiling(df):
    return df.profile_report()



if __name__=="__main__":
    type = st.sidebar.radio(
        "Choose Machine Learning type",
        ("Classification", "Regression")
        )
    if type == "Classification":
        main_page("Classification")
    if type == "Regression":
        main_page("Regression")
