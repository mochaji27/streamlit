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
            tune_model, download = None, None
            if st.button('Go', key="key_classification_btn"):
                #pipeline_optimizer = TPOTClassifier(generations=5, population_size=20, cv=5,
                #                    random_state=42, verbosity=2, max_eval_time_mins=60)
                #pipeline_optimizer.fit(X_train, y_train)
                #score = pipeline_optimizer.score(X_test, y_test)
                #st.success(score)
                #csf = pycc.set
                st.info("This is Experimental Machine Learning")
                pycc.setup(df, target = target, silent=True, train_size=0.75, normalize=True, normalize_method='zscore', feature_selection=True, fix_imbalance=True)
                setup_df = pycc.pull()
                best_clf = pycc.compare_models()
                compare_df = pycc.pull()
                model_name = compare_df.iloc[0, 0]
                st.dataframe(compare_df)
                st.info("Use {} model to Hyperparameter Tuning".format(model_name))
                tune_model = pycc.tune_model(best_clf, choose_better = True)
                st.dataframe(pycc.pull())
                st.write(pycc.plot_model(tune_model, plot = 'parameter', display_format='streamlit'))
                pycc.plot_model(tune_model, plot = 'feature', display_format='streamlit')
                pycc.plot_model(tune_model, plot = 'auc', display_format='streamlit')
                pycc.plot_model(tune_model, plot = 'confusion_matrix', display_format='streamlit')
                st.write(tune_model)

                #st.info("Best Parameter : " + tune_model.get_params)
                #pycc.plot_model(best_clf, plot = 'auc')
            if tune_model is not None:
                pycc.finalize_model(tune_model)
                pycc.save_model(tune_model, "{}_{}".format(file_upload, model_name))
                download = st.button("Download")
            else:
                del download

    
#@st.cache()
#def init_h2o():
#    h2o.connect()

@st.cache
def data_profiling(df):
    return df.profile_report()

#def download_model():


if __name__=="__main__":
    type = st.sidebar.radio(
        "Choose Machine Learning type",
        ("Classification", "Regression")
        )
    if type == "Classification":
        main_page("Classification")
    if type == "Regression":
        main_page("Regression")
