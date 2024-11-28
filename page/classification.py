import streamlit as st
import pandas as pd
import pandas_profiling 
from streamlit_pandas_profiling import st_profile_report
import utils.utils as utils
#from tpot import TPOTClassifier, TPOTRegressor
import numpy as np
#import pycaret.classification as pycc
#import h2o 
#from h2o.automl import H2OAutoML
import pycaret.classification as pycc
import pickle

def main_page():
    st.title("Simple Classification AutoML using PyCaret")
    st.write("Upload your data")
    file_upload = st.file_uploader("Upload a XLSX file", type=["xlsx", "csv"])
    if file_upload is None:
        if 'run_model' in st.session_state:
            del st.session_state['run_model']
        if 'data_profiling' in st.session_state:
            del st.session_state['data_profiling']
        if 'result_predict' in st.session_state:
            del st.session_state['result_predict']
        return

    df = utils.check_file(file_upload)
    tab_view, tab_profile, tab_experimental, tab_prediction = st.tabs(["View Data", "Data Profiling", "Experimental ML", "Prediction"])

    with tab_view:
        st.dataframe(df)

    with tab_profile:
        if 'data_profiling' not in st.session_state:
            st.session_state['data_profiling'] = data_profiling(df)
        st_profile_report(st.session_state['data_profiling'])

    with tab_prediction:
        """
        st.radio(
            "Select Model From",
            ("Server", "Upload"),
            key="select_model_from",
            horizontal=True
        )
        if st.session_state["select_model_from"] == "Server":   
            model_selected_name = st.selectbox(
                    'Select Unique ID Model',
                    utils.get_list_model_result()
                    )
            model_selected_name = "model_result\\" + model_selected_name.replace(".pkl", "")
        else :
            model_selected = st.file_uploader("Upload a Model ML", )
            model_selected_name = model_selected.name.replace(".pkl", "") if model_selected is not None else model_selected
        """
        model_selected_name = st.selectbox(
                    'Select Unique ID Model',
                    utils.get_list_model_result()
                    )
        model_selected_name = "model_result\\" + model_selected_name.replace(".pkl", "")
        st.button('Go', key="key_prediction_btn", on_click=prediction, args=(model_selected_name, df))
        placeholder_result = st.empty()
        placeholder_result.button(
            "Download Result Prediction",
            disabled = True
        )
        if 'result_predict' in st.session_state:
            result_predict = st.session_state['result_predict']
            st.dataframe(result_predict)
            placeholder_result.download_button(
                "Download Result",
                data=result_predict.to_csv(),
                file_name="Result.csv"
            )
        

    with tab_experimental:
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
        
        list_col_exc = selected_col_exc_cat + selected_col_exc_num

        st.radio(
            "Choose Option for Data Preprocessing",
            ("Default", "Advanced"),
            key="preprocess_option",
            horizontal=True
        )
        if st.session_state["preprocess_option"] == "Advanced":
            st.markdown("""---""")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.checkbox("preprocess", key="preprocess")
                st.checkbox("normalize", key="normalize")
                st.checkbox("pca", key="pca")
                st.checkbox("feature_selection", key="feature_selection")
            with col2:
                st.checkbox("remove_multicollinearity", key="remove_multicollinearity")
                st.checkbox("polynomial_features", key="polynomial_features")
                st.checkbox("fold_shuffle", key="fold_shuffle")
            with col3: 
                st.checkbox("fix_imbalance", key="fix_imbalance")
                st.checkbox("data_split_shuffle", key="data_split_shuffle")
                st.checkbox("remove_outliers", key="remove_outliers")
                st.checkbox("data_split_stratify", key="data_split_stratify")
                
        st.markdown("""---""")

        unique_code = st.text_input('Unique Code', '')
        model_unique_code = "model_{}".format(unique_code)
        placeholder_unique_code_error = st.empty()
        st.button('Go', key="key_classification_btn", on_click=run_model, kwargs=dict(model_unique_code = model_unique_code, placeholder_unique_code_error = placeholder_unique_code_error))
        placeholder = st.empty()
        placeholder.button(
            "Download Model",
            disabled = True
        )
        if 'error_unique_code' in st.session_state:
            placeholder_unique_code_error.error("Unique Code has been used!")
            return
        
        if 'run_model' not in st.session_state:
            return

        if 'best_clf' not in st.session_state:
            st.info("This is Experimental Machine Learning")
            st.session_state['best_clf'], st.session_state['compare_df'], st.session_state['tune_model'] = model(df, target, list_col_exc)

        placeholder_unique_code_error.empty()
        model_name = st.session_state['compare_df'].iloc[0, 0]
        st.dataframe(st.session_state['compare_df'])
        st.info("Use {} model to Hyperparameter Tuning".format(model_name))
        st.dataframe(pycc.pull())
        pycc.plot_model(st.session_state['tune_model'], plot = 'feature', display_format='streamlit')
        try:
            pycc.plot_model(st.session_state['tune_model'], plot = 'auc', display_format='streamlit')
        except:
            st.error('Cannot show the AUC plot for {} model, run model again'.format(model_name))
        pycc.plot_model(st.session_state['tune_model'], plot = 'confusion_matrix', display_format='streamlit')
        st.write(st.session_state['tune_model'])

        pycc.save_model(st.session_state['tune_model'], "model_result\\" + model_unique_code)
        with open('model_result\\{}.pkl'.format(model_unique_code), 'rb') as f:
            placeholder.download_button(
                "Download Model {}".format(unique_code),
                data=f,
                file_name=model_unique_code + ".pkl"
            )
                



def data_profiling(df):
    return df.profile_report()

def prediction(model_upload, file_upload):
    #model = pickle.load(model_upload)
    #print(model_upload)
    model = pycc.load_model(model_upload)
    
    st.session_state['result_predict'] = pycc.predict_model(model, file_upload)
    

def model(df, target, list_col_exc):
    if st.session_state["preprocess_option"] == "Default":
        print("------------------- ini adalah default ------------------------")
        pycc.setup(df, target = target, train_size=0.75, use_gpu = False, ignore_features=(list_col_exc), session_id = 123)
    else:
        print("------------------- ini adalah advanced ------------------------")
        pycc.setup(df, target = target, train_size=0.75, 
                   preprocess = st.session_state['preprocess'],
                   normalize = st.session_state['normalize'],
                   pca = st.session_state['pca'],
                   remove_outliers = st.session_state['remove_outliers'],
                   remove_multicollinearity = st.session_state['remove_multicollinearity'],
                   polynomial_features = st.session_state['polynomial_features'],
                   feature_selection = st.session_state['feature_selection'], 
                   fix_imbalance = st.session_state['fix_imbalance'],
                   data_split_shuffle = st.session_state['data_split_shuffle'],
                   data_split_stratify = st.session_state['data_split_stratify'],
                   fold_shuffle = st.session_state['fold_shuffle'],
                   use_gpu = False,
                   ignore_features=(list_col_exc),
                   session_id = 123
                   )
                
    setup_df = pycc.pull()
    best_clf = pycc.compare_models(n_select=3)
    compare_df = pycc.pull()
    tuned_top3 = [pycc.tune_model(i) for i in best_clf]
    blend = pycc.blend_models(tuned_top3)
    stack = pycc.stack_models(tuned_top3)
    #tune_model = pycc.tune_model(best_clf, choose_better = True)
    best_acc_model = pycc.automl(optimize = 'Accuracy')
    pycc.finalize_model(best_acc_model)
    return best_clf, compare_df, best_acc_model

def run_model(model_unique_code, placeholder_unique_code_error):
    list_history_model_name = utils.get_list_model_result()
    model_unique_code += '.pkl'
    print(model_unique_code)
    print(list_history_model_name)
    #print(model_unique_code in list_history_model_name)
    if model_unique_code in list_history_model_name:
        st.session_state['error_unique_code'] = True
        return
    st.session_state['run_model'] = True
    if "error_unique_code" in st.session_state:
        del st.session_state['error_unique_code']
    if 'best_clf' in st.session_state:
        del st.session_state['best_clf']
