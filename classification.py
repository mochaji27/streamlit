import streamlit as st
import pandas as pd
import pandas_profiling 
from streamlit_pandas_profiling import st_profile_report
import utils
#from tpot import TPOTClassifier, TPOTRegressor
import numpy as np
#import pycaret.classification as pycc
#import h2o 
#from h2o.automl import H2OAutoML
import pycaret.classification as pycc

def main_page():
    st.title("Simple Classification AutoML using PyCaret")
    st.write("Upload your data")
    file_upload = st.file_uploader("Choose a XLSX file", type=["xlsx", "csv"])
    if file_upload is None:
        if 'run_model' in st.session_state:
            del st.session_state['run_model']
        if 'data_profiling' in st.session_state:
            del st.session_state['data_profiling']
        return

    df = utils.check_file(file_upload)
    tab_view, tab_profile, tab_predict = st.tabs(["View Data", "Data Profiling", "ML"])
    with tab_view:
        st.dataframe(df)
    with tab_profile:
        if 'data_profiling' not in st.session_state:
            st.session_state['data_profiling'] = data_profiling(df)
        st_profile_report(st.session_state['data_profiling'])
    with tab_predict:
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
                st.checkbox("handle_unknown_categorical", key="handle_unknown_categorical")
                st.checkbox("pca", key="pca")
                st.checkbox("ignore_low_variance", key="ignore_low_variance")
                st.checkbox("feature_selection", key="feature_selection")
            with col2:
                st.checkbox("remove_multicollinearity", key="remove_multicollinearity")
                st.checkbox("remove_perfect_collinearity", key="remove_perfect_collinearity")
                st.checkbox("create_clusters", key="create_clusters")
                st.checkbox("polynomial_features", key="polynomial_features")
                st.checkbox("trigonometry_features", key="trigonometry_features")
                st.checkbox("fold_shuffle", key="fold_shuffle")
            with col3: 
                st.checkbox("feature_interaction", key="feature_interaction")
                st.checkbox("feature_ratio", key="feature_ratio")
                st.checkbox("fix_imbalance", key="fix_imbalance")
                st.checkbox("data_split_shuffle", key="data_split_shuffle")
                st.checkbox("remove_outliers", key="remove_outliers")
                st.checkbox("data_split_stratify", key="data_split_stratify")
                
        st.markdown("""---""")
        #y = df[target].values
        #X = df.drop(columns=target)
        #x = pd.get_dummies(data=x, columns=x.select_dtypes(exclude=[np.float64, np.int8, np.int16, np.int32, np.int64]).columns)
        #X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75, test_size=0.25, random_state=2022)
        st.button('Go', key="key_classification_btn", on_click=run_model)
        placeholder = st.empty()
        placeholder.button(
            "Download Model",
            disabled = True
        )
        if 'run_model' not in st.session_state:
            return

        if 'best_clf' not in st.session_state:
            st.info("This is Experimental Machine Learning")
            st.session_state['best_clf'], st.session_state['compare_df'], st.session_state['tune_model'] = model(df, target)
        #pipeline_optimizer = TPOTClassifier(generations=5, population_size=20, cv=5,
        #                    random_state=42, verbosity=2, max_eval_time_mins=60)
        #pipeline_optimizer.fit(X_train, y_train)
        #score = pipeline_optimizer.score(X_test, y_test)
        #st.success(score)
        #csf = pycc.set

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

        #st.info("Best Parameter : " + tune_model.get_params)
        #pycc.plot_model(best_clf, plot = 'auc')
        #pycc.save_model(tune_model, "{}_{}".format(file_upload, model_name))
        placeholder.download_button(
            "Download Model {}".format(model_name),
            data=utils.download_model(st.session_state['tune_model']),
            file_name="model_{}.pkl".format(model_name)
        )
        #del st.session_state['run_model']
                

    
#@st.cache()
#def init_h2o():
#    h2o.connect()


def data_profiling(df):
    return df.profile_report()


def model(df, target):
    if st.session_state["preprocess_option"] == "Default":
        print("------------------- ini adalah default ------------------------")
        pycc.setup(df, target = target, train_size=0.75, silent = True, use_gpu = True)
    else:
        print("------------------- ini adalah advanced ------------------------")
        pycc.setup(df, target = target, train_size=0.75, 
                   preprocess = st.session_state['preprocess'],
                   normalize = st.session_state['normalize'],
                   handle_unknown_categorical = st.session_state['handle_unknown_categorical'],
                   pca = st.session_state['pca'],
                   ignore_low_variance = st.session_state['ignore_low_variance'],
                   remove_outliers = st.session_state['remove_outliers'],
                   remove_multicollinearity = st.session_state['remove_multicollinearity'],
                   remove_perfect_collinearity = st.session_state['remove_perfect_collinearity'],
                   create_clusters = st.session_state['create_clusters'],
                   polynomial_features = st.session_state['polynomial_features'],
                   trigonometry_features = st.session_state['trigonometry_features'],
                   feature_selection = st.session_state['feature_selection'], 
                   feature_interaction = st.session_state['feature_interaction'],
                   feature_ratio = st.session_state['feature_ratio'],
                   fix_imbalance = st.session_state['fix_imbalance'],
                   data_split_shuffle = st.session_state['data_split_shuffle'],
                   data_split_stratify = st.session_state['data_split_stratify'],
                   fold_shuffle = st.session_state['fold_shuffle'],
                   use_gpu = True,
                   silent = True
                   )
                
    setup_df = pycc.pull()
    best_clf = pycc.compare_models()
    compare_df = pycc.pull()
    tune_model = pycc.tune_model(best_clf, choose_better = True)
    pycc.finalize_model(tune_model)
    return best_clf, compare_df, tune_model

def run_model():
    st.session_state['run_model'] = True
    if 'best_clf' in st.session_state:
        del st.session_state['best_clf']