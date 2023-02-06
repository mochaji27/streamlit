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
import pycaret.clustering as pyccl

def main_page():
    st.title("Simple Clustering AutoML using PyCaret")
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
            
        choose_model = st.selectbox("Select Model Clustering", ['hclust', 'kmeans'])
        #df = df.drop(columns=selected_col_exc_cat).drop(columns=selected_col_exc_num)
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
        placeholder2 = st.empty()
        placeholder.button(
            "Download Model",
            disabled = True
        )
        placeholder2.button(
            "Download Result",
            disabled = True
        )
        if 'run_model' not in st.session_state:
            return

        if 'model' not in st.session_state:
            st.info("This is Experimental Machine Learning")
            col_exclude = np.concatenate((selected_col_exc_cat, selected_col_exc_num))
            st.session_state['model'] = model(df, choose_model, col_exclude)
        #pipeline_optimizer = TPOTClassifier(generations=5, population_size=20, cv=5,
        #                    random_state=42, verbosity=2, max_eval_time_mins=60)
        #pipeline_optimizer.fit(X_train, y_train)
        #score = pipeline_optimizer.score(X_test, y_test)
        #st.success(score)
        #csf = pycc.set

        pyccl.plot_model(st.session_state['model'], display_format='streamlit')
        pyccl.plot_model(st.session_state['model'], plot = 'elbow', display_format='streamlit')
        pyccl.plot_model(st.session_state['model'], plot = 'tsne', display_format='streamlit')
        pyccl.plot_model(st.session_state['model'], plot = 'silhouette', display_format='streamlit')
        pyccl.plot_model(st.session_state['model'], plot = 'distribution', display_format='streamlit')
        pyccl.plot_model(st.session_state['model'], plot = 'distance', display_format='streamlit')
        st.write(st.session_state['model'])

        #st.info("Best Parameter : " + tune_model.get_params)
        #pycc.plot_model(best_clf, plot = 'auc')
        #pycc.save_model(tune_model, "{}_{}".format(file_upload, model_name))
        kmean_results = pyccl.assign_model(st.session_state['model'])
        placeholder2.download_button(
            "Download Result Dataframe",
            data=kmean_results.to_csv().encode('utf-8'),
            file_name="result_dataframe_{}.csv".format(choose_model)
        )

        placeholder.download_button(
            "Download Model {}".format(choose_model),
            data=utils.download_model(st.session_state['model'], "model_{}".format(choose_model)),
            file_name="model_{}.pkl".format(choose_model)
        )
        #del st.session_state['run_model']
            

    
#@st.cache()
#def init_h2o():
#    h2o.connect()


def data_profiling(df):
    return df.profile_report()


def model(df, choose_model, col_exclude):
    if st.session_state["preprocess_option"] == "Default":
        print("------------------- ini adalah default ------------------------")
        pyccl.setup(df, silent = True, use_gpu = True)
    else:
        print("------------------- ini adalah advanced ------------------------")
        pyccl.setup(df, 
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
                   silent = True,
                   ignore_features=col_exclude
                   )
                
    model = pyccl.create_model(choose_model)
    return model

def run_model():
    st.session_state['run_model'] = True
    if 'model' in st.session_state:
        del st.session_state['model']