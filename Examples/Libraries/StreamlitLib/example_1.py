import numpy as np
import pandas as pd
import streamlit as st
from pandas_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report

from global_params import *
from pathlib import Path

# Web App Title
st.markdown('''
# **The EDA App**
''')

# # Upload CSV data
# with st.sidebar.header('1. Upload your CSV data'):
#     uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
#     st.sidebar.markdown("""
# [Example CSV input file](https://raw.githubusercontent.com/dataprofessor/data/master/delaney_solubility_with_descriptors.csv)
# """)

def apply_pandas_profiling_to_input():

    @st.cache
    def load_csv():
        csv = pd.read_csv(str(Path(BASEPATH) / "Data/Raw/COVID19Cases/StateLevels/us-states.csv"))
        return csv
    df = load_csv()
    
    # selected_state = st.selectbox("select state", ('florida', 'la'))
    state_name = st.selectbox("select state", np.unique(df.state.values).tolist())

    @st.cache
    def group_by_state():
        selected_df = df[df['state'] == state_name]
        return selected_df
    selected_df = group_by_state()

    pr = ProfileReport(selected_df, explorative=True)
    st.header('**Input DataFrame**')
    st.write(selected_df)

    enable_eda = st.checkbox('show pandas profiling')
    # if enable_eda:
    #     st.write('---')
    #     st.header('**Pandas Profiling Report**')
    #     st_profile_report(pr)

    enable_baseline_performance_output = st.checkbox('show baseline model performance')
    if enable_baseline_performance_output:
        st.write('ok')
        all_models_name = ['mlp', 'linear regression', 'xgboost','previous_day']
        # model_name = st.selectbox("select model", all_models_name)

        for model_name in all_models_name:
            model_name = 'previous_val' if model_name == 'previous_day' else model_name
            st.write(f"## {model_name}")
            model_name = "_".join(model_name.split(' '))
            try:
                img = str(Path(BASEPATH) / f"Outputs/Models/Performances/Baselines/{state_name}/Images/{state_name}_{model_name}_forcasting.jpg")
                st.image(img)
            except:
                img = str(Path(BASEPATH) / f"Outputs/Models/Performances/Baselines/{state_name}/Images/{state_name}_{model_name}_model_forcasting.jpg")
               st.image(img)

apply_pandas_profiling_to_input()

