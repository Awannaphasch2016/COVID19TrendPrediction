import numpy as np
import pandas as pd
import streamlit as st
from pandas_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report

from global_params import *
from pathlib import Path


def display_baseline_performance_output(**kwargs):
    st.write('## Baseline Performance Output')
    notes = []
    performance_table = []
    for model_name in ALL_BASELINES_MODELS:
        model_name = 'previous_val' if model_name == 'previous_day' else model_name
        model_name = "_".join(model_name.split(' '))
        params = kwargs['params'].copy()
        params.append(model_name)
        selected_metrics = kwargs['selected_metrics']
        sort_by_metric = kwargs['sort_by_metric']
        try:
            performance_result = pd.read_csv(str(Path(BASEPATH + FRAME_PERFORMANCE_PATH.format(*params))))
            performance_result.index = [model_name]
            performance_table.append(performance_result[selected_metrics])
        except:
            try:
                FRAME_PERFORMANCE_PATH_2 =  "/Outputs/Models/Performances/Baselines/PredictNext{}/{}/{}_{}_model_performance.csv"
                # st.write(str(Path(BASEPATH + FRAME_PERFORMANCE_PATH_2.format(*params))))
                performance_result = pd.read_csv(str(Path(BASEPATH + FRAME_PERFORMANCE_PATH_2.format(*params))))
                performance_result.index = [model_name]
                performance_table.append(performance_result[selected_metrics])
            except:
                notes.append(f'{model_name} performance result is not recorded')
    if len(performance_table) > 0:
        performance_table_df = pd.concat(performance_table)
        st.write(performance_table_df.sort_values(by=[sort_by_metric]))
    if len(notes) > 0: 
        st.write( 'Note: \n' + '\n\t'.join(notes))

def display_baseline_plot(**kwargs):
    st.write('## Baseline Plot')
    for model_name in ALL_BASELINES_MODELS:
        model_name = 'previous_val' if model_name == 'previous_day' else model_name
        st.write(f"## {model_name}")
        model_name = "_".join(model_name.split(' '))

        params = kwargs['params'].copy()
        params.append(model_name)

        try:
            img = str(Path(BASEPATH + PLOT_PATH.format(*params)))
            st.image(img)
        except:
            try:
                PLOT_PATH_2 =  "/Outputs/Models/Performances/Baselines/PredictNext{}/{}/Images/{}_{}_model_forcasting.jpg"
                img = str(Path(BASEPATH + PLOT_PATH_2.format(*params)))
                st.image(img)
            except:
                st.write('not yet exist')

def display_eda(selected_df):
    pr = ProfileReport(selected_df, explorative=True)
    st.write('---')
    st.header('**Pandas Profiling Report**')
    st_profile_report(pr)


# Web App Title
st.markdown('''
# **The EDA App**
''')


def apply_pandas_profiling_to_input():

    @st.cache
    def load_csv():
        csv = pd.read_csv(str(Path(BASEPATH) / "Data/Raw/COVID19Cases/StateLevels/us-states.csv"))
        return csv
    df = load_csv()
    
    # selected_state = st.selectbox("select state", ('florida', 'la'))
    state_name = st.sidebar.selectbox("select state", np.unique(df.state.values).tolist())

    # selected_state = st.selectbox("select state", ('florida', 'la'))
    pred_length = st.sidebar.selectbox("prediction length", [1,7])

    @st.cache
    def group_by_state():
        selected_df = df[df['state'] == state_name]
        return selected_df
    selected_df = group_by_state()

    st.header('**Input DataFrame**')
    st.write(selected_df)

    # st.write()
    
    enable_eda = st.sidebar.checkbox('show pandas profiling')
    enable_baseline_performance_output = st.sidebar.checkbox('show baseline model performance')
    enable_baseline_plot = st.sidebar.checkbox('show baseline plot')
    selected_metrics = st.sidebar.multiselect('select evaluation metrics', ALL_METRICS, default='mse')
    sort_by_metric = st.sidebar.selectbox('sort_by_metric', ALL_METRICS)

    if enable_eda:
        display_eda(selected_df)
    if enable_baseline_performance_output:
        display_baseline_performance_output(params=[pred_length, state_name, state_name], selected_metrics=selected_metrics, sort_by_metric=sort_by_metric)
    if enable_baseline_plot:
        display_baseline_plot(params=[pred_length, state_name, state_name])

if __name__ == '__main__':
    apply_pandas_profiling_to_input()

