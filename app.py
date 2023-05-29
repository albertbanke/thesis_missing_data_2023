import streamlit as st
import pandas as pd
import plotly.express as px
import glob

# Load your DataFrames
filenames = glob.glob("results_*.csv")
df_list = []

# Read each file, cast 'target' to string, and append to list
for filename in filenames:
    df_temp = pd.read_csv(filename)
    df_temp['target'] = df_temp['target'].astype(str)
    df_list.append(df_temp)

# Concatenate all the dataframes
df = pd.concat(df_list)

def main():
    st.title('My Modeling Results')

    # Use a select box for user to select a target to view model results
    st.sidebar.title('Select a target to view model results')
    targets = ['All'] + df['target'].unique().tolist()
    select_box = st.sidebar.selectbox('Targets', targets, index=0)

    # Filter the DataFrame based on the selected target
    if select_box == 'All':
        selected_target_df = df
    else:
        selected_target_df = df[df['target'] == select_box]

    # Use a select box for user to select a data type
    st.sidebar.title('Select a data type')
    data_types = ['All'] + df['data'].unique().tolist()
    data_select_box = st.sidebar.selectbox('Data Types', data_types, index=0)

    # Filter the DataFrame based on the selected data type
    if data_select_box == 'All':
        selected_data_df = selected_target_df
    else:
        selected_data_df = selected_target_df[selected_target_df['data'] == data_select_box]

    # Use a select box for user to select a cv method
    st.sidebar.title('Select a CV method')
    cv_methods = ['All'] + df['cv_method'].unique().tolist()
    cv_select_box = st.sidebar.selectbox('CV Methods', cv_methods, index=0)

    # Filter the DataFrame based on the selected cv method
    if cv_select_box == 'All':
        selected_df = selected_data_df
    else:
        selected_df = selected_data_df[selected_data_df['cv_method'] == cv_select_box]

    # Display model results for the selected target, data type, and cv method
    st.dataframe(selected_df)

    # Create a histogram of the selected column for the selected target, data type, and cv method
    fig = px.histogram(selected_df, x='model', color='class_label')
    st.plotly_chart(fig)
    
    # Calculate the average macro_f1 score per model
    avg_macro_f1_per_model = df.groupby('model')['macro_f1'].mean().reset_index()

    # Create a bar plot of average macro_f1 score per model
    fig_model = px.bar(avg_macro_f1_per_model, x='model', y='macro_f1', title='Average Macro F1 Score per Model')
    st.plotly_chart(fig_model)

    # Calculate the average macro_f1 score per data type
    avg_macro_f1_per_data = df.groupby('data')['macro_f1'].mean().reset_index()

    # Create a bar plot of average macro_f1 score per data type
    fig_data = px.bar(avg_macro_f1_per_data, x='data', y='macro_f1', title='Average Macro F1 Score per Data Type')
    st.plotly_chart(fig_data)

if __name__ == "__main__":
    main()
