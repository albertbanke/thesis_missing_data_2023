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

    # Use a select box for user to select a column to view statistics
    st.sidebar.title('Select a column to view statistics')
    select_box = st.sidebar.selectbox('Columns', df.columns.tolist(), index=0)
    st.sidebar.write(df[select_box].describe())

    # Use a select box for user to select a data type
    st.sidebar.title('Select a data type')
    data_select_box = st.sidebar.selectbox('Data', df['data'].unique().tolist(), index=0)
    selected_data_df = df[df['data'] == data_select_box]

    # Display the selected DataFrame in the app
    st.dataframe(selected_data_df)

    # Create a histogram of the selected column for the selected data type
    fig = px.histogram(selected_data_df, x=select_box)
    st.plotly_chart(fig)

if __name__ == "__main__":
    main()
