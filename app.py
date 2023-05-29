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

    # Display model results for the selected target
    st.dataframe(selected_target_df)

    # Create a histogram of the selected column for the selected target
    fig = px.histogram(selected_target_df, x='model', color='class_label')
    st.plotly_chart(fig)

if __name__ == "__main__":
    main()
