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

    # Use a select box for user to select a target
    st.sidebar.title('Select a target to view model results')
    select_target_box = st.sidebar.selectbox('Targets', ['All'] + df['target'].unique().tolist(), index=0)

    # Use a select box for user to select a data type
    st.sidebar.title('Select a data type')
    select_data_box = st.sidebar.selectbox('Data', df['data'].unique().tolist(), index=0)

    # Use a select box for user to select a CV method
    st.sidebar.title('Select a CV method')
    select_cv_box = st.sidebar.selectbox('CV method', df['cv_method'].unique().tolist(), index=0)

    # Apply selected filters to the DataFrame
    selected_df = df.copy()
    if select_target_box != 'All':
        selected_df = selected_df[selected_df['target'] == select_target_box]
    selected_df = selected_df[selected_df['data'] == select_data_box]
    selected_df = selected_df[selected_df['cv_method'] == select_cv_box]

    # Display the selected DataFrame in the app
    st.dataframe(selected_df)

    # Group by model and calculate average macro_f1 score
    avg_f1_by_model = selected_df.groupby('model')['macro_f1'].mean().reset_index()
    avg_f1_by_model_fig = px.bar(avg_f1_by_model, x='model', y='macro_f1', title='Average Macro F1 Score per Model')
    st.plotly_chart(avg_f1_by_model_fig)

    # Group by data type and calculate average macro_f1 score
    avg_f1_by_data = selected_df.groupby('data')['macro_f1'].mean().reset_index()
    avg_f1_by_data_fig = px.bar(avg_f1_by_data, x='data', y='macro_f1', title='Average Macro F1 Score per Data Type')
    st.plotly_chart(avg_f1_by_data_fig)

    # Code to count frequency of top features per model
    top_features_cols = ['model', 'top1_feature', 'top2_feature', 'top3_feature']
    top_features_df = selected_df[top_features_cols].melt(id_vars='model').dropna()

    # Group by top features and count their frequency
    top_features_counts = top_features_df.value_counts().rename('counts').reset_index()

    # Filter the top_features_counts DataFrame based on the selected model
    selected_top_features_counts = top_features_counts[top_features_counts['model'] == selected_df['model'].iloc[0]]

    # Create a bar plot of top features for the selected model
    fig_model_features = px.bar(selected_top_features_counts, x='value', y='counts', title=f'Top Features Importance for {selected_df["model"].iloc[0]} Model')
    st.plotly_chart(fig_model_features)

if __name__ == "__main__":
    main()
