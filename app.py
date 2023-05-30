import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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
    select_data_box = st.sidebar.selectbox('Data', ['All'] + df['data'].unique().tolist(), index=0)

    # Use a select box for user to select a CV method
    st.sidebar.title('Select a CV method')
    select_cv_box = st.sidebar.selectbox('CV method', ['All'] + df['cv_method'].unique().tolist(), index=0)

    # Use a select box for user to select a class label
    st.sidebar.title('Select a class label')
    select_class_label_box = st.sidebar.selectbox('Class Label', ['All'] + df['class_label'].unique().tolist(), index=0)

    # Apply selected filters to the DataFrame
    selected_df = df.copy()
    if select_target_box != 'All':
        selected_df = selected_df[selected_df['target'] == select_target_box]
    if select_data_box != 'All':
        selected_df = selected_df[selected_df['data'] == select_data_box]
    if select_cv_box != 'All':
        selected_df = selected_df[selected_df['cv_method'] == select_cv_box]
    if select_class_label_box != 'All':
        selected_df = selected_df[selected_df['class_label'] == select_class_label_box]

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
    
    # Group by model and CV method, then calculate average macro_f1 score
    avg_f1_by_model_cv = selected_df.groupby(['model', 'cv_method'])['macro_f1'].mean().reset_index()
    avg_f1_by_model_cv_fig = px.bar(avg_f1_by_model_cv, x='model', y='macro_f1', color='cv_method', 
                                    barmode='group', title='Average Macro F1 Score per Model and CV Method')
    st.plotly_chart(avg_f1_by_model_cv_fig)

    # Group by model and target, then calculate average macro_f1 score
    avg_f1_by_model_target = selected_df.groupby(['model', 'target'])['macro_f1'].mean().reset_index()
    avg_f1_by_model_target_fig = px.bar(avg_f1_by_model_target, x='model', y='macro_f1', color='target',
                                        barmode='group', title='Average Macro F1 Score per Model and Target')
    st.plotly_chart(avg_f1_by_model_target_fig)

    # Code to count frequency of top features per model
    top_features_cols = ['model', 'top1_feature', 'top2_feature', 'top3_feature']
    top_features_df = selected_df[top_features_cols].melt(id_vars='model').dropna()

    # Group by top features and count their frequency
    top_features_counts = top_features_df.value_counts().rename('counts').reset_index()

    # Sort features by counts in descending order and select top 5
    top_features_counts = top_features_counts.sort_values('counts', ascending=False).head(5)

    # Create a bar plot of top 5 features
    fig_model_features = px.bar(top_features_counts, x='value', y='counts', title=f'Top 5 Features Importance')
    st.plotly_chart(fig_model_features)

if __name__ == "__main__":
    main()