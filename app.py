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

    # Apply selected filters to the DataFrame
    selected_df = df.copy()
    if select_target_box != 'All':
        selected_df = selected_df[selected_df['target'] == select_target_box]
    if select_data_box != 'All':
        selected_df = selected_df[selected_df['data'] == select_data_box]
    if select_cv_box != 'All':
        selected_df = selected_df[selected_df['cv_method'] == select_cv_box]

    # Display the selected DataFrame in the app
    st.dataframe(selected_df)

    # Group by model and calculate average macro_f1 score
    avg_f1_by_model = selected_df.groupby('model')['macro_f1'].mean().reset_index()

    # Sort by average macro_f1 and keep top 10 models
    top_10_models = avg_f1_by_model.sort_values('macro_f1', ascending=False).head(10)['model'].tolist()

    # Filter selected_df to keep only top 10 models
    top_10_models_df = selected_df[selected_df['model'].isin(top_10_models)]

    # Code to count frequency of top features per model
    top_features_cols = ['model', 'top1_feature', 'top2_feature', 'top3_feature']
    top_features_df = top_10_models_df[top_features_cols].melt(id_vars='model').dropna()

    # Group by model and top features and count their frequency
    top_features_counts = top_features_df.groupby(['model', 'value']).size().reset_index(name='counts')
    
    top_features_counts = top_features_counts.groupby('model').apply(lambda x: x.sort_values('counts', ascending=False).head(5)).reset_index(drop=True)

    # Create a subplot for each model
    fig = make_subplots(rows=2, cols=5, subplot_titles=top_10_models)

    # Iterate over models and add a bar plot for each one
    for i, model in enumerate(top_10_models):
        df_temp = top_features_counts[top_features_counts['model'] == model]
        fig.add_trace(go.Bar(x=df_temp['value'], y=df_temp['counts'], name=model), row=(i//5)+1, col=(i%5)+1)
    
    fig.update_layout(height=800, width=1500, title_text="Top 5 Feature Importances per Model")
    st.plotly_chart(fig)
    
if __name__ == "__main__":
    main()