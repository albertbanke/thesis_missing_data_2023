import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import glob
import geopandas as gpd

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

    # Create two columns layout
    col1, col2 = st.columns(2)

    # First column
    with col1:
        # Display the selected DataFrame in the app
        st.dataframe(selected_df)

    # Second column
    with col2:
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
        
        # Sort features by counts in descending order and select top 5
        top_features_counts = top_features_counts.sort_values('counts', ascending=False).head(5)

        # Create a bar plot of top 5 features
        fig_model_features = px.bar(top_features_counts, x='value', y='counts', title=f'Top 5 Features Importance')
        st.plotly_chart(fig_model_features)

        # Additional code for the second pane (visualization with maps)
        gdf_nonproc = pd.read_csv("your_spatial_data.csv")

        df_2020 = gdf_nonproc[gdf_nonproc['year'] == 2020]

        # Create a map of the selected column
        selected_column = col2.selectbox('Select a column to explore', df_2020.columns.tolist(), index=0)
        fig_map = px.choropleth_mapbox(df_2020, geojson=df_2020.geometry, locations=df_2020.index,
                                       color=selected_column, color_continuous_scale='Viridis',
                                       mapbox_style='carto-positron', zoom=1, center={"lat": 0, "lon": 0})
        fig_map.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
        st.plotly_chart(fig_map)

if __name__ == "__main__":
    main()