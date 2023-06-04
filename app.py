# Imports to get the streamlit app to run
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import glob
import geopandas as gpd
import matplotlib
import mapclassify
from datetime import datetime
import streamlit.components.v1 as components

# Set page layout
st.set_page_config(layout="wide")

# Load the DataFrames
filenames = glob.glob("results_*.csv")
df_list = []

# Read each file, cast 'target' to string, and append to list
for filename in filenames:
    df_temp = pd.read_csv(filename)
    df_temp['target'] = df_temp['target'].astype(str)
    df_list.append(df_temp)

# Concatenate all the dataframes
df = pd.concat(df_list)

# Load the GeoDataFrame
gdf_engineered = gpd.read_parquet('gdf_nonproc.parquet')  

def main():
    
    st.title('My Modeling Results')
    
    # Define two columns
    col1, col2 = st.sidebar.columns(2)

    # Use a select box for user to select a target
    col1.title('Select a target to model')
    select_target_box = col1.selectbox('Targets', ['All'] + df['target'].unique().tolist(), index=0)

    # Use a select box for user to select a data type
    col2.title('Select a data type')
    select_data_box = col2.selectbox('Data', ['All'] + df['data'].unique().tolist(), index=0)

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
    top_features_cols = ['top1_feature', 'top2_feature', 'top3_feature']
    top_features_df = selected_df[top_features_cols].melt().dropna()

    # Group by top features and count their frequency
    top_features_counts = top_features_df['value'].value_counts().reset_index()
    top_features_counts.columns = ['feature', 'counts']

    # Sort features by counts in descending order and select top 5
    top_features_counts = top_features_counts.sort_values('counts', ascending=False).head(5)

    # Create a bar plot of top 5 features
    fig_model_features = px.bar(top_features_counts, x='feature', y='counts', title='Top 5 Features Importance')
    st.plotly_chart(fig_model_features)

    # Create a pivot table with 'model' and 'target' as index and columns respectively, and 'balanced_accuracy' as values
    pivot = selected_df.pivot_table(values='balanced_accuracy', index='model', columns='target')

    # Round off values to one decimal place
    pivot_rounded = pivot.round(2)

    # Create a heatmap using plotly
    heatmap_fig = ff.create_annotated_heatmap(z=pivot_rounded.values, x=pivot.columns.tolist(), y=pivot.index.tolist(), colorscale='YlGnBu', 
                                            annotation_text=pivot_rounded.values.astype(str))

    # Add title
    heatmap_fig.update_layout(title='Average Balanced Accuracy Heatmap per Model and Target')

    # Render the plot
    st.plotly_chart(heatmap_fig)
    
    # Create a violin plot
    violin_fig = px.violin(selected_df, x="model", y="matthews_corr", box=True, points="all")

    # Add title
    violin_fig.update_layout(title='Model Performance Distribution', xaxis_title='Model', yaxis_title='Matthews Correlation')

    # Render the plot
    st.plotly_chart(violin_fig)

    # Add a new section for the map
    st.sidebar.title('Interactive Map Settings')

    # Use a select box for user to select a feature to plot
    select_feature_box = st.sidebar.selectbox('Feature', gdf_engineered.columns.tolist())

    # Get unique years in the DataFrame
    unique_years = sorted([int(year) for year in gdf_engineered['year'].unique().tolist()])

    # Use a select box for user to select a year
    st.sidebar.title('Select a year')
    select_year_box = st.sidebar.selectbox('Year', unique_years)

    # Filter the GeoDataFrame based on the selected year
    selected_gdf = gdf_engineered[gdf_engineered['year'] == select_year_box]
    
    # Add a new section for the map
    st.subheader('Interactive Map')
    
    # Wrap the map in an expand section
    with st.expander("Click to expand map"):
        # Use the .explore() function from GeoPandas
        m = selected_gdf.explore(column=select_feature_box, legend=True)
        
        # Render the map in Streamlit
        components.html(m._repr_html_(), height=800, width=1000)

if __name__ == "__main__":
    main()