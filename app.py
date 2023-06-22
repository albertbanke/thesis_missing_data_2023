# Imports to get the streamlit app to run
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import glob
import numpy as np
import geopandas as gpd
import matplotlib
import mapclassify
from datetime import datetime
import streamlit.components.v1 as components

# Set page layout
st.set_page_config(layout="wide")

# Load the DataFrames
filenames = glob.glob("results/results_*.csv")
df_list = []

# Read each file, cast 'target' to string, and append to list
for filename in filenames:
    df_temp = pd.read_csv(filename)
    df_temp['target'] = df_temp['target'].astype(str)
    df_list.append(df_temp)

# Concatenate all the dataframes
df = pd.concat(df_list)

# Load the GeoDataFrame
gdf_engineered = gpd.read_parquet('data_frames/gdf_nonproc.parquet')  

def main():
    
    st.title('Missing Data Modeling Results')
    
    # Define two columns
    col1, col2 = st.sidebar.columns(2)

    # Use a select box for user to select a target
    col1.title('Select a target to model')
    select_target_box = col1.selectbox('Targets', ['All'] + df['target'].unique().tolist(), index=0)

    # Use a select box for user to select a data type
    col2.title('Select a data type')
    select_data_box = col2.selectbox('Data', ['All'] + df['data'].unique().tolist(), index=0)

    # Use a select box for user to select a CV method
    col1.title('Select a CV method')
    select_cv_box = col1.selectbox('CV method', ['All'] + df['cv_method'].unique().tolist(), index=0)

    # Use a select box for user to select a class label
    col2.title('Select a class label')
    select_class_label_box = col2.selectbox('Class Label', ['All'] + df['class_label'].unique().tolist(), index=0)

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
    st.title('Data View')
    
        # Wrap the map in an expand section
    with st.expander("Click to select columns"):
        # Add a multi-select box for user to select/deselect columns for viewing
        columns_to_display = st.multiselect('Columns', selected_df.columns.tolist(), default = selected_df.columns.tolist())
        
    # Display DataFrame with selected columns
    st.dataframe(selected_df[columns_to_display])
    
    # Group by data type and calculate average macro_f1 score
    avg_f1_by_data = selected_df.groupby('data')['macro_f1'].mean().reset_index()
    avg_f1_by_data_fig = px.bar(avg_f1_by_data, x='data', y='macro_f1', title='Average Macro F1 Score per Data Type')
    st.plotly_chart(avg_f1_by_data_fig)

    # Group by model and calculate average macro_f1 score
    avg_f1_by_model = selected_df.groupby('model')['macro_f1'].mean().reset_index()
    avg_f1_by_model_fig = px.bar(avg_f1_by_model, x='model', y='macro_f1', title='Average Macro F1 Score per Model')
    st.plotly_chart(avg_f1_by_model_fig)
    
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
    
    # Add a slider to adjust the range of the matthews_corr
    matthews_corr_range = st.sidebar.slider('Select Matthews Correlation Coefficient Range (only for Top 5 features)', 
                                            min_value=float(selected_df['matthews_corr'].min()), 
                                            max_value=float(selected_df['matthews_corr'].max()), 
                                            value=(float(selected_df['matthews_corr'].min()), float(selected_df['matthews_corr'].max())))

    # Filter the DataFrame according to the selected matthews_corr range
    selected_df = selected_df[(selected_df['matthews_corr'] >= matthews_corr_range[0]) & (selected_df['matthews_corr'] <= matthews_corr_range[1])]

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

    # List of available metrics 
    available_metrics = ['balanced_accuracy', 'recall', 'macro_f1', 'matthews_corr']  # selected attributes

    # Add a select box in the main area of the app 
    selected_metric = st.selectbox('Select a metric for the heatmap', available_metrics)

    # Create a pivot table with 'model' and 'target' as index and columns respectively, and the selected metric as values
    pivot = selected_df.pivot_table(values=selected_metric, index='model', columns='target')

    # Round off values to one decimal place
    pivot_rounded = pivot.round(2)
    
    # Create a heatmap using plotly
    heatmap_fig = ff.create_annotated_heatmap(z=pivot_rounded.values, x=pivot.columns.tolist(), y=pivot.index.tolist(), colorscale='YlGnBu', 
                                            annotation_text=pivot_rounded.values.astype(str))

    # Add title
    heatmap_fig.update_layout(title=f'Average {selected_metric.capitalize()} Heatmap per Model and Target', title_font=dict(size=18))

    # Render the plot
    st.plotly_chart(heatmap_fig)
    
    # Create a violin plot
    violin_fig = px.violin(selected_df, y="model", x="matthews_corr", box=True, points="all")

    # Add title
    violin_fig.update_layout(title='Model Performance Distribution', xaxis_title='Matthews Correlation', yaxis_title='Model')

    # Render the plot
    st.plotly_chart(violin_fig)
    
    # Convert 'time' from seconds to minutes
    selected_df['time'] = selected_df['time'] / 60

    # Create a scatter plot of 'time' vs 'matthews_corr' with colored markers based on 'data' category
    time_vs_matthews_fig = px.scatter(selected_df, x='time', y='matthews_corr', color='model')
    time_vs_matthews_fig.update_layout(title='Trade-off between Time and Matthews Correlation Coefficient',
                                    yaxis_title='Matthews Correlation Coefficient')
    time_vs_matthews_fig.update_xaxes(type="log", title_text = 'Time (in log minutes)')  # log scale for x-axis

    # Calculate the polynomial fit (degree 2 for a curve)
    poly_fit = np.polyfit(selected_df['time'], selected_df['matthews_corr'], 2)

    # Get the polynomial function
    poly_func = np.poly1d(poly_fit)

    # Generate x values
    x_poly = np.linspace(min(selected_df['time']), max(selected_df['time']), 400)

    # Generate y values
    y_poly = poly_func(x_poly)

    # Add the polynomial line to the figure
    time_vs_matthews_fig.add_trace(go.Scatter(x=x_poly, y=y_poly, mode='lines', name='Trendline'))

    st.plotly_chart(time_vs_matthews_fig)

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