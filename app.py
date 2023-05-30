import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import glob
import geopandas as gpd
import pydeck as pdk

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

# Read the nonprocessed GeoParquet file
gdf_nonproc = gpd.read_parquet('gdf_nonproc.parquet')

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

    # Create a subplot
    fig = go.Figure()

    # Create a plot for each model
    for model in selected_df['model'].unique():
        selected_top_features_counts = top_features_counts[top_features_counts['model'] == model]
        fig.add_trace(
            go.Bar(x=selected_top_features_counts['value'], y=selected_top_features_counts['counts'], name=model, visible=False)
        )

    # Make first model visible
    fig.data[0].visible = True

    # Create dropdown menu
    buttons = []
    for i, model in enumerate(selected_df['model'].unique()):
        visibility = [False]*len(fig.data)
        visibility[i] = True
        button = dict(
            label = model,
            method = 'update',
            args = [{'visible': visibility}]
        )
        buttons.append(button)

    fig.update_layout(
        updatemenus=[
            dict(
                type="buttons",
                direction="down",
                showactive=True,
                buttons=buttons
            )
        ]
    )

    st.plotly_chart(fig)

    # Create a Pydeck chart
    map_chart = pdk.Deck(
        map_style='mapbox://styles/mapbox/light-v9',
        initial_view_state=pdk.ViewState(
            latitude=selected_gdf_df['lat'].mean(),
            longitude=selected_gdf_df['lon'].mean(),
            zoom=11,
            pitch=50,
        ),
        layers=[
            pdk.Layer(
                'ScatterplotLayer',
                data=selected_gdf_df,
                get_position=['lon', 'lat'],
                get_color=[200, 30, 0, 160],
                get_radius=100,
            ),
        ],
    )

    # Show the Pydeck chart in the app
    st.pydeck_chart(map_chart)
    
if __name__ == "__main__":
    main()