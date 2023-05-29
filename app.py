import streamlit as st
import pandas as pd
import plotly.express as px

# Load your DataFrame
# (Replace this with the actual code to get your data)
df = pd.read_csv("your_modeling_results.csv")

def main():
    st.title('My Modeling Results')

    # Use a select box for user to select a column to view statistics
    st.sidebar.title('Select a column to view statistics')
    select_box = st.sidebar.selectbox('Columns', df.columns.tolist(), index=0)
    st.sidebar.write(df[select_box].describe())

    # Display the DataFrame in the app
    st.dataframe(df)

    # Create a histogram of the selected column
    fig = px.histogram(df, x=select_box)
    st.plotly_chart(fig)

if __name__ == "__main__":
    main()
    
