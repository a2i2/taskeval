"""
TaskEval - A Streamlit app for LLM task evaluation
"""

import pandas as pd
import streamlit as st

# Page config
st.set_page_config(page_title="TaskEval - Figure Visualisation", layout="wide")

# Main app
st.title("🔬 TaskEval")
st.subheader("Figure Visualisation")


# Input section
col1, col2 = st.columns([1, 1])
with col1:
    # Original chart image
    original_chart_path = st.text_input(
        "Enter the path to the original chart image",
        help="Enter the path to the original chart image",
        value="./data/new_chart_images/",
    )
with col2:
    # Extracted table data
    extracted_table_path = st.text_input(
        "Enter the path to the generated results",
        help="Enter the path to the generated results",
        value="./outputs/figure_extractions/",
    )

col1, _ = st.columns([1, 1])
with col1:
    # Index
    data_index = st.number_input(
        "Select the index of the chart to visualise",
        help="Use the arrows to increase or decrease the index",
        min_value=1,
        value=1,
        step=1,
        format="%d",
    )

# Process button
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    process_button = st.button(
        "🔍 Visualise",
        type="primary",
        use_container_width=True,
        disabled=not (original_chart_path and extracted_table_path and data_index),
    )

# Processing logic
if process_button:
    with st.spinner("Visualising..."):
        st.header("📊 Extraction Results")
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Original chart image")
            original_chart_image = f"{original_chart_path}/{data_index}.png"
            st.image(original_chart_image)
        with col2:
            st.subheader("Regenerated Chart Image")
            regenerated_chart_image = f"{extracted_table_path}/{data_index}.png"
            st.image(regenerated_chart_image)

        df = pd.read_csv(f"{extracted_table_path}/{data_index}.csv")

        st.subheader("Extracted table data")
        st.data_editor(df, hide_index=True, num_rows="dynamic")
