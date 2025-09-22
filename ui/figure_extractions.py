import cairosvg
import streamlit as st


def figure_extractions_results(image_path, results):
    """
    Display the output results for the figure extractions.
    """
    st.header("📊 Extraction Results")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original chart image")
        st.image(image_path)
    with col2:
        st.subheader("Regenerated Chart Image")
        if results["regenerated_chart_image"] == "":
            st.error("No regenerated chart image available")
        else:
            png_bytes = cairosvg.svg2png(
                bytestring=results["regenerated_chart_image"].encode("utf-8")
            )
            st.image(png_bytes)

    st.subheader("Extracted table data")
    st.data_editor(results["extracted_table_data"], hide_index=True, num_rows="dynamic")
