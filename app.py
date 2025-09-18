"""
TaskEval - A Streamlit app for LLM task evaluation
"""

import os
from datetime import datetime

import streamlit as st
from PIL import Image

from taskevals.figure_extractions import figure_extractions
from taskevals.qa_answering import qa_answering
from ui.figure_extractions import figure_extractions_inputs, figure_extractions_results
from ui.qa_answering import qa_answering_inputs, qa_answering_results

CACHE_DIR = "cache"
os.makedirs(CACHE_DIR, exist_ok=True)

# Page config
st.set_page_config(page_title="TaskEval - LLM Task Evaluation", layout="wide")


# Helper functions
def process_uploaded_image(file_upload):
    """Process uploaded image and return PIL Image object"""
    try:
        img = Image.open(file_upload)
        return img, None
    except (IOError, OSError) as e:
        return None, f"Error processing image: {str(e)}"


# Main app
st.title("🔬 TaskEval")
st.subheader("LLM Task Evaluation Platform")

# Initialize session state
if "processed" not in st.session_state:
    st.session_state.processed = False
if "task" not in st.session_state:
    st.session_state.task = "figure_extractions"

# Input section
st.header("📝 Inputs")

tab1, tab2 = st.tabs(["Figure Extraction", "QA Answering"])
with tab1:
    instruction, uploaded_file = figure_extractions_inputs()
with tab2:
    qa_question, expected_answer, uploaded_pdf = qa_answering_inputs()

if tab1:
    st.session_state.task = "figure_extractions"
if tab2:
    st.session_state.task = "qa_answering"

# Process button
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    process_button = st.button(
        "🚀 Process with LLM",
        type="primary",
        use_container_width=True,
        disabled=not (
            (instruction and uploaded_file)
            or (qa_question and uploaded_pdf and expected_answer)
        ),
    )

# Processing logic
if (
    process_button
    and (instruction and uploaded_file)
    or (qa_question and uploaded_pdf and expected_answer)
):
    with st.spinner("Processing with LLM..."):
        if st.session_state.task == "figure_extractions":
            # Process uploaded image
            image, error = process_uploaded_image(uploaded_file)
            # Save image to local directory
            image_path = os.path.join(
                CACHE_DIR, f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            )
            image.save(image_path)

            if error:
                st.error(f"❌ {error}")
            else:
                # Simulate LLM processing
                results = figure_extractions(instruction, image_path)
                figure_extractions_results(image_path, results)
                st.session_state.processed = True

        elif st.session_state.task == "qa_answering":
            # Process uploaded pdf
            pdf_path = os.path.join(
                CACHE_DIR, f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
            )
            # Save pdf to local directory
            with open(pdf_path, "wb") as f:
                f.write(uploaded_pdf.getvalue())

            # Get answer from the PDF
            results = qa_answering(qa_question, pdf_path, expected_answer)
            qa_answering_results(results)
            st.session_state.processed = True
        else:
            st.error(f"❌ Invalid task: {st.session_state.task}")
