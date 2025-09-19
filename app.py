"""
TaskEval - A Streamlit app for LLM task evaluation
"""

import os
import json

from typing import Tuple, Optional
from datetime import datetime

import streamlit as st
from PIL import Image

from taskevals.figure_extractions import figure_extractions
from taskevals.qa_answering import qa_answering
from taskevals.task_classification import classify_task
from ui.figure_extractions import figure_extractions_results
from ui.qa_answering import qa_answering_results

CACHE_DIR = "cache"
os.makedirs(CACHE_DIR, exist_ok=True)

# Page config
st.set_page_config(page_title="TaskEval - LLM Task Evaluation", layout="wide")


# Helper functions
def process_uploaded_file(file_upload) -> Tuple[Optional[str], Optional[str]]:
    """Process uploaded file and return PIL Image object when it's an image"""
    if file_upload.type == "application/pdf":
        # Process uploaded pdf
        pdf_path = os.path.join(
            CACHE_DIR, f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        )
        # Save pdf to local directory
        with open(pdf_path, "wb") as f:
            f.write(file_upload.getvalue())
        return pdf_path, None
    try:
        img = Image.open(file_upload)
        # Save image to local directory
        image_path = os.path.join(
            CACHE_DIR, f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        )
        img.save(image_path)
        return image_path, None
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
col1, col2 = st.columns([1, 1])
with col1:
    # Text input for LLM instructions
    instruction = st.text_area(
        "Enter your instructions for the LLM to perform:",
        placeholder=(
            "e.g., Extract the data points in the chart "
            "image and provide the output as a table or "
            "Answer this question by using the document: "
            "How do they perform multilingual training?"
        ),
        height=150,
        help=(
            "Provide clear instructions for what you want "
            "the LLM to do with the uploaded image"
        ),
    )
with col2:
    # File upload
    uploaded_file = st.file_uploader(
        "Choose an image or pdf file",
        type=["png", "jpg", "jpeg", "pdf"],
        help="Upload the diagram for the LLM to process",
    )
expected_answer = st.text_area(
    "Enter your expected results for evaluating the LLM's answer:",
    placeholder="e.g., table results from the image",
    height=100,
    help="Provide clear expected results from the uploaded image",
)

# Process button
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    process_button = st.button(
        "🚀 Perform task and evaluate",
        type="primary",
        use_container_width=True,
        disabled=not (instruction and uploaded_file),
    )

# Processing logic
if process_button:
    with st.spinner("Processing with LLM..."):
        processed_file_path, error = process_uploaded_file(uploaded_file)
        if error:
            st.error(f"❌ {error}")
        else:
            # Get the task classification
            try:
                task_classification = classify_task(instruction)
                parsed_classification = json.loads(task_classification)
                if "task_classification" not in parsed_classification:
                    raise Exception("Expected keyword 'task_classification' not found")
                task = parsed_classification["task_classification"]
            except Exception as e:
                st.error(f"❌ Failed to classify task: {str(e)}")

            if task == "FIGURE_EXTRACTION":
                results = figure_extractions(instruction, processed_file_path)
                figure_extractions_results(processed_file_path, results)
                st.session_state.processed = True
            elif task == "QA_ANSWERING":
                results = qa_answering(
                    instruction, processed_file_path, expected_answer
                )
                qa_answering_results(results)
                st.session_state.processed = True
            else:
                st.error(f"❌ Invalid task: {st.session_state.task}")
