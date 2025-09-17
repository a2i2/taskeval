import streamlit as st


def qa_answering_inputs():
    """
    Display the input fields for the QA answering.
    """
    col1, col2 = st.columns([1, 1])
    with col1:
        # Text input for LLM instructions
        qa_question = st.text_area(
            "Enter your question for the LLM to answer:",
            placeholder="e.g., How do they perform multilingual training?",
            height=150,
            help="Provide clear question from the uploaded pdf",
        )

    with col2:
        # File upload
        uploaded_pdf = st.file_uploader(
            "Choose an pdf file",
            type=["pdf"],
            help="Upload the pdf for the LLM to process",
        )
    return qa_question, uploaded_pdf
