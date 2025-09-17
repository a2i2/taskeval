import json

import streamlit as st


def figure_extractions_inputs():
    """
    Display the input fields for the figure extractions.
    """
    col1, col2 = st.columns([1, 1])
    with col1:
        # Text input for LLM instructions
        instruction = st.text_area(
            "Enter your instructions for the LLM to perform:",
            placeholder="e.g., Extract data points from this chart...",
            height=150,
            help=(
                "Provide clear instructions for what you want "
                "the LLM to do with the uploaded image"
            ),
        )
    with col2:
        # File upload
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=["png", "jpg", "jpeg", "gif", "bmp", "tiff"],
            help="Upload the diagram for the LLM to process",
        )
    return instruction, uploaded_file


def figure_extractions_results(image_path, results):
    """
    Display the output results for the figure extractions.
    """
    st.header("📊 Output Results")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original chart image")
        st.image(image_path)
    with col2:
        st.subheader("Extracted table data")
        st.dataframe(results["extracted_table_data"])

    st.header("Evaluation strategy")
    # Add notes to say that only visualize strategy is supported for now
    st.markdown(
        (
            "**Note:** Only 'visualize' strategy is supported for now. "
            "Other strategies will be added in the future."
        )
    )
    critical_errors = results["critical_errors"]
    evaluation_strategies = results["error_strategy_mapping"]
    rationales = results["mapping_rationale"]

    if (
        critical_errors
        and len(
            [eval for eval in evaluation_strategies.values() if eval == "visualize"]
        )
        > 0
    ):
        # Create tabs dynamically
        tab_names = []
        for error in critical_errors:
            if evaluation_strategies[error["error_name"]] != "visualize":
                continue
            error_name = error["error_name"]
            display_error_name = f"{error_name[0].upper()}{error_name[1:]}".replace(
                "_", " "
            )
            tab_names.append(f"{display_error_name} ({error['likelihood']})")

        tabs = st.tabs(tab_names)
        tab_counter = 0
        for error in critical_errors:
            if evaluation_strategies[error["error_name"]] != "visualize":
                continue
            with tabs[tab_counter]:
                error_name = error["error_name"]
                evaluation_strategy = evaluation_strategies[error_name]
                evaluation_strategy = (
                    f"{evaluation_strategy[0].upper()}{evaluation_strategy[1:]}"
                )
                evaluation_strategy = evaluation_strategy.replace("_", " ")

                st.markdown(f"**Description:** {error['description']}")
                st.markdown(
                    f"**Recommended evaluation strategy:** *{evaluation_strategy}*"
                )
                st.markdown(f"**Rationale:**\n{rationales[error_name]}")
                tab_counter += 1

        st.subheader("Regenerated chart image")
        # TODO: Add regenerated chart image
        st.error("No regenerated chart image available")
    else:
        st.info("No critical errors found.")

    temp_results = results
    temp_results["extracted_table_data"] = results["extracted_table_data"].to_dict(
        orient="records"
    )

    # Make this collapsible
    with st.expander("Raw evaluation results"):
        st.code(
            json.dumps(temp_results, indent=2, ensure_ascii=False),
            language="json",
        )
