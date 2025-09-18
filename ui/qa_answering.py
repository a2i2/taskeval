import json

import streamlit as st

from ui.utils import (
    HIGHLIGHT_COLORS,
    get_section_specific_matches,
    highlight_text_multicolor,
    highlight_section_text,
)


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
    expected_answer = st.text_area(
        "Enter your expected answer for evaluating the LLM's answer:",
        placeholder="e.g., Multilingual training is performed by randomly alternating between languages for every new minibatch",
        height=100,
        help="Provide clear expected answer from the uploaded pdf",
    )
    return qa_question, expected_answer, uploaded_pdf


eval_label_mapping = {
    "AnswerRelevancyMetric": "Answer Relevancy",
    "FaithfulnessMetric": "Faithfulness",
    "ContextualPrecisionMetric": "Contextual Precision",
    "ContextualRecallMetric": "Contextual Recall",
}


def qa_answering_results(results):
    """
    Display the results of the QA answering.
    """
    qa_results = results["qa_results"]
    shortened_sources = [
        {
            "text": (
                node["text"].strip()[:200] + "..."
                if len(node["text"]) > 200
                else node["text"].strip()
            ),
            "score": node["score"],
        }
        for node in qa_results["source_nodes"]
    ]
    # Get section-specific matches
    section_matches = get_section_specific_matches(
        qa_results["answer"], shortened_sources
    )
    st.subheader("QA Answering Results")
    st.write(qa_results["answer"])

    critical_errors = results["critical_errors"]
    evaluation_strategies = results["error_strategy_mapping"]
    rationales = results["mapping_rationale"]

    if (
        critical_errors
        and len(
            [
                eval
                for eval in evaluation_strategies.values()
                if eval == "visualize" or eval == "automate"
            ]
        )
        > 0
    ):
        # Create tabs dynamically
        tab_names = []
        for error in critical_errors:
            if (
                evaluation_strategies[error["error_name"]] != "visualize"
                and evaluation_strategies[error["error_name"]] != "automate"
            ):
                continue
            error_name = error["error_name"]
            display_error_name = f"{error_name[0].upper()}{error_name[1:]}".replace(
                "_", " "
            )
            tab_names.append(f"{display_error_name} ({error['likelihood']})")

        tabs = st.tabs(tab_names)
        tab_counter = 0
        for error in critical_errors:
            if (
                evaluation_strategies[error["error_name"]] != "visualize"
                and evaluation_strategies[error["error_name"]] != "automate"
            ):
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

                if evaluation_strategies[error_name].lower() == "visualize":
                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader("Highlighted sections from the sources")
                        highlighted_output = highlight_text_multicolor(
                            qa_results["answer"], section_matches
                        )
                        st.markdown(
                            highlighted_output,
                            unsafe_allow_html=True,
                        )
                    with col2:
                        st.subheader("Sources")
                        for i, node in enumerate(shortened_sources):
                            color = HIGHLIGHT_COLORS[i % len(HIGHLIGHT_COLORS)]
                            # Highlight keywords in this section if it has matches
                            if i in section_matches:
                                highlighted_section = highlight_section_text(
                                    node["text"].strip(), section_matches[i], color
                                )
                            else:
                                highlighted_section = node["text"].strip()
                            st.markdown(highlighted_section, unsafe_allow_html=True)
                elif evaluation_strategies[error_name].lower() == "automate":
                    st.subheader("Evaluation Metrics")
                    metrics = results["qa_scores"]
                    cols = st.columns(len(metrics))
                    for i, metric in enumerate(metrics.items()):
                        with cols[i]:
                            st.markdown(
                                f"#### {eval_label_mapping[metric[0]]}: {metric[1]}"
                            )
    else:
        st.info("No critical errors found.")

    # Make this collapsible
    with st.expander("Raw Results"):
        st.code(
            json.dumps(results, indent=2, ensure_ascii=False),
            language="json",
        )
