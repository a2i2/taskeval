import streamlit as st

from ui.utils import (
    HIGHLIGHT_COLORS,
    get_section_specific_matches,
    highlight_text_multicolor,
    highlight_section_text,
)

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
    # Get section-specific matches
    section_matches = get_section_specific_matches(
        qa_results["answer"], qa_results["extracted_source_nodes"]
    )
    st.header("QA Answering Results")
    st.write(qa_results["answer"])

    st.subheader("Evaluation Metrics")
    metrics = results["qa_scores"]
    cols = st.columns(len(metrics))
    for i, metric in enumerate(metrics.items()):
        with cols[i]:
            st.markdown(f"#### {eval_label_mapping[metric[0]]}: {metric[1]:.2f}")

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
        for i, node in enumerate(qa_results["extracted_source_nodes"]):
            color = HIGHLIGHT_COLORS[i % len(HIGHLIGHT_COLORS)]
            # Highlight keywords in this section if it has matches
            if i in section_matches:
                highlighted_section = highlight_section_text(
                    node.strip(), section_matches[i], color
                )
            else:
                highlighted_section = node.strip()
            # Style the source with a frame and some padding
            styled_section = f"""
                <div style="
                    border: 2px solid {color};
                    border-radius: 8px;
                    padding: 4px;
                    margin-bottom: 2px;
                    background-color: #f9f9f9;
                    box-shadow: 0 2px 8px rgba(0,0,0,0.04);
                    ">
                    {highlighted_section}
                </div>
            """
            st.markdown(styled_section, unsafe_allow_html=True)
