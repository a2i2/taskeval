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
    st.header("QA Answering Results")

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
