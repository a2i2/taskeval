"""
TaskEval - A Streamlit app for LLM task evaluation
"""

import ast
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Add the project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
from ui.utils import (
    HIGHLIGHT_COLORS,
    highlight_text_multicolor,
    highlight_section_text,
)
from ui.qa_answering import eval_label_mapping

# Page config
st.set_page_config(page_title="TaskEval - QA Answering Visualisation", layout="wide")

# Main app
st.title("🔬 TaskEval")
st.subheader("QA Answering Visualisation")


# Input section
col1, col2 = st.columns([1, 1])
with col1:
    # Original chart image
    qa_evals_path = st.text_input(
        "Enter the path to the QA evals",
        help="Enter the path to the original chart image",
        value="./outputs/qa_evals/expanded_qa_evals.csv",
    )
with col2:
    # Index
    data_index = st.number_input(
        "Select the index of the chart to visualise, start from 1",
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
        disabled=not (qa_evals_path and data_index),
    )

# Processing logic
if process_button:
    with st.spinner("Visualising..."):
        st.header("📊 Extraction Results")
        # Prepare the results
        df = pd.read_csv(qa_evals_path)
        df["qa_section_matches"] = df["qa_section_matches"].apply(ast.literal_eval)
        df["qa_extracted_source_nodes"] = df["qa_extracted_source_nodes"].apply(
            ast.literal_eval
        )
        results = df.iloc[data_index - 1]

        # # Limit the source nodes to 200 characters
        # source_nodes = [
        #     (node[:200] + "..." if len(node) > 200 else node)
        #     for node in results["qa_extracted_source_nodes"]
        # ]
        source_nodes = results["qa_extracted_source_nodes"]
        # Get section-specific matches
        section_matches = results["qa_section_matches"]
        # Reorder source nodes based on the number of section matches length
        sorted_section_matches = sorted(
            section_matches.items(), key=lambda x: len(x[1]), reverse=True
        )
        # Convert the tuples back to a dictionary
        sorted_section_matches = {
            idx: section_matches[idx] for idx, _ in sorted_section_matches
        }

        sorted_source_nodes = [
            source_nodes[idx] for idx in sorted_section_matches.keys()
        ]

        # Start displaying the results
        st.header("QA Answering Results")
        st.write(results["qa_answer"])

        st.subheader("Evaluation Metrics")
        metrics = {
            "AnswerRelevancyMetric": results["qa_scores_relevant"],
            "FaithfulnessMetric": results["qa_scores_faithful"],
            "ContextualPrecisionMetric": results["qa_scores_contextual_precision"],
            "ContextualRecallMetric": results["qa_scores_contextual_recall"],
        }
        cols = st.columns(len(metrics))
        for i, metric in enumerate(metrics.items()):
            with cols[i]:
                st.markdown(f"#### {eval_label_mapping[metric[0]]}: {metric[1]:.2f}")
        st.write(results["id"])
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Question")
            st.write(results["question"])
        with col2:
            st.subheader("Expected Answer")
            st.write(results["answer"])
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Highlighted sections from the sources")
            highlighted_output = highlight_text_multicolor(
                results["qa_answer"], sorted_section_matches
            )
            st.markdown(
                highlighted_output,
                unsafe_allow_html=True,
            )
        with col2:
            st.subheader("Sources")
            for i, node in enumerate(sorted_source_nodes):
                color = HIGHLIGHT_COLORS[i % len(HIGHLIGHT_COLORS)]
                # Highlight keywords in this section if it has matches
                if i in sorted_section_matches:
                    highlighted_section = highlight_section_text(
                        node.strip(), sorted_section_matches[i], color
                    )
                else:
                    highlighted_section = node.strip()
                match_count = len(list(sorted_section_matches.values())[i])
                # Style the source with a frame and some padding
                styled_section = f"""
                    <div style="
                        position: relative;
                        border: 2px solid {color};
                        border-radius: 8px;
                        padding: 4px;
                        margin-bottom: 4px;
                        background-color: #f9f9f9;
                        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
                        ">
                        <div style="
                            position: absolute;
                            top: 4px;
                            right: 4px;
                            background: #fff;
                            border-radius: 6px;
                            padding: 2px 8px;
                            font-size: 0.95em;
                            color: #555;
                            border: 1px solid {color};
                            box-shadow: 0 1px 4px rgba(0,0,0,0.04);
                            z-index: 20;
                        ">
                            {match_count} matches
                        </div>
                        <div style="margin-right: 92px; margin-left: 4px;">
                          {highlighted_section}
                        </div>
                        <div style="
                            position: absolute;
                            bottom: 2px;
                            right: 8px;
                            display: flex;
                            gap: 8px;
                            z-index: 10;
                        ">
                            <span style="font-size: 1.2em; cursor: pointer;" title="Thumbs up">&#128077;</span>
                            <span style="font-size: 1.2em; cursor: pointer;" title="Thumbs down">&#128078;</span>
                        </div>
                    </div>
                """
                st.markdown(styled_section, unsafe_allow_html=True)
