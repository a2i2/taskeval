import ast
import sys
from pathlib import Path

import pandas as pd

from tqdm import tqdm
from fire import Fire

# Add the project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
from taskevals.qa_answering import qa_answering
from ui.utils import get_section_specific_matches


def main(
    qa_csv_path: str,
    pdf_dir_path: str,
    output_dir_path: str = "./outputs/qa_evals",
    n_samples: int = 5,
):
    """
    Main function to evaluate the figure extractions.
    """
    qa_csv = Path(qa_csv_path)
    if not qa_csv.exists():
        raise FileNotFoundError(f"QA CSV file {qa_csv_path} not found")
    pdf_dir = Path(pdf_dir_path)
    if not pdf_dir.exists():
        raise FileNotFoundError(f"PDF directory {pdf_dir_path} not found")
    output_dir = Path(output_dir_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_csv = output_dir / "qa_evals.csv"
    if output_csv.exists():
        qa_df = pd.read_csv(output_csv)
    else:
        qa_df = pd.read_csv(qa_csv)
    qa_df["document_paths"] = qa_df["document_paths"].apply(ast.literal_eval)

    pdf_files = list(pdf_dir.glob("*.pdf"))
    pdf_files = [f.name for f in pdf_files]
    table_columns = qa_df.columns

    for idx, row in tqdm(qa_df.iterrows(), total=len(qa_df), desc="Evaluating QA"):
        if "qa_answer" in table_columns and not pd.isna(row["qa_answer"]):
            print(f"Skipping {idx} because it already has a QA answer")
            continue
        question = row["question"]
        expected_answer = row["answer"]
        document_path = Path(row["document_paths"][0]).name
        if document_path not in pdf_files:
            print(f"PDF file {document_path} not found")
            continue
        document_path = pdf_dir / document_path
        results = qa_answering(question, document_path, expected_answer)
        section_matches = get_section_specific_matches(
            results["qa_results"]["answer"],
            results["qa_results"]["extracted_source_nodes"],
        )

        section_matches_str = str(section_matches) if section_matches else ""
        extracted_source_nodes_str = str(
            results["qa_results"]["extracted_source_nodes"]
        )

        qa_df.at[idx, "qa_answer"] = results["qa_results"]["answer"]
        qa_df.at[idx, "qa_scores_relevant"] = results["qa_scores"][
            "AnswerRelevancyMetric"
        ]
        qa_df.at[idx, "qa_scores_faithful"] = results["qa_scores"]["FaithfulnessMetric"]
        qa_df.at[idx, "qa_scores_contextual_precision"] = results["qa_scores"][
            "ContextualPrecisionMetric"
        ]
        qa_df.at[idx, "qa_scores_contextual_recall"] = results["qa_scores"][
            "ContextualRecallMetric"
        ]

        for i in range(n_samples):
            qa_df[f"qa_section_matches_{i+1}"] = (
                str(section_matches[i]) if i in section_matches else set()
            )
        for i in range(n_samples):
            qa_df[f"qa_section_matches_{i+1}_len"] = (
                len(section_matches[i]) if i in section_matches else 0
            )
        for i in range(n_samples):
            qa_df[f"qa_extracted_source_nodes_{i+1}"] = (
                results["qa_results"]["extracted_source_nodes"][i]
                if i in results["qa_results"]["extracted_source_nodes"]
                else ""
            )

        qa_df.at[idx, "qa_section_matches"] = section_matches_str
        qa_df.at[idx, "qa_extracted_source_nodes"] = extracted_source_nodes_str

        qa_df.to_csv(output_dir / "qa_evals.csv", index=False)


if __name__ == "__main__":
    Fire(main)
