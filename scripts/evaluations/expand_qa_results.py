from pathlib import Path
import ast

import pandas as pd
from fire import Fire


def main(
    result_csv_path: str,
    output_csv_path: str = "./outputs/qa_evals/expanded_qa_evals.csv",
):
    """
    Main function to expand the QA results.
    """
    input_path = Path(result_csv_path)
    output_path = Path(output_csv_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    qa_df = pd.read_csv(input_path)
    qa_df["qa_extracted_source_nodes"] = qa_df["qa_extracted_source_nodes"].apply(
        ast.literal_eval
    )
    max_matches = qa_df["qa_extracted_source_nodes"].apply(len).max()
    qa_df["qa_section_matches"] = qa_df["qa_section_matches"].apply(ast.literal_eval)
    max_matches = max(max_matches, qa_df["qa_section_matches"].apply(len).max())

    for i in range(max_matches):
        qa_df[f"qa_section_matches_{i+1}"] = qa_df["qa_section_matches"].apply(
            lambda x: list(x[i]) if i in x else []
        )
    for i in range(max_matches):
        qa_df[f"qa_section_matches_{i+1}_len"] = qa_df["qa_section_matches"].apply(
            lambda x: len(x[i]) if i in x else 0
        )

    for i in range(max_matches):
        qa_df[f"qa_extracted_source_nodes_{i+1}"] = qa_df[
            "qa_extracted_source_nodes"
        ].apply(lambda x: x[i] if len(x) > i else "")

    columns = qa_df.columns
    # Move 'qa_section_matches', 'qa_extracted_source_nodes' to the end
    columns = [
        col
        for col in columns
        if col not in ["qa_section_matches", "qa_extracted_source_nodes"]
    ]
    columns.extend(["qa_section_matches", "qa_extracted_source_nodes"])
    qa_df = qa_df[columns]

    qa_df.to_csv(output_path, index=False)


if __name__ == "__main__":
    Fire(main)
