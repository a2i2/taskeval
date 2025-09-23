import ast
import shutil

from pathlib import Path
from typing import Optional

import pandas as pd

from fire import Fire


def main(
    qa_csv_path: str,
    pdf_dir_path: str,
    existing_qa_csv_path: Optional[str] = None,
    n_samples: int = 30,
    output_dir_path: str = "./data",
):
    """
    Main function to sample QA data.
    """
    qa_csv = Path(qa_csv_path)
    if not qa_csv.exists():
        raise FileNotFoundError(f"QA CSV file {qa_csv_path} not found")
    existing_qa_csv = Path(existing_qa_csv_path)
    pdf_dir = Path(pdf_dir_path)
    if not pdf_dir.exists():
        raise FileNotFoundError(f"PDF directory {pdf_dir_path} not found")
    output_dir = Path(output_dir_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_output_dir = output_dir / "qa_pdfs"
    pdf_output_dir.mkdir(parents=True, exist_ok=True)

    # Load existing QA CSV
    used_question_ids = []
    if existing_qa_csv.exists():
        existing_qa_df = pd.read_csv(existing_qa_csv)
        used_question_ids = existing_qa_df["id"].tolist()

    # Load QA CSV
    qa_df = pd.read_csv(qa_csv)

    # Exclude used question ids
    qa_df = qa_df[~qa_df["id"].isin(used_question_ids)]

    # Prepare the QA CSV
    qa_df["document_paths"] = qa_df["document_paths"].apply(ast.literal_eval)
    # Only pick longer answers samples
    qa_df["answer"] = qa_df["answer"].astype(str)
    qa_df["answer_len"] = qa_df["answer"].apply(len)
    qa_df = qa_df[qa_df["answer_len"] > 300]

    sampled_qa_df = qa_df.sample(n=n_samples)
    sampled_qa_df.to_csv(output_dir / "qa_set.csv", index=False)

    pdf_files = list(pdf_dir.glob("*.pdf"))
    pdf_files = [f.name for f in pdf_files]
    for _, row in sampled_qa_df.iterrows():
        pdf_file = Path(row["document_paths"][0]).name
        if pdf_file not in pdf_files:
            raise ValueError(f"PDF file {pdf_file} not found")
        shutil.copy(pdf_dir / pdf_file, pdf_output_dir / f"{pdf_file}")


if __name__ == "__main__":
    Fire(main)
