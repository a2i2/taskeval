import sys
from pathlib import Path

import cairosvg
from tqdm import tqdm
from fire import Fire

# Add the project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
from taskevals.figure_extractions import figure_extractions


def main(input_dir_path: str, output_dir_path: str = "./outputs/figure_extractions"):
    """
    Main function to evaluate the figure extractions.
    """
    input_dir = Path(input_dir_path)
    output_dir = Path(output_dir_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(list(input_dir.glob("*.png")))
    existing_files = list(output_dir.glob("*.png"))
    existing_files = [f.name for f in existing_files]
    task = (
        "Extract the data points in the chart image and provide the output as a table."
    )
    for file in tqdm(files, desc="Evaluating figure extractions", total=len(files)):
        if file.name in existing_files:
            print(f"Skipping {file} because it already exists")
            continue
        try:
            print(f"Evaluating {file}")
            results = figure_extractions(task, file)

            results["extracted_table_data"].to_csv(
                output_dir / f"{file.stem}.csv", index=False
            )
            png_bytes = cairosvg.svg2png(
                bytestring=results["regenerated_chart_image"].encode("utf-8")
            )
            with open(output_dir / f"{file.stem}.png", "wb") as f:
                f.write(png_bytes)
        except Exception as e:
            print(f"Error evaluating {file}: {e}")
            continue


if __name__ == "__main__":
    Fire(main)
