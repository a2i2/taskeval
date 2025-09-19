import os
import base64

from io import StringIO
from typing import Dict

import openai
import pandas as pd

from anthropic import Anthropic

from taskevals.pipeline import OutputGenerator


def get_base64_image(task_input: str) -> str:
    """
    Get the base64 image from the task input.
    """
    with open(task_input, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def extract_figure_into_table(task_input: str) -> pd.DataFrame:
    """
    Extract the figure into a table.
    """
    # Use Anthropic model to extract chart data from image and return as DataFrame

    # Prepare the prompt for the LLM
    prompt = (
        "You are given an image of a chart. "
        "Extract all the data points from the chart and return them as a table. "
        "The table should be in CSV format, with the first row as column headers. "
        "Do not include any explanations or extra text, only the CSV table with the correct column headers.\n\n"
        "CSV ONLY:"
    )

    # Initialize the OutputGenerator (assumes ANTHROPIC_API_KEY is set)
    api_key = os.getenv("ANTHROPIC_API_KEY")
    client = Anthropic(api_key=api_key)

    # Use the Anthropic model to generate the output
    try:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=400,
            temperature=0,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": get_base64_image(task_input),
                            },
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
        )

        return pd.read_csv(StringIO(response.content[0].text))
    except Exception as e:
        print(f"Error extracting table from chart: {e}")
    return pd.DataFrame()


def _dataframe_to_table_description(df: pd.DataFrame) -> str:
    # We'll describe the chart type and data, and ask to mimic the style of the original image
    cols = df.columns.tolist()
    rows = df.values.tolist()
    col_str = ", ".join([f'"{c}"' for c in cols])
    row_strs = []
    for row in rows:
        row_strs.append(", ".join([str(x) for x in row]))
    table_desc = f"Columns: {col_str}. Data rows: " + "; ".join(row_strs)
    return table_desc


def regenerate_chart_image(task_input: str, extracted_table_data: pd.DataFrame) -> str:
    """
    Regenerate the chart image.
    """
    # Compose the prompt
    table_desc = _dataframe_to_table_description(extracted_table_data)
    prompt = (
        "Regenerate a chart image in the same style as the reference image.\n"
        "The chart should visualize the following data table:\n"
        f"{table_desc}.\n\n"
        "Match the chart type, axis labels, and overall style of the reference image as closely as possible, "
        "but using the data from the table provided above.\n"
        "Draw the chart in SVG for displaying in the markdown with no ```.\n"
        "SVG code only ready for display:\n"
    )
    image_data = get_base64_image(task_input)
    try:

        openai_api_key = os.getenv("OPENAI_API_KEY")
        client = openai.OpenAI(api_key=openai_api_key)

        # Compose the message for OpenAI's GPT-4o vision model
        response = client.chat.completions.create(
            model="gpt-5",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": "data:image/png;base64," + image_data,
                            },
                        },
                        {
                            "type": "text",
                            "text": prompt,
                        },
                    ],
                }
            ],
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error generating chart image: {e}")
        return ""


def figure_extractions(task: str, task_input: str, llm_output: str = "csv") -> Dict:
    """
    Perform figure extractions.
    """

    outputs = {}
    # output_generator = OutputGenerator(os.getenv("ANTHROPIC_API_KEY"))
    # domain_keywords = output_generator.generate_domain_keywords_interactive(task, False)
    # outputs = output_generator.generate_single_output(
    #     task=task,
    #     task_input=task_input,
    #     llm_output=llm_output,
    #     domain_keywords=domain_keywords,
    # )
    # Add additional outputs
    outputs["extracted_table_data"] = extract_figure_into_table(task_input)
    new_chart = regenerate_chart_image(task_input, outputs["extracted_table_data"])
    new_chart = new_chart.replace("```svg", "").replace("```", "")
    outputs["regenerated_chart_image"] = new_chart
    return outputs
