import os
import base64
import pandas as pd

from io import StringIO
from typing import Dict

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


def regenerate_chart_image(
    task_input: str,
    extracted_table_data: pd.DataFrame,
    output_dir: str = "./outputs",
) -> str:
    """
    Regenerate the chart image.
    """
    # Compose the prompt
    table_desc = _dataframe_to_table_description(extracted_table_data)
    prompt = (
        # "Regenerate a chart image in the same style as the reference image. "
        # "The chart should visualize the following data table: "
        # f"{table_desc}. "
        # "Match the chart type, axis labels, and overall style of the reference image as closely as possible."
        "Create a simple 2D chart that is best to visualize the following data table in the same style as the reference image:\n"
        f"{table_desc}.\n\n"
        "Keep the chart simple, basic and clean, don't add any extra unnecessary elements.\n"
        "Base64 encoded string of the chart image:\n"
    )

    try:
        # response = openai.images.generate(
        #     model="gpt-image-0721-mini-alpha",
        #     prompt=prompt,
        #     n=1,
        #     size="1024x1024",
        # )
        # response = openai.chat.completions.create(
        #     model="gpt-5",
        #     messages=[
        #         {
        #             "role": "system",
        #             "content": (
        #                 "You are a helpful assistant that generates chart images from data table descriptions. "
        #                 "Return the image as a base64 encoded string of the chart image."
        #             ),
        #         },
        #         {
        #             "role": "user",
        #             "content": [
        #                 {
        #                     "type": "image_url",
        #                     "image_url": {
        #                         "url": f"data:image/jpeg;base64,{get_base64_image(task_input)}",
        #                     },
        #                 },
        #                 {
        #                     "type": "text",
        #                     "text": prompt,
        #                 },
        #             ],
        #         },
        #     ],
        # )
        # response = openai.responses.create(
        #     model="gpt-5",
        #     input=prompt,
        #     # tools=[{"type": "image_generation"}],
        # )
        # print(response)
        # image_base64 = response.choices[0].message.content
        # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # os.makedirs(output_dir, exist_ok=True)
        # output_path = os.path.join(output_dir, f"regenerated_chart_{timestamp}.png")
        # with open(output_path, "wb") as f:
        #     f.write(base64.b64decode(image_base64))
        # return output_path
        return ""
    except Exception as e:
        print(f"Error generating chart image: {e}")
        return ""


def figure_extractions(task: str, task_input: str, llm_output: str = "csv") -> Dict:
    """
    Perform figure extractions.
    """

    output_generator = OutputGenerator(os.getenv("ANTHROPIC_API_KEY"))

    domain_keywords = output_generator.generate_domain_keywords_interactive(task, False)
    outputs = output_generator.generate_single_output(
        task=task,
        task_input=task_input,
        llm_output=llm_output,
        domain_keywords=domain_keywords,
    )
    # Add additional outputs
    outputs["extracted_table_data"] = extract_figure_into_table(task_input)
    new_chart_path = regenerate_chart_image(task_input, outputs["extracted_table_data"])
    outputs["regenerated_chart_image"] = new_chart_path
    return outputs
