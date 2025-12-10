import pandas as pd
import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
from openai import OpenAI
import matplotlib.pyplot as plt
import requests
import subprocess
import os
import re
import base64


class StrategyExecutor:
    """
    Strategy Executor class that processes Excel inputs and implements various strategies
    including visualization, summarization, code generation, and automation using LLM calls
    """

    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o"):
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is not set. Pass api_key=... or export OPENAI_API_KEY.")
        self.client = OpenAI(api_key=api_key)  # one-liner init (GPT-4)
        self.model = model
        self.available_strategies = {
            "summarize": "Generate a summary or condensed version",
            "visualize": "Create a visual representation",
            "automate": "Use LLM as a judge",
            "generate": "Generate verification code or scripts",
            "other": "Apply techniques not covered by the above methods"
        }

    # --------------------- helpers ---------------------
    def _extract_json(self, text: str) -> Dict[str, Any]:
        if not text:
            return {}
        t = text.strip()
        if t.startswith("```"):
            t = re.sub(r"^```[a-zA-Z0-9]*\n?", "", t)
            if t.endswith("```"):
                t = t[:-3]
        m = re.search(r"\{.*\}", t, re.DOTALL)
        return json.loads(m.group()) if m else {}

    def _normalize_list(self, items):
        seen, out = set(), []
        for x in items or []:
            s = re.sub(r"\s+", " ", str(x).strip())
            key = s.lower()
            if s and key not in seen:
                seen.add(key)
                out.append(s)
        return out

    def _read_file_content(self, file_path: str) -> Dict[str, Any]:
        """Generalized reader for images, Excel, text; returns summary info."""
        if not os.path.exists(file_path):
            return {"error": f"File not found: {file_path}", "content": "", "type": "unknown"}

        try:
            _, ext = os.path.splitext(file_path.lower())

            if ext in [".png", ".jpg", ".jpeg", ".gif", ".webp"]:
                with open(file_path, "rb") as f:
                    image_b64 = base64.b64encode(f.read()).decode("utf-8")
                return {
                    "type": "image",
                    "content": image_b64,
                    "format": ext[1:],
                    "description": f"Image file ({ext})"
                }

            elif ext in [".xlsx", ".xls"]:
                try:
                    df = pd.read_excel(file_path)
                    return {
                        "type": "excel",
                        "content": f"Excel file with {df.shape[0]} rows and {df.shape[1]} columns",
                        "shape": df.shape,
                        "columns": list(df.columns),
                        "preview": df.head(10).to_dict(orient="records"),
                        "description": f"Excel file ({df.shape[0]}x{df.shape[1]})"
                    }
                except Exception as e:
                    return {"error": f"Could not read Excel file: {str(e)}", "content": "", "type": "excel"}

            elif ext in [".txt", ".csv"]:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                return {
                    "type": "text",
                    "content": content[:5000],
                    "description": f"Text file ({ext})"
                }

            else:
                return {
                    "error": f"Unsupported file type: {ext}",
                    "content": f"File path: {file_path}",
                    "type": "unknown"
                }

        except Exception as e:
            return {"error": f"Error reading file: {str(e)}", "content": "", "type": "unknown"}

    def validate_files(self, test_scenario: Dict[str, str]) -> Dict[str, Any]:
        """
        Checks that task_input and llm_output exist and are readable; returns brief info.
        test_scenario must contain keys: 'task_input', 'llm_output'
        """
        results = {
            "task_input_valid": False,
            "llm_output_valid": False,
            "errors": [],
            "file_info": {}
        }

        task_input = test_scenario.get("task_input", "")
        if task_input:
            input_content = self._read_file_content(task_input)
            if "error" not in input_content:
                results["task_input_valid"] = True
                results["file_info"]["task_input"] = input_content.get("description", "")
            else:
                results["errors"].append(f"Task input: {input_content['error']}")

        llm_output = test_scenario.get("llm_output", "")
        if llm_output:
            output_content = self._read_file_content(llm_output)
            if "error" not in output_content:
                results["llm_output_valid"] = True
                results["file_info"]["llm_output"] = output_content.get("description", "")
            else:
                results["errors"].append(f"LLM output: {output_content['error']}")

        return results

    # ----------------- single-sheet mappings loader -----------------
    def load_mappings(
        self,
        mappings_path: str,
        sheet: str | int = 0,
        required_cols: Dict[str, List[str]] | None = None,
    ) -> Dict[str, List[str]]:
        """
        Load mappings from a single sheet and extract three logical columns:
          - Critical_Errors_With_Descriptions
          - Strategy_Mappings
          - Strategy_Rationales

        Returns a dict of lists for each logical key. Missing columns -> empty lists.
        """
        if required_cols is None:
            required_cols = {
                "Critical_Errors_With_Descriptions": [
                    "Critical_Errors_With_Descriptions",
                    "Critical Errors With Descriptions",
                    "Critical_Errors",
                    "Critical Errors",
                    "Critical Error Descriptions",
                ],
                "Strategy_Mappings": [
                    "Strategy_Mappings",
                    "Strategy Mappings",
                    "Mappings",
                ],
                "Strategy_Rationales": [
                    "Strategy_Rationales",
                    "Strategy Rationales",
                    "Rationales",
                ],
            }

        try:
            df = pd.read_excel(mappings_path, sheet_name=sheet)
        except FileNotFoundError:
            raise FileNotFoundError(f"Mappings file not found: {mappings_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to read Excel: {e}")

        # normalize headers
        norm_to_orig = {}
        for c in df.columns:
            norm = " ".join(str(c).strip().split()).lower()
            norm_to_orig[norm] = c

        def pick_column(variants: List[str]) -> Optional[str]:
            for v in variants:
                key = " ".join(v.strip().split()).lower()
                if key in norm_to_orig:
                    return norm_to_orig[key]
            return None

        out: Dict[str, List[str]] = {}
        for logical_key, variants in required_cols.items():
            col = pick_column(variants)
            if not col:
                out[logical_key] = []
                continue
            series = (
                df[col]
                .astype(str)
                .map(lambda s: " ".join(s.strip().split()))
                .replace({"nan": ""})
            )
            # drop empties and dedupe (preserve order)
            seen = set()
            cleaned = []
            for v in series:
                if v and v.lower() not in seen:
                    seen.add(v.lower())
                    cleaned.append(v)
            out[logical_key] = cleaned

        return out
    
    def automate(self, task, task_input, llm_output, error, strategy_rationale):
        
        prompt = f"""You are an intelligent automation system capable of processing, analyzing, and executing tasks based on the provided context.

            
            LLM task: {task}
            Input:{task_input}
            LLM generated output : {llm_output}
            Critical error detected: {error}
            Rationale for evalute the crtical error: {strategy_rationale}
         
            Based on the task, input task, and llm generated output this critical error identified as to be evaluated using this strategy rationale.

            1. Analyze what needs to be done to evaluate the critical error according to the strategy rationale provided
            2. Identify the best approach for handling this specific scenario
            3. Process the data and generate the most appropriate response
            4. Provide actionable output that directly addresses the evaluation according to the strategy rationale of the detected critical error

            Return the final result only (any format is fine)."""

        resp = self.client.chat.completions.create(
                    model=self.model,  # e.g., "gpt-4"
                    messages=[
                        {"role": "system", "content": "You are a precise automation engine. Output only the final result."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0,
                )
        
        content = (resp.choices[0].message.content or "").rstrip()
        out_dir = os.path.join(os.getcwd(), "automation_outputs")
        os.makedirs(out_dir, exist_ok=True)
        filename = f"automation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        save_to = os.path.join(out_dir, filename)

        with open(save_to, "w", encoding="utf-8") as f:
            f.write(content)

  
        return content
    
    def summarize(self, task, task_input, llm_output, error, strategy_rationale):
        prompt = f"""You are an intelligent summarization system that distills the essential information from the provided context.

    LLM task: {task}
    Input: {task_input}
    LLM generated output: {llm_output}
    Critical error detected: {error}
    Strategy rationale: {strategy_rationale}

    Based on the task, input task, and llm generated output this critical error identified as to be evaluated using this strategy rationale.

    1. Analyze what needs to be summarized to support evaluation of the critical error according to the strategy rationale provided
    2. Identify the best summarization approach for this specific scenario (e.g., bullet points, short paragraph, or brief table if appropriate)
    3. Process the data and produce a concise, structured summary
    4. Provide actionable summary output that directly supports evaluation according to the strategy rationale of the detected critical error

    Return the final result only."""

        resp = self.client.chat.completions.create(
            model=self.model,  # e.g., "gpt-4"
            messages=[
                {"role": "system", "content": "You are a precise summarization engine. Output only the final result."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
        )

        content = (resp.choices[0].message.content or "").rstrip()

        # Save summary to a timestamped text file
        out_dir = os.path.join(os.getcwd(), "summarize_outputs")
        os.makedirs(out_dir, exist_ok=True)
        filename = f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        save_to = os.path.join(out_dir, filename)

        with open(save_to, "w", encoding="utf-8") as f:
            f.write(content)

        return content


    def code_generate(self, task, task_input, llm_output, error, strategy_rationale):
        prompt = f"""You are an intelligent code generation system that produces runnable validator code based on the provided context.

    LLM task: {task}
    Input: {task_input}
    LLM generated output: {llm_output}
    Critical error detected: {error}
    Strategy rationale: {strategy_rationale}

    Based on the task, input task, and llm generated output this critical error identified as to be evaluated using this strategy rationale.

    1. Analyze what needs to be implemented in code to evaluate the critical error according to the strategy rationale provided
    2. Identify the best approach for handling this specific scenario (e.g., which libraries, checks, and outputs)
    3. Process the data and generate a robust, runnable Python script that performs the evaluation end-to-end
    4. Provide actionable output through the code 

    Return ONLY a Python code block (```python ... ```)."""

        resp = self.client.chat.completions.create(
            model=self.model,  # e.g., "gpt-4"
            messages=[
                {"role": "system", "content": "You generate robust, runnable Python validators. Output only a Python code block."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
        )

        content = (resp.choices[0].message.content or "").rstrip()

        # Save raw response to .txt
        out_dir = os.path.join(os.getcwd(), "codegen_outputs")
        os.makedirs(out_dir, exist_ok=True)
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        txt_path = os.path.join(out_dir, f"codegen_{ts}.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(content)

        # If a Python fenced block exists, also save it as a .py file
        py_match = re.search(r"```python\s*([\s\S]*?)```", content, flags=re.IGNORECASE)
        if py_match:
            py_code = py_match.group(1).strip()
            py_path = os.path.join(out_dir, f"codegen_{ts}.py")
            with open(py_path, "w", encoding="utf-8") as f:
                f.write(py_code)

        return content
    
    def _extract_data_url(self, text: str):
        """
        Returns (ext, base64_str) from a data URL like:
        data:image/png;base64,<BASE64>
        Accepts png/jpeg/jpg/webp. Cleans and pads base64.
        """
        import re
        m = re.search(
            r"(data:image/(?:png|jpeg|jpg|webp);base64,)([A-Za-z0-9+/=\s]+)",
            text, flags=re.IGNORECASE
        )
        if not m:
            return None, None
        prefix = m.group(1).lower()
        b64 = m.group(2)

        # sanitize: remove whitespace and non-base64 junk
        b64 = re.sub(r"\s+", "", b64)
        b64 = re.sub(r"[^A-Za-z0-9+/=]", "", b64)

        # pad to multiple of 4
        if len(b64) % 4:
            b64 += "=" * (4 - (len(b64) % 4))

        if "png" in prefix:
            ext = "png"
        elif "jpeg" in prefix or "jpg" in prefix:
            ext = "jpg"
        else:
            ext = "webp"
        return ext, b64


    def visualize(self, task, task_input, llm_output, error, strategy_rationale):
        """
        Generate Python code to create visualization based on the strategy rationale.
        """
        try:
            img_data_url = self._image_to_small_data_url(task_input)         # tiny image
            csv_snippet  = self._table_to_csv_snippet(llm_output)            # tiny table
            
            prompt = f"""Generate executable Python code to create a side-by-side visualization.

    Requirements:
    - Create a figure with size (20.48, 6.82)
    - LEFT subplot: Display the original chart image from the provided data URL
    - RIGHT subplot: Create a new chart from the CSV data according to the strategy rationale
    - Save as PNG file and return the file path

    Context:
    Task: {task}
    Critical error: {error}
    Strategy rationale: {strategy_rationale}

    Original image data URL:
    {img_data_url}

    CSV data to plot:
    {csv_snippet}

    Generate complete Python code that:
    1. Imports all necessary libraries (matplotlib, pandas, PIL, base64, io, etc.)
    2. Decodes the base64 image data URL
    3. Creates a figure with two subplots side by side
    4. Left subplot shows the original image
    5. Right subplot creates a chart from CSV data following the strategy rationale
    6. Adds appropriate titles and labels
    7. Saves as PNG with timestamp
    8. Returns the saved file path

    Return ONLY the executable Python code, no explanations."""

            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a Python code generator. Generate only clean, executable Python code with proper imports. No markdown formatting, no explanations, no comments except essential ones."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0,
            )
            
            generated_code = (resp.choices[0].message.content or "").strip()
            
            # Clean up any markdown formatting if present
            if "```python" in generated_code:
                generated_code = generated_code.split("```python")[1].split("```")[0].strip()
            elif "```" in generated_code:
                generated_code = generated_code.split("```")[1].split("```")[0].strip()
            
            # Save the generated code
            out_dir = os.path.join(os.getcwd(), "visualization_code")
            os.makedirs(out_dir, exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            code_filename = os.path.join(out_dir, f"viz_code_{timestamp}.py")
            
            with open(code_filename, 'w', encoding='utf-8') as f:
                f.write(generated_code)
            
            print(f"💾 Generated visualization code saved to: {code_filename}")
            
            # Try to execute the generated code
            try:
                print("🔄 Executing generated visualization code...")
                
                # Create a safe execution environment
                exec_globals = {
                    '__builtins__': __builtins__,
                    'matplotlib': __import__('matplotlib'),
                    'plt': __import__('matplotlib.pyplot'),
                    'pandas': __import__('pandas'),
                    'pd': __import__('pandas'),
                    'PIL': __import__('PIL'),
                    'Image': __import__('PIL.Image').Image,
                    'base64': __import__('base64'),
                    'io': __import__('io'),
                    'os': __import__('os'),
                    'datetime': __import__('datetime'),
                    'numpy': __import__('numpy'),
                    'np': __import__('numpy'),
                }
                
                # Execute the generated code
                exec(generated_code, exec_globals)
                
                print(" Code executed successfully!")
                return code_filename
                
            except Exception as exec_error:
                print(f" Code execution failed: {str(exec_error)}")
                
                # Save execution error for debugging
                error_filename = os.path.join(out_dir, f"viz_error_{timestamp}.txt")
                with open(error_filename, 'w', encoding='utf-8') as f:
                    f.write(f"Execution Error:\n{str(exec_error)}\n\n")
                    f.write(f"Generated Code:\n{generated_code}")
                
                return code_filename  # Still return the code file for manual inspection
                
        except Exception as e:
            print(f" Visualization generation failed: {str(e)}")
            
            # Create error log
            out_dir = os.path.join(os.getcwd(), "visualization_code")
            os.makedirs(out_dir, exist_ok=True)
            error_filename = os.path.join(out_dir, f"viz_generation_error_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
            
            with open(error_filename, 'w', encoding='utf-8') as f:
                f.write(f"Visualization generation error: {str(e)}\n")
                f.write(f"Task: {task}\n")
                f.write(f"Error: {error}\n")
                f.write(f"Strategy rationale: {strategy_rationale}\n")
            
            return error_filename

    # def visualize(self, task, task_input, llm_output, error, strategy_rationale):
    #     """
    #     Generate Python code to create visualization based on the strategy rationale.
    #     """
        
    #     img_data_url = self._image_to_small_data_url(task_input)         # tiny image
    #     csv_snippet  = self._table_to_csv_snippet(llm_output)            # tiny table
    #     prompt = f""" You are a visualization engine.

    #         Goal: Produce a single 2048x682 PNG with a SIDE-BY-SIDE layout.
    #         LEFT  = the original input image (use it exactly as given).
    #         RIGHT = a clean, Matplotlib-style redrawn from the provided points according to the strategy rationale provided, same axis ranges as the left.

    #         Return ONLY Python code that creates this visualization.

    #             LLM task: {task}
    #             Input: {task_input}
    #             LLM generated output: {llm_output}
    #             Critical error detected: {error}
    #             Strategy rationale: {strategy_rationale}
    #             Original image (small data URL):
    # {img_data_url}

    # Tabular points (compact CSV; header row included):
    # {csv_snippet}

                
    #             Based on the task, input task, and llm generated output this critical error identified as to be evaluated using this strategy rationale.

    #             1. Analyze what needs to be visualized to support evaluation of the critical error according to the strategy rationale provided
    #             2. generate the regenerated image according to the rationale by taking appropriate inputs.
    #             3. To the visual strategy to evaluate, create original input nd generated image side by side.
    #             4. Provide visual output that directly supports evaluation according to the strategy rationale of the detected critical error

    #             Return executable Python code only that:
    #             - Creates figure with size (20.48, 6.82)
    #             - Decodes the provided data URL for left image
    #             - Uses the CSV data for right chart
    #             - Saves as PNG file
    #             - Returns the output file path"""

    #     resp = self.client.chat.completions.create(
    #         model=self.model,
    #         messages=[
    #             {"role": "system", "content": "Generate only executable Python code. Include necessary imports. No explanations or markdown formatting."},
    #             {"role": "user", "content": prompt},
    #         ],
    #         temperature=0,
    #     )
        
    #     generated_code = resp.choices[0].message.content.strip()
        
    #     # Clean up potential markdown formatting
    #     if "```python" in generated_code:
    #         generated_code = generated_code.split("```python")[1].split("```")[0].strip()
    #     elif "```" in generated_code:
    #         generated_code = generated_code.split("```")[1].split("```")[0].strip()
        
    #     try:
    #         # Execute the generated code
    #         exec(generated_code)
            
    #         # Assume the code creates a file - find the most recent PNG in output directory
    #         out_dir = os.path.join(os.getcwd(), "visualize_outputs")
    #         os.makedirs(out_dir, exist_ok=True)
            
    #         # Look for generated PNG files
    #         png_files = [f for f in os.listdir(out_dir) if f.endswith('.png')]
    #         if png_files:
    #             latest_file = max([os.path.join(out_dir, f) for f in png_files], key=os.path.getctime)
    #             return latest_file
    #         else:
    #             # If no PNG found, save with timestamp
    #             output_path = os.path.join(out_dir, f"visualize_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    #             return output_path
                
    #     except Exception as e:
    #         # Save generated code for debugging
    #         out_dir = os.path.join(os.getcwd(), "visualize_outputs")
    #         os.makedirs(out_dir, exist_ok=True)
            
    #         with open(os.path.join(out_dir, f"debug_code_{datetime.now().strftime('%Y%m%d_%H%M%S')}.py"), 
    #                 "w", encoding="utf-8") as f:
    #             f.write(generated_code)
            
    #         print(f"Generated code saved for inspection. Error: {e}")
    #         raise RuntimeError(f"Failed to execute visualization code: {e}")






    # def visualize(self, task, task_input, llm_output, error, strategy_rationale):
    #     """
    #     Single GPT-4 prompt that returns a base64 PNG; we keep the prompt tiny by:
    #     - downscaling/compressing the image to a small data URL
    #     - sending only a compact CSV snippet of the table
    #     """
        

        

    #     img_data_url = self._image_to_small_data_url(task_input)         # tiny image
    #     csv_snippet  = self._table_to_csv_snippet(llm_output)            # tiny table
    #     prompt = f""" You are a visualization engine.

    #         Goal: Produce a single 2048x682 PNG with a SIDE-BY-SIDE layout.
    #         LEFT  = the original chart image (use it exactly as given).
    #         RIGHT = a clean, Matplotlib-style redrawn from the provided points according to the strategy rationale provided, same axis ranges as the left.


    #         Return ONLY one line in this exact format:
    #         data:image/png;base64,<BASE64>

    #             LLM task: {task}
    #             Input: {task_input}
    #             LLM generated output: {llm_output}
    #             Critical error detected: {error}
    #             Strategy rationale: {strategy_rationale}
    #             Original image (small data URL):
    # {img_data_url}

    # Tabular points (compact CSV; header row included):
    # {csv_snippet}

                
    #             Based on the task, input task, and llm generated output this critical error identified as to be evaluated using this strategy rationale.

    #             1. Analyze what needs to be visualized to support evaluation of the critical error according to the strategy rationale provided
    #             2. generate the regenerated image according to the rationale by taking appropriate inputs.
    #             3. To the visual strategy to evaluate, create original input nd generated image side by side.
    #             4. Provide visual output that directly supports evaluation according to the strategy rationale of the detected critical error

    #             Return the final result only."""
   

   

    
    
    #     resp = self.client.chat.completions.create(
    #         model=self.model,
    #         messages=[
    #             {"role": "system", "content": "Output only a single data URL: data:image/png;base64,<BASE64>"},
    #             {"role": "user", "content": prompt},
    #         ],
    #         temperature=0,
    #     )
    #     content = (resp.choices[0].message.content or "").strip()
    #     # print(content)
        
    #     try:
    #         from PIL import Image
    #         import base64
    #         import io
            
    #         if content.startswith("data:image/png;base64,"):
    #             base64_data = content.split(',')[1]
    #             img_data = base64.b64decode(base64_data)
    #             debug_img = Image.open(io.BytesIO(img_data))
                
    #             print(f"Generated image size: {debug_img.size}")
    #             debug_img.show()  # This will open the image for preview
    #         else:
    #             print(f"Unexpected response format: {content[:100]}...")
                
    #     except Exception as e:
    #         print(f"Debug preview failed: {e}")
    #         print(f"Raw content: {content[:200]}...")
    #     ext, b64 = self._extract_data_url(content)
    #     if not b64:
    #         # Save raw for inspection if the model didn’t follow the format
    #         out_dir = os.path.join(os.getcwd(), "visualize_outputs")
    #         os.makedirs(out_dir, exist_ok=True)
    #         with open(os.path.join(out_dir, f"visualize_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"),
    #                 "w", encoding="utf-8") as f:
    #             f.write(content)
    #         raise RuntimeError("Model did not return a valid image data URL.")

    #     # decode (with a final padding safeguard)
    #     try:
    #         raw = base64.b64decode(b64, validate=True)
    #     except Exception:
    #         b64 += "=" * (4 - (len(b64) % 4)) % 4
    #         raw = base64.b64decode(b64)

    #     out_dir = os.path.join(os.getcwd(), "visualize_outputs")
    #     os.makedirs(out_dir, exist_ok=True)
    #     out_path = os.path.join(out_dir, f"visualize_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{ext}")
    #     with open(out_path, "wb") as f:
    #         f.write(raw)
    #     return out_path
    
    
#     def visualize(self, task, task_input, llm_output, error, strategy_rationale):
#         prompt = f""" You are a visualization engine.

# Goal: Produce a single 2048x682 PNG with a SIDE-BY-SIDE layout.
# LEFT  = the original chart image (use it exactly as given).
# RIGHT = a clean, Matplotlib-style redrawn from the provided points according to the strategy rationale provided, same axis ranges as the left.


# Return ONLY one line in this exact format:
# data:image/png;base64,<BASE64>

#     LLM task: {task}
#     Input: {task_input}
#     LLM generated output: {llm_output}
#     Critical error detected: {error}
#     Strategy rationale: {strategy_rationale}

    
#     Based on the task, input task, and llm generated output this critical error identified as to be evaluated using this strategy rationale.

#     1. Analyze what needs to be visualized to support evaluation of the critical error according to the strategy rationale provided
#     2. generate the regenerated image according to the rationale by taking appropriate inputs.
#     3. To the visual strategy to evaluate, create original input nd generated image side by side.
#     4. Provide visual output that directly supports evaluation according to the strategy rationale of the detected critical error

#     Return the final result only."""

#         resp = self.client.chat.completions.create(
#             model=self.model,  # e.g., "gpt-4"
#             messages=[
#                 {"role": "system", "content": "Output only a single data URL: data:image/png;base64,<BASE64"},
#                 {"role": "user", "content": prompt}
#             ],
#             temperature=0,
#         )

#         content = (resp.choices[0].message.content or "").rstrip()

#         m = re.search(r"data:image/png;base64,([A-Za-z0-9+/=\n\r]+)", content)
#         if not m:
#             # Model didn’t return a data URL; save the text so you can inspect it.
#             out_dir = os.path.join(os.getcwd(), "visualize_outputs")
#             os.makedirs(out_dir, exist_ok=True)
#             txt_path = os.path.join(out_dir, f"visualize_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
#             with open(txt_path, "w", encoding="utf-8") as f:
#                 f.write(content)
#             raise RuntimeError("Model did not return a PNG data URL. Saved raw text for inspection.")

#         b64 = m.group(1)
#         out_dir = os.path.join(os.getcwd(), "visualize_outputs")
#         os.makedirs(out_dir, exist_ok=True)
#         out_path = os.path.join(out_dir, f"visualize_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
#         with open(out_path, "wb") as f:
#             f.write(base64.b64decode(b64))
#         return out_path
    
    def execute_error_strategies(self, mappings, test_scenario, task, task_input, llm_output ) -> Dict[str, Any]:
        """
        Single function to load mappings, parse errors, and execute appropriate strategies
        
        Args:
            test_scenario: Dictionary containing task, task_input, llm_output, mappings
            
        Returns:
            Dictionary with all execution results
        """
        try:
            # Load mappings
            #mappings = self.load_mappings(test_scenario["mappings"])
            
            # Get task details
            # task = test_scenario.get("task", "")
            # task_input = test_scenario.get("task_input", "")
            # llm_output = test_scenario.get("llm_output", "")
            
            results = {"execution_results": {}}
            
            # Parse and execute for each error
            critical_errors = mappings.get("Critical_Errors_With_Descriptions", [])
            strategy_mappings = mappings.get("Strategy_Mappings", [])
            strategy_rationales = mappings.get("Strategy_Rationales", [])
            
            # Parse strategy mappings JSON
            strategies = {}
            rationales = {}
            
            for item in strategy_mappings:
                try:
                    mapping_dict = json.loads(item.strip().strip('"\''))
                    strategies.update(mapping_dict)
                except:
                    continue
                    
            for item in strategy_rationales:
                try:
                    rationale_dict = json.loads(item.strip().strip('"\''))
                    rationales.update(rationale_dict)
                except:
                    continue
            
            # Execute strategy for each error
            for error_item in critical_errors:
                if ":" not in error_item:
                    continue
                    
                error_code = error_item.split(":")[0].strip()
                error_description = error_item.split(":", 1)[1].split("|")[0].strip()
                
                strategy = strategies.get(error_code)
                rationale = rationales.get(error_code, "No rationale provided")
                
                if not strategy:
                    continue
                
                # Execute the strategy
                try:
                    if strategy == "visualize":
                        output = self.visualize(task, task_input, llm_output, error_description, rationale)
                    elif strategy == "summarize":
                        output = self.summarize(task, task_input, llm_output, error_description, rationale)
                    elif strategy == "generate":
                        output = self.code_generate(task, task_input, llm_output, error_description, rationale)
                    elif strategy == "automate":
                        output = self.automate(task, task_input, llm_output, error_description, rationale)
                    else:
                        output = f"Unknown strategy: {strategy}"
                    
                    results["execution_results"][error_code] = {
                        "strategy": strategy,
                        "output": output,
                        "status": "success"
                    }
                    
                except Exception as e:
                    results["execution_results"][error_code] = {
                        "strategy": strategy,
                        "error": str(e),
                        "status": "failed"
                    }
            
            return results
            
        except Exception as e:
            return {"error": str(e)}

    def _image_to_small_data_url(self, path: str, max_side: int = 768, quality: int = 60) -> str:
        """Downscale & JPEG-compress the image and return a small data URL."""
        from PIL import Image
        import io, base64
        img = Image.open(path).convert("RGB")
        w, h = img.size
        scale = min(1.0, max_side / max(w, h))
        if scale < 1.0:
            img = img.resize((int(w*scale), int(h*scale)), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality, optimize=True)
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        return f"data:image/jpeg;base64,{b64}"

    def _table_to_csv_snippet(self, path: str, max_rows: int = 200, max_chars: int = 5000) -> str:
            """Read Excel/CSV and return a compact CSV (first 2–3 numeric columns, up to max_rows)."""
            import pandas as pd, os
            ext = os.path.splitext(path)[1].lower()
            if ext in (".xlsx", ".xls"):
                df = pd.read_excel(path)
            elif ext == ".csv":
                df = pd.read_csv(path)
            else:
                raise ValueError("llm_output must be .xlsx/.xls/.csv")

            num = df.select_dtypes(include=["number", "float", "int"])
            if num.shape[1] == 0:
                # fall back to first 3 cols
                num = df.iloc[:, :3]
            else:
                num = num.iloc[:, :min(3, num.shape[1])]
            snippet = num.head(max_rows).to_csv(index=False)
            return snippet[:max_chars]


def main():
    
    api_key = os.getenv("ANTHROPIC_API_KEY")
    

    test_scenario = {
        "task": "Extract the data points in the chart image and provide the output as a table.",
        "task_input": "./data/new_chart_images/1.png",
        "llm_output": "./outputs/1.xlsx",
        "mappings": "./outputs/2.xlsx",
    }

    print(" Starting Strategy Executor...")
    executor = StrategyExecutor(api_key, model="gpt-4o")

    print(" Validating input files...")
    validation = executor.validate_files(test_scenario)
    print("Validation:", validation)
    
    # Check if files are valid before proceeding
    if not validation["task_input_valid"] or not validation["llm_output_valid"]:
        print(" File validation failed:", validation["errors"])
        return

    print(" Reading file contents...")
    task = test_scenario.get("task", "")
    # task = test_scenario.get("task", "")
    task_input = test_scenario.get("task_input", "")   # keep as path
    llm_output = test_scenario.get("llm_output", "")   # keep as path

    # task_input = executor._read_file_content(test_scenario.get("task_input", ""))
    # llm_output = executor._read_file_content(test_scenario.get("llm_output", ""))
    
    # print("Task:", task)
    # print("Task input type:", task_input.get("type", "unknown"))
    # print("LLM output type:", llm_output.get("type", "unknown"))

    print(" Loading mappings...")
    try:
        sheet = test_scenario.get("mappings_sheet", 0)
        mappings = executor.load_mappings(test_scenario["mappings"], sheet=sheet)
        
        critical = mappings["Critical_Errors_With_Descriptions"]
        strat_map = mappings["Strategy_Mappings"]
        strat_rat = mappings["Strategy_Rationales"]

        print(f"Found {len(critical)} critical errors")
        print(f"Found {len(strat_map)} strategy mappings")
        print(f"Found {len(strat_rat)} strategy rationales")
        
        print("\nCritical_Errors_With_Descriptions:", critical)
        print("Strategy_Mappings:", strat_map)
        print("Strategy_Rationales:", strat_rat)

    except Exception as e:
        print(f" Error loading mappings: {str(e)}")
        return

    print(" Executing error strategies...")
    try:
        results = executor.execute_error_strategies(mappings, test_scenario, task, task_input, llm_output)
        
        print(" Execution completed!")
        print("Results:", json.dumps(results, indent=2))
        
        # Print summary
        execution_results = results.get("execution_results", {})
        print(f"\n Summary: Processed {len(execution_results)} errors")
        
        for error_code, result in execution_results.items():
            status = result.get('status', 'unknown')
            strategy = result.get('strategy', 'unknown')
            print(f"  • {error_code}: {strategy} -> {status}")
            
            if status == 'failed':
                print(f"    Error: {result.get('error', 'Unknown error')}")
        
    except Exception as e:
        print(f" Error executing strategies: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
