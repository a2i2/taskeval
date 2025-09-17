import os
import re
import json

from datetime import datetime
from typing import Dict, List

import anthropic
import pandas as pd

from fire import Fire


class OutputGenerator:
    def __init__(self, api_key: str):
        self.client = anthropic.Anthropic(api_key=api_key)
        # self.available_strategies = ["summarize", "visualize", "automate", "generate", "other"]
        self.available_strategies = {
            "summarize": "Generate a summary or condensed version",
            "visualize": "Create a visual representation",
            "automate": "Use LLM as a judge",
            "generate": "Generate verification code or scripts",
            "other": "Apply techniques not covered by the above methods",
        }

    def _extract_json(self, text: str):
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
        # trim, collapse spaces, dedupe (keep order)
        seen, out = set(), []
        for x in items or []:
            s = re.sub(r"\s+", " ", str(x).strip())
            key = s.lower()
            if s and key not in seen:
                seen.add(key)
                out.append(s)
        return out

    def propose_elements(self, task: str, min_k=3, max_k=8):
        prompt = f"""
Return ONLY JSON with a single key "elements".
- "elements" must be {min_k}-{max_k} SHORT noun phrases (1–3 words each).
- They should name the concrete parts/entities that must be evaluated for correctness for this task.
- Derive them strictly from the task description.
- No verbs, no explanations, no extra text.

Task: {task}

JSON ONLY:
{{ "elements": ["..."] }}
"""
        try:
            response = self.client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=400,
                temperature=0,
                messages=[{"role": "user", "content": prompt}],
            )
            text = response.content[0].text.strip()

            data = self._extract_json(text) or {}
            elems = self._normalize_list(data.get("elements", []))

            # light guidance only (no defaults)
            notes = []
            if len(elems) < min_k:
                notes.append(
                    f"Only {len(elems)} elements; consider adding {min_k - len(elems)} more."
                )
            if len(elems) > max_k:
                notes.append(f"{len(elems)} elements; consider trimming to ≤ {max_k}.")
            # flag obvious verbs/numbers
            non_nounish = [
                e
                for e in elems
                if re.search(r"\b(ing|ed)\b", e.lower()) or re.search(r"\d", e)
            ]
            if non_nounish:
                notes.append(f"Some items may not be noun phrases: {non_nounish}")

            return {
                "elements": elems,
                "elements_string": ", ".join(elems),
                "notes": notes,
            }

        except Exception as e:
            return {
                "elements": [],
                "elements_string": "",
                "notes": [f"Error generating elements: {str(e)}"],
            }

    def generate_domain_keywords_interactive(
        self, task: str, required_confirmation: bool = True
    ) -> str:
        """Generate domain keywords with interactive editing capability"""
        print(f"\nGenerating domain keywords for task: {task}")

        proposal = self.propose_elements(task)
        if required_confirmation:
            print("\nProposed elements to evaluate:", proposal["elements_string"])
            if proposal["notes"]:
                print("Notes:", proposal["notes"])

            # Let the developer accept or edit the single line:
            edited = input(
                "\nPress Enter to accept, or edit the comma-separated list: "
            ).strip()
        else:
            edited = proposal["elements_string"]
        final_line = edited if edited else proposal["elements_string"]

        print(f"Final domain keywords: {final_line}")
        return final_line

    def generate_single_output(
        self, task: str, task_input: str, llm_output: str, domain_keywords: str
    ) -> Dict:
        """Generate a single analysis output"""

        # Step 1: Generate task description
        task_description = self._generate_task_description(
            task, task_input, llm_output, domain_keywords
        )
        if task_description.startswith("Error"):
            return {"error": task_description}

        # Step 2: Identify potential errors
        potential_errors = self._identify_errors(
            task_description, task_input, llm_output, domain_keywords
        )
        if isinstance(potential_errors, str):
            return {"error": potential_errors}

        # Step 3: Rank error criticality
        ranked_errors = self._rank_error_criticality(
            task, potential_errors, domain_keywords
        )
        if isinstance(ranked_errors, str):
            return {"error": ranked_errors}

        # Step 4: Filter errors
        filtered_errors = self._filter_errors(ranked_errors)
        critical_errors = self._filter_critical_errors(filtered_errors)

        # Step 5: Map strategies
        if critical_errors:
            strategy_mapping = self._map_strategies(
                task, critical_errors, domain_keywords
            )
            if isinstance(strategy_mapping, str):
                return {"error": strategy_mapping}
        else:
            strategy_mapping = {"error_strategy_mapping": {}, "mapping_rationale": {}}

        return {
            "task_description": task_description,
            "domain_keywords": domain_keywords,
            "potential_errors": filtered_errors,
            "critical_errors": critical_errors,
            "error_strategy_mapping": strategy_mapping.get(
                "error_strategy_mapping", {}
            ),
            "mapping_rationale": strategy_mapping.get("mapping_rationale", {}),
        }

    def generate_output(self, test_scenario: Dict, output_dir: str):
        """Generate single output and save it"""

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Generate scenario name from task if not provided
        scenario_name = test_scenario.get(
            "name",
            re.sub(r"[^\w\s-]", "", test_scenario["task"])[:30].replace(" ", "_"),
        )
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Interactive domain keywords generation
        if (
            "domain_keywords" not in test_scenario
            or not test_scenario["domain_keywords"]
        ):
            test_scenario["domain_keywords"] = (
                self.generate_domain_keywords_interactive(test_scenario["task"])
            )

        print(f"\nGenerating output for scenario: {scenario_name}")
        print("=" * 60)

        try:
            output = self.generate_single_output(
                test_scenario["task"],
                test_scenario["task_input"],
                test_scenario["llm_output"],
                test_scenario["domain_keywords"],
            )

            if "error" not in output:
                # Format data for single output
                row_data = self._format_iteration_row(1, output, test_scenario)

                # Save to Excel
                excel_file = f"{output_dir}/{scenario_name}_{timestamp}.xlsx"
                self._save_single_output_to_excel(row_data, test_scenario, excel_file)

                print("✓ Generation completed successfully")
                print(f"\nResults saved to: {excel_file}")

                return excel_file
            else:
                error_msg = f"✗ Error: {output['error']}"
                print(error_msg)
                return {"error": output["error"]}

        except Exception as e:
            error_msg = f"✗ Exception: {str(e)}"
            print(error_msg)
            return {"error": str(e)}

    def _format_iteration_row(
        self, iteration: int, output: Dict, scenario: Dict
    ) -> Dict:
        """Format output data as a single row for Excel"""

        # Extract critical errors info
        critical_errors = output["critical_errors"]
        critical_error_names = [e.get("error_name", "") for e in critical_errors]
        critical_error_descriptions = [
            e.get("description", "") for e in critical_errors
        ]
        critical_error_likelihoods = [e.get("likelihood", "") for e in critical_errors]
        critical_errors_with_descriptions = [
            f"{e.get('error_name', '')}: {e.get('description', '')}"
            for e in critical_errors
        ]
        # Extract all errors info
        all_errors = output["potential_errors"]
        all_error_names = [e.get("error_name", "") for e in all_errors]
        all_errors_with_descriptions = [
            f"{e.get('error_name', '')}: {e.get('description', '')}" for e in all_errors
        ]
        error_counts_by_impact = {
            "critical": len(
                [e for e in all_errors if e.get("impact", "").lower() == "critical"]
            ),
            "major": len(
                [e for e in all_errors if e.get("impact", "").lower() == "major"]
            ),
            "minor": len(
                [e for e in all_errors if e.get("impact", "").lower() == "minor"]
            ),
        }

        # Extract strategy mappings
        mappings = output["error_strategy_mapping"]
        rationales = output["mapping_rationale"]
        mapped_strategies = list(mappings.values())

        # Create row data
        row_data = {
            "Iteration": iteration,
            "Timestamp": datetime.now().isoformat(),
            # Task Description Info
            "Task_Description": output["task_description"],
            "Task_Description_Length": len(output["task_description"]),
            "Task_Description_Word_Count": len(output["task_description"].split()),
            # Generated Domain Keywords
            "Generated_Domain_Keywords": output.get("domain_keywords", ""),
            # Error Summary Stats
            "Total_Errors": len(all_errors),
            "Critical_Error_Count": error_counts_by_impact["critical"],
            "Major_Error_Count": error_counts_by_impact["major"],
            "Minor_Error_Count": error_counts_by_impact["minor"],
            # All Error Names (pipe-separated for easy parsing)
            "All_Error_Names": "|".join(all_error_names),
            "All_Errors_With_Descriptions": "|".join(all_errors_with_descriptions),
            # Critical Errors Details
            "Critical_Error_Names": "|".join(critical_error_names),
            "Critical_Error_Descriptions": "|".join(critical_error_descriptions),
            "Critical_Error_Likelihoods": "|".join(critical_error_likelihoods),
            "Critical_Errors_With_Descriptions": "|".join(
                critical_errors_with_descriptions
            ),
            # Strategy Mappings
            "Strategy_Mappings": json.dumps(mappings),  # JSON string for exact mapping
            "Strategy_Rationales": json.dumps(rationales),
            "Mapped_Strategies": "|".join(mapped_strategies),
            "Strategy_Count": len(mappings),
            # Scenario Info (for reference)
            "Scenario_Name": scenario.get("name", ""),
            "Task": scenario.get("task", ""),
            "Domain_Keywords": scenario.get("domain_keywords", ""),
        }

        return row_data

    def _save_single_output_to_excel(
        self, output_data: Dict, scenario: Dict, excel_file: str
    ):
        """Save single output data to Excel"""

        with pd.ExcelWriter(excel_file, engine="openpyxl") as writer:

            # Main sheet: Single Output
            df_output = pd.DataFrame([output_data])
            df_output.to_excel(writer, sheet_name="Output", index=False)

            # Scenario Info sheet
            scenario_info = pd.DataFrame(
                [
                    {
                        "Scenario_Name": scenario.get("name", ""),
                        "Task": scenario.get("task", ""),
                        "Task_Input": scenario.get("task_input", ""),
                        "LLM_Output": scenario.get("llm_output", ""),
                        "Domain_Keywords": scenario.get("domain_keywords", ""),
                        "Generated_Successfully": True,
                        "Generation_Timestamp": datetime.now().isoformat(),
                    }
                ]
            )
            scenario_info.to_excel(writer, sheet_name="Scenario_Info", index=False)

    def _generate_task_description(
        self, task: str, task_input: str, llm_output: str, domain_keywords: str
    ) -> str:
        """Generate detailed task description"""
        prompt = f"""
You are an expert requirements analyst.

TASK: {task}
DOMAIN KEYWORDS: {domain_keywords}

Write a structured task description for this type of task with:
1. Primary Objective: [What is the main goal?]
2. Key Processing Steps: [What steps must be performed?]
3. Critical Success Factors: [What determines if this task is done correctly?]
4. Output Requirements: [What format and content is expected?]
5. Domain-Specific Considerations: [Based on keywords: {domain_keywords}]

Be specific and systematic. Focus on what makes this task successful vs failed.
Note: This is for analyzing the task type in general, not specific files.
"""
        try:
            response = self.client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=3000,
                temperature=0,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.content[0].text.strip()
        except Exception as e:
            return f"Error generating task description: {str(e)}"

    def _identify_errors(
        self,
        task_description: str,
        task_input: str,
        llm_output: str,
        domain_keywords: str,
    ) -> List[Dict]:
        """Identify potential errors"""
        prompt = f"""
You are a systematic error analysis expert. Your job is to identify ALL possible errors.

TASK ANALYSIS:
Task Description: {task_description}
Domain Keywords: {domain_keywords}

For this type of task (input type: {task_input}, output type: {llm_output}):
Comprehensive error identification:
Think about what could go wrong at each stage based on the task description. Be comprehensive and include both obvious and subtle potential errors.

IMPORTANT: You must return only valid JSON in exactly this format:

{{
  "potential_errors": [
    {{
      "error_name": "short descriptive name",
      "description": "detailed explanation of what goes wrong and why",
      "likelihood": "high",
      "impact": "critical",
      "error_stage": "input_interpretation"
    }}
  ]
}}

Return only the JSON structure above, no additional text or explanations.
"""
        try:
            response = self.client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=3000,
                temperature=0,
                messages=[{"role": "user", "content": prompt}],
            )
            parsed = self._parse_json(response.content[0].text)
            if "error" in parsed:
                return parsed["error"]
            return parsed.get("potential_errors", [])
        except Exception as e:
            return f"Error identification failed: {str(e)}"

    def _rank_error_criticality(
        self, task: str, potential_errors: List[Dict], domain_keywords: str
    ) -> List[Dict]:
        """Rank error criticality"""
        prompt = f"""
You are a criticality assessment expert. Rank these errors by how severely they would break the CORE OBJECTIVE.

CORE OBJECTIVE: {task}
Domain Context: {domain_keywords}

ERRORS TO RANK:
{json.dumps(potential_errors, indent=2)}

RANKING CRITERIA:
- CRITICAL: Significantly distorts or omits core task requirements, leading to misleading, incomplete, or invalid outputs.
- MAJOR: Violates important but non-core requirements; the output is still partially useful but requires correction. 
- MINOR: Minor inaccuracies or formatting issues that don't materially affect task understanding or usability.

IMPORTANT: Return ONLY valid JSON in exactly this format:

{{
  "potential_errors": [
    {{
      "error_name": "same name as input",
      "description": "same description as input",
      "likelihood": "same likelihood as input", 
      "impact": "critical or major or minor",
      "error_stage": "same stage as input",
      "criticality_reasoning": "brief explanation why this impact level"
    }}
  ]
}}

Return only the JSON structure above, no additional text.
"""
        try:
            response = self.client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=3000,
                temperature=0,
                messages=[{"role": "user", "content": prompt}],
            )
            parsed = self._parse_json(response.content[0].text)
            if "error" in parsed:
                return parsed["error"]
            return parsed.get("potential_errors", [])
        except Exception as e:
            return f"Criticality ranking failed: {str(e)}"

    def _filter_errors(self, errors: List[Dict]) -> List[Dict]:
        """Filter out infrastructure errors"""
        exclude_keywords = [
            "path",
            "file",
            "accessibility",
            "network",
            "connection",
            "memory",
            "infrastructure",
            "system",
            "input validation",
            "format validation",
            "api",
            "server",
            "storage",
            "permission",
            "access",
        ]
        filtered_errors = []
        for error in errors:
            text = (
                error.get("error_name", "") + " " + error.get("description", "")
            ).lower()
            if not any(k in text for k in exclude_keywords):
                filtered_errors.append(error)
        return filtered_errors

    def _filter_critical_errors(self, errors: List[Dict]) -> List[Dict]:
        """Get only critical errors"""
        return [e for e in errors if e.get("impact", "").lower() == "critical"]

    def _map_strategies(
        self, task: str, critical_errors: List[Dict], domain_keywords: str
    ) -> Dict:
        """Map critical errors to strategies"""
        strategies_list = ", ".join(self.available_strategies)
        prompt = f"""
You are a validation strategy expert. For each critical error, choose the single most appropriate validation strategy.

IMPORTANT CONSTRAINT: You are working in validation mode where no ground truth is available. 
Only select strategies that can work without access to correct answers or reference data.

TASK CONTEXT:
Task: {task}
Domain Keywords: {domain_keywords}

CRITICAL ERRORS TO MAP:
{json.dumps(critical_errors, indent=2)}

AVAILABLE STRATEGIES: {strategies_list}

For each critical error, think:
1. What would be the most effective way to validate/check this error without ground truth?
2. Which strategy name best matches that validation approach?
3. Which strategy would allow an LLM to perform the needed validation independently?

Choose the single most appropriate strategy for each error.

Return ONLY valid JSON in this exact format:
{{
  "error_strategy_mapping": {{
    "error_name_from_list": "chosen_strategy"
  }},
  "mapping_rationale": {{
    "error_name_from_list": "reason for this choice and why it works without ground truth"
  }}
}}

Make sure to map ALL critical errors that were provided. Return only the JSON structure above.
"""
        try:
            response = self.client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=3000,
                temperature=0,
                messages=[{"role": "user", "content": prompt}],
            )
            parsed = self._parse_json(response.content[0].text)
            if "error" in parsed:
                return parsed["error"]
            return parsed
        except Exception as e:
            return f"Strategy mapping failed: {str(e)}"

    def _parse_json(self, response: str) -> Dict:
        """Parse JSON response"""
        try:
            cleaned = response.strip()

            # Remove markdown formatting
            if cleaned.startswith("```json"):
                cleaned = cleaned.replace("```json", "").replace("```", "").strip()
            elif cleaned.startswith("```"):
                cleaned = cleaned.replace("```", "").strip()

            # Extract JSON if there's extra text
            json_match = re.search(r"\{.*\}", cleaned, re.DOTALL)
            if json_match:
                cleaned = json_match.group()

            return json.loads(cleaned)

        except json.JSONDecodeError as e:
            return {"error": f"Could not parse JSON response: {str(e)}"}
        except Exception as e:
            return {"error": f"Unexpected parsing error: {str(e)}"}


def main(
    task: str = "Extract the data points in the chart image and provide the output as a table.",
    task_input: str = "./data/new_chart_images/1.png",
    llm_output: str = "./outputs/1.xlsx",
):
    """
    Main function to generate a single output for a given task, task input, and LLM output.
    Args:
        task: The task to generate an output for.
        task_input: The input to the task.
        llm_output: The output of the LLM.
    Returns:
        The output file path.
    """
    # Minimal test scenario - only essential fields needed
    test_scenario = {
        "task": task,
        "task_input": task_input,
        "llm_output": llm_output,
        # domain_keywords will be generated interactively
    }

    # Initialize generator
    api_key = os.getenv("ANTHROPIC_API_KEY")
    generator = OutputGenerator(api_key)

    # Generate single output
    print("Starting output generation...")
    output_file = generator.generate_output(
        test_scenario=test_scenario, output_dir="single_output"
    )

    print(f"\n{'='*60}")
    print("OUTPUT GENERATION COMPLETED")
    print(f"{'='*60}")
    if isinstance(output_file, str):
        print(f"Excel file: {output_file}")
        print("\nSheet structure:")
        print("- Output: Single row with all generated data")
        print("- Scenario_Info: Test scenario metadata")
    else:
        print(f"Generation failed: {output_file.get('error', 'Unknown error')}")


if __name__ == "__main__":
    Fire(main)
