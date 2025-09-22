import os
import re
import shutil
from typing import Optional

from anthropic import Anthropic
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.llms.openai import OpenAI
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.response_synthesizers import ResponseMode
from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
)
from deepeval.test_case import LLMTestCase

from taskevals.pipeline import OutputGenerator

output_generator = OutputGenerator(os.getenv("ANTHROPIC_API_KEY"))


def _get_answer(qa_question, pdf_path):
    """
    Answer the question from the uploaded pdf using RAG with LlamaIndex.
    """
    try:
        # Initialize OpenAI LLM
        llm = OpenAI(model="gpt-4o-mini", temperature=0, max_tokens=500)

        # Create new index
        temp_dir = os.path.join(os.getcwd(), "temp_pdf")
        os.makedirs(temp_dir, exist_ok=True)
        shutil.copy(pdf_path, temp_dir)
        documents = SimpleDirectoryReader(temp_dir).load_data()
        index = VectorStoreIndex.from_documents(documents, llm=llm)
        shutil.rmtree(temp_dir)

        # Create query engine
        retriever = VectorIndexRetriever(index=index, similarity_top_k=5)
        query_engine = RetrieverQueryEngine.from_args(
            retriever=retriever, llm=llm, response_mode=ResponseMode.COMPACT
        )

        # Query the document
        response = query_engine.query(qa_question)

        return {
            "question": qa_question,
            "answer": str(response),
            "source_nodes": (
                [
                    {
                        "text": node.text,
                        "score": node.score,
                    }
                    for node in response.source_nodes
                ]
                if hasattr(response, "source_nodes")
                else []
            ),
        }
    except Exception as e:
        return {
            "question": qa_question,
            "answer": f"Error processing PDF: {str(e)}",
            "source_nodes": [],
        }


def extract_source_nodes(question, answer, source_nodes: list[dict]) -> list[str]:
    """
    Extract key sentences from the source nodes.
    """
    client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    extracted_source_nodes = []
    for node in source_nodes:
        task = f"""Extract up to 3 key sentences (words for words) from this source that are relevant to the following question and answer:
Question:
```
{question}

```
Answer:
```
{answer}
```

Source:
```
{node["text"]}
```
If there are no key sentences, find the most relevant sentences. Do not make up any sentences.
Output the extracted key sentences as continuous text.
DON NOT SAY YOU CANNOT FIND ANY KEY SENTENCES OR INCLUDE ANY OTHER TEXT THAT ARE NOT FROM THE SOURCE.

Just extracted sentences:
"""

        response = client.messages.create(
            model="claude-opus-4-1-20250805",
            max_tokens=1000,
            temperature=0,
            messages=[{"role": "user", "content": task}],
        )
        extracted_source_nodes.append(response.content[0].text)
    return extracted_source_nodes


def evaluate_qa_answer(
    question: str,
    expected_answer: Optional[str],
    answer: str,
    extracted_source_nodes: list[str],
):
    """
    Evaluate the QA answer.
    """
    if expected_answer and expected_answer.strip() == "":
        expected_answer = None

    context = "\n".join([f'"{node}"' for node in extracted_source_nodes])
    if expected_answer:
        # Evaluate with expected answer
        metrics = [
            AnswerRelevancyMetric(),
            FaithfulnessMetric(),
            ContextualPrecisionMetric(),
            ContextualRecallMetric(),
        ]
        test_case = LLMTestCase(
            input=question,
            actual_output=answer,
            expected_output=expected_answer,
            retrieval_context=[context],
        )
    else:
        # Evaluate without expected answer
        metrics = [
            AnswerRelevancyMetric(),
        ]
        test_case = LLMTestCase(
            input=question,
            actual_output=answer,
            retrieval_context=[context],
        )

    # Run metrics one by one
    scores = {}
    for metric in metrics:
        score = metric.measure(test_case)
        scores[metric.__class__.__name__] = score
    return scores


def qa_answering(instruction, pdf_path, expected_answer):
    """
    Display the results of the QA answering.
    """
    outputs = {}
    # domain_keywords = output_generator.generate_domain_keywords_interactive(instruction, False)
    # outputs = output_generator.generate_single_output(
    #     task=instruction,
    #     task_input=pdf_path,
    #     llm_output="text",
    #     domain_keywords=domain_keywords,
    # )
    answer = _get_answer(instruction, pdf_path)
    outputs["qa_results"] = answer

    # Clean up the source nodes to remove multiple spaces and newlines in between words
    source_nodes = [
        {
            "text": re.sub(r"\s+", " ", node["text"].strip()),
            "score": node["score"],
        }
        for node in answer["source_nodes"]
    ]
    # Limit the source nodes to 5000 characters
    source_nodes = [
        {
            "text": (
                node["text"][:5000] + "..."
                if len(node["text"]) > 5000
                else node["text"]
            ),
            "score": node["score"],
        }
        for node in source_nodes
    ]

    # extracted_source_nodes = extract_source_nodes(
    #     instruction, answer["answer"], source_nodes
    # )
    extracted_source_nodes = [node["text"] for node in source_nodes]
    outputs["qa_results"]["extracted_source_nodes"] = extracted_source_nodes

    outputs["qa_scores"] = evaluate_qa_answer(
        instruction, expected_answer, answer["answer"], extracted_source_nodes
    )
    return outputs
