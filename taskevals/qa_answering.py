import os
import shutil
from typing import Optional

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


def evaluate_qa_answer(
    question: str, expected_answer: Optional[str], answer: str, source_nodes: list[dict]
):
    """
    Evaluate the QA answer.
    """
    if expected_answer and expected_answer.strip() == "":
        expected_answer = None

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
            retrieval_context=[node["text"] for node in source_nodes],
        )
    else:
        # Evaluate without expected answer
        metrics = [
            AnswerRelevancyMetric(),
        ]
        test_case = LLMTestCase(
            input=question,
            actual_output=answer,
            retrieval_context=[node["text"] for node in source_nodes],
        )

    # Run metrics one by one
    scores = {}
    for metric in metrics:
        score = metric.measure(test_case)
        scores[metric.__class__.__name__] = score
    return scores


def qa_answering(question, pdf_path, expected_answer):
    """
    Display the results of the QA answering.
    """
    task = f"Answer the question from the uploaded pdf for this question: {question}"
    domain_keywords = output_generator.generate_domain_keywords_interactive(task, False)
    outputs = output_generator.generate_single_output(
        task=task,
        task_input=pdf_path,
        llm_output="text",
        domain_keywords=domain_keywords,
    )
    answer = _get_answer(question, pdf_path)
    outputs["qa_results"] = answer
    outputs["qa_scores"] = evaluate_qa_answer(
        question, expected_answer, answer["answer"], answer["source_nodes"]
    )
    return outputs
