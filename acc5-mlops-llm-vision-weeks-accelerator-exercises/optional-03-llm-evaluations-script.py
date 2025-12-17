import json
import logging
import os
import csv
import datetime
import numpy as np
import pandas as pd
import plotly.express as px
from typing import Dict, List, Any
from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    ContextualRelevancyMetric,
    BiasMetric,
    ToxicityMetric,
    HallucinationMetric
)
from deepeval.test_case import LLMTestCase
from deepeval.models import DeepEvalBaseLLM
from utils_evaluation import OpenAIEvaluationModel

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

CSV_FILE = "rag_results.csv"

class RAGEvaluator:
    """Class to evaluate RAG-generated responses using DeepEval and OpenAIEvaluationModel."""

    def __init__(self, csv_file: str, llm_config: Dict[str, Any]):
        self.csv_file = csv_file
        self.results = []
        self.llm_model = OpenAIEvaluationModel(llm_config)
        self.load_results()

    def load_results(self):
        """Load results from the CSV file into memory."""
        try:
            with open(self.csv_file, newline='', encoding='utf-8') as file:
                reader = csv.DictReader(file)
                for row in reader:
                    self.results.append(row)
            logging.info(f"Loaded {len(self.results)} results from CSV.")
        except Exception as e:
            logging.error(f"Failed to load results from CSV: {str(e)}")

    def create_metrics(self):
        """Initialize the evaluation metrics with OpenAIEvaluationModel."""
        return {
            'answer_relevancy': AnswerRelevancyMetric(threshold=0.75, model=self.llm_model, include_reason=True),
            'faithfulness': FaithfulnessMetric(threshold=0.75, model=self.llm_model, include_reason=True),
            'contextual_relevancy': ContextualRelevancyMetric(threshold=0.80, model=self.llm_model, include_reason=True),
            'bias': BiasMetric(threshold=0.60, model=self.llm_model, include_reason=True),
            'toxicity': ToxicityMetric(threshold=0.50, model=self.llm_model, include_reason=True),
            'hallucination': HallucinationMetric(threshold=0.65, model=self.llm_model, include_reason=True)
        }

    def evaluate_results(self):
        """Evaluate each RAG result using DeepEval metrics."""
        evaluation_results = []
        metrics = self.create_metrics()

        for row in self.results:
            query = row["Question"]
            context = row["Context"]
            response = row["Response"]

            result_entry = {
                "query": query,
                "response": response,
                "metrics": {}
            }

            for metric_name, metric in metrics.items():
                print(f"Evaluating {metric_name} for query '{query}'")
                try:
                    if metric_name == 'hallucination':
                        test_case = LLMTestCase(
                            input=query,
                            actual_output=response,
                            context=context
                            )
                    else:
                        # Create test case
                        test_case = LLMTestCase(
                            input=query,
                            actual_output=response,
                            retrieval_context=context.split(" | ")
                        )                  
                                    
                    metric.measure(test_case)
                    result_entry["metrics"][metric_name] = {
                        "score": float(metric.score),
                        "reason": metric.reason if hasattr(metric, 'reason') else None
                    }
                except Exception as e:
                    logging.error(f"Error evaluating {metric_name} for query '{query}': {str(e)}")
                    result_entry["metrics"][metric_name] = {"score": 0.0, "reason": "Evaluation error"}

            evaluation_results.append(result_entry)

        return evaluation_results

    def save_results(self, results: List[Dict]):
        """Save the evaluation results to a JSON file."""
        try:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"evaluation_results/rag_evaluation_{timestamp}.json"
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2)

            logging.info(f"Evaluation results saved to {output_file}")
        except Exception as e:
            logging.error(f"Error saving results to file: {str(e)}")

    def visualize_results(self, results: List[Dict]):
        """Generate visualizations for the evaluation results."""
        try:
            data = []
            for result in results:
                for metric, details in result["metrics"].items():

                    data.append({
                        "query": result["query"],
                        "metric": metric,
                        "score": details["score"]
                    })

            df = pd.DataFrame(data)
            fig = px.bar(
                df,
                x="query",
                y="score",
                color="metric",
                barmode="group",
                title="Metric Scores by Query"
            )

            fig.write_image("evaluation_results/metric_scores.png")
            logging.info("Saved evaluation visualizations.")
        except Exception as e:
            logging.error(f"Error generating visualizations: {str(e)}")


def main():
    """Execute the RAG evaluation and visualization pipeline."""
    try:
        LLM_CONFIG = {
            "model": "meta-llama/Meta-Llama-3.1-8B-Instruct",
            "temperature": 0.5,
            "max_tokens": 2048,
            "api_key": os.getenv("LLM_API_KEY", "test"),
            "api_base": os.getenv("LLM_API_BASE", "http://localhost:8000/v1"),
            "model_kwargs": {
                "stop": ["<|eot_id|>"],
                "top_p": 0.99,

            }
        }

        evaluator = RAGEvaluator(CSV_FILE, LLM_CONFIG)
        results = evaluator.evaluate_results()
        evaluator.save_results(results)
        evaluator.visualize_results(results)
    except Exception as e:
        logging.error(f"Error running evaluation pipeline: {str(e)}")

if __name__ == "__main__":
    main()
