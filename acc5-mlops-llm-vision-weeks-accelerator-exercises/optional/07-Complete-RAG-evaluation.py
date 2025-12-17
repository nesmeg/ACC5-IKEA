import json
import logging
import os
from typing import Dict, List, Any, Optional
import time
from json import JSONDecodeError
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
import numpy as np
import datetime
from utils_evaluation import OpenAIEvaluationModel
from multiprocessing import Pool, cpu_count

# Basic logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def extract_full_context(retrieval_info: Dict) -> List[str]:
    """Extract context from retrieval information as a list of strings"""
    context = []
    
    # Add all retrieved documents
    if 'retrieved_documents' in retrieval_info:
        for doc in retrieval_info['retrieved_documents']:
            context_str = (
                f"Product: {doc.get('product_name', 'Unknown')} | "
                f"Type: {doc.get('product_type', 'Unknown')} | "
                f"Price: ${doc.get('price', 'N/A')} | "
                f"Benefits: {doc.get('benefits', 'No description')} | "
                f"Measurements: {doc.get('measurements', {}).get('metric', 'N/A')} / "
                f"{doc.get('measurements', {}).get('imperial', 'N/A')}"
            )
            if context_str not in context:  # Avoid duplicates
                context.append(context_str)
    
    # Add all used documents
    if 'used_documents' in retrieval_info:
        for doc in retrieval_info['used_documents']:
            context_str = (
                f"Product: {doc.get('product_name', 'Unknown')} | "
                f"Type: {doc.get('product_type', 'Unknown')} | "
                f"Price: ${doc.get('price', 'N/A')} | "
                f"Benefits: {doc.get('benefits', 'No description')} | "
                f"Measurements: {doc.get('measurements', {}).get('metric', 'N/A')} / "
                f"{doc.get('measurements', {}).get('imperial', 'N/A')}"
            )
            if context_str not in context:  # Avoid duplicates
                context.append(context_str)
    
    return context

def create_metrics(evaluation_model: OpenAIEvaluationModel) -> Dict[str, Any]:
    """Initialize the metrics with the provided evaluation model"""
    return {
        'answer_relevancy': AnswerRelevancyMetric(
            threshold=0.75,  
            model=evaluation_model,
            include_reason=True
        ),
        'faithfulness': FaithfulnessMetric(
            threshold=0.75,  
            model=evaluation_model,
            include_reason=True
        ),
        'contextual_relevancy': ContextualRelevancyMetric(
            threshold=0.80,  
            model=evaluation_model,
            include_reason=True
        ),
        'bias': BiasMetric(
            threshold=0.60, 
            model=evaluation_model,
            include_reason=True
        ),
        'toxicity': ToxicityMetric(
            threshold=0.50,  
            model=evaluation_model,
            include_reason=True
        ),
        'hallucination': HallucinationMetric(
            threshold=0.65, 
            model=evaluation_model,
            include_reason=True
        )
    }

def evaluate_single_question(args):
    """Evaluate all strategies for a single question"""
    json_path, llm_config, question_index = args
    try:
        # Load JSON data
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Initialize evaluation model
        evaluation_model = OpenAIEvaluationModel(llm_config)
        
        # Initialize results dictionary for this question
        results = {
            'question_index': question_index,
            'strategies': {}
        }
        
        # Get the question from the first strategy (they all share the same question)
        query = data['results'][question_index]['execution_info']['query']
        results['query'] = query
        
        # Initialize metrics template
        metrics = create_metrics(evaluation_model)
        
        # Evaluate each strategy
        strategies = ['basic', 'multi_query', 'hypothetical', 'step_back']
        
        for strategy in strategies:
            # Find the result for this strategy
            strategy_result = None
            for result in data['results']:
                if result['execution_info']['strategy'] == strategy:
                    strategy_result = result
                    break
            
            if not strategy_result:
                logging.warning(f"Strategy {strategy} not found in results for question {question_index}")
                continue
            
            # Extract information
            generated_response = strategy_result['response_info']['generated_response']
            retrieval_context = extract_full_context(strategy_result['retrieval_info'])
            
            # Initialize strategy results
            results['strategies'][strategy] = {
                'model_response': generated_response,
                'context': '\n'.join(retrieval_context),  # Store as string for JSON
                'metrics': {},
                'execution_info': {
                    'latency': strategy_result['execution_info']['latency'],
                    'status': strategy_result['execution_info']['status'],
                    'context_size': strategy_result['retrieval_info'].get('context_size', 0),
                    'used_context_size': strategy_result['retrieval_info'].get('used_context_size', 0)
                }
            }
            
            # Evaluate with each metric
            for metric_name, metric in metrics.items():
                try:
                    if metric_name == 'hallucination':
                        test_case = LLMTestCase(
                            input=query,
                            actual_output=generated_response,
                            context=retrieval_context
                        )
                    else:
                        # Create test case
                        test_case = LLMTestCase(
                            input=query,
                            actual_output=generated_response,
                            retrieval_context=retrieval_context
                        )
                        
                    # Measure the metric
                    metric.measure(test_case)
                    
                    # Extract score and reason
                    score = metric.score if hasattr(metric, 'score') else 0.0
                    reason = metric.reason if hasattr(metric, 'reason') else None
                    
                    # Convert numpy types to Python types for JSON serialization
                    if isinstance(score, (np.float32, np.float64)):
                        score = float(score)
                    
                    # Store results
                    results['strategies'][strategy]['metrics'][metric_name] = {
                        'score': score,
                        'reason': str(reason) if reason is not None else None
                    }
                    
                    logging.info(f"Question {question_index} | Strategy: {strategy} | {metric_name} Score: {score}")
                    
                except Exception as e:
                    logging.error(f"Error evaluating {metric_name} for {strategy} in question {question_index}: {e}")
                    results['strategies'][strategy]['metrics'][metric_name] = {
                        'score': 0.0,
                        'reason': f"Error: {str(e)}"
                    }
        
        # Add summary statistics
        results['summary'] = {strategy: {} for strategy in strategies}
        for strategy in strategies:
            if strategy in results['strategies']:
                metric_scores = [
                    metric['score'] 
                    for metric in results['strategies'][strategy]['metrics'].values()
                    if isinstance(metric['score'], (int, float))
                ]
                avg_score = np.mean(metric_scores) if metric_scores else 0.0
                results['summary'][strategy] = {
                    'average_score': float(avg_score),
                    'latency': results['strategies'][strategy]['execution_info']['latency'],
                    'context_size': results['strategies'][strategy]['execution_info']['context_size'],
                    'used_context_size': results['strategies'][strategy]['execution_info']['used_context_size']
                }
        
        return results
    
    except Exception as e:
        logging.error(f"Evaluation failed for question {question_index}: {e}")
        return {
            'question_index': question_index,
            'error': str(e)
        }

def evaluate_all_questions(json_path: str, llm_config: Dict[str, Any]) -> List[Dict]:
    """
    Evaluate all questions in the JSON file across all strategies
    
    Args:
        json_path (str): Path to the RAG results JSON file
        llm_config (Dict[str, Any]): Configuration for the LLM
        
    Returns:
        List[Dict]: List of evaluation results for each question
    """
    try:
        # Load JSON data to determine the number of questions
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        total_questions = len(data['results'])
        logging.info(f"Total questions to evaluate: {total_questions}")
        
        # Prepare arguments for parallel processing
        args = [(json_path, llm_config, idx) for idx in range(total_questions)]
        
        # Determine the number of processes
        num_processes = min(cpu_count(), 8)  # Limit to 8 to prevent overconsumption
        
        # Use multiprocessing Pool to parallelize evaluations
        with Pool(processes=num_processes) as pool:
            results = pool.map(evaluate_single_question, args)
        
        return results
    
    except JSONDecodeError as jde:
        logging.error(f"Invalid JSON format: {jde}")
        return []
    except Exception as e:
        logging.error(f"Failed to evaluate all questions: {e}")
        return []

if __name__ == "__main__":
    # LLM Configuration
    LLM_CONFIG = {
        "model": "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "temperature": 0.2,
        "max_tokens": 2048,
        "api_key": os.getenv("LLM_API_KEY", "test"),
        "api_base": os.getenv("LLM_API_BASE", "http://localhost:8001/v1"),
        "model_kwargs": {
            "stop": ["<|eot_id|>"],
            "top_p": 0.95,
            "presence_penalty": 0.6,
            "frequency_penalty": 0.3
        }
    }
    
    # Example usage
    json_path = "rag_system/results/rag_detailed_20250117_183623.json"
    
    try:
        all_results = evaluate_all_questions(json_path, LLM_CONFIG)
        
        if not all_results:
            logging.error("No results to save.")
        else:
            # Save results to JSON with timestamp
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"evaluation_results/strategies_evaluation_{timestamp}.json"
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            
            with open(output_file, 'w') as f:
                json.dump(all_results, f, indent=2)
            
            logging.info(f"All evaluation results saved to {output_file}")
        
    except Exception as e:
        logging.error(f"Evaluation process failed: {e}")
