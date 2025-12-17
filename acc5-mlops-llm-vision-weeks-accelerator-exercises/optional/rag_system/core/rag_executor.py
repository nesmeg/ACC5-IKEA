import logging
from typing import Dict, List, Any
from datetime import datetime, date
import time
from pathlib import Path
import json
import csv

from models.query_result import QueryResult
from strategies.basic_rag import BasicRAGStrategy
from strategies.multi_query_rag import MultiQueryRAGStrategy
from strategies.hypothetical_rag import HypotheticalRAGStrategy
from strategies.step_back_rag import StepBackRAGStrategy
from utils.logging_utils import LoggerUtils
from utils.metrics_utils import MetricsCalculator

logger = logging.getLogger(__name__)

class RAGExecutor:
    """Enhanced RAG executor with comprehensive logging and metrics."""

    def __init__(self, retriever, embedding_service, llm, config: Dict[str, Any]):
        self.retriever = retriever
        self.embedding_service = embedding_service
        self.llm = llm
        self.config = config
        self.results_history: List[QueryResult] = []
        self.logger_utils = LoggerUtils()
        self.metrics_calculator = MetricsCalculator()
        
        # Initialize strategies
        self.strategies = self._initialize_strategies()
        
        logger.info("Initialized RAG Executor", extra={
            "config": config,
            "strategies": list(self.strategies.keys())
        })

    def _initialize_strategies(self) -> Dict[str, Any]:
        """Initialize all RAG strategies."""
        strategies = {
            "basic": BasicRAGStrategy(
                retriever=self.retriever,
                llm=self.llm,
                config=self.config.get("basic", {})
            ),
            "multi_query": MultiQueryRAGStrategy(
                retriever=self.retriever,
                llm=self.llm,
                config=self.config.get("multi_query", {})
            ),
            "hypothetical": HypotheticalRAGStrategy(
                retriever=self.retriever,
                llm=self.llm,
                config=self.config.get("hypothetical", {})
            ),
            "step_back": StepBackRAGStrategy(
                retriever=self.retriever,
                llm=self.llm,
                config=self.config.get("step_back", {})
            )
        }
        
        logger.info("Initialized strategies", extra={
            "strategy_configs": {
                name: strategy.config for name, strategy in strategies.items()
            }
        })
        
        return strategies

    def execute_strategy(self, strategy_name: str, query: str) -> QueryResult:
        """Execute a specific RAG strategy with comprehensive logging."""
        start_time = time.time()
        
        try:
            logger.info(f"Executing strategy", extra={
                "strategy": strategy_name,
                "query": query,
                "timestamp": datetime.now().isoformat()
            })

            if strategy_name not in self.strategies:
                raise ValueError(f"Unknown strategy: {strategy_name}")

            # Execute strategy
            strategy = self.strategies[strategy_name]
            result = strategy.execute(query)

            # Calculate and log metrics
            execution_metrics = self.metrics_calculator.calculate_execution_metrics(
                start_time=start_time,
                end_time=time.time(),
                result=result
            )
            
            logger.info("Strategy execution completed", extra={
                "strategy": strategy_name,
                "metrics": execution_metrics,
                "duration": f"{time.time() - start_time:.3f}s"
            })

            # Store result
            self.results_history.append(result)
            return result

        except Exception as e:
            error_msg = f"Strategy execution failed: {str(e)}"
            logger.error(error_msg, extra={
                "strategy": strategy_name,
                "query": query,
                "duration": f"{time.time() - start_time:.3f}s"
            })
            
            # Create error result
            error_result = QueryResult(
                strategy=strategy_name,
                query=query,
                context=[],
                used_context=[],
                generated_response="",
                prompt_used="",
                latency=time.time() - start_time,
                timestamp=datetime.now(),
                metadata={},
                error=error_msg
            )
            
            self.results_history.append(error_result)
            return error_result


    def _calculate_strategy_breakdown(self) -> Dict[str, Dict[str, Any]]:
        """Calculate detailed metrics for each strategy."""
        breakdown = {}
        for strategy_name in self.strategies.keys():
            strategy_results = [r for r in self.results_history 
                              if r.strategy == strategy_name]
            
            if strategy_results:
                breakdown[strategy_name] = {
                    "total_executions": len(strategy_results),
                    "success_rate": self._calculate_success_rate(strategy_results),
                    "avg_latency": self._calculate_avg_latency(strategy_results),
                    "avg_context_size": self._calculate_avg_context_size(strategy_results),
                    "avg_response_length": self._calculate_avg_response_length(strategy_results)
                }
        
        return breakdown

    def _calculate_success_rate(self, results: List[QueryResult]) -> float:
        """Calculate success rate for a set of results."""
        if not results:
            return 0.0
        successful = sum(1 for r in results if not r.error)
        return (successful / len(results)) * 100

    def _calculate_avg_latency(self, results: List[QueryResult]) -> float:
        """Calculate average latency for a set of results."""
        if not results:
            return 0.0
        return sum(r.latency for r in results) / len(results)

    def _calculate_avg_context_size(self, results: List[QueryResult]) -> float:
        """Calculate average context size for a set of results."""
        if not results:
            return 0.0
        return sum(r.context_size for r in results) / len(results)

    def _calculate_avg_response_length(self, results: List[QueryResult]) -> float:
        """Calculate average response length for a set of results."""
        if not results:
            return 0.0
        return sum(r.response_length for r in results) / len(results)

    def _calculate_performance_metrics(self) -> Dict[str, Any]:
        """Calculate overall performance metrics."""
        if not self.results_history:
            return {}
        
        return {
            "overall_success_rate": self._calculate_success_rate(self.results_history),
            "avg_latency": self._calculate_avg_latency(self.results_history),
            "avg_context_size": self._calculate_avg_context_size(self.results_history),
            "avg_response_length": self._calculate_avg_response_length(self.results_history),
            "total_execution_time": sum(r.latency for r in self.results_history)
        }
            
    def export_results(self, output_dir: str = "results") -> tuple[str, str]:
        """
        Export comprehensive results with detailed metrics.
        Handles non-serializable objects and model outputs properly.
        
        Args:
            output_dir: Directory for output files
            
        Returns:
            Tuple of (json_path, csv_path)
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        def serialize_value(value: Any) -> Any:
            """Helper function to ensure all values are JSON serializable."""
            if hasattr(value, 'model_dump'):  # Handle Pydantic models
                return value.model_dump()
            elif hasattr(value, 'to_dict'):   # Handle objects with to_dict method
                return value.to_dict()
            elif isinstance(value, (datetime, date)):
                return value.isoformat()
            elif isinstance(value, (set, frozenset)):
                return list(value)
            elif hasattr(value, '__dict__'):  # Handle generic objects
                return {k: serialize_value(v) for k, v in value.__dict__.items() 
                    if not k.startswith('_')}
            return value

        def clean_dict(d: Dict) -> Dict:
            """Recursively clean dictionary of non-serializable objects."""
            cleaned = {}
            for k, v in d.items():
                if isinstance(v, dict):
                    cleaned[k] = clean_dict(v)
                elif isinstance(v, list):
                    cleaned[k] = [clean_dict(i) if isinstance(i, dict) else serialize_value(i) for i in v]
                else:
                    cleaned[k] = serialize_value(v)
            return cleaned

        try:
            # Get model name safely
            llm_config = {}
            for attr in ['model', 'model_name', 'engine', '_model_name']:
                if hasattr(self.llm, attr):
                    llm_config['model'] = getattr(self.llm, attr)
                    break
            
            # Add other LLM parameters safely
            for param in ['temperature', 'max_tokens', 'top_p', 'frequency_penalty', 'presence_penalty']:
                if hasattr(self.llm, param):
                    llm_config[param] = getattr(self.llm, param)
            
            # Calculate detailed metrics
            results_data = {
                "summary": {
                    "total_queries": len(self.results_history),
                    "unique_queries": len(set(r.query for r in self.results_history)),
                    "strategy_breakdown": clean_dict(self._calculate_strategy_breakdown()),
                    "performance_metrics": clean_dict(self._calculate_performance_metrics()),
                    "model_configuration": {
                        "llm": llm_config,
                        "embedding": {
                            "model": self.embedding_service.model,
                            "embedding_params": clean_dict(getattr(self.embedding_service, 'params', {}))
                        },
                        "retrieval": {
                            "vector_index": self.retriever.vector_index,
                            "search_index": self.retriever.search_index,
                            "retrieval_params": {
                                "hybrid_search": True,
                                "text_weight": 0.3,
                                "default_k": 3
                            }
                        }
                    },
                    "timestamp": datetime.now().isoformat()
                },
                "results": []
            }

            # Process each result
            for result in self.results_history:
                detailed_result = {
                    "execution_info": {
                        "timestamp": result.timestamp.isoformat(),
                        "strategy": result.strategy,
                        "query": result.query,
                        "latency": result.latency,
                        "status": "Success" if not result.error else "Failed",
                        "error": str(result.error) if result.error else None
                    },
                    "retrieval_info": {
                        "total_context": len(result.context),
                        "used_context": len(result.used_context),
                        "retrieved_documents": [
                            {
                                "item_no": str(doc.get("item_no")),
                                "product_name": str(doc.get("product_name")),
                                "product_type": str(doc.get("product_type", "")),
                                "price": float(doc.get("price", 0)),
                                "benefits": str(doc.get("benefits_summary", "")),
                                "measurements": doc.get("measurements", {}),
                                "score": float(doc.get("score", 0)),
                                "vector_score": float(doc.get("vector_score", 0)),
                                "text_score": float(doc.get("text_score", 0)),
                                "hybrid_score": float(doc.get("hybrid_score", 0))
                            } for doc in result.context
                        ],
                        "used_documents": [
                            {
                                "item_no": str(doc.get("item_no")),
                                "product_name": str(doc.get("product_name")),
                                "product_type": str(doc.get("product_type", "")),
                                "price": float(doc.get("price", 0)),
                                "benefits": str(doc.get("benefits_summary", "")),
                                "measurements": doc.get("measurements", {}),
                                "score": float(doc.get("score", 0)),
                                "vector_score": float(doc.get("vector_score", 0)),
                                "text_score": float(doc.get("text_score", 0)),
                                "hybrid_score": float(doc.get("hybrid_score", 0))
                            } for doc in result.used_context
                        ]
                    },
                    "response_info": {
                        "prompt_used": str(result.prompt_used),
                        "response_length": len(result.generated_response),
                        "generated_response": str(result.generated_response),
                        "token_count": int(result.token_count)
                    },
                    "metadata": clean_dict(result.metadata)
                }
                results_data["results"].append(detailed_result)

            # Export JSON with detailed results
            json_path = f"{output_dir}/rag_detailed_{timestamp}.json"
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(clean_dict(results_data), f, indent=2, ensure_ascii=False)

            # Export CSV with metrics
            csv_path = f"{output_dir}/rag_metrics_{timestamp}.csv"
            with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'Timestamp',
                    'Strategy',
                    'Query',
                    'Retrieved Context Size',
                    'Used Context Size',
                    'Retrieved Documents',
                    'Used Documents',
                    'Response Length',
                    'Token Count',
                    'Prompt Length',
                    'Embedding Model',
                    'LLM Model',
                    'Relevance Scores',
                    'Latency (s)',
                    'Status',
                    'Error Message'
                ])
                
                for result in self.results_history:
                    retrieved_docs = '; '.join([
                        f"{doc.get('product_name')}({doc.get('item_no')})"
                        for doc in result.context
                    ])
                    used_docs = '; '.join([
                        f"{doc.get('product_name')}({doc.get('item_no')})"
                        for doc in result.used_context
                    ])
                    relevance_scores = '; '.join([
                        f"{float(doc.get('score', 0)):.3f}"
                        for doc in result.context
                    ])
                    
                    writer.writerow([
                        result.timestamp.isoformat(),
                        result.strategy,
                        result.query,
                        len(result.context),
                        len(result.used_context),
                        retrieved_docs,
                        used_docs,
                        len(result.generated_response),
                        result.token_count,
                        len(result.prompt_used),
                        self.embedding_service.model,
                        llm_config.get('model', 'unknown'),
                        relevance_scores,
                        f"{result.latency:.3f}",
                        'Success' if not result.error else 'Failed',
                        str(result.error) if result.error else ''
                    ])

            logger.info("Exported results successfully", extra={
                "json_path": json_path,
                "csv_path": csv_path,
                "total_results": len(self.results_history)
            })
            
            return json_path, csv_path

        except Exception as e:
            logger.error(f"Failed to export results: {str(e)}", extra={
                "error_type": type(e).__name__,
                "output_dir": output_dir
            })
            raise