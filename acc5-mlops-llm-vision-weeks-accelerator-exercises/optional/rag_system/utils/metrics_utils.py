from typing import List, Dict, Any
import numpy as np
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class MetricsCalculator:
    """Utility class for calculating various RAG metrics."""

    @staticmethod
    def calculate_retrieval_metrics(context: List[Dict], used_context: List[Dict]) -> Dict[str, float]:
        """
        Calculate retrieval-related metrics.
        
        Args:
            context: All retrieved context documents
            used_context: Context documents actually used in response
            
        Returns:
            Dictionary of retrieval metrics
        """
        metrics = {
            "total_retrieved": len(context),
            "total_used": len(used_context),
            "usage_ratio": len(used_context) / len(context) if context else 0,
        }
        logger.debug("Retrieval Metrics:", extra={"metrics": metrics})
        return metrics

    @staticmethod
    def calculate_latency_metrics(start_time: float, end_time: float) -> Dict[str, float]:
        """
        Calculate latency-related metrics.
        
        Args:
            start_time: Operation start timestamp
            end_time: Operation end timestamp
            
        Returns:
            Dictionary of latency metrics
        """
        latency = end_time - start_time
        metrics = {
            "total_latency": latency,
            "latency_seconds": round(latency, 3)
        }
        logger.debug("Latency Metrics:", extra={"metrics": metrics})
        return metrics

    @staticmethod
    def calculate_response_metrics(response: str, prompt: str) -> Dict[str, int]:
        """
        Calculate response-related metrics.
        
        Args:
            response: Generated response text
            prompt: Input prompt text
            
        Returns:
            Dictionary of response metrics
        """
        metrics = {
            "response_length": len(response),
            "prompt_length": len(prompt),
            "total_tokens": len(response.split())
        }
        logger.debug("Response Metrics:", extra={"metrics": metrics})
        return metrics

    @staticmethod
    def calculate_execution_metrics(start_time: float, end_time: float, result: Any) -> Dict[str, Any]:
        """
        Calculate comprehensive execution metrics.
        
        Args:
            start_time: Execution start timestamp
            end_time: Execution end timestamp
            result: QueryResult object
            
        Returns:
            Dictionary of execution metrics
        """
        execution_time = end_time - start_time
        
        metrics = {
            "execution_time": execution_time,
            "response_length": len(result.generated_response),
            "context_size": len(result.context),
            "used_context_size": len(result.used_context),
            "success": not bool(result.error),
            "timestamp": datetime.now().isoformat()
        }
        
        logger.info("Execution Metrics:", extra={"metrics": metrics})
        return metrics

    @staticmethod
    def calculate_strategy_metrics(results: List[Any]) -> Dict[str, Dict[str, Any]]:
        """
        Calculate strategy-specific metrics across multiple results.
        
        Args:
            results: List of QueryResult objects
            
        Returns:
            Dictionary of strategy metrics
        """
        strategy_metrics = {}
        
        for result in results:
            strategy = result.strategy
            if strategy not in strategy_metrics:
                strategy_metrics[strategy] = {
                    "total_executions": 0,
                    "successful_executions": 0,
                    "total_latency": 0,
                    "total_context": 0,
                    "total_used_context": 0
                }
            
            metrics = strategy_metrics[strategy]
            metrics["total_executions"] += 1
            if not result.error:
                metrics["successful_executions"] += 1
            metrics["total_latency"] += result.latency
            metrics["total_context"] += len(result.context)
            metrics["total_used_context"] += len(result.used_context)
        
        # Calculate averages
        for strategy, metrics in strategy_metrics.items():
            total = metrics["total_executions"]
            if total > 0:
                metrics["avg_latency"] = metrics["total_latency"] / total
                metrics["avg_context_size"] = metrics["total_context"] / total
                metrics["avg_used_context"] = metrics["total_used_context"] / total
                metrics["success_rate"] = (metrics["successful_executions"] / total) * 100
        
        logger.info("Strategy Metrics:", extra={"metrics": strategy_metrics})
        return strategy_metrics