import logging
from typing import Any, Dict, Optional
import json
from datetime import datetime

logger = logging.getLogger(__name__)

class LoggerUtils:
    """Utility class for enhanced logging functionality."""
    
    @staticmethod
    def log_model_config(config: Dict[str, Any], model_type: str) -> None:
        """
        Log model configuration details.
        
        Args:
            config: Model configuration dictionary
            model_type: Type of model being configured
        """
        logger.info(f"{model_type} Configuration:", extra={
            "config": config,
            "timestamp": datetime.now().isoformat()
        })

    @staticmethod
    def log_prompt(prompt: str, strategy: str, metadata: Optional[Dict] = None) -> None:
        """
        Log prompt with detailed metadata.
        
        Args:
            prompt: Generated prompt text
            strategy: Name of the strategy
            metadata: Additional metadata to log
        """
        log_data = {
            "timestamp": datetime.now().isoformat(),
            "strategy": strategy,
            "prompt_length": len(prompt),
            "prompt": prompt
        }
        if metadata:
            log_data.update(metadata)
        
        logger.debug("Generated Prompt:", extra={"data": log_data})

    @staticmethod
    def log_llm_response(response: str, metadata: Dict[str, Any]) -> None:
        """
        Log LLM response with performance metrics.
        
        Args:
            response: Generated response text
            metadata: Response metadata and metrics
        """
        log_data = {
            "timestamp": datetime.now().isoformat(),
            "response_length": len(response),
            "response": response,
            **metadata
        }
        logger.debug("LLM Response:", extra={"data": log_data})

    @staticmethod
    def log_retrieval_results(results: list, query: str, strategy: str) -> None:
        """
        Log retrieval results with metadata.
        
        Args:
            results: Retrieved documents/context
            query: Original query
            strategy: Strategy name
        """
        log_data = {
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "strategy": strategy,
            "num_results": len(results),
            "results": results
        }
        logger.debug("Retrieved Context:", extra={"data": log_data})

    @staticmethod
    def log_error(error: Exception, context: Dict[str, Any]) -> None:
        """
        Log error with detailed context.
        
        Args:
            error: Exception object
            context: Error context information
        """
        error_data = {
            "timestamp": datetime.now().isoformat(),
            "error_type": type(error).__name__,
            "error_message": str(error),
            **context
        }
        logger.error("Error occurred:", extra={"error": error_data})