from abc import ABC, abstractmethod
from typing import Dict, List, Any
import logging
from datetime import datetime

from models.query_result import QueryResult
from utils.logging_utils import LoggerUtils

logger = logging.getLogger(__name__)

class BaseRAGStrategy(ABC):
    """Base class for RAG strategies with enhanced logging and metrics."""
    
    def __init__(self, retriever, llm, config: Dict[str, Any]):
        """
        Initialize the base RAG strategy.
        
        Args:
            retriever: Vector store retriever instance
            llm: Language model instance
            config: Strategy configuration parameters
        """
        self.retriever = retriever
        self.llm = llm
        self.config = config
        self.logger_utils = LoggerUtils()
        
        logger.info(f"Initialized {self.__class__.__name__}", extra={
            "config": config,
            "timestamp": datetime.now().isoformat()
        })

    @abstractmethod
    def execute(self, query: str) -> QueryResult:
        """
        Execute the RAG strategy.
        
        Args:
            query: User query string
            
        Returns:
            QueryResult object containing execution results and metrics
        """
        pass

    @abstractmethod
    def generate_prompt(self, query: str, context: List[Dict]) -> str:
        """
        Generate prompt for the strategy.
        
        Args:
            query: User query string
            context: Retrieved context documents
            
        Returns:
            Formatted prompt string
        """
        pass

    def _log_execution_start(self, query: str):
        """Log strategy execution start with metadata."""
        logger.info(f"Executing {self.__class__.__name__}", extra={
            "query": query,
            "timestamp": datetime.now().isoformat(),
            "strategy_config": self.config
        })

    def _log_execution_complete(self, result: QueryResult):
        """Log strategy execution completion with metrics."""
        logger.info(f"Completed {self.__class__.__name__}", extra={
            "metrics": {
                "latency": result.latency,
                "response_length": result.response_length,
                "context_size": result.context_size,
                "used_context_size": len(result.used_context)
            },
            "timestamp": datetime.now().isoformat()
        })

    def _log_error(self, error: Exception, query: str, context: Dict[str, Any] = None):
        """Log strategy execution error with context."""
        error_context = {
            "error_type": type(error).__name__,
            "error_message": str(error),
            "query": query,
            "timestamp": datetime.now().isoformat()
        }
        if context:
            error_context.update(context)
        
        logger.error(f"{self.__class__.__name__} execution failed", extra=error_context)

    def _validate_config(self, required_params: List[str] = None):
        """
        Validate strategy configuration parameters.
        
        Args:
            required_params: List of required parameter names
        
        Raises:
            ValueError: If required parameters are missing
        """
        if required_params:
            missing_params = [param for param in required_params if param not in self.config]
            if missing_params:
                error_msg = f"Missing required configuration parameters: {missing_params}"
                logger.error(error_msg, extra={
                    "strategy": self.__class__.__name__,
                    "config": self.config
                })
                raise ValueError(error_msg)