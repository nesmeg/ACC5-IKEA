import time
import logging
from typing import Dict, List, Any
from datetime import datetime

from .base_strategy import BaseRAGStrategy
from models.query_result import QueryResult
from utils.data_utils import DataProcessor

logger = logging.getLogger(__name__)

class BasicRAGStrategy(BaseRAGStrategy):
    """Basic RAG strategy implementation with enhanced logging and metrics."""

    def __init__(self, retriever, llm, config: Dict[str, Any]):
        """
        Initialize basic RAG strategy.
        
        Args:
            retriever: Vector store retriever instance
            llm: Language model instance
            config: Strategy configuration parameters
        """
        super().__init__(retriever, llm, config)
        self._validate_config(['k', 'use_hybrid'])
        
        logger.info("Initialized Basic RAG Strategy", extra={
            "config": config,
            "timestamp": datetime.now().isoformat()
        })

    def execute(self, query: str) -> QueryResult:
        """
        Execute basic RAG strategy.
        
        Args:
            query: User query string
            
        Returns:
            QueryResult object containing execution results and metrics
        """
        start_time = time.time()
        self._log_execution_start(query)

        try:
            # Retrieve context
            context = self.retriever.retrieve(
                query=query,
                k=self.config["k"],
                use_hybrid=self.config.get("use_hybrid", True),
                text_weight=self.config.get("text_weight", 0.3)
            )
            
            # Log retrieval results
            logger.debug("Retrieved context", extra={
                "context_size": len(context),
                "query": query
            })

            # Generate prompt
            prompt = self.generate_prompt(query, context)
            self.logger_utils.log_prompt(prompt, "basic", {
                "context_size": len(context)
            })

            # Generate response using the new `invoke` method
            response = self.llm.invoke(prompt)
            self.logger_utils.log_llm_response(response, {
                "strategy": "basic",
                "latency": time.time() - start_time
            })

            # Process used context
            used_context = DataProcessor.extract_product_mentions(response, context)

            # Create result
            result = QueryResult(
                strategy="basic",
                query=query,
                context=context,
                used_context=used_context,
                generated_response=response,
                prompt_used=prompt,
                latency=time.time() - start_time,
                timestamp=datetime.now(),
                metadata={
                    "config": self.config,
                    "context_size": len(context),
                    "used_context_size": len(used_context)
                }
            )

            self._log_execution_complete(result)
            return result

        except Exception as e:
            self._log_error(e, query, {
                "duration": time.time() - start_time
            })
            raise

    def generate_prompt(self, query: str, context: List[Dict]) -> str:
        """
        Generate prompt for basic RAG.
        
        Args:
            query: User query string
            context: Retrieved context documents
            
        Returns:
            Formatted prompt string
        """
        formatted_context = DataProcessor.format_context(context)
        
        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
        You are an expert IKEA product advisor. Consider the full context of the customer's needs and provide detailed, complete recommendations.
        Never stop mid-explanation and always include comprehensive product details including measurements, prices, and specific benefits.

        Important: Always end your response with "End of recommendations."<|eot_id|>

        <|start_header_id|>user<|end_header_id|>
        Here is the product context and customer query:

        Product Context:
        {formatted_context}

        Customer Query: {query}<|eot_id|>

        <|start_header_id|>assistant<|end_header_id|>
        Let me help you find the perfect IKEA products for your needs.

        Based on the available products, here are my detailed recommendations:
        """

        logger.debug("Generated prompt", extra={
            "prompt_length": len(prompt),
            "context_length": len(formatted_context)
        })

        return prompt