import time
import logging
from typing import Dict, List, Any
from datetime import datetime

from .base_strategy import BaseRAGStrategy
from models.query_result import QueryResult
from utils.data_utils import DataProcessor

logger = logging.getLogger(__name__)

class HypotheticalRAGStrategy(BaseRAGStrategy):
    """Hypothetical document RAG strategy with enhanced logging."""

    def __init__(self, retriever, llm, config: Dict[str, Any]):
        """
        Initialize hypothetical RAG strategy.
        
        Args:
            retriever: Vector store retriever instance
            llm: Language model instance
            config: Strategy configuration parameters
        """
        super().__init__(retriever, llm, config)
        self._validate_config(['k', 'use_hybrid'])
        
        logger.info("Initialized Hypothetical RAG Strategy", extra={
            "config": config,
            "timestamp": datetime.now().isoformat()
        })

    def _generate_hypothetical_document(self, query: str) -> str:
        """
        Generate hypothetical ideal product description.
        
        Args:
            query: User query string
            
        Returns:
            Generated hypothetical document
        """
        try:
            prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
            You are a product design expert. Your goal is to create a concise and specific description of an ideal product.<|eot_id|>
            <|start_header_id|>user<|end_header_id|>
            Design an ideal IKEA product based on this request:
            {query}

            Keep your description concise (maximum 3-5 sentences) and focus only on the following:
            1. Key features and dimensions
            2. Materials and style
            3. Primary use case and value proposition

            Respond with a short, clear description and end with "End of specification."<|eot_id|>
            <|start_header_id|>assistant<|end_header_id|>"""


            logger.debug("Generating hypothetical document", extra={"query": query})
            response = self.llm.invoke(prompt)
            
            logger.info("Generated hypothetical document", extra={
                "response_length": len(response)
            })

            return response

        except Exception as e:
            self._log_error(e, query)
            raise

    def execute(self, query: str) -> QueryResult:
        """
        Execute hypothetical RAG strategy.
        
        Args:
            query: User query string
            
        Returns:
            QueryResult object containing execution results and metrics
        """
        start_time = time.time()
        self._log_execution_start(query)

        try:
            # Generate hypothetical document
            hypothetical_doc = self._generate_hypothetical_document(query)
            
            # Retrieve context using both original query and hypothetical doc
            context_query = f"{query} {hypothetical_doc}"
            context = self.retriever.retrieve(
                query=context_query,
                k=self.config["k"],
                use_hybrid=self.config.get("use_hybrid", False)
            )

            logger.info("Retrieved context", extra={
                "context_size": len(context),
                "hypothetical_doc_length": len(hypothetical_doc)
            })

            # Generate prompt
            prompt = self.generate_prompt(query, context, hypothetical_doc)
            self.logger_utils.log_prompt(prompt, "hypothetical", {
                "context_size": len(context),
                "hypothetical_length": len(hypothetical_doc)
            })

            # Generate response
            response = self.llm.invoke(prompt)
            self.logger_utils.log_llm_response(response, {
                "strategy": "hypothetical",
                "latency": time.time() - start_time
            })

            # Process used context
            used_context = DataProcessor.extract_product_mentions(response, context)

            # Create result
            result = QueryResult(
                strategy="hypothetical",
                query=query,
                context=context,
                used_context=used_context,
                generated_response=response,
                prompt_used=prompt,
                latency=time.time() - start_time,
                timestamp=datetime.now(),
                metadata={
                    "config": self.config,
                    "hypothetical_doc": hypothetical_doc,
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

    def generate_prompt(self, query: str, context: List[Dict], hypothetical_doc: str) -> str:
        """
        Generate prompt for hypothetical RAG.
        
        Args:
            query: User query string
            context: Retrieved context documents
            hypothetical_doc: Generated hypothetical document
            
        Returns:
            Formatted prompt string
        """
        formatted_context = DataProcessor.format_context(context)
        
        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
        You are an expert IKEA product advisor. Consider the full context of the customer's needs and provide detailed, complete recommendations.
        Never stop mid-explanation and always include comprehensive product details including measurements, prices, and specific benefits.

        Important: Always end your response with "End of recommendations."<|eot_id|>

        <|start_header_id|>user<|end_header_id|>
        Help me find products that match this ideal description.

        Customer Query: {query}

        Ideal Product Description:
        {hypothetical_doc}

        Available Products:
        {formatted_context}<|eot_id|>

        <|start_header_id|>assistant<|end_header_id|>
        I'll help you find products that best match your ideal requirements.

        Let me analyze each available product against your ideal specifications:
        """

        logger.debug("Generated hypothetical prompt", extra={
            "prompt_length": len(prompt),
            "context_length": len(formatted_context),
            "hypothetical_length": len(hypothetical_doc)
        })

        return prompt