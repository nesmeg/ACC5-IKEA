import time
import logging
from typing import Dict, List, Any
from datetime import datetime

from .base_strategy import BaseRAGStrategy
from models.query_result import QueryResult
from utils.data_utils import DataProcessor

logger = logging.getLogger(__name__)

class MultiQueryRAGStrategy(BaseRAGStrategy):
    """Multi-query RAG strategy implementation with enhanced logging."""

    def __init__(self, retriever, llm, config: Dict[str, Any]):
        """
        Initialize multi-query RAG strategy.
        
        Args:
            retriever: Vector store retriever instance
            llm: Language model instance
            config: Strategy configuration parameters
        """
        super().__init__(retriever, llm, config)
        self._validate_config(['k', 'use_hybrid'])
        
        logger.info("Initialized Multi-Query RAG Strategy", extra={
            "config": config,
            "timestamp": datetime.now().isoformat()
        })

    def _generate_query_aspects(self, query: str) -> List[str]:
        """
        Generate different aspects of the query.
        
        Args:
            query: Original user query
            
        Returns:
            List of query aspects
        """
        try:
            prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a product search expert. Create specific and detailed search queries.<|eot_id|>
<|start_header_id|>user<|end_header_id|>
Generate three different search perspectives for this query:

Original Query: {query}

Focus on:
1. Core functionality and primary requirements
2. Alternative product types that could serve the same need
3. Specific features, materials, or measurements

Respond with exactly three search queries, one per line.
End your response with "End of search queries."<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>"""

            logger.debug("Generating query aspects", extra={"original_query": query})
            response = self.llm.invoke(prompt)  # Updated to use invoke
            
            aspects = [
                line.strip("- ").strip() 
                for line in response.split("\n") 
                if line.strip() and "End of search queries" not in line
            ]

            logger.info("Generated query aspects", extra={
                "original_query": query,
                "aspects": aspects,
                "aspects_count": len(aspects)
            })

            return aspects or [query]

        except Exception as e:
            self._log_error(e, query)
            return [query]

    def execute(self, query: str) -> QueryResult:
        """
        Execute multi-query RAG strategy.
        
        Args:
            query: User query string
            
        Returns:
            QueryResult object containing execution results and metrics
        """
        start_time = time.time()
        self._log_execution_start(query)

        try:
            # Generate query aspects
            query_aspects = self._generate_query_aspects(query)
            
            # Retrieve context for each aspect
            all_context = []
            for idx, aspect in enumerate(query_aspects):
                context = self.retriever.retrieve(
                    query=aspect,
                    k=self.config["k"],
                    use_hybrid=self.config.get("use_hybrid", True),
                    text_weight=self.config.get("text_weight", 0.4)
                )
                all_context.extend(context)
                
                logger.debug(f"Retrieved context for aspect {idx + 1}", extra={
                    "aspect": aspect,
                    "context_size": len(context)
                })

            # Deduplicate context
            seen_items = set()
            unique_context = []
            for item in all_context:
                item_no = item.get('item_no')
                if item_no not in seen_items:
                    seen_items.add(item_no)
                    unique_context.append(item)

            logger.info("Context retrieval completed", extra={
                "total_context": len(all_context),
                "unique_context": len(unique_context)
            })

            # Generate prompt
            prompt = self.generate_prompt(query, unique_context, query_aspects)
            self.logger_utils.log_prompt(prompt, "multi_query", {
                "aspects": query_aspects,
                "context_size": len(unique_context)
            })

            # Generate response
            response = self.llm.invoke(prompt)  # Updated to use invoke
            self.logger_utils.log_llm_response(response, {
                "strategy": "multi_query",
                "latency": time.time() - start_time
            })

            # Process used context
            used_context = DataProcessor.extract_product_mentions(response, unique_context)

            # Create result
            result = QueryResult(
                strategy="multi_query",
                query=query,
                context=unique_context,
                used_context=used_context,
                generated_response=response,
                prompt_used=prompt,
                latency=time.time() - start_time,
                timestamp=datetime.now(),
                metadata={
                    "config": self.config,
                    "query_aspects": query_aspects,
                    "aspects_count": len(query_aspects),
                    "context_size": len(unique_context),
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

    def generate_prompt(self, query: str, context: List[Dict], query_aspects: List[str]) -> str:
        """
        Generate prompt for multi-query RAG.
        
        Args:
            query: Original user query
            context: Retrieved context documents
            query_aspects: Generated query aspects
            
        Returns:
            Formatted prompt string
        """
        formatted_context = DataProcessor.format_context(context)
        formatted_aspects = "\n".join([f"- {aspect}" for aspect in query_aspects])
        
        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
        You are an expert IKEA product advisor. Consider the full context of the customer's needs and provide detailed, complete recommendations.
        Never stop mid-explanation and always include comprehensive product details including measurements, prices, and specific benefits.

        Important: Always end your response with "End of recommendations."<|eot_id|>

        <|start_header_id|>user<|end_header_id|>
        I need recommendations based on multiple aspects of this query.

        Original Query: {query}

        We've explored these aspects:
        {formatted_aspects}

        Available Products:
        {formatted_context}<|eot_id|>

        <|start_header_id|>assistant<|end_header_id|>
        I'll provide comprehensive recommendations considering all aspects of your needs.

        Let me analyze each available product based on the different perspectives:
        """

        logger.debug("Generated multi-query prompt", extra={
            "prompt_length": len(prompt),
            "aspects_count": len(query_aspects),
            "context_length": len(formatted_context)
        })

        return prompt
