import time
import logging
from typing import Dict, List, Any
from datetime import datetime

from .base_strategy import BaseRAGStrategy
from models.query_result import QueryResult
from utils.data_utils import DataProcessor

logger = logging.getLogger(__name__)

class StepBackRAGStrategy(BaseRAGStrategy):
    """Step-back RAG strategy with enhanced logging."""

    def __init__(self, retriever, llm, config: Dict[str, Any]):
        """
        Initialize step-back RAG strategy.
        
        Args:
            retriever: Vector store retriever instance
            llm: Language model instance
            config: Strategy configuration parameters
        """
        super().__init__(retriever, llm, config)
        self._validate_config(['k', 'use_hybrid'])
        
        logger.info("Initialized Step-Back RAG Strategy", extra={
            "config": config,
            "timestamp": datetime.now().isoformat()
        })

    def _analyze_broader_context(self, query: str) -> Dict[str, str]:
        """
        Analyze broader context of the query.
        
        Args:
            query: User query string
            
        Returns:
            Dictionary containing broader context analysis
        """
        try:
            prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
            You are an interior design expert. Analyze the broader context of this request.<|eot_id|>
            <|start_header_id|>user<|end_header_id|>
            Analyze the broader context of this IKEA product request:
            {query}

            Provide concise analysis for:
            1. Space/Environment Context (size, layout, lighting)
            2. Functional Requirements (usage patterns, storage needs)
            3. Practical Constraints (budget, accessibility)

            Format: One detailed aspect per line.
            End with "End of analysis."<|eot_id|>
            <|start_header_id|>assistant<|end_header_id|>"""

            logger.debug("Analyzing broader context", extra={"query": query})
            response = self.llm.invoke(prompt)
            
            aspects = [line.strip() for line in response.split('\n') 
                      if line.strip() and 'End of analysis' not in line]
            
            context_dict = {
                "space_context": aspects[0] if len(aspects) > 0 else "",
                "functional_needs": aspects[1] if len(aspects) > 1 else "",
                "constraints": aspects[2] if len(aspects) > 2 else ""
            }

            logger.info("Analyzed broader context", extra={
                "context": context_dict
            })

            return context_dict

        except Exception as e:
            self._log_error(e, query)
            raise

    def execute(self, query: str) -> QueryResult:
        """
        Execute step-back RAG strategy.
        
        Args:
            query: User query string
            
        Returns:
            QueryResult object containing execution results and metrics
        """
        start_time = time.time()
        self._log_execution_start(query)

        try:
            # Analyze broader context
            broader_context = self._analyze_broader_context(query)
            
            # Retrieve context with broader context
            context_query = f"{query} {' '.join(broader_context.values())}"
            context = self.retriever.retrieve(
                query=context_query,
                k=self.config["k"],
                use_hybrid=self.config.get("use_hybrid", True),
                text_weight=self.config.get("text_weight", 0.5)
            )

            logger.info("Retrieved context", extra={
                "context_size": len(context),
                "broader_context": broader_context
            })

            # Generate prompt
            prompt = self.generate_prompt(query, context, broader_context)
            self.logger_utils.log_prompt(prompt, "step_back", {
                "context_size": len(context),
                "broader_context": broader_context
            })

            # Generate response
            response = self.llm.invoke(prompt)
            self.logger_utils.log_llm_response(response, {
                "strategy": "step_back",
                "latency": time.time() - start_time
            })

            # Process used context
            used_context = DataProcessor.extract_product_mentions(response, context)

            # Create result
            result = QueryResult(
                strategy="step_back",
                query=query,
                context=context,
                used_context=used_context,
                generated_response=response,
                prompt_used=prompt,
                latency=time.time() - start_time,
                timestamp=datetime.now(),
                metadata={
                    "config": self.config,
                    "broader_context": broader_context,
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

    def generate_prompt(self, query: str, context: List[Dict], broader_context: Dict[str, str]) -> str:
        """
        Generate prompt for step-back RAG.
        
        Args:
            query: User query string
            context: Retrieved context documents
            broader_context: Dictionary containing broader context analysis
            
        Returns:
            Formatted prompt string
        """
        formatted_context = DataProcessor.format_context(context)
        
        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are an expert IKEA product advisor. Consider the full context of the customer's needs and provide detailed, complete recommendations.
Never stop mid-explanation and always include comprehensive product details including measurements, prices, and specific benefits.

Important: Always end your response with "End of recommendations."<|eot_id|>

<|start_header_id|>user<|end_header_id|>
Consider the broader context of this request.

Customer Query: {query}

Context Analysis:
1. Space/Environment: {broader_context['space_context']}
2. Functional Needs: {broader_context['functional_needs']}
3. Constraints: {broader_context['constraints']}

Available Products:
{formatted_context}<|eot_id|>

<|start_header_id|>assistant<|end_header_id|>
I'll provide recommendations that consider the full context of your needs.

Let me break down the solution for you:
"""

        logger.debug("Generated step-back prompt", extra={
            "prompt_length": len(prompt),
            "context_length": len(formatted_context)
        })

        return prompt