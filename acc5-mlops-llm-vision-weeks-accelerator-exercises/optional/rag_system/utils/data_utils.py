from typing import List, Dict, Any
import json
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class DataProcessor:
    """Utility class for data processing and formatting."""

    @staticmethod
    def format_context(context: List[Dict[str, Any]]) -> str:
        """
        Format context data for prompt insertion.
        
        Args:
            context: List of context documents
            
        Returns:
            Formatted context string
        """
        formatted_context = []
        for item in context:
            formatted_item = (
                f"Product: {item.get('product_name', 'N/A')}\n"
                f"Item No: {item.get('item_no', 'N/A')}\n"
                f"Type: {item.get('product_type', 'N/A')}\n"
                f"Price: ${item.get('price', 'N/A')}\n"
                f"Benefits: {item.get('benefits_summary', 'N/A')}\n"
                f"Score: {item.get('score', 'N/A')}\n"
            )
            formatted_context.append(formatted_item)
        
        logger.debug("Formatted Context:", extra={"context": formatted_context})
        return "\n\n".join(formatted_context)

    @staticmethod
    def extract_product_mentions(response: str, context: List[Dict]) -> List[Dict]:
        """
        Extract and log products mentioned in the response.
        
        Args:
            response: Generated response text
            context: Original context documents
            
        Returns:
            List of mentioned products
        """
        mentioned_products = []
        for product in context:
            if (str(product.get("item_no")) in response or 
                product.get("product_name", "").upper() in response.upper()):
                mentioned_products.append(product)
        
        logger.debug("Mentioned Products:", extra={
            "total_mentioned": len(mentioned_products),
            "products": mentioned_products
        })
        return mentioned_products

    @staticmethod
    def clean_product_data(product: Dict[str, Any]) -> Dict[str, Any]:
        """
        Clean and validate product data.
        
        Args:
            product: Raw product data dictionary
            
        Returns:
            Cleaned product data dictionary
        """
        cleaned = {}
        required_fields = ["item_no", "product_name", "product_type", "price"]
        
        try:
            # Clean and validate required fields
            for field in required_fields:
                value = product.get(field)
                if value is None:
                    logger.warning(f"Missing required field: {field}", extra={
                        "product": product
                    })
                    value = "N/A"
                cleaned[field] = value

            # Clean optional fields
            cleaned["benefits_summary"] = product.get("benefits_summary", "")
            cleaned["measurements"] = product.get("measurements", {})
            cleaned["score"] = float(product.get("score", 0.0))

            return cleaned

        except Exception as e:
            logger.error("Error cleaning product data", extra={
                "error": str(e),
                "product": product
            })
            return product

    @staticmethod
    def validate_query_result(result: Any) -> bool:
        """
        Validate query result data.
        
        Args:
            result: QueryResult object to validate
            
        Returns:
            Boolean indicating validation success
        """
        required_fields = [
            "strategy", "query", "context", "generated_response",
            "prompt_used", "latency", "timestamp"
        ]
        
        try:
            # Check required fields
            for field in required_fields:
                if not hasattr(result, field):
                    logger.error(f"Missing required field in result: {field}")
                    return False

            # Validate types and values
            if not isinstance(result.context, list):
                logger.error("Context must be a list")
                return False

            if result.latency < 0:
                logger.error("Latency cannot be negative")
                return False

            if not isinstance(result.timestamp, datetime):
                logger.error("Timestamp must be a datetime object")
                return False

            return True

        except Exception as e:
            logger.error("Error validating query result", extra={
                "error": str(e),
                "result": str(result)
            })
            return False