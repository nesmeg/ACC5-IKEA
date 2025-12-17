import logging
import sys
from typing import List
import time
from datetime import datetime
from pathlib import Path

from langchain_openai import OpenAI
from config.settings import Settings
from config.logging_config import setup_logging
from models.embedding_service import EmbeddingService
from models.vector_store import MongoDBVectorStore
from core.rag_executor import RAGExecutor

logger = logging.getLogger(__name__)

def initialize_components(settings: Settings):
    """Initialize components with proper model configuration."""
    try:
        # Setup embedding service
        logger.info("Initializing embedding service...")
        embedding_service = EmbeddingService(
            api_key=settings.EMBEDDING_CONFIG["api_key"],
            base_url=settings.EMBEDDING_CONFIG["base_url"],
            model=settings.EMBEDDING_CONFIG["model"]
        )

        # Setup vector store
        logger.info("Initializing vector store...")
        vector_store = MongoDBVectorStore(
            mongodb_uri=settings.MONGODB_CONFIG["uri"],
            db_name=settings.MONGODB_CONFIG["db_name"],
            collection_name=settings.MONGODB_CONFIG["collection_name"],
            embedding_service=embedding_service
        )

        # Setup LLM with explicit model name
        logger.info("Initializing LLM...")
        llm = OpenAI(
            model_name=settings.LLM_CONFIG["model"],  
            temperature=settings.LLM_CONFIG["temperature"],
            max_tokens=settings.LLM_CONFIG["max_tokens"],
            api_key=settings.LLM_CONFIG["api_key"],
            base_url=settings.LLM_CONFIG["api_base"],
            top_p=settings.LLM_CONFIG["model_kwargs"].get("top_p", 1.0), 
            presence_penalty=settings.LLM_CONFIG["model_kwargs"].get("presence_penalty", 0.0),  
            frequency_penalty=settings.LLM_CONFIG["model_kwargs"].get("frequency_penalty", 0.0)  
        )

        # Initialize RAG executor
        logger.info("Initializing RAG executor...")
        executor = RAGExecutor(
            retriever=vector_store,
            embedding_service=embedding_service,
            llm=llm,
            config=settings.STRATEGY_CONFIGS
        )

        return executor

    except Exception as e:
        logger.error(f"Failed to initialize components: {str(e)}", 
                    extra={"error_type": type(e).__name__})
        raise

def process_queries(executor: RAGExecutor, queries: List[str]):
    """Process a list of queries using all available strategies."""
    try:
        total_start_time = time.time()
        results = []

        for query in queries:
            logger.info(f"Processing query: {query}")
            
            for strategy_name in executor.strategies.keys():
                try:
                    logger.info(f"Executing strategy: {strategy_name}")
                    result = executor.execute_strategy(strategy_name, query)
                    results.append(result)
                    
                    logger.info(f"Strategy {strategy_name} completed", extra={
                        "latency": result.latency,
                        "context_size": result.context_size,
                        "response_length": result.response_length
                    })

                except Exception as e:
                    logger.error(f"Error processing query with strategy {strategy_name}", extra={
                        "error": str(e),
                        "query": query
                    })

        # Export results
        json_path, csv_path = executor.export_results()
        
        total_duration = time.time() - total_start_time
        logger.info("Query processing completed", extra={
            "total_queries": len(queries),
            "total_executions": len(results),
            "total_duration": f"{total_duration:.3f}s",
            "results_paths": {
                "json": json_path,
                "csv": csv_path
            }
        })

    except Exception as e:
        logger.error(f"Failed to process queries: {str(e)}")
        raise

def main():
    """Main execution flow with comprehensive logging."""
    try:
        # Initialize logging
        setup_logging()
        logger.info("Starting RAG system...")

        # Load settings
        settings = Settings()
        logger.info("Loaded settings", extra={
            "base_dir": str(settings.BASE_DIR),
            "log_dir": str(settings.LOG_DIR),
            "results_dir": str(settings.RESULTS_DIR)
        })

        # Initialize components
        executor = initialize_components(settings)

        # Test queries
        test_queries = [
            # General Recommendation Questions
            "I need storage solutions for a small bathroom",
            "Looking for bedroom storage ideas",
            "Need a comfortable office chair under $200",
            "Kitchen organization solutions for small spaces",
            "Modern living room furniture for apartments",
            
            # Specific Product Questions
            "What colors are available for the SPJUTROCKA ice cube tray with lid?",
            "What is the maximum load per shelf for the LOMMARP bookcase?",
            "What materials are used in the MALM glass top?",
            "How should the VÄLVÅRDAD dish brush be cared for?",
            "What are the dimensions and color options for the HILJA curtains?"
        ]


        # Process queries
        process_queries(executor, test_queries)

        logger.info("RAG system execution completed successfully")

    except Exception as e:
        logger.error(f"RAG system execution failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()