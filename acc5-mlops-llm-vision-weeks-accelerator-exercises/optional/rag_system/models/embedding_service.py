import logging
from typing import List, Optional
import time
from openai import OpenAI
import numpy as np

logger = logging.getLogger(__name__)

class EmbeddingService:
    def __init__(self, api_key: str, base_url: str, model: str):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.cache = {}
        
        logger.info("Initialized embedding service", extra={
            "model": model,
            "base_url": base_url
        })

    def create_embedding(self, text: str) -> List[float]:
        """Create embedding"""
        try:
            # Input validation
            if not text or not isinstance(text, str):
                raise ValueError("Invalid input text")

            cache_key = hash(text)
            if cache_key in self.cache:
                logger.debug("Retrieved embedding from cache", extra={
                    "text_length": len(text),
                    "cache_key": cache_key
                })
                return self.cache[cache_key]

            start_time = time.time()
            response = self.client.embeddings.create(
                input=text,
                model=self.model,
            )
            embedding = response.data[0].embedding
            
            duration = time.time() - start_time
            logger.debug("Created embedding", extra={
                "text_length": len(text),
                "embedding_dim": len(embedding),
                "duration": f"{duration:.3f}s"
            })
            
            self.cache[cache_key] = embedding
            return embedding

        except Exception as e:
            error_msg = f"Embedding creation failed: {str(e)}"
            logger.error(error_msg, extra={
                "text_length": len(text) if text else 0,
                "error_type": type(e).__name__
            })
            raise

    def batch_create_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Create embeddings for multiple texts with batching."""
        start_time = time.time()
        results = []
        batch_size = 100  

        try:
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                response = self.client.embeddings.create(
                    input=batch,
                    model=self.model
                )
                batch_embeddings = [d.embedding for d in response.data]
                results.extend(batch_embeddings)

            duration = time.time() - start_time
            logger.info("Batch embedding creation completed", extra={
                "total_texts": len(texts),
                "total_batches": len(range(0, len(texts), batch_size)),
                "duration": f"{duration:.3f}s"
            })
            
            return results

        except Exception as e:
            logger.error("Batch embedding creation failed", extra={
                "error": str(e),
                "total_texts": len(texts),
                "duration": f"{time.time() - start_time:.3f}s"
            })
            raise