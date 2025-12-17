import logging
import uuid
from typing import List, Dict, Any, Optional
from datetime import datetime
import time
from pymongo import MongoClient
from pymongo.collection import Collection

from models.embedding_service import EmbeddingService

logger = logging.getLogger(__name__)

class MongoDBVectorStore:
    """MongoDB vector store with comprehensive search and logging."""
    
    def __init__(self, 
                mongodb_uri: str, 
                db_name: str, 
                collection_name: str,
                embedding_service: EmbeddingService):
        """Initialize MongoDB vector store."""
        try:
            self.client = MongoClient(mongodb_uri)
            self.db = self.client[db_name]
            self.collection = self.db[collection_name]
            self.embedding_service = embedding_service
            
            # Index configurations
            self.vector_index = "vector_index"
            self.search_index = "default"
            
            # Verify indexes
            self._verify_indexes()
            
            logger.info("Initialized MongoDB Vector Store", extra={
                "db_name": db_name,
                "collection": collection_name,
                "vector_index": self.vector_index,
                "search_index": self.search_index
            })

        except Exception as e:
            logger.error("Failed to initialize MongoDB connection", extra={
                "error": str(e),
                "db_name": db_name,
                "collection": collection_name
            })
            raise

    def _verify_indexes(self) -> None:
        """Verify required indexes exist."""
        try:
            indexes = list(self.collection.list_indexes())
            logger.info("Verified indexes", extra={
                "total_indexes": len(indexes),
                "index_names": [idx.get("name") for idx in indexes]
            })
        except Exception as e:
            logger.error("Failed to verify indexes", extra={"error": str(e)})
            raise

    def retrieve(self, query: str, k: int = 3, use_hybrid: bool = True, text_weight: float = 0.3) -> List[Dict]:
        """
        Retrieve documents using vector and/or text search.
        
        Args:
            query: Search query
            k: Number of results to return
            use_hybrid: Whether to use hybrid search
            text_weight: Weight for text search scores in hybrid search
            
        Returns:
            List of retrieved documents with scores
        """
        start_time = time.time()
        request_id = str(uuid.uuid4())
        
        try:
            logger.info("Starting retrieval", extra={
                "request_id": request_id,
                "query": query,
                "k": k,
                "use_hybrid": use_hybrid,
                "text_weight": text_weight
            })

            query_embedding = self.embedding_service.create_embedding(query)
            
            if use_hybrid:
                # Vector search pipeline
                vector_results = self._vector_search(query_embedding, k * 2)
                
                # Text search pipeline
                text_results = self._text_search(query, k * 2)
                
                # Combine results with scores
                results = self._combine_search_results(
                    vector_results=vector_results,
                    text_results=text_results,
                    text_weight=text_weight,
                    k=k
                )
            else:
                results = self._vector_search(query_embedding, k)

            # Add retrieval metadata
            for result in results:
                result["_retrieval_metadata"] = {
                    "timestamp": datetime.now().isoformat(),
                    "request_id": request_id,
                    "query": query,
                    "search_type": "hybrid" if use_hybrid else "vector",
                    "retrieval_latency": time.time() - start_time
                }

            # Log results
            logger.debug("Retrieved results", extra={
                "request_id": request_id,
                "results_count": len(results),
                "top_scores": [r.get("score", "N/A") for r in results[:3]]
            })

            return results

        except Exception as e:
            logger.error("Retrieval failed", extra={
                "error": str(e),
                "request_id": request_id,
                "duration": f"{time.time() - start_time:.3f}s"
            })
            return []

    def _vector_search(self, query_embedding: List[float], k: int) -> List[Dict]:
        """
        Perform vector search.
        
        Args:
            query_embedding: Query embedding vector
            k: Number of results
            
        Returns:
            List of documents with vector search scores
        """
        pipeline = [
            {
                "$vectorSearch": {
                    "index": self.vector_index,
                    "path": "embedding",
                    "queryVector": query_embedding,
                    "numCandidates": k * 10,
                    "limit": k
                }
            },
            {
                "$project": {
                    "score": {"$meta": "vectorSearchScore"},
                    "item_no": 1,
                    "product_name": 1,
                    "product_type": 1,
                    "benefits_summary": 1,
                    "price": 1,
                    "measurements": 1,
                    "categories": 1,
                    "description": 1,
                    "specifications": 1,
                    "availability": 1,
                    "ratings": 1,
                    "reviews": 1,
                    "metadata": "$$ROOT",
                    "_id": 0
                }
            }
        ]

        return list(self.collection.aggregate(pipeline))

    def _text_search(self, query: str, k: int) -> List[Dict]:
        """
        Perform text search.
        
        Args:
            query: Search query
            k: Number of results
            
        Returns:
            List of documents with text search scores
        """
        pipeline = [
            {
                "$search": {
                    "index": self.search_index,
                    "text": {
                        "query": query,
                        "path": ["product_name", "product_type", "benefits_summary"],
                        "fuzzy": {
                            "maxEdits": 1,
                            "prefixLength": 2
                        }
                    }
                }
            },
            {
                "$project": {
                    "score": {"$meta": "searchScore"},
                    "item_no": 1,
                    "product_name": 1,
                    "product_type": 1,
                    "benefits_summary": 1,
                    "price": 1,
                    "measurements": 1,
                    "categories": 1,
                    "description": 1,
                    "specifications": 1,
                    "availability": 1,
                    "ratings": 1,
                    "reviews": 1,
                    "metadata": "$$ROOT",
                    "_id": 0
                }
            },
            {"$limit": k}
        ]

        return list(self.collection.aggregate(pipeline))

    def _combine_search_results(
        self, 
        vector_results: List[Dict], 
        text_results: List[Dict],
        text_weight: float,
        k: int
    ) -> List[Dict]:
        """
        Combine vector and text search results with detailed scoring.
        
        Args:
            vector_results: Results from vector search
            text_results: Results from text search
            text_weight: Weight for text scores
            k: Number of results to return
        
        Returns:
            Combined and scored results
        """
        combined_results = {}
        
        # Process vector results
        for doc in vector_results:
            item_no = doc.get("item_no")
            if item_no not in combined_results:
                doc["vector_score"] = doc.get("score", 0)
                doc["text_score"] = 0
                doc["hybrid_score"] = 0
                combined_results[item_no] = doc

        # Process text results
        for doc in text_results:
            item_no = doc.get("item_no")
            if item_no in combined_results:
                combined_results[item_no]["text_score"] = doc.get("score", 0)
            else:
                doc["vector_score"] = 0
                doc["text_score"] = doc.get("score", 0)
                doc["hybrid_score"] = 0
                combined_results[item_no] = doc

        # Calculate hybrid scores
        for doc in combined_results.values():
            vector_score = float(doc["vector_score"])
            text_score = float(doc["text_score"])
            
            # Normalize scores
            max_vector_score = max(d["vector_score"] for d in combined_results.values())
            max_text_score = max(d["text_score"] for d in combined_results.values())
            
            if max_vector_score > 0:
                vector_score = vector_score / max_vector_score
            if max_text_score > 0:
                text_score = text_score / max_text_score

            # Calculate hybrid score
            hybrid_score = (
                (1 - text_weight) * vector_score +
                text_weight * text_score
            )
            
            doc["hybrid_score"] = hybrid_score
            doc["score"] = hybrid_score  # For backward compatibility
            
            # Add score explanations
            doc["score_explanation"] = {
                "vector_score": vector_score,
                "text_score": text_score,
                "hybrid_score": hybrid_score,
                "vector_weight": 1 - text_weight,
                "text_weight": text_weight
            }

        # Sort and limit results
        results = sorted(
            combined_results.values(),
            key=lambda x: x["hybrid_score"],
            reverse=True
        )[:k]

        return results