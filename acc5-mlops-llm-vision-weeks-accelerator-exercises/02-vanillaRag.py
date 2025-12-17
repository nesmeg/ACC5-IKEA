import os
import csv
from typing import List, Dict
import logging
from datetime import datetime
from pymongo import MongoClient
import openai
from langchain.llms import OpenAI

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

CSV_FILE = "rag_results.csv"

# Ask MLOps team for these details (API_KEY and MONGODB_URI)
API_KEY = ""
MONGODB_URI = "mongodb+srv://demo-fe-genai-model:NusukYJoAw573v9d@cluster-dev.fipfr.mongodb.net/?retryWrites=true&w=majority&appName=cluster-dev"

# -------------------------
# 🔹 Part 1: Embedding Service
# -------------------------
class EmbeddingService:
    """Service for creating embeddings using OpenAI format."""
    def __init__(self, api_key: str, base_url: str, model: str):
        # ✅ TODO: Initialize OpenAI client with API key and model -> https://platform.openai.com/docs/libraries
        self.client = openai.OpenAI(
            api_key=API_KEY,
            base_url=base_url
        )
        self.model = model
        logger.info(f"Initialized embedding service with model: {model}")

    def create_embedding(self, text: str) -> List[float]:
        """Generate embeddings for input text."""
        try:
            # ✅ TODO: Call OpenAI embedding API 
            response = self.client.embeddings.create(input=text, model="sentence-transformers/all-MiniLM-L6-v2")
            logger.info(f"Created embedding for text: {text}")
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Failed to create embedding: {str(e)}")
            raise

# -------------------------
# 🔹 Part 2: Vector Search in MongoDB
# -------------------------
class MongoDBVectorRetriever:
    """MongoDB vector retriever for product recommendations."""
    def __init__(self, mongodb_uri: str, db_name: str, collection_name: str, embedding_service: EmbeddingService):
        try:
            self.client = MongoClient(mongodb_uri)
            self.db = self.client[db_name]
            self.collection = self.db[collection_name]
            self.embedding_service = embedding_service
            self.index_name = "vector_index"
            logger.info(f"Initialized MongoDB retriever with index: {self.index_name}")
        except Exception as e:
            logger.error(f"Failed to initialize MongoDB connection: {str(e)}")
            raise

    def retrieve(self, query: str, k: int = 3) -> List[Dict]:
        try:
            # Generate embedding for query
            query_embedding = self.embedding_service.create_embedding(query)
            
            # Define search pipeline
            pipeline = [
                {
                    "$vectorSearch": {
                        "index": self.index_name,
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
                        "benefits": 1,
                        "measurements": 1,
                        "price": 1,
                        "_id": 0
                    }
                }
            ]

            results = list(self.collection.aggregate(pipeline))
            logger.debug(f"Vector search returned {len(results)} results")
            return results

        except Exception as e:
            logger.error(f"Vector search failed: {str(e)}")
            raise

# -------------------------
# 🔹 Part 3: RAG System with Context-Based Prompts
# -------------------------
class SimpleRAG:
    """Simple RAG implementation using MongoDB vector store."""
    def __init__(self, embedding_service: EmbeddingService, retriever: MongoDBVectorRetriever, llm: OpenAI):
        self.embedding_service = embedding_service
        self.retriever = retriever
        self.llm = llm
        logger.info("Successfully initialized SimpleRAG")

    def generate_response(self, query: str, context: List[Dict]) -> str:
        """Generate response using LLM and retrieved context."""
        try:
            formatted_context = "\n".join([str(doc) for doc in context])

            # ✅ TODO: Create Prompt
            prompt = f"""
            ...
            You are an IKEA customer service representative. A customer has asked for product recommendations.
            Please provide a response to the user based on the request he is making "customer query" and the provided relevant information "product context".

            Product Context:
            {formatted_context}

            Customer Query: {query}<|eot_id|>

            ...
            """
            
            # ✅ TODO: Send the prompt to the LLM and retrieve response -> https://python.langchain.com/v0.1/docs/modules/model_io/llms/quick_start/
            response = self.llm.invoke(prompt)
            return response
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            raise

    def query(self, user_query: str, k: int = 3) -> Dict:
        """Retrieve documents and generate a RAG response."""
        try:
            # ✅ TODO: Retrieve top-K documents using MongoDB using the "retrieve" method from the retriever
            context = self.retriever.retrieve(user_query, k)
            
            # ✅ TODO: Generate final response with retrieved context using the "generate_response" method
            response = self.generate_response(user_query, context)

            result = {
                "query": user_query,
                "context": context,
                "response": response,
                "timestamp": datetime.now().isoformat()
            }
            self.save_to_csv(result)
            return result
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            raise

    def save_to_csv(self, result: Dict):
        """Save query results to a CSV file."""
        try:
            file_exists = os.path.isfile(CSV_FILE)
            with open(CSV_FILE, mode='a', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                if not file_exists:
                    writer.writerow(["Question", "Context", "Response", "Timestamp"])
                formatted_context = " | ".join([str(doc) for doc in result["context"]])
                writer.writerow([result["query"], formatted_context, result["response"], result["timestamp"]])
            logger.info("Saved query result to CSV file.")
        except Exception as e:
            logger.error(f"Error saving results to CSV: {str(e)}")

# -------------------------
# 🔹 Part 4: Execute RAG Pipeline with Test Queries
# -------------------------
def main():
    logger.info("Initializing embedding service...")
    embedding_service = EmbeddingService(
        api_key=API_KEY,
        base_url="https://mini-l6-v2-embedding-mlops-core.dev.inference.genai.mlops.ingka.com/v1",
        model="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    logger.info("Initializing MongoDB vector retriever...")
    retriever = MongoDBVectorRetriever(
        mongodb_uri=MONGODB_URI,
        db_name="acc5-mlops-llm-vision-weeks-db",
        collection_name="acc5-mlops-llm-vision-weeks-collection",
        embedding_service=embedding_service
    )
    
    logger.info("Initializing LLM...")
    # ✅ TODO: Tune the parameters like temperature and max_tokens
    llm = OpenAI(
        model="meta-llama/Meta-Llama-3.1-8B-Instruct",
        temperature= 0.5,
        api_key=API_KEY,
        max_tokens= 100,
        openai_api_base="https://llama31-inst-mlops-core.dev.inference.genai.mlops.ingka.com/v1"
    )
    
    # ✅ TODO: Initialize RAG system using SimpleRAG class
    rag = SimpleRAG(embedding_service, retriever, llm)

    # ✅ TODO: Define test queries
    test_queries = [
        "I have a small bathroom and need space-efficient storage solutions. What are the best options available?",
        "I am looking for a modern sofa with a Scandinavian design. Can you recommend some products?",
    ]

    
    for query in test_queries:
        logger.info(f"\nProcessing query: {query}")
        try:
            # ✅ TODO: Iterate through queries and process them using RAG system -> "query" method from "rag" previous class
            result = rag.query(query)
            logger.info(f"Response: {result['response']}")
        except Exception as e:
            logger.error(f"Error processing test query: {str(e)}")
            continue

if __name__ == "__main__":
    main()
