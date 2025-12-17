import os
import csv
import faiss
import numpy as np
import pandas as pd
from typing import List, Dict
import logging
from datetime import datetime
from langchain.llms import OpenAI
import openai

# ✅ TODO: Configure logging for debugging and insights
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

CSV_FILE = "rag_results.csv"
PROCESSED_DATA_PATH = "../data/processed_data.csv"

# Ask MLOps team for these details
API_KEY = ""

class EmbeddingService:
    """Service for creating embeddings using OpenAI format."""
    def __init__(self, api_key: str, base_url: str, model: str):
        self.client = openai.OpenAI(
            api_key=api_key,
            base_url=base_url
        )
        self.model = model
        logger.info(f"Initialized embedding service with model: {model}")

    def create_embedding(self, text: str) -> List[float]:
        """Generate embeddings for input text."""
        try:
            # Ensure text is not None and is a string
            if text is None or not isinstance(text, str):
                text = ""
            
            # Truncate text if it's too long (rough approximation of token limit)
            if len(text) > 200:
                text = text[:200] + "..."
            
            response = self.client.embeddings.create(
                input=text,
                model=self.model
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Failed to create embedding: {str(e)}")
            raise

class FAISSVectorRetriever:
    """FAISS vector retriever for product recommendations."""
    def __init__(self, embedding_service: EmbeddingService):
        self.embedding_service = embedding_service
        self.index = None
        self.documents = []
        self.dimension = None
        logger.info("Initialized FAISS vector retriever")

    def is_missing_scalar(self, value) -> bool:
        """Check if a value is missing (None or NaN)."""
        return pd.isna(value)

    def safe_get_string(self, value, default="") -> str:
        """Safely get string value, handling NaN and None."""
        if self.is_missing_scalar(value):
            return default
        try:
            return str(value).strip()
        except:
            return default

    def safe_get_float(self, value, default=0.0) -> float:
        """Safely convert value to float."""
        if self.is_missing_scalar(value):
            return default
        try:
            return float(value)
        except (ValueError, TypeError):
            return default

    def process_benefits_list(self, benefits) -> List[str]:
        """Process benefits from string or list format into clean list."""
        if self.is_missing_scalar(benefits):
            return []
        
        try:
            if isinstance(benefits, str):
                if benefits.startswith('[') and benefits.endswith(']'):
                    # Clean and split the string representation of list
                    items = benefits.strip('[]').split(',')
                    # Clean each item and filter empty ones
                    return [item.strip().strip("'\"") for item in items if item.strip()]
            elif isinstance(benefits, list):
                return [str(item).strip() for item in benefits if str(item).strip()]
            return []
        except Exception as e:
            logger.warning(f"Error processing benefits list: {str(e)}")
            return []

    def prepare_embedding_text(self, row: pd.Series) -> str:
        """Prepare text for embedding within token limits."""
        try:
            # Safely get all text fields
            product_name = self.safe_get_string(row.get('productName'))
            product_type = self.safe_get_string(row.get('productType'))
            benefit_summary = self.safe_get_string(row.get('benefitSummary'))
            
            # Process benefits list
            benefits = self.process_benefits_list(row.get('benefits'))
            main_benefits = benefits[:2] if benefits else []  # Take only first two benefits
            
            # Combine fields with proper truncation
            parts = []
            if product_name:
                parts.append(product_name)
            if product_type:
                parts.append(product_type)
            if benefit_summary:
                parts.append(benefit_summary[:100])  # Truncate summary
            if main_benefits:
                parts.append("Key benefits: " + ". ".join(main_benefits))
            
            # Join all parts and ensure final length limit
            text = " | ".join(filter(None, parts))
            if len(text) > 200:
                text = text[:200] + "..."
                
            return text
        except Exception as e:
            logger.error(f"Error preparing embedding text: {str(e)}")
            return f"{product_name} {product_type}"  # Fallback to basic info

    def load_and_index_data(self, csv_path: str):
        """Load data from CSV and create FAISS index."""
        try:
            # Load the CSV data
            df = pd.read_csv(csv_path)
            logger.info(f"Loaded {len(df)} rows from CSV")
            
            # Create embeddings for all documents
            documents = []
            embeddings = []
            
            for idx, row in df.iterrows():
                try:
                    # Prepare text for embedding
                    text = self.prepare_embedding_text(row)
                    if not text.strip():
                        logger.warning(f"Empty text for row {idx}, skipping")
                        continue
                    
                    # Create embedding
                    embedding = self.embedding_service.create_embedding(text)
                    
                    # Store document data
                    doc = {
                        "item_no": self.safe_get_string(row.get('itemNo')),
                        "product_name": self.safe_get_string(row.get('productName')),
                        "product_type": self.safe_get_string(row.get('productType')),
                        "benefit_summary": self.safe_get_string(row.get('benefitSummary')),
                        "benefits": self.process_benefits_list(row.get('benefits')),
                        "measurements": self.safe_get_string(row.get('measurementFilters')),
                        "price": self.safe_get_float(row.get('price')),
                        "colour": self.safe_get_string(row.get('colour')),
                        "material": self.safe_get_string(row.get('material'))
                    }
                    
                    documents.append(doc)
                    embeddings.append(embedding)
                    
                except Exception as e:
                    logger.error(f"Error processing row {idx}: {str(e)}")
                    continue
            
            if not documents:
                raise ValueError("No valid documents were processed")
            
            # Convert embeddings to numpy array
            embeddings_array = np.array(embeddings).astype('float32')
            self.dimension = embeddings_array.shape[1]
            
            # Initialize and train FAISS index
            self.index = faiss.IndexFlatL2(self.dimension)
            self.index.add(embeddings_array)
            self.documents = documents
            
            logger.info(f"Successfully indexed {len(documents)} documents")
            
        except Exception as e:
            logger.error(f"Failed to load and index data: {str(e)}")
            raise

    def retrieve(self, query: str, k: int = 3) -> List[Dict]:
        """Retrieve similar documents using FAISS."""
        try:
            # Generate embedding for query
            query_embedding = np.array([self.embedding_service.create_embedding(query)]).astype('float32')
            
            # Search in FAISS index
            distances, indices = self.index.search(query_embedding, k)
            
            # Get corresponding documents
            results = []
            for i, idx in enumerate(indices[0]):
                doc = self.documents[idx].copy()
                doc['score'] = float(distances[0][i])  # Add similarity score
                results.append(doc)
            
            logger.debug(f"Vector search returned {len(results)} results")
            return results

        except Exception as e:
            logger.error(f"Vector search failed: {str(e)}")
            raise

# Rest of the code remains the same
class SimpleRAG:
    """Simple RAG implementation using FAISS vector store."""
    def __init__(self, embedding_service: EmbeddingService, retriever: FAISSVectorRetriever, llm: OpenAI):
        self.embedding_service = embedding_service
        self.retriever = retriever
        self.llm = llm
        logger.info("Successfully initialized SimpleRAG")

    def generate_response(self, query: str, context: List[Dict]) -> str:
        """Generate response using LLM and retrieved context."""
        try:
            formatted_context = "\n".join([str(doc) for doc in context])

            prompt = f"""
You are an IKEA bathroom furniture expert advisor. Consider the full context of the customer's needs and provide detailed, complete recommendations.
            Never stop mid-explanation and always include:
            - Product name and type
            - Price
            - Key measurements and dimensions
            - Material and color options
            - Specific benefits and features
            - Installation requirements
            - Compatibility with other items
            
            Format prices in USD with $ symbol. 

            Important: Always end your response with "Let me know if you want to explore other IKEA bathroom solutions"<|eot_id|>

            Here is the product context and customer query:

            Product Context:
            {formatted_context}

            Customer Query: {query}<|eot_id|>

            Let me help you find the perfect IKEA bathroom products for your needs.

            Based on the available products, here are my detailed recommendations:
            """
            
            response = self.llm.invoke(prompt)
            return response
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            raise

    def query(self, user_query: str, k: int = 3) -> Dict:
        """Retrieve documents and generate a RAG response."""
        try:
            context = self.retriever.retrieve(user_query, k)
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

def main():
    logger.info("Initializing embedding service...")
    embedding_service = EmbeddingService(
        api_key=API_KEY,
        base_url="https://mini-l6-v2-embedding-mlops-core.dev.inference.genai.mlops.ingka.com/v1",
        model="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    logger.info("Initializing FAISS vector retriever...")
    retriever = FAISSVectorRetriever(embedding_service)
    
    # Load and index data
    logger.info("Loading and indexing data...")
    retriever.load_and_index_data(PROCESSED_DATA_PATH)
    
    logger.info("Initializing LLM...")
    llm = OpenAI(
        model="meta-llama/Meta-Llama-3.1-8B-Instruct",
        temperature=0.5,
        api_key=API_KEY,
        max_tokens=2048,
        openai_api_base="https://llama31-inst-mlops-core.dev.inference.genai.mlops.ingka.com/v1"
    )
    
    # Initialize RAG system
    rag = SimpleRAG(embedding_service, retriever, llm)

    # Define test queries
    test_queries = [
        "I have a small bathroom and need space-efficient storage solutions. What are the best options available?",
        "I'm looking for a bathroom vanity under $400 with good storage and modern design. What do you recommend?",
        "I want to remodel my bathroom with a classic look. What are some elegant vanity options with traditional styling?"
    ]

    # Process test queries
    for query in test_queries:
        logger.info(f"\nProcessing query: {query}")
        try:
            result = rag.query(query)
            logger.info(f"Response: {result['response']}")
        except Exception as e:
            logger.error(f"Error processing test query: {str(e)}")
            continue

if __name__ == "__main__":
    main()