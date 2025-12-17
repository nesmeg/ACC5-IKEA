# RAG System for IKEA Product Recommendations

## Overview

This project implements a Retrieval-Augmented Generation (RAG) system for IKEA product recommendations. The system uses multiple RAG strategies to provide context-aware, accurate product recommendations based on user queries. It features comprehensive logging, metrics tracking, and different retrieval strategies to optimize the recommendation quality.

## Table of Contents

- [Project Structure](#project-structure)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [RAG Strategies](#rag-strategies)
- [Logging and Metrics](#logging-and-metrics)
- [Contributing](#contributing)

## Project Structure

```
rag_system/
│
├── config/
│   ├── __init__.py
│   ├── logging_config.py      # Logging configuration
│   └── settings.py            # System settings and configurations
│
├── utils/
│   ├── __init__.py
│   ├── logging_utils.py       # Logging utilities
│   ├── metrics_utils.py       # Metrics calculation utilities
│   └── data_utils.py         # Data processing utilities
│
├── models/
│   ├── __init__.py
│   ├── embedding_service.py   # Embedding creation service
│   ├── vector_store.py       # MongoDB vector store implementation
│   └── query_result.py       # Query result data structure
│
├── strategies/
│   ├── __init__.py
│   ├── base_strategy.py      # Base RAG strategy
│   ├── basic_rag.py         # Basic RAG implementation
│   ├── multi_query_rag.py   # Multi-query strategy
│   ├── hypothetical_rag.py  # Hypothetical document strategy
│   └── step_back_rag.py     # Step-back reasoning strategy
│
├── core/
│   ├── __init__.py
│   └── rag_executor.py       # Main RAG execution logic
│
├── logs/                     # Log files directory
├── results/                  # Results and metrics output
├── cache/                    # Cache directory
│
├── main.py                   # Application entry point
├── requirements.txt          # Project dependencies
└── README.md              
```

## Features

- Multiple RAG strategies for diverse query handling
- Hybrid search combining vector and text-based retrieval
- Comprehensive logging system with rotation
- Detailed metrics and performance tracking
- Query result caching
- Export functionality for results and metrics
- MongoDB vector store integration
- Configurable system settings

## Requirements

- Python 3.8+
- MongoDB Atlas cluster with vector search capability
- OpenAI API or compatible API endpoint
- Required Python packages (see requirements.txt)

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up environment variables (Ask MLOps team for details):
```bash
export MONGODB_URI="your_mongodb_uri"
export LLM_API_KEY="your_llm_api_key"
export EMBEDDING_API_KEY="your_embedding_api_key"
```

## Configuration

The system can be configured through the `config/settings.py` file or by providing a JSON configuration file. Key configuration sections include:

- LLM Configuration
- Embedding Service Configuration
- MongoDB Configuration
- Strategy Configurations
- Logging Configuration
- Cache Configuration

Example configuration:
```python
{
    "llm": {
        "model": "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "temperature": 0.2,
        "max_tokens": 2048
    },
    "embedding": {
        "model": "sentence-transformers/all-MiniLM-L6-v2",
        "cache_embeddings": true
    },
    "mongodb": {
        "collection_name": "products-collection",
        "vector_index": "vector_index"
    }
}
```

## Usage

1. Basic usage:
```python
from config.settings import Settings
from main import initialize_components, process_queries

# Initialize system
settings = Settings()
executor = initialize_components(settings)

# Process queries
queries = [
    "I need storage solutions for a small bathroom",
    "Looking for bedroom storage ideas"
]
process_queries(executor, queries)
```

2. Running from command line:
```bash
python main.py
```

## RAG Strategies

# RAG Strategies: Detailed Analysis and Implementation

## 1. Basic RAG Strategy

### Overview
Basic RAG performs direct retrieval and generation based on the user's query without additional processing or context manipulation.

### Implementation Details
```python
def execute(self, query: str):
    # 1. Direct retrieval from vector store
    context = self.retriever.retrieve(query, k=3)
    
    # 2. Generate response using retrieved context
    response = self.llm(prompt + context + query)
```

### Advantages
- Fast execution time (lowest latency)
- Simple to implement and maintain
- Predictable results
- Good for clear, specific queries
- Low computational overhead

### Disadvantages
- May miss relevant context
- Limited understanding of complex requirements
- Can be too literal in interpretation
- May struggle with ambiguous queries

### Best Use Cases
- Product-specific queries ("Show me MALM bed frames")
- Price inquiries ("What's the cost of BILLY bookcase?")
- Simple feature requests ("Show me blue chairs")

## 2. Multi-Query RAG Strategy

### Overview
Generates multiple perspectives of the original query to capture different aspects and interpretations.

### Implementation Details
```python
def execute(self, query: str):
    # 1. Generate query aspects
    aspects = self._generate_query_aspects(query)
    # e.g., "Need storage for small bathroom" becomes:
    # - "Compact bathroom storage solutions"
    # - "Wall-mounted bathroom organizers"
    # - "Space-saving bathroom furniture"

    # 2. Retrieve context for each aspect
    all_context = []
    for aspect in aspects:
        context = self.retriever.retrieve(aspect)
        all_context.extend(context)

    # 3. Deduplicate and generate response
    unique_context = self._deduplicate_context(all_context)
    response = self.llm(prompt + unique_context + query)
```

### Advantages
- Better coverage of different query aspects
- Captures implicit requirements
- Reduces query ambiguity
- Better handling of complex requests
- More diverse product recommendations

### Disadvantages
- Higher latency (multiple retrievals)
- More computational resources
- Can retrieve irrelevant context
- Might overwhelm with too many options

### Best Use Cases
- Complex requirements ("Need furniture for small apartment living room")
- Multi-purpose queries ("Looking for storage that works in both bedroom and office")
- Exploratory searches ("Show me options for kid's room organization")

## 3. Hypothetical RAG Strategy

### Overview
Creates an ideal/hypothetical product description first, then matches real products against this ideal specification.

### Implementation Details
```python
def execute(self, query: str):
    # 1. Generate hypothetical ideal product
    ideal_product = self._generate_hypothetical_document(query)
    # e.g., "Perfect desk would be: 120cm wide, adjustable height..."

    # 2. Use ideal product description for retrieval
    context = self.retriever.retrieve(f"{query} {ideal_product}")

    # 3. Generate response comparing real products to ideal
    response = self.llm(prompt + ideal_product + context + query)
```

### Advantages
- Better understanding of user requirements
- More precise matching of features
- Can handle abstract requirements
- Good for comparison-based recommendations
- Helps identify gaps in product catalog

### Disadvantages
- Highest latency (complex generation)
- May set unrealistic expectations
- Can be overly specific
- Might miss good alternatives

### Best Use Cases
- Detailed requirements ("Need a desk for gaming setup with cable management")
- Feature-rich requests ("Looking for a sofa bed with storage and USB ports")
- Custom solutions ("Want a modular kitchen storage system")

## 4. Step-Back RAG Strategy

### Overview
Analyzes broader context and environment before making specific recommendations.

### Implementation Details
```python
def execute(self, query: str):
    # 1. Analyze broader context
    context_analysis = self._analyze_broader_context(query)
    # e.g., for "small bathroom storage":
    # - Space context: Limited floor space, moisture environment
    # - Functional needs: Daily access, multiple users
    # - Constraints: Budget, installation requirements

    # 2. Enhanced retrieval with context
    context = self.retriever.retrieve(
        f"{query} {context_analysis['space']} {context_analysis['needs']}"
    )

    # 3. Generate contextual response
    response = self.llm(prompt + context_analysis + context + query)
```

### Advantages
- Better understanding of environmental context
- More holistic recommendations
- Considers practical constraints
- Good for solution-oriented queries
- Helps prevent impractical suggestions

### Disadvantages
- Higher latency
- May overanalyze simple requests
- Can be too broad in some cases
- Requires more prompt engineering

### Best Use Cases
- Room planning ("Help me furnish my small bedroom")
- Solution-oriented queries ("Need storage solutions for growing family")
- Context-dependent requests ("Office furniture for hybrid work setup")

## Strategy Selection Guidelines

1. **Use Basic RAG when**:
   - Query is clear and specific
   - Quick response needed
   - Simple product lookup
   - Direct matching is sufficient

2. **Use Multi-Query RAG when**:
   - Query has multiple aspects
   - Need diverse options
   - Requirements are complex
   - User is exploring options

3. **Use Hypothetical RAG when**:
   - Detailed requirements given
   - Specific features needed
   - Custom solution required
   - Comparison shopping

4. **Use Step-Back RAG when**:
   - Context is crucial
   - Solution-oriented query
   - Environmental constraints
   - Long-term planning

## Performance Comparison

| Strategy      | Latency | Context Quality | Response Depth | Resource Usage |
|--------------|---------|-----------------|----------------|----------------|
| Basic        | Low     | Good           | Moderate       | Low            |
| Multi-Query  | High    | Excellent      | High           | High           |
| Hypothetical | High    | Very Good      | Very High      | High           |
| Step-Back    | Medium  | Excellent      | High           | Medium         |


## Logging and Metrics

### Logging

The system provides comprehensive logging with different levels:
- DEBUG: Detailed debugging information
- INFO: General operational information
- ERROR: Error conditions

Logs are stored in the `logs/` directory with rotation:
```
logs/
├── debug_YYYYMMDD_HHMMSS.log
├── info_YYYYMMDD_HHMMSS.log
└── error_YYYYMMDD_HHMMSS.log
```

### Metrics

The system exports detailed metrics in two formats:

1. **CSV Format** (`results/rag_metrics_TIMESTAMP.csv`):
   - Query execution details
   - Strategy performance
   - Context utilization
   - Response metrics

2. **JSON Format** (`results/rag_detailed_TIMESTAMP.json`):
   - Detailed execution information
   - Strategy-specific metrics
   - Model configurations
   - Retrieved and used contexts


# Additional Resources
- https://medium.aiplanet.com/advanced-rag-improving-retrieval-using-hypothetical-document-embeddings-hyde-1421a8ec075a
- https://huggingface.co/spaces/mteb/leaderboard