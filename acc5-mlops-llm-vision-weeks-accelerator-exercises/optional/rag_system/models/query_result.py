from dataclasses import dataclass, field
from typing import List, Dict, Optional
from datetime import datetime
import json

@dataclass
class QueryResult:
    """Enhanced query result with complete context information."""
    strategy: str
    query: str
    context: List[Dict]  # Full context from vector store
    used_context: List[Dict]
    generated_response: str
    prompt_used: str
    latency: float
    timestamp: datetime
    metadata: Dict
    error: Optional[str] = None
    token_count: int = field(init=False)
    response_length: int = field(init=False)
    context_size: int = field(init=False)
    raw_vector_results: List[Dict] = field(default_factory=list)  # Store raw vector results

    def __post_init__(self):
        """Calculate additional metrics after initialization."""
        self.response_length = len(self.generated_response)
        self.context_size = len(self.context)
        self.token_count = len(self.generated_response.split())

    def to_dict(self) -> Dict:
        """Convert to dictionary with detailed metrics."""
        return {
            "strategy": self.strategy,
            "query": self.query,
            "response": self.generated_response,
            "metrics": {
                "token_count": self.token_count,
                "response_length": self.response_length,
                "context_size": self.context_size,
                "used_context_size": len(self.used_context),
                "latency": self.latency
            },
            "context": self._format_context(self.context),
            "used_context": self._format_context(self.used_context),
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
            "error": self.error
        }

    def _format_context(self, context: List[Dict]) -> List[Dict]:
        """Format context information."""
        return [{
            "item_no": product.get("item_no"),
            "product_name": product.get("product_name"),
            "product_type": product.get("product_type"),
            "price": product.get("price"),
            "benefits_summary": product.get("benefits_summary"),
            "score": product.get("score", "N/A")
        } for product in context]