from typing import Dict, Any
from pathlib import Path
import os
import json
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

API_KEY = "" 

MONGODB_URI = ""



class Settings:
    """Configuration settings for the RAG system."""
    
    def __init__(self, config_path: str = None):
        """
        Initialize settings with optional config file.
        
        Args:
            config_path: Path to JSON configuration file (optional)
        """
        # Base directories
        self.BASE_DIR = Path(__file__).parent.parent
        self.LOG_DIR = self.BASE_DIR / "logs"
        self.RESULTS_DIR = self.BASE_DIR / "results"
        self.CACHE_DIR = self.BASE_DIR / "cache"
        
        # Create necessary directories
        for directory in [self.LOG_DIR, self.RESULTS_DIR, self.CACHE_DIR]:
            directory.mkdir(parents=True, exist_ok=True)

        # Load external config if provided
        self.config_path = config_path
        if config_path and Path(config_path).exists():
            self._load_config(config_path)
        else:
            self._set_default_config()

        # Log configuration
        logger.info("Initialized settings", extra={
            "config": self.to_dict(),
            "timestamp": datetime.now().isoformat()
        })

    def _load_config(self, config_path: str) -> None:
        """
        Load configuration from JSON file.
        
        Args:
            config_path: Path to JSON configuration file
        """
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            # Update configurations
            self.LLM_CONFIG.update(config.get('llm', {}))
            self.EMBEDDING_CONFIG.update(config.get('embedding', {}))
            self.MONGODB_CONFIG.update(config.get('mongodb', {}))
            self.STRATEGY_CONFIGS.update(config.get('strategies', {}))
            
            logger.info("Loaded configuration from file", extra={
                "config_path": config_path
            })

        except Exception as e:
            logger.error("Failed to load configuration", extra={
                "error": str(e),
                "config_path": config_path
            })
            self._set_default_config()

    def _set_default_config(self) -> None:
        """Set default configuration values."""
        # LLM Configuration
        self.LLM_CONFIG = {
            "model": "meta-llama/Meta-Llama-3.1-8B-Instruct",
            "temperature": 0.2,
            "max_tokens": 2048,
            "api_key": API_KEY, #Ask MLOps team for API key
            "api_base": os.getenv("LLM_API_BASE", "https://llama31-inst-mlops-core.dev.inference.genai.mlops.ingka.com/v1"),
            "model_kwargs": {
                "stop": ["<|eot_id|>", "<|eom_id|>", "End of recommendations"],
                "top_p": 0.95,
                "presence_penalty": 0.6,
                "frequency_penalty": 0.3
            }
        }
        
        # Embedding Service Configuration
        self.EMBEDDING_CONFIG = {
            "api_key": API_KEY, #Ask MLOps team for API key
            "base_url": os.getenv("EMBEDDING_API_BASE", "https://mini-l6-v2-embedding-mlops-core.dev.inference.genai.mlops.ingka.com/v1"),
            "model": "sentence-transformers/all-MiniLM-L6-v2",
            "cache_embeddings": True,
            "batch_size": 32
        }
        
        # MongoDB Configuration
        self.MONGODB_CONFIG = {
            "uri": MONGODB_URI, #Ask MLOps team for MongoDB URI
            "db_name": "acc5-mlops-llm-vision-weeks-db",
            "collection_name": "acc5-mlops-llm-vision-weeks-collection",
            "vector_index": "vector_index",
            "text_index": "default"
        }
        
        # Strategy Configurations
        self.STRATEGY_CONFIGS = {
            "basic": {
                "k": 3,
                "use_hybrid": True,
                "text_weight": 0.3,
                "response_template": "default_response.txt"
            },
            "multi_query": {
                "k": 2,
                "use_hybrid": True,
                "text_weight": 0.4,
                "max_aspects": 3,
                "response_template": "multi_query_response.txt"
            },
            "hypothetical": {
                "k": 3,
                "use_hybrid": False,
                "doc_template": "hypothetical_doc.txt",
                "response_template": "hypothetical_response.txt"
            },
            "step_back": {
                "k": 4,
                "use_hybrid": True,
                "text_weight": 0.5,
                "context_template": "step_back_context.txt",
                "response_template": "step_back_response.txt"
            }
        }

        # Logging Configuration
        self.LOGGING_CONFIG = {
            "level": os.getenv("LOG_LEVEL", "INFO"),
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            "file_logging": True,
            "console_logging": True,
            "log_rotation": True,
            "max_bytes": 10485760, 
            "backup_count": 1
        }

        # Cache Configuration
        self.CACHE_CONFIG = {
            "enabled": True,
            "max_size": 1000,
            "ttl": 3600,  
            "persistence": True
        }

        logger.info("Set default configuration")

    def get_strategy_config(self, strategy_name: str) -> Dict[str, Any]:
        """
        Get configuration for a specific strategy.
        
        Args:
            strategy_name: Name of the strategy
            
        Returns:
            Strategy configuration dictionary
        """
        config = self.STRATEGY_CONFIGS.get(strategy_name, {})
        logger.debug(f"Retrieved config for strategy: {strategy_name}", extra={
            "config": config
        })
        return config

    def update_strategy_config(self, strategy_name: str, updates: Dict[str, Any]) -> None:
        """
        Update configuration for a specific strategy.
        
        Args:
            strategy_name: Name of the strategy
            updates: Dictionary of configuration updates
        """
        if strategy_name in self.STRATEGY_CONFIGS:
            self.STRATEGY_CONFIGS[strategy_name].update(updates)
            logger.info(f"Updated config for strategy: {strategy_name}", extra={
                "updates": updates
            })
        else:
            logger.warning(f"Strategy not found: {strategy_name}")

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert settings to dictionary.
        
        Returns:
            Dictionary of all configuration settings
        """
        return {
            "llm": self.LLM_CONFIG,
            "embedding": self.EMBEDDING_CONFIG,
            "mongodb": self.MONGODB_CONFIG,
            "strategies": self.STRATEGY_CONFIGS,
            "logging": self.LOGGING_CONFIG,
            "cache": self.CACHE_CONFIG,
            "directories": {
                "base": str(self.BASE_DIR),
                "logs": str(self.LOG_DIR),
                "results": str(self.RESULTS_DIR),
                "cache": str(self.CACHE_DIR)
            }
        }

    def save_config(self, path: str = None) -> None:
        """
        Save current configuration to file.
        
        Args:
            path: Path to save configuration file (optional)
        """
        save_path = path or self.config_path or "config.json"
        try:
            with open(save_path, 'w') as f:
                json.dump(self.to_dict(), f, indent=2)
            
            logger.info("Saved configuration", extra={
                "path": save_path
            })

        except Exception as e:
            logger.error("Failed to save configuration", extra={
                "error": str(e),
                "path": save_path
            })

    @staticmethod
    def validate_config(config: Dict[str, Any]) -> bool:
        """
        Validate configuration dictionary.
        
        Args:
            config: Configuration dictionary to validate
            
        Returns:
            Boolean indicating validation success
        """
        required_sections = ["llm", "embedding", "mongodb", "strategies"]
        required_strategies = ["basic", "multi_query", "hypothetical", "step_back"]
        
        try:
            # Check required sections
            for section in required_sections:
                if section not in config:
                    logger.error(f"Missing required section: {section}")
                    return False

            # Check strategy configurations
            strategies = config.get("strategies", {})
            for strategy in required_strategies:
                if strategy not in strategies:
                    logger.error(f"Missing required strategy: {strategy}")
                    return False

            return True

        except Exception as e:
            logger.error("Configuration validation failed", extra={
                "error": str(e)
            })
            return False