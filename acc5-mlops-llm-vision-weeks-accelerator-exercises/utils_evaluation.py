import json
import logging
import os
from typing import Dict, Any
import time
from json import JSONDecodeError

from deepeval.models import DeepEvalBaseLLM
from openai import OpenAI
import re
import random
import numpy as np

# Basic logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


class OpenAIEvaluationModel(DeepEvalBaseLLM):
    def __init__(self, llm_config: Dict[str, Any]):
        """Initialize OpenAI-like model wrapper"""
        self.client = OpenAI(
            api_key=llm_config.get("api_key"),
            base_url=llm_config.get("api_base")
        )
        
        self.model = llm_config.get("model", "meta-llama/Meta-Llama-3.1-8B-Instruct")
        self.temperature = llm_config.get("temperature", 0.2)
        self.max_tokens = llm_config.get("max_tokens", 2048)
        
        model_kwargs = llm_config.get("model_kwargs", {})
        self.top_p = model_kwargs.get("top_p", 0.95)
        # self.presence_penalty = model_kwargs.get("presence_penalty", 0.6)
        # self.frequency_penalty = model_kwargs.get("frequency_penalty", 0.3)
        self.stop = model_kwargs.get("stop", ["<|eot_id|>"])
        
        # Retry configuration
        self.max_retries = 15
        self.fixed_delay = 1 

    def _is_valid_json(self, text: str) -> bool:
        """Validate if the text is proper JSON"""
        try:
            json.loads(text)
            return True
        except JSONDecodeError:
            return False

    def _format_system_prompt(self) -> str:
        """Format the system prompt to strongly encourage JSON output"""
        return """You are an AI evaluation system analyzing RAG responses. 
        You MUST generate feedback ONLY in valid JSON format.
        
        Guidelines:
        1. Always wrap the entire response in curly braces {}
        2. Use double quotes for all keys and string values
        3. Include only valid JSON data types (string, number, boolean, null, object, array)
        4. Ensure all objects and arrays are properly closed
        
        Example format:
        "metric_name": {
          "score": 0.7777777777777778,
          "reason": "The score is 0.78 because the actual output contradicts the retrieval context by stating that the product washstand has a similar or smaller footprint than expected."
        }
        """

    def generate(self, prompt: str) -> str:
        """Generate response with retry logic and JSON validation."""
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": self._format_system_prompt()},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    top_p=self.top_p,
                    # presence_penalty=self.presence_penalty,
                    # frequency_penalty=self.frequency_penalty,
                    stop=self.stop
                )

                content = response.choices[0].message.content

                # 1. Check if the raw content is already valid JSON
                if self._is_valid_json(content):
                    return content

                # 2. Try to fix common JSON issues
                fixed_content = self._attempt_json_fix(content)
                if fixed_content and self._is_valid_json(fixed_content):
                    logging.info("Successfully fixed malformed JSON response")
                    return fixed_content

                # 3. If still not valid JSON, either retry or fail on the last attempt
                if attempt < self.max_retries:
                    logging.warning(
                        f"Attempt {attempt}: Invalid JSON response. "
                        f"Retrying in {self.fixed_delay} second(s)..."
                    )
                    time.sleep(self.fixed_delay)
                else:
                    raise ValueError("Failed to generate valid JSON after all attempts")

            except Exception as e:
                # Catch any other exceptions (like network errors) and retry up to max_retries
                if attempt < self.max_retries:
                    logging.warning(
                        f"Attempt {attempt} failed: {str(e)}. "
                        f"Retrying in {self.fixed_delay} second(s)..."
                    )
                    time.sleep(self.fixed_delay)
                else:
                    logging.error(f"All attempts failed: {str(e)}")
                    raise


    def _attempt_json_fix(self, content: str) -> str:
        """Attempt to fix common JSON formatting issues."""
        try:
            # 1. Extract the JSON-like portion between the outermost curly braces
            if '{' in content and '}' in content:
                content = content[content.find('{'):content.rfind('}')+1]

            # 2. Replace single quotes with double quotes
            #    so that strings like 'text' become "text"
            content = content.replace("'", '"')

            # 3. Collapse repeated double quotes into a single double quote
            #    This fixes issues like: ""The context..."" => "The context..."
            content = re.sub(r'(?<!\\)"{2,}', '"', content)

            # 4. Remove trailing commas before closing braces/brackets
            #    This handles cases like: {"key": "value",}
            content = re.sub(r',(\s*[}\]])', r'\1', content)

            # 5. Finally, check if this "fixed" content parses as valid JSON
            if self._is_valid_json(content):
                return content
            return None

        except Exception as e:
            logging.error(f"Error attempting to fix JSON: {str(e)}")
            return None


    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    def get_model_name(self):
        return self.model
        
    def load_model(self) -> Any:
        """
        Load the evaluation model and return the client instance
        
        Returns:
            OpenAI: Configured OpenAI client instance
        """
        try:
            # Log model configuration
            logging.info(f"Loading model {self.model} with base URL: {self.client.base_url}")
            
            # Return the configured client
            return self.client
            
        except Exception as e:
            logging.error(f"Failed to load model: {str(e)}")
            raise
