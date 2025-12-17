import asyncio
import os
import sys
import vertexai
import pandas as pd
from vertexai.generative_models import (
    GenerationConfig,
    GenerativeModel,
    HarmBlockThreshold,
    HarmCategory,
)
import time
import json  # Import the json module for saving results

PROJECT_ID = "ingka-b2bda-iifb-dev"  # Replace with your PROJECT_ID if different
LOCATION = "europe-west1"  # Replace with your LOCATION if different
MODEL_ID = "gemini-1.5-flash-001"


class LLModel:
    def __init__(self, model_name=MODEL_ID, system_instructions="", chat=False):
        
        vertexai.init(project=PROJECT_ID, location=LOCATION)
        generation_config, safety_settings = configure_generation_and_safety()
        self.model = GenerativeModel(model_name, system_instruction=system_instructions)
        if chat:
            self.chat = self.model.start_chat(history=[])
        
        # Apply generation config and safety settings when the model is initialized
        self.generation_config = generation_config
        self.safety_settings = safety_settings

    def generate_response(self, prompt):
        try:
            response = self.model.generate_content(
                prompt, generation_config=self.generation_config, safety_settings=self.safety_settings
            )
            return response.text
        except Exception as e:
            print(f"Error generating response: {e}")
            return None
    

def configure_generation_and_safety():
    """Initialize generation config and safety settings."""
    # 1. Configure Generation Parameters
    generation_config = GenerationConfig(
        temperature=0.7,
        top_p=0.95,
        top_k=40,
        max_output_tokens=256,
    )

    # 2. Configure Safety Settings (Adjust as needed!)
    safety_settings = {
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    }

    return generation_config, safety_settings

    