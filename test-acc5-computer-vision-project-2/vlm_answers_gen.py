import torch
from tqdm import tqdm
import os
from PIL import Image
from models.mllava import MLlavaProcessor, LlavaForConditionalGeneration, chat_mllava
from models.janus.janus_utils import ask_janus
import pandas as pd

from transformers import BlipProcessor, BlipForConditionalGeneration
from transformers import LlavaProcessor, AutoTokenizer
from transformers import AutoConfig, AutoModelForCausalLM
from janus.models import MultiModalityCausalLM, VLChatProcessor

CACHE_DIR = "cache"

def load_model(model_name, processor_type, model_type, attn_implementation=None):
    """
    Load the model and processor.
    
    Args:
        model_name (str): Name of the model from Hugging Face.
        processor_type (class): Processor class.
        model_type (class): Model class.
        attn_implementation (str, optional): Attention implementation type.

    Returns:
        processor, model: Loaded processor and model.
    """
    processor = processor_type.from_pretrained(model_name, cache_dir=CACHE_DIR)
    model = model_type.from_pretrained(
        model_name, 
        cache_dir=CACHE_DIR, 
        device_map="cuda", 
        torch_dtype=torch.float16, 
        attn_implementation=attn_implementation
    )
    return processor, model

def run_inference(image_path, question, question_type, model, processor, generation_kwargs):
    """
    Using the provided image and question, generate an answer.

    Args:
        image_path (str): Path to the image file.
        question (str): The question to be answered.
        model: Loaded model.
        processor: Loaded processor.
        generation_kwargs (dict): Generation parameters.

    Returns:
        str: Generated answer.
    """
    image = Image.open(image_path).convert("RGB")
    images = [image]

    prompt = f"""
    You will be provided an image and a question about the image.
    You will also be told if the answer expected is a qualitative or quantitative answer.
    Please provide an answer to the question. 

    Example 1:
    Question: "How far is the yellow finger from the silver can?"
    Expected Answer Type: Quantitative
    Output: "The yellow finger is one and a half meters away from the silver can"

    Example 2:
    Question: "What is next to the blue sofa?"
    Expected Answer Type: Qualitative
    Output: "The yellow chair is next to the blue sofa"

    Your task: Provide an answer to the question. No greetings or comments. Just provide the answer.
    Question: {question}
    Question Type: {question_type}
    """
    if model_name.lower().startswith("deepseek-ai/janus"):
        response = ask_janus(model, processor, image_path, question)
    else:
        response, _ = chat_mllava(prompt, images, model, processor, **generation_kwargs)
    return response

def process_questions(questions_csv, model, processor, generation_kwargs, output_csv, checkpoint_interval=50):
    """
    Process questions from a CSV file and generate answers.

    Args:
        questions_csv (str): Path to the CSV file containing questions.
        model: Loaded model.
        processor: Loaded processor.
        generation_kwargs (dict): Generation parameters.
        output_csv (str): Path to save the CSV file with answers.
        checkpoint_interval (int): Interval for saving checkpoints.

    Returns:
        list: List of dictionaries containing image paths, questions, and answers.
    """
    df = pd.read_csv(questions_csv)
    results = []
    
    # Load existing results if the file exists (for resuming)
    for index, row in enumerate(tqdm(df.itertuples(), desc="Processing Questions", unit="question")):        
        answer = run_inference(row.path, row.question, row.type_question, model, processor, generation_kwargs)
        results.append({'path': row.path, 'question': row.question, 'true_answer': row.true_answer, 'type_question': row.type_question,'answer': answer})

        if index % checkpoint_interval == 0 or index == len(df) - 1:
            save_results(results, output_csv)
    
    return results

def save_results(results, output_csv):
    """
    Save the generated answers to a CSV file.

    Args:
        results (list): List of dictionaries containing image paths, questions, and answers.
        output_csv (str): Path to save the CSV file.
    """
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"Checkpoint saved to {output_csv}")

# Example usage
if __name__ == "__main__":
    model_name = "deepseek-ai/Janus-Pro-7B"
    processor, model = load_model(model_name, MLlavaProcessor, LlavaForConditionalGeneration)

    generation_kwargs = {
        "max_new_tokens": 1024,
        "num_beams": 1,
        "do_sample": False
    }

    questions_csv = "output_files/questions.csv"
    output_csv = f"output_files/model_answers_{model_name}.csv"

    results = process_questions(questions_csv, model, processor, generation_kwargs, output_csv)
    save_results(results, output_csv)

