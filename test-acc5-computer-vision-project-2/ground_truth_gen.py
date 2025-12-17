import torch
from PIL import Image
from models.mllava import MLlavaProcessor, LlavaForConditionalGeneration, chat_mllava
import pandas as pd
from tqdm import tqdm
import os
from transformers import BlipProcessor, BlipForConditionalGeneration
from transformers import LlavaProcessor, AutoTokenizer


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

def get_image_paths(directory):
    """
    Get all image file paths from the specified directory.
    
    Args:
        directory (str): Path to the directory containing images.

    Returns:
        list: List of image file paths.
    """
    return [os.path.join(directory, f) for f in os.listdir(directory) if f.lower().endswith(('png', 'jpg', 'jpeg'))]

def run_inference(image_path, model, processor, generation_kwargs):
    """
    Perform multi-step reasoning on a single image.

    Args:
        image_path (str): Path to the image file.
        model: Loaded model.
        processor: Loaded processor.
        generation_kwargs (dict): Generation parameters.

    Returns:
        str: Final generated description.
    """
    image = Image.open(image_path).convert("RGB")
    images = [image]

    steps = [
        "Describe the image in great detail. Do not make anything up and do not assume anything. Only generate useful descriptive information.",
        "Based on the previous description: \n\n'{}'.\n\n Now, modify it by listing all distinct objects in the image, specifying colors and positions in the image.",
        "Based on the identified objects: \n\n'{}'.\n\n Now, modify it by describing the semantic relationships between objects (e.g., one object is on top of another, next to, behind, etc.).",
        "Using the previous information: \n\n'{}'.\n\n Now, modify it by specifying the distances between objects in meters or feet."
    ]
    
    response = ""
    for step in steps:
        step_prompt = step.format(response)
        response, _ = chat_mllava(step_prompt, images, model, processor, **generation_kwargs)
    
    return response

def process_images(image_dir, model, processor, generation_kwargs, output_csv, checkpoint_interval=50):
    """
    Process images from a directory and generate descriptions.

    Args:
        image_dir (str): Path to the directory containing images.
        model: Loaded model.
        processor: Loaded processor.
        generation_kwargs (dict): Generation parameters.

    Returns:
        list: List of dictionaries containing image paths and descriptions.
    """
    image_paths = get_image_paths(image_dir)
    results = []
    
    # Load existing results if the file exists (for resuming)
    if os.path.exists(output_csv):
        df_existing = pd.read_csv(output_csv)
        processed_images = set(df_existing["path"].tolist())
        results = df_existing.to_dict(orient="records")
    else:
        processed_images = set()

    for index, image_path in enumerate(tqdm(image_paths, desc="Processing Images", unit="image")):
        if image_path in processed_images:
            continue
        
        description = run_inference(image_path, model, processor, generation_kwargs)
        results.append({'path': image_path, 'description': description})

        if index % checkpoint_interval == 0 or index == len(image_paths):
            save_results(results, output_csv)
    
    return results

def save_results(results, output_csv):
    """
    Save the generated descriptions to a CSV file.

    Args:
        results (list): List of dictionaries containing image paths and descriptions.
        output_csv (str): Path to save the CSV file.
    """
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"Checkpoint saved to {output_csv}")

# Example usage
if __name__ == "__main__":
    model_name = "remyxai/SpaceMantis"
    processor, model = load_model(model_name, MLlavaProcessor, LlavaForConditionalGeneration)

    generation_kwargs = {
        "max_new_tokens": 1024,
        "num_beams": 1,
        "do_sample": False
    }

    image_dir = "img/test" 
    output_csv = "data/ground_truth_SpaceMantis-8B_4.csv"

    results = process_images(image_dir, model, processor, generation_kwargs)
    save_results(results, output_csv)

# # Model: Salesforce/blip-image-captioning-base
# if __name__ == "__main__":
#     model_name = "Salesforce/blip-image-captioning-base"
#     processor, model = load_model(model_name, BlipProcessor, BlipForConditionalGeneration)  # Adjust if BLIP needs different processor/model classes

#     generation_kwargs = {
#         "max_new_tokens": 512,
#         "num_beams": 5,
#         "do_sample": False
#     }

#     image_dir = "img/test_blip"
#     output_csv = "data/ground_truth_blip-base.csv"

#     results = process_images(image_dir, model, processor, generation_kwargs, output_csv)
#     save_results(results, output_csv)

# # Model: llava-hf/llava-1.5-7b-hf
# if __name__ == "__main__":
#     model_name = "llava-hf/llava-1.5-7b-hf"
#     processor, model = load_model(model_name, LlavaProcessor, LlavaForConditionalGeneration)

#     generation_kwargs = {
#         "max_new_tokens": 1024,
#         "num_beams": 1,
#         "do_sample": False
#     }

#     image_dir = "img/test_llava"
#     output_csv = "data/ground_truth_llava-1.5-7b-hf.csv"

#     results = process_images(image_dir, model, processor, generation_kwargs, output_csv)
#     save_results(results, output_csv)