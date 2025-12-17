import numpy as np
from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor 
from janus.models import MultiModalityCausalLM, VLChatProcessor
import torch
from PIL import Image

CACHE_DIR = "cache"

def load_model_janus(model_path: str = "deepseek-ai/Janus-1.3B"):
    """
    Loads the Janus model and processor.
    """
    config = AutoConfig.from_pretrained(model_path)
    # For improved performance or if you get CUDA-related errors:
    # config.language_config._attn_implementation = 'eager'
    language_config = config.language_config
    language_config._attn_implementation = 'eager'

    # Load the base model
    vl_gpt = AutoModelForCausalLM.from_pretrained(
        model_path,
        cache_dir=CACHE_DIR,
        language_config=language_config,
        trust_remote_code=True
    )

    # Move model to GPU if available, otherwise CPU
    if torch.cuda.is_available():
        vl_gpt = vl_gpt.to(torch.bfloat16).cuda()
    else:
        vl_gpt = vl_gpt.to(torch.float16)

    # Load the processor
    vl_chat_processor = VLChatProcessor.from_pretrained(model_path)

    return vl_gpt, vl_chat_processor

def load_model_janus(model_path: str = "microsoft/Florence-2-large"):
    """
    Loads the Janus model and processor.
    """
    config = AutoConfig.from_pretrained(model_path)
    # For improved performance or if you get CUDA-related errors:
    # config.language_config._attn_implementation = 'eager'
    language_config = config.language_config
    language_config._attn_implementation = 'eager'

    # Load the base model
    vl_gpt = AutoModelForCausalLM.from_pretrained(
        model_path,
        cache_dir=CACHE_DIR,
        language_config=language_config,
        trust_remote_code=True
    )

    # Move model to GPU if available, otherwise CPU
    if torch.cuda.is_available():
        vl_gpt = vl_gpt.to(torch.bfloat16).cuda()
    else:
        vl_gpt = vl_gpt.to(torch.float16)

    # Load the processor
    vl_chat_processor = VLChatProcessor.from_pretrained(model_path)

    return vl_gpt, vl_chat_processor

@torch.inference_mode()
def ask_janus(
    vl_gpt: MultiModalityCausalLM,
    vl_chat_processor: VLChatProcessor,
    image_path: str,
    question: str,
    seed: int = 42,
    top_p: float = 0.95,
    temperature: float = 0.1
) -> str:
    """
    Given an image path and a question, returns the answer predicted by the Janus model.
    """
    # Set random seeds for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    # Read and prepare the image
    pil_image = Image.open(image_path).convert("RGB")

    # Build conversation data
    conversation = [
        {
            "role": "User",
            "content": f"<image_placeholder>\n{question}",
            "images": [pil_image],
        },
        {"role": "Assistant", "content": ""},
    ]

    # Move data to correct device / dtype
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float16

    # Process input (tokenization, image features, etc.)
    prepare_inputs = vl_chat_processor(
        conversations=conversation, images=[pil_image], force_batchify=True
    ).to(device, dtype=dtype)

    # Convert tokens + embeddings for the model
    inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)

    # Generate the answer
    tokenizer = vl_chat_processor.tokenizer
    outputs = vl_gpt.language_model.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=prepare_inputs.attention_mask,
        pad_token_id=tokenizer.eos_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        max_new_tokens=10124,
        do_sample=(temperature != 0.0),
        use_cache=True,
        temperature=temperature,
        top_p=top_p,
    )

    # Decode model output into text
    answer = tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
    return answer