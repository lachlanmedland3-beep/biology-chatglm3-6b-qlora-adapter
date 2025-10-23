# Model Card: QLoRA Fine-tuned ChatGLM3-6B Adapter for Primary School Biology QA

# Model Description
This repository contains a QLoRA adapter for the ChatGLM3-6B large language model, fine-tuned on a custom dataset of primary school biology knowledge questions and answers. The aim of this adapter is to improve the model's performance on domain-specific knowledge retrieval for junior high school biology.

# Base Model
The base model used for fine-tuning is [THUDM/chatglm3-6b](https://huggingface.co/THUDM/chatglm3-6b). Users must download this base model from Hugging Face to use this adapter.

# Fine-tuning Details
- **Methodology:** QLoRA (Quantized Low-Rank Adaptation)
- **Quantization:** 4-bit (NF4)
- **LoRA Target Modules:** `query_key_value`
- **LoRA Rank:** 8
- **LoRA Alpha:** 16
- **Training Framework:** LLaMA-Factory
- **Dataset:** A custom dataset of over 2000 primary school biology question-answer pairs (manually collected), split into 80% training, 10% validation, and 10% testing sets.
- **Training Epochs:** 30
- **Learning Rate:** 1e-4
- **Batch Size:** 4 (effective batch size: 32 due to gradient accumulation steps of 8)
- **Hardware:** NVIDIA GeForce RTX 4090 (single GPU was used for training on GPU 1)
- **Training Time:** Approximately 53 minutes.
- **Best Validation Loss:** 1.3791 (achieved at Epoch 6)

# How to Use This Adapter
To use this adapter, you need to load the original `THUDM/chatglm3-6b` model in 4-bit quantization, and then attach this LoRA adapter to it.

**Prerequisites:**
Install the necessary Python libraries:
```bash
pip install torch transformers peft bitsandbytes accelerate


from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel, PeftConfig
import torch

# 1. Define paths and quantization config
base_model_id = "THUDM/chatglm3-6b"
# Assuming this adapter is cloned/downloaded into a directory, e.g., "biology-chatglm3-6b-qlora-adapter"
# And the adapter files (adapter_config.json, adapter_model.safetensors) are directly within it.
adapter_path = "./biology-chatglm3-6b-qlora-adapter/" # Or adjust to your local path

# QLoRA configuration (must match training setup)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4", # Use "nf4" or "fp4" as per your training
    bnb_4bit_compute_dtype=torch.float16, # Or torch.bfloat16 if used (RTX 4090 supports bfloat16)
    bnb_4bit_use_double_quant=True,
)

# 2. Load the base ChatGLM3-6B model in 4-bit quantization
print("Loading base model...")
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    quantization_config=bnb_config,
    device_map="auto", # or specify a single GPU like {"": 0}
    trust_remote_code=True,
    torch_dtype=torch.float16 # Must match compute_dtype in bnb_config
)
tokenizer = AutoTokenizer.from_pretrained(
    base_model_id,
    trust_remote_code=True
)
print("Base model and tokenizer loaded.")

# 3. Load the LoRA adapter
print(f"Loading LoRA adapter from {adapter_path}...")
model = PeftModel.from_pretrained(base_model, adapter_path)
model = model.eval() # Set model to evaluation mode
print("LoRA adapter loaded and merged with base model.")

# 4. Perform inference
def generate_response(instruction, input_text=""):
    # Use the ChatGLM3 template
    prompt = f"<|user|>\n{instruction} {input_text}\n<|assistant|>\n"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Generation configuration
    generation_config = model.generation_config
    generation_config.max_new_tokens = 512 # Adjust as needed
    generation_config.do_sample = True
    generation_config.top_p = 0.7
    generation_config.temperature = 0.95
    generation_config.pad_token_id = tokenizer.eos_token_id # ChatGLM3 specific
    generation_config.num_beams = 1 # Simple greedy decoding for generation
    
    outputs = model.generate(
        **inputs,
        generation_config=generation_config
    )
    response = tokenizer.decode(outputs[0][len(inputs["input_ids"][0]):], skip_special_tokens=True)
    return response.strip()

# Example usage:
question1 = "光合作用的产物是什么？"
answer1 = generate_response(question1)
print(f"\n问题: {question1}")
print(f"回答: {answer1}")

question2 = "缺乏哪种维生素会导致夜盲症？"
answer2 = generate_response(question2)
print(f"\n问题: {question2}")
print(f"回答: {answer2}")
