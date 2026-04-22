from huggingface_hub import login
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, TextStreamer
import os
from dotenv import load_dotenv

load_dotenv()

# Uncomment and set your Hugging Face token if needed:
# hf_token = os.getenv("HUGGINGFACE_TOKEN")
# login(token=hf_token)


def get_model_llama(model_name):
    print(f"[+] Loading model {model_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        use_auth_token=True,
    )
    return model, tokenizer


def get_model_pipeline_for_attack(model_name):
    print(f"[+] Loading model {model_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    text_pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device_map="auto",
    )
    return text_pipe


def get_model_tokenizer_for_attack(model_name):
    print(f"[+] Loading model {model_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    return (model, tokenizer)


def get_model_for_attack(model_name):
    print(f"[+] Loading model {model_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    if model_name == "lmsys/vicuna-7b-v1.5":
        return (model, tokenizer)
    if model_name in ("meta-llama/Meta-Llama-3-8B-Instruct", "mistralai/Mistral-7B-Instruct-v0.3"):
        text_pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            device_map="auto",
        )
        return text_pipe

    return None