from typing import Any, Dict, List, Union

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_MODEL_NAME = "Qwen/Qwen3-8B"
SUPPORTS_BATCH = True


def model_load(
    model_name_or_path: str = DEFAULT_MODEL_NAME,
    *,
    torch_dtype: Union[str, torch.dtype] = torch.bfloat16,
    device_map: Union[str, Dict[str, int]] = "auto",
    trust_remote_code: bool = True,
) -> Dict[str, Any]:
    # alias mapping
    if (model_name_or_path or "").strip().lower() in {"qwen3-8b", "qwen/qwen3-8b"}:
        model_name_or_path = DEFAULT_MODEL_NAME
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=trust_remote_code,
    )
    # left padding for decoder-only batch gen
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        dtype=(torch_dtype if isinstance(torch_dtype, torch.dtype) else torch.bfloat16),
        device_map=device_map,
        trust_remote_code=trust_remote_code,
    )
    return {"model": model, "tokenizer": tokenizer}


def _render_chat(tokenizer: AutoTokenizer, messages: List[Dict[str, str]], *, enable_thinking: bool = False) -> str:
    # Enable or disable thinking mode via chat template flag
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )


def _prepare_inputs(tokenizer: AutoTokenizer, prompt: str, device) -> Dict[str, torch.Tensor]:
    return tokenizer(prompt, return_tensors="pt").to(device)


def generate(
    state: Dict[str, Any],
    messages: Union[str, List[Dict[str, str]], List[str], List[List[Dict[str, str]]]],
    *,
    max_new_tokens: int = 1024,
    temperature: float = 0.7,
    top_p: float = 0.95,
    top_k: int | None = None,
    do_sample: bool = True,
    batch_size: int = 1,
    enable_thinking: bool = False,
) -> Union[str, List[str]]:
    model = state["model"]
    tokenizer = state["tokenizer"]
    model.eval()

    def _parse_thinking_from_ids(gen_token_ids: List[int]) -> tuple[str, str]:
        # Split by the special token id for </think> (151668). If not found, thinking="", content=all
        try:
            idx = len(gen_token_ids) - gen_token_ids[::-1].index(151668)
        except ValueError:
            idx = 0
        thinking_text = tokenizer.decode(gen_token_ids[:idx], skip_special_tokens=True).strip("\n")
        content_text = tokenizer.decode(gen_token_ids[idx:], skip_special_tokens=True).strip("\n")
        return thinking_text, content_text

    def _gen_single(single: Union[str, List[Dict[str, str]]]) -> str:
        if isinstance(single, list):
            prompt = _render_chat(tokenizer, single, enable_thinking=enable_thinking)
        else:
            # wrap raw text as a single user message
            prompt = _render_chat(tokenizer, [{"role": "user", "content": single}], enable_thinking=enable_thinking)

        inputs = _prepare_inputs(tokenizer, prompt, model.device)
        with torch.no_grad():
            gen_kwargs = dict(
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                pad_token_id=tokenizer.eos_token_id,
            )
            if do_sample:
                if top_k is not None:
                    gen_kwargs.update(temperature=temperature, top_p=top_p, top_k=top_k)
                else:
                    gen_kwargs.update(temperature=temperature, top_p=top_p)
            output_ids = model.generate(**inputs, **gen_kwargs)

        if model.config.is_encoder_decoder:
            decoded = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        else:
            gen_ids = output_ids[0][inputs["input_ids"].shape[1] :]
            if enable_thinking:
                _thinking, content = _parse_thinking_from_ids(gen_ids.tolist())
                decoded = content
            else:
                decoded = tokenizer.decode(gen_ids, skip_special_tokens=True)
        return decoded

    # Single input
    if isinstance(messages, str) or (isinstance(messages, list) and messages and isinstance(messages[0], dict)):
        return _gen_single(messages)  # type: ignore[arg-type]

    # Batch input (list[str] or list[list[dict]])
    if not isinstance(messages, list):
        raise ValueError("messages must be str, list[dict], list[str], or list[list[dict]]")

    outputs: List[str] = []
    for i in range(0, len(messages), batch_size):
        chunk = messages[i : i + batch_size]
        prompts: List[str] = []
        for item in chunk:
            if isinstance(item, list):
                prompts.append(_render_chat(tokenizer, item, enable_thinking=enable_thinking))
            elif isinstance(item, str):
                prompts.append(_render_chat(tokenizer, [{"role": "user", "content": item}], enable_thinking=enable_thinking))
            else:
                raise ValueError("Unsupported message type in batch")

        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            gen_kwargs = dict(
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                pad_token_id=tokenizer.eos_token_id,
            )
            if do_sample:
                if top_k is not None:
                    gen_kwargs.update(temperature=temperature, top_p=top_p, top_k=top_k)
                else:
                    gen_kwargs.update(temperature=temperature, top_p=top_p)
            output_ids = model.generate(**inputs, **gen_kwargs)

        for b in range(output_ids.size(0)):
            if model.config.is_encoder_decoder:
                decoded = tokenizer.decode(output_ids[b], skip_special_tokens=True)
            else:
                input_len = inputs["input_ids"][b].shape[0]
                gen_ids = output_ids[b][input_len:]
                if enable_thinking:
                    _thinking, content = _parse_thinking_from_ids(gen_ids.tolist())
                    decoded = content
                else:
                    decoded = tokenizer.decode(gen_ids, skip_special_tokens=True)
            outputs.append(decoded)

    return outputs
