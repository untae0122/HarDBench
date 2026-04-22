import os
from typing import Any, Dict, List, Union

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"

# Batch generation supported
SUPPORTS_BATCH = True


def model_load(
	model_name_or_path: str = DEFAULT_MODEL_NAME,
	*,
	torch_dtype: Union[str, torch.dtype] = torch.bfloat16,
	device_map: Union[str, Dict[str, int]] = "auto",
	trust_remote_code: bool = True,
) -> Dict[str, Any]:
	"""Load DeepSeek-R1-Distill-Qwen-32B model and tokenizer.

	Returns a dict with keys: {"model", "tokenizer"}.
	"""
	if model_name_or_path == "deepseek-r1-32b":
		model_name_or_path = DEFAULT_MODEL_NAME
	tokenizer = AutoTokenizer.from_pretrained(
		model_name_or_path,
		trust_remote_code=trust_remote_code,
	)
	# Decoder-only models require left padding for correct batched generation
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


def _render_chat(tokenizer: AutoTokenizer, messages: List[Dict[str, str]]) -> str:
	return tokenizer.apply_chat_template(
		messages,
		tokenize=False,
		add_generation_prompt=True,
	)


def _prepare_inputs(tokenizer: AutoTokenizer, prompt: str, device) -> Dict[str, torch.Tensor]:
	return tokenizer(prompt, return_tensors="pt").to(device)


def generate(
	state: Dict[str, Any],
	messages: Union[str, List[Dict[str, str]], List[str], List[List[Dict[str, str]]]],
	*,
	max_new_tokens: int = 2048,
	temperature: float = 0.7,
	top_p: float = 0.95,
	do_sample: bool = True,
	batch_size: int = 1,
) -> Union[str, List[str]]:
	"""Generate text from prompt or chat messages.

	messages can be:
	  - str (single prompt)
	  - list[dict] (single chat messages)
	  - list[str] (batch of prompts)
	  - list[list[dict]] (batch of chats)

	Returns str for single input, or list[str] for batch.
	"""
	model = state["model"]
	tokenizer = state["tokenizer"]
	model.eval()

	def _gen_single(single: Union[str, List[Dict[str, str]]]) -> str:
		if isinstance(single, list):
			prompt = _render_chat(tokenizer, single)
		else:
			prompt = single

		inputs = _prepare_inputs(tokenizer, prompt, model.device)
		with torch.no_grad():
			gen_kwargs = dict(
				max_new_tokens=max_new_tokens,
				do_sample=do_sample,
				pad_token_id=tokenizer.eos_token_id,
			)
			if do_sample:
				gen_kwargs.update(temperature=temperature, top_p=top_p)
			output_ids = model.generate(**inputs, **gen_kwargs)

		if model.config.is_encoder_decoder:
			decoded = tokenizer.decode(output_ids[0], skip_special_tokens=True)
		else:
			gen_ids = output_ids[0][inputs["input_ids"].shape[1] :]
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
				prompts.append(_render_chat(tokenizer, item))
			elif isinstance(item, str):
				prompts.append(item)
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
				gen_kwargs.update(temperature=temperature, top_p=top_p)
			output_ids = model.generate(**inputs, **gen_kwargs)

		for b in range(output_ids.size(0)):
			if model.config.is_encoder_decoder:
				decoded = tokenizer.decode(output_ids[b], skip_special_tokens=True)
			else:
				input_len = inputs["input_ids"][b].shape[0]
				gen_ids = output_ids[b][input_len:]
				decoded = tokenizer.decode(gen_ids, skip_special_tokens=True)
			outputs.append(decoded)

	return outputs
