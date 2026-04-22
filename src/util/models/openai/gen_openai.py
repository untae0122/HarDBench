import os
from typing import Any, Dict, List, Union

from dotenv import load_dotenv
from openai import OpenAI
from openai import BadRequestError

load_dotenv()

SUPPORTS_BATCH = False


def model_load(model_name_or_path: str, **kwargs) -> Dict[str, Any]:
	client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
	return {"client": client, "model": model_name_or_path}


def _to_chat(msg: Union[str, List[Dict[str, str]]]) -> List[Dict[str, str]]:
	if isinstance(msg, str):
		return [{"role": "user", "content": msg}]
	return msg


def generate(
	state: Dict[str, Any],
	messages: Union[str, List[Dict[str, str]], List[str], List[List[Dict[str, str]]]],
	*,
	max_new_tokens: int = 1024,
	temperature: float = 0.7,
	do_sample: bool = True,
	batch_size: int = 1,
) -> Union[str, List[str]]:
	client: OpenAI = state["client"]
	model: str = state["model"]

	temp = 0.0 if not do_sample else temperature

	def _gen_single(msg) -> str:
		"""Generate a single completion with graceful fallback across new API parameter names.

		Order tried:
		1) chat.completions (legacy) with max_tokens
		2) responses API with max_completion_tokens (as hinted by error)
		3) responses API with max_output_tokens (older Responses spec)
		4) chat.completions without explicit max tokens (let server decide)
		"""
		chat = _to_chat(msg)
		model_lower = model.lower()
		is_gpt5 = ("gpt5" in model_lower) or ("gpt-5" in model_lower)
		# Fast path: gpt5 series uses responses API with max_completion_tokens directly
		if is_gpt5:
			try:
				resp = client.chat.completions.create(
					model=model,
					messages=chat,
					reasoning_effort="minimal",
					#temperature=temp,
					#max_completion_tokens=max_new_tokens,
				)
				# print("Using chat.completions with max_completion_tokens")
				# print(resp)
				return resp.choices[0].message.content
			except Exception as e:
				# Re-raise original error for visibility
				raise e
		else:
			try:
				resp = client.chat.completions.create(
					model=model,
					messages=chat,
					temperature=temp,
					max_tokens=max_new_tokens,
				)
				return resp.choices[0].message.content
			except Exception as e:
				# Re-raise original error for visibility
				raise e

	if isinstance(messages, (str, list)) and (isinstance(messages, str) or (messages and isinstance(messages[0], dict))):
		return _gen_single(messages)  # single

	if not isinstance(messages, list):
		raise ValueError("messages must be str, list[dict], list[str], or list[list[dict]]")

	outputs: List[str] = []
	for i in range(0, len(messages), batch_size):
		chunk = messages[i : i + batch_size]
		for item in chunk:
			outputs.append(_gen_single(item))
	return outputs

