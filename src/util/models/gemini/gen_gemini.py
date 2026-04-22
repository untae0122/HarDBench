import os
from typing import Any, Dict, List, Union

from dotenv import load_dotenv
from google import genai

load_dotenv()

SUPPORTS_BATCH = False


def model_load(model_name_or_path: str, **kwargs) -> Dict[str, Any]:
	client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
	return {"client": client, "model": model_name_or_path}


def _to_text(msg: Union[str, List[Dict[str, str]]]) -> str:
	if isinstance(msg, str):
		return msg
	# single chat: join user contents (simple)
	return "\n".join(m.get("content", "") for m in msg)


def generate(
	state: Dict[str, Any],
	messages: Union[str, List[Dict[str, str]], List[str], List[List[Dict[str, str]]]],
	*,
	max_new_tokens: int = 1024,
	temperature: float = 0.7,
	do_sample: bool = True,
	batch_size: int = 1,
) -> Union[str, List[str]]:
	client: genai.Client = state["client"]
	model: str = state["model"]

	def _gen_single(msg) -> str:
		text = _to_text(msg)
		resp = client.models.generate_content(model=model, contents=text)
		return getattr(resp, "text", "")

	if isinstance(messages, (str, list)) and (isinstance(messages, str) or (messages and isinstance(messages[0], dict))):
		return _gen_single(messages)

	if not isinstance(messages, list):
		raise ValueError("messages must be str, list[dict], list[str], or list[list[dict]]")

	outputs: List[str] = []
	for i in range(0, len(messages), batch_size):
		chunk = messages[i : i + batch_size]
		for item in chunk:
			outputs.append(_gen_single(item))
	return outputs

