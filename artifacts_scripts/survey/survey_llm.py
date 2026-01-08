"""
python -m artifacts_scripts.survey.survey_llm artifacts_scripts/survey/survey.qsf --model hf:meta-llama/Llama-3.3-70B-Instruct --hf-4bit --prompt-mode triangular --run
python -m artifacts_scripts.survey.survey_llm artifacts_scripts/survey/survey.qsf --model openai:gpt-5.2 --prompt-mode triangular --run
python -m artifacts_scripts.survey.survey_llm artifacts_scripts/survey/survey.qsf --model openai:gpt-4o-mini --prompt-mode triangular --run
"""


import json
import random
import re
import argparse
import os
import time
import logging
from typing import List, Dict, Any, Set, Tuple, Optional

OpenAI = None  # Deferred import when provider is openai
LOGGER = logging.getLogger(__name__)

# ==========================================
# CONFIGURATION & PROMPTS
# ==========================================

SYSTEM_PROMPT = """You are an expert academic reviewer evaluating textbook proposals. 
Your task is to read the provided Title and Abstract and evaluate it on three metrics.

The scale is from 1 to 7:
1 = Not at all
2 = Slightly
3 = Somewhat
4 = Moderately
5 = Quite
6 = Very
7 = Extremely

You must output your answer in strict JSON format with no additional text.
Structure your JSON exactly like this:
{
  "interesting": <1-7>,
  "well_written": <1-7>,
  "worth_publishing": <1-7>
}
"""

# Expected trailing prompt in Qualtrics question text that should be removed prior to parsing.
EXPECTED_SUFFIX = "\n\n<br><br><b>Do you personally think this book is...</b>"


def build_triangular_prompt(
    history: List[Tuple[str, Dict[str, Any]]], current_question: Dict[str, str]
) -> str:
    """Build a prompt that includes prior questions and model answers.

    The model is asked to continue by answering only the *next* question. This
    is meant to mimic the sequential experience a human rater would have while
    progressing through the survey without seeing all questions upfront.
    """

    lines: List[str] = []
    if history:
        for idx, (q_text, answer) in enumerate(history, start=1):
            lines.append(f"Previous question {idx}:")
            lines.append(q_text)
            lines.append("Previous answer:")
            lines.append(json.dumps(answer))
            lines.append("---")

    lines.append(
        "Return a single JSON object with keys: interesting (1-7), well_written (1-7), worth_publishing (1-7). No extra text."
    )
    lines.append(current_question["text"])
    return "\n".join(lines)


def parse_model_spec(spec: str) -> Tuple[str, str]:
    """Parse model spec of the form provider:model. Default provider=openai if missing."""
    if ":" in spec:
        provider, name = spec.split(":", 1)
        provider = provider.strip()
        name = name.strip()
    else:
        provider = "openai"
        name = spec.strip()
    if provider not in {"openai", "hf", "dummy"}:
        raise ValueError(f"Unsupported provider '{provider}'. Use 'openai', 'hf', or 'dummy'.")
    if not name:
        raise ValueError("Model name cannot be empty.")
    return provider, name


def ensure_openai():
    """Lazy import OpenAI client so local-only runs don't require the package."""
    global OpenAI
    if OpenAI is None:
        try:
            from openai import OpenAI as _OpenAI  # type: ignore
        except ImportError as exc:
            raise ImportError("openai package not installed; pip install openai") from exc
        OpenAI = _OpenAI
    return OpenAI

# ==========================================
# HELPER FUNCTIONS
# ==========================================

def clean_html(raw_html: str) -> str:
    """Removes HTML tags and cleans up whitespace."""
    if not raw_html:
        return ""
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, '', raw_html)
    return cleantext.replace('&nbsp;', ' ').strip()

def parse_qsf_questions(file_path: str) -> List[Dict[str, str]]:
    """
    Parses a Qualtrics QSF file to extract question text and IDs.
    Filters specifically for textbook proposals containing 'Abstract'.
    """
    LOGGER.info("Reading file: %s", file_path)
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    questions = []
    
    # Locate SurveyElements
    elements = data.get('SurveyElements', [])
    
    for element in elements:
        # We only care about Survey Questions ("SQ")
        if element.get('Element') == 'SQ':
            payload = element.get('Payload', {})
            q_text = payload.get('QuestionText', '')
            q_id = payload.get('QuestionID', 'Unknown')
            
            # Filter logic: We only want the textbook proposals.
            # Based on your file, these always contain "Title:" and "Abstract:"
            if "Abstract" in q_text and "Title" in q_text:
                if not q_text.endswith(EXPECTED_SUFFIX):
                    raise ValueError(f"Question {q_id} missing expected suffix")
                q_text = q_text[: -len(EXPECTED_SUFFIX)]
                clean_text = clean_html(q_text)
                questions.append({
                    "id": q_id,
                    "text": clean_text
                })

    LOGGER.info("Found %d relevant questions.", len(questions))
    return questions


def build_triangular_messages(
    history: List[Tuple[str, Dict[str, Any]]],
    current_question_text: str,
) -> List[Dict[str, str]]:
    """Build chat messages for triangular mode as true user/assistant turns.

    Structure:
    - system: SYSTEM_PROMPT
    - (user q1, assistant a1), ..., (user q_{k-1}, assistant a_{k-1})
    - user q_k + trailing 'Your answer:'
    """

    messages: List[Dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    for q_text, answer in history:
        messages.append({"role": "user", "content": q_text})
        messages.append({"role": "assistant", "content": json.dumps(answer)})

    messages.append({"role": "user", "content": f"{current_question_text}\n\nYour answer:"})
    return messages


def build_standard_messages(question_text: str) -> List[Dict[str, str]]:
    """Build chat messages for standard mode.

    We always include the system prompt explicitly so debug logging can emit the
    full payload that providers actually receive.
    """

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Here is the proposal:\n\n{question_text}"},
    ]

def query_openai(client: Any, model: str, question_text: str) -> dict[str, Any]:
    """OpenAI provider: chat.completions with JSON mode and retries."""
    retries = 3
    content = None
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"Here is the proposal:\n\n{question_text}"}
                ],
                response_format={"type": "json_object"},
                temperature=0.3,
            )
            content = response.choices[0].message.content
            if content is None:
                return {"error": "No content returned from LLM."}
            return json.loads(content)
        except Exception as e:
            LOGGER.warning("Attempt %d/%d failed: %s", attempt + 1, retries, e)
            if attempt < retries - 1:
                time.sleep(2)
            else:
                return {"error": str(e), "raw_content": content}
    return {"error": "No retries left."}


def query_openai_messages(client: Any, model: str, messages: List[Dict[str, str]]) -> dict[str, Any]:
    """OpenAI provider using explicit chat messages and retries."""
    retries = 3
    content = None
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                response_format={"type": "json_object"},
                temperature=0.3,
            )
            content = response.choices[0].message.content
            if content is None:
                return {"error": "No content returned from LLM."}
            return json.loads(content)
        except Exception as e:
            LOGGER.warning("Attempt %d/%d failed: %s", attempt + 1, retries, e)
            if attempt < retries - 1:
                time.sleep(2)
            else:
                return {"error": str(e), "raw_content": content}
    return {"error": "No retries left."}


def query_openai_batch(client: Any, model: str, user_prompt: str) -> Any:
    retries = 3
    content = None
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0.3,
            )
            content = response.choices[0].message.content
            if content is None:
                return {"error": "No content returned from LLM."}
            return content
        except Exception as e:
            LOGGER.warning("Attempt %d/%d failed: %s", attempt + 1, retries, e)
            if attempt < retries - 1:
                time.sleep(2)
            else:
                return {"error": str(e), "raw_content": content}
    return {"error": "No retries left."}


def load_hf_model(model_id: str, load_4bit: bool):
    """Load a local HuggingFace causal LM with optional 4-bit quantization."""
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

    quantization_config = None
    if load_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    # Ensure generation padding safety for batch/inference
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=quantization_config,
        device_map="auto",
    )
    model.eval()
    return tokenizer, model


def query_hf(tokenizer: Any, model: Any, question_text: str, max_new_tokens: int, temperature: float) -> dict[str, Any]:
    """HuggingFace provider using chat templates and JSON parsing."""
    import torch

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Here is the proposal:\n\n{question_text}"},
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        return_tensors="pt",
        add_generation_prompt=True,
    )

    # apply_chat_template may return a tensor or a dict; normalize to explicit tensors
    if isinstance(inputs, torch.Tensor):
        input_ids = inputs.to(model.device)
        attention_mask = torch.ones_like(input_ids, device=model.device)
    else:
        input_ids = inputs["input_ids"].to(model.device)
        attention_mask = inputs.get("attention_mask")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, device=model.device)
        else:
            attention_mask = attention_mask.to(model.device)

    input_length = input_ids.shape[1]
    with torch.no_grad():
        generated = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature,
        )

    gen_tokens = generated[0, input_length:]
    text = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        return {"error": f"Failed to parse JSON: {exc}", "raw_content": text}


def query_hf_messages(
    tokenizer: Any,
    model: Any,
    messages: List[Dict[str, str]],
    max_new_tokens: int,
    temperature: float,
) -> dict[str, Any]:
    """HuggingFace provider using chat templates from explicit chat messages."""
    import torch

    inputs = tokenizer.apply_chat_template(
        messages,
        return_tensors="pt",
        add_generation_prompt=True,
    )

    # apply_chat_template may return a tensor or a dict; normalize to explicit tensors
    if isinstance(inputs, torch.Tensor):
        input_ids = inputs.to(model.device)
        attention_mask = torch.ones_like(input_ids, device=model.device)
    else:
        input_ids = inputs["input_ids"].to(model.device)
        attention_mask = inputs.get("attention_mask")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, device=model.device)
        else:
            attention_mask = attention_mask.to(model.device)

    input_length = input_ids.shape[1]
    with torch.no_grad():
        generated = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature,
        )

    gen_tokens = generated[0, input_length:]
    text = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        return {"error": f"Failed to parse JSON: {exc}", "raw_content": text}


def query_hf_batch(tokenizer: Any, model: Any, user_prompt: str, max_new_tokens: int, temperature: float) -> Any:
    import torch

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        return_tensors="pt",
        add_generation_prompt=True,
    )

    if isinstance(inputs, torch.Tensor):
        input_ids = inputs.to(model.device)
        attention_mask = torch.ones_like(input_ids, device=model.device)
    else:
        input_ids = inputs["input_ids"].to(model.device)
        attention_mask = inputs.get("attention_mask")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, device=model.device)
        else:
            attention_mask = attention_mask.to(model.device)

    with torch.no_grad():
        generated = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature,
        )

    gen_tokens = generated[0, input_ids.shape[1]:]
    text = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
    return text


def load_results(output_path: str) -> List[Dict[str, Any]]:
    """Load existing results if the output file already exists."""
    if not os.path.exists(output_path):
        return []

    with open(output_path, 'r') as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Output file must contain a list of results.")
    return data


def index_existing_results(results: List[Dict[str, Any]]) -> Set[Tuple[str, str, int, str]]:
    """Return a set of (question_id, model, sample_index, prompt_mode) for resuming."""
    index: Set[Tuple[str, str, int, str]] = set()
    for entry in results:
        qid = entry.get("question_id")
        model = entry.get("model")
        sample_idx = entry.get("sample_index")

        if qid is None or model is None or sample_idx is None:
            raise ValueError("Existing result is missing 'question_id', 'model', or 'sample_index'.")

        try:
            sample_int = int(sample_idx)
        except (TypeError, ValueError) as exc:
            raise ValueError("Existing result has a non-integer 'sample_index'.") from exc

        prompt_mode = str(entry.get("prompt_mode") or "standard")
        index.add((str(qid), str(model), sample_int, prompt_mode))
    return index


def build_batch_prompt(batch_questions: List[Dict[str, str]]) -> str:
    lines = []
    lines.append("You will receive multiple textbook proposals. For EACH proposal, produce an object with: question_id, interesting (1-7), well_written (1-7), worth_publishing (1-7). Return a single JSON array in the same order as provided. No extra text.")
    lines.append("")
    for idx, q in enumerate(batch_questions, start=1):
        lines.append(f"Proposal {idx} | question_id={q['id']}")
        lines.append(q['text'])
        lines.append("---")
    return "\n".join(lines)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a QSF survey against an LLM agent.")
    parser.add_argument("qsf_file", help="Path to the .qsf file")
    parser.add_argument(
        "--model",
        dest="models",
        action="append",
        help=(
            "Model spec. Use 'openai:model' (e.g. openai:gpt-5.2), 'hf:model_id', or 'dummy:anything'. "
            "If provider omitted, defaults to openai. Repeatable."
        ),
    )
    parser.add_argument("--hf-4bit", action="store_true", help="Load HF model with 4-bit quantization (bitsandbytes)")
    parser.add_argument("--hf-max-new-tokens", type=int, default=256, help="HF: max new tokens per completion (default: 256)")
    parser.add_argument("--hf-temperature", type=float, default=0.3, help="HF: sampling temperature (default: 0.3)")
    parser.add_argument(
        "--prompt-mode",
        choices=["batch", "standard", "triangular"],
        default="standard",
        help="Prompt style: batch (all Qs at once), standard (independent), or triangular (sequential with prior Q/A).",
    )
    parser.add_argument("--log-file", default="survey_llm.log", help="Path to log file (default: survey_llm.log)")
    parser.add_argument("--samples", type=int, default=1, help="Number of samples per question/model (default: 1)")
    parser.add_argument("--run", action="store_true", help="Execute calls (default is dry-run)")
    parser.add_argument("--output", default="survey_results.json", help="Output file path")
    parser.add_argument(
        "--debug-prompts",
        action="store_true",
        help="Log full prompts/messages (including triangular history) at DEBUG level.",
    )
    return parser


def configure_logging(log_file: str, *, debug_prompts: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if debug_prompts else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(),
        ],
        force=True,
    )


def dummy_evaluation() -> Dict[str, int]:
    return {
        "interesting": random.randint(1, 7),
        "well_written": random.randint(1, 7),
        "worth_publishing": random.randint(1, 7),
    }


def parse_models(model_specs: Optional[List[str]]) -> List[Tuple[str, str]]:
    specs = model_specs or ["gpt-4o-mini"]
    return [parse_model_spec(m) for m in specs]


def init_openai_client_if_needed(
    *,
    dry_run: bool,
    parsed_models: List[Tuple[str, str]],
) -> Any:
    if dry_run or not any(p == "openai" for p, _ in parsed_models):
        return None

    api_key = os.environ.get("OPENAI_API_KEY")
    OpenAIClient = ensure_openai()
    if not api_key:
        LOGGER.error("OPENAI_API_KEY environment variable not found. Export it or use dry-run.")
        raise SystemExit(1)
    return OpenAIClient(api_key=api_key)


def load_questions_or_exit(qsf_file: str) -> List[Dict[str, str]]:
    questions = parse_qsf_questions(qsf_file)
    if not questions:
        LOGGER.error("No questions found. Please check the QSF format.")
        raise SystemExit(1)
    return questions


def load_resume_state(output_path: str) -> tuple[List[Dict[str, Any]], Set[Tuple[str, str, int, str]]]:
    existing_results = load_results(output_path)
    existing_index = index_existing_results(existing_results)
    return list(existing_results), existing_index


def save_results(output_path: str, results: List[Dict[str, Any]]) -> None:
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)


def get_hf_backend(
    hf_cache: Dict[str, tuple[Any, Any]],
    *,
    model_id: str,
    load_4bit: bool,
) -> tuple[Any, Any]:
    if model_id not in hf_cache:
        LOGGER.info("Loading HF model %s (4-bit=%s)...", model_id, load_4bit)
        hf_cache[model_id] = load_hf_model(model_id, load_4bit)
    return hf_cache[model_id]


def handle_batch_mode(
    *,
    provider: str,
    model: str,
    sample_idx: int,
    questions: List[Dict[str, str]],
    existing_index: Set[Tuple[str, str, int, str]],
    results: List[Dict[str, Any]],
    output_path: str,
    dry_run: bool,
    prompt_mode: str,
    client: Any,
    hf_cache: Dict[str, tuple[Any, Any]],
    hf_4bit: bool,
    hf_max_new_tokens: int,
    hf_temperature: float,
) -> None:
    if prompt_mode != "batch":
        raise ValueError("handle_batch_mode called with non-batch prompt_mode")

    missing_questions = [
        q
        for q in questions
        if (q["id"], f"{provider}:{model}", sample_idx, "batch") not in existing_index
    ]
    if not missing_questions:
        LOGGER.info(
            "All questions already present for %s:%s sample %d; skipping batch",
            provider,
            model,
            sample_idx,
        )
        return

    if dry_run:
        LOGGER.info(
            "[DRY RUN] Would evaluate %s:%s sample %d with batch of %d questions",
            provider,
            model,
            sample_idx,
            len(missing_questions),
        )
        return

    if provider == "dummy":
        LOGGER.debug(
            "Dummy batch prompt for %s:%s sample %d:\n%s",
            provider,
            model,
            sample_idx,
            build_batch_prompt(missing_questions),
        )
        for q_item in missing_questions:
            key = (q_item["id"], f"{provider}:{model}", sample_idx, "batch")
            if key in existing_index:
                continue
            results.append(
                {
                    "question_id": q_item["id"],
                    "model": f"{provider}:{model}",
                    "sample_index": sample_idx,
                    "batch_size": len(missing_questions),
                    "prompt_mode": "batch",
                    "original_text_truncated": q_item["text"][:100] + "...",
                    "evaluation": dummy_evaluation(),
                }
            )
            existing_index.add(key)
        save_results(output_path, results)
        return

    # Log the full prompt payload for non-dummy providers when debug is enabled.
    # This includes the system prompt and the full user prompt.
    LOGGER.debug(
        "Batch messages for %s:%s sample %d:\n%s",
        provider,
        model,
        sample_idx,
        json.dumps(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_batch_prompt(missing_questions)},
            ],
            indent=2,
        ),
    )

    user_prompt = build_batch_prompt(missing_questions)
    if provider == "openai":
        assert client is not None, "Client must be initialized in run mode"
        raw = query_openai_batch(client, model, user_prompt)
    else:
        tokenizer, hf_model = get_hf_backend(hf_cache, model_id=model, load_4bit=hf_4bit)
        batch_tokens = hf_max_new_tokens * len(missing_questions)
        raw = query_hf_batch(
            tokenizer,
            hf_model,
            user_prompt,
            max_new_tokens=batch_tokens,
            temperature=hf_temperature,
        )

    try:
        parsed = json.loads(raw)
    except Exception as exc:
        LOGGER.error("Failed to parse batch response: %s", exc)
        LOGGER.error("Raw batch response: %s", raw)
        parsed = None

    candidate_map = None
    if isinstance(parsed, dict):
        candidate_map = parsed
    elif isinstance(parsed, list) and len(parsed) == 1 and isinstance(parsed[0], dict):
        candidate_map = parsed[0]

    if candidate_map is not None:
        LOGGER.info("Batch response was a dict keyed by question_id; converting to per-question entries.")
        for q_item in missing_questions:
            key = (q_item["id"], f"{provider}:{model}", sample_idx, "batch")
            if key in existing_index:
                continue
            eval_item = candidate_map.get(q_item["id"])
            if eval_item is None:
                LOGGER.error("Missing entry for %s in batch dict; skipping.", q_item["id"])
                continue
            result_entry = {
                "question_id": q_item["id"],
                "model": f"{provider}:{model}",
                "sample_index": sample_idx,
                "batch_size": len(missing_questions),
                "prompt_mode": "batch",
                "original_text_truncated": q_item["text"][:100] + "...",
                "evaluation": eval_item,
            }
            results.append(result_entry)
            existing_index.add(key)
        save_results(output_path, results)
        return

    if not isinstance(parsed, list):
        LOGGER.error("Batch response is not a list; skipping save. Raw: %s", raw)
        return

    if len(parsed) != len(missing_questions):
        LOGGER.error(
            "Batch response length %d does not match questions %d; skipping save. Raw: %s",
            len(parsed),
            len(missing_questions),
            raw,
        )
        return

    for q_item, eval_item in zip(missing_questions, parsed):
        key = (q_item["id"], f"{provider}:{model}", sample_idx, "batch")
        if key in existing_index:
            continue
        result_entry = {
            "question_id": q_item["id"],
            "model": f"{provider}:{model}",
            "sample_index": sample_idx,
            "batch_size": len(missing_questions),
            "prompt_mode": "batch",
            "original_text_truncated": q_item["text"][:100] + "...",
            "evaluation": eval_item,
        }
        results.append(result_entry)
        existing_index.add(key)
    save_results(output_path, results)


def handle_per_question_mode(
    *,
    provider: str,
    model: str,
    sample_idx: int,
    questions: List[Dict[str, str]],
    existing_index: Set[Tuple[str, str, int, str]],
    results: List[Dict[str, Any]],
    output_path: str,
    dry_run: bool,
    prompt_mode: str,
    client: Any,
    hf_cache: Dict[str, tuple[Any, Any]],
    hf_4bit: bool,
    hf_max_new_tokens: int,
    hf_temperature: float,
) -> None:
    if prompt_mode == "batch":
        raise ValueError("handle_per_question_mode called with batch prompt_mode")

    # Carry history of (question_text, model_answer) per model/sample when in triangular mode.
    history: List[Tuple[str, Dict[str, Any]]] = []

    for i, q in enumerate(questions):
        LOGGER.info("Processing %s (%d/%d)...", q["id"], i + 1, len(questions))
        if dry_run:
            LOGGER.info("Question payload: %s", json.dumps(q, indent=2))

        key = (q["id"], f"{provider}:{model}", sample_idx, prompt_mode)
        if key in existing_index:
            LOGGER.info("Skipping %s sample %d (already present)", model, sample_idx)
            continue

        if dry_run:
            LOGGER.info("[DRY RUN] Would evaluate %s sample %d", model, sample_idx)
            continue

        if provider == "dummy":
            if prompt_mode == "triangular":
                messages = build_triangular_messages(history, q["text"])
                LOGGER.debug(
                    "Triangular messages for %s:%s sample %d %s:\n%s",
                    provider,
                    model,
                    sample_idx,
                    q["id"],
                    json.dumps(messages, indent=2),
                )
            else:
                messages = build_standard_messages(q["text"])
                LOGGER.debug(
                    "Standard messages for %s:%s sample %d %s:\n%s",
                    provider,
                    model,
                    sample_idx,
                    q["id"],
                    json.dumps(messages, indent=2),
                )
            evaluation = dummy_evaluation()
        elif provider == "openai":
            assert client is not None, "Client must be initialized in run mode"
            if prompt_mode == "triangular":
                messages = build_triangular_messages(history, q["text"])
                LOGGER.debug(
                    "Triangular messages for %s:%s sample %d %s:\n%s",
                    provider,
                    model,
                    sample_idx,
                    q["id"],
                    json.dumps(messages, indent=2),
                )
                evaluation = query_openai_messages(client, model, messages)
            else:
                messages = build_standard_messages(q["text"])
                LOGGER.debug(
                    "Standard messages for %s:%s sample %d %s:\n%s",
                    provider,
                    model,
                    sample_idx,
                    q["id"],
                    json.dumps(messages, indent=2),
                )
                evaluation = query_openai_messages(client, model, messages)
        else:
            tokenizer, hf_model = get_hf_backend(hf_cache, model_id=model, load_4bit=hf_4bit)
            if prompt_mode == "triangular":
                messages = build_triangular_messages(history, q["text"])
                LOGGER.debug(
                    "Triangular messages for %s:%s sample %d %s:\n%s",
                    provider,
                    model,
                    sample_idx,
                    q["id"],
                    json.dumps(messages, indent=2),
                )
                evaluation = query_hf_messages(
                    tokenizer,
                    hf_model,
                    messages,
                    max_new_tokens=hf_max_new_tokens,
                    temperature=hf_temperature,
                )
            else:
                messages = build_standard_messages(q["text"])
                LOGGER.debug(
                    "Standard messages for %s:%s sample %d %s:\n%s",
                    provider,
                    model,
                    sample_idx,
                    q["id"],
                    json.dumps(messages, indent=2),
                )
                evaluation = query_hf_messages(
                    tokenizer,
                    hf_model,
                    messages,
                    max_new_tokens=hf_max_new_tokens,
                    temperature=hf_temperature,
                )

        batch_size = 1 if prompt_mode != "triangular" else len(history) + 1
        result_entry = {
            "question_id": q["id"],
            "model": f"{provider}:{model}",
            "sample_index": sample_idx,
            "batch_size": batch_size,
            "prompt_mode": prompt_mode,
            "original_text_truncated": q["text"][:100] + "...",
            "evaluation": evaluation,
        }
        results.append(result_entry)
        existing_index.add(key)

        if prompt_mode == "triangular":
            history.append((q["text"], evaluation))

        # Incremental save (prevents data loss if script crashes)
        save_results(output_path, results)

# ==========================================
# MAIN EXECUTION
# ==========================================

def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    configure_logging(args.log_file, debug_prompts=args.debug_prompts)

    if args.samples < 1:
        raise ValueError("--samples must be at least 1")

    dry_run = not args.run
    parsed_models = parse_models(args.models)

    client = init_openai_client_if_needed(dry_run=dry_run, parsed_models=parsed_models)
    hf_cache: Dict[str, tuple[Any, Any]] = {}

    questions = load_questions_or_exit(args.qsf_file)
    results, existing_index = load_resume_state(args.output)

    # 3. Loop through questions
    mode_msg = "DRY RUN" if dry_run else "RUN"
    readable = ", ".join([f"{prov}:{name}" for prov, name in parsed_models])
    LOGGER.info("Starting evaluation (%s) using models: %s", mode_msg, readable)
    LOGGER.info("Samples per question/model: %d", args.samples)
    LOGGER.info("%s", "-" * 50)

    for provider, model in parsed_models:
        for sample_idx in range(1, args.samples + 1):

            if args.prompt_mode == "batch":
                handle_batch_mode(
                    provider=provider,
                    model=model,
                    sample_idx=sample_idx,
                    questions=questions,
                    existing_index=existing_index,
                    results=results,
                    output_path=args.output,
                    dry_run=dry_run,
                    prompt_mode=args.prompt_mode,
                    client=client,
                    hf_cache=hf_cache,
                    hf_4bit=args.hf_4bit,
                    hf_max_new_tokens=args.hf_max_new_tokens,
                    hf_temperature=args.hf_temperature,
                )
            else:
                handle_per_question_mode(
                    provider=provider,
                    model=model,
                    sample_idx=sample_idx,
                    questions=questions,
                    existing_index=existing_index,
                    results=results,
                    output_path=args.output,
                    dry_run=dry_run,
                    prompt_mode=args.prompt_mode,
                    client=client,
                    hf_cache=hf_cache,
                    hf_4bit=args.hf_4bit,
                    hf_max_new_tokens=args.hf_max_new_tokens,
                    hf_temperature=args.hf_temperature,
                )

    LOGGER.info("%s", "-" * 50)
    LOGGER.info("Done! Results saved to %s", args.output)

if __name__ == "__main__":
    main()