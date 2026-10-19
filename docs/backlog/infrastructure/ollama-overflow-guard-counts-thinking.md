# The Ollama overflow guard leaves out the thinking tokens

`OllamaProposalGenerator._chat_once` (`bookgap/llm/generators/ollama_local.py`) raises `ContextOverflowError` when `prompt_eval_count + eval_count` exceeds `num_ctx`, since a reply that runs past the window makes Ollama shift the context and drop the start of the prompt.
With gemma4's thinking and a `format` schema, `eval_count` and `eval_duration` cover only the answer: a typical proposal reports 151 tokens in 1.8 s, while its `thinking` field holds about 2,700 characters (some 700 tokens) and the request takes 13 s.
So the guard under-counts by the thinking, and REPRODUCING.md's "fails any reply that runs past the window" is stronger than what is checked.

No run of the paper's has overflowed.
Counting the thinking at a conservative 3 characters a token, the closest at 16,384 tokens is EmbeddingGemma JAD on T12026, at about 15,760 (96%); at 65,536 the largest is T13062 at radius fraction 1/2, about 31,300.

## Fix

- Count the thinking in the guard: tokenize the returned `thinking` with the model's tokenizer, or have Ollama report it, if a version does.
- Or stop the shift at its source, if the Ollama in use takes an option to end generation at the window instead of shifting (then `done_reason` says so, and the generator can fail on it).
- Either way, re-check the October 2026 runs with the corrected count before claiming no reply overflowed.
