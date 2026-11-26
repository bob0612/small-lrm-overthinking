"""Model backends. VLLMBackend runs the real model; FakeBackend is a tiny deterministic
stand-in used by the tests so the whole pipeline can be checked without a GPU."""
from __future__ import annotations

import random

# Qwen3 sampling settings recommended for thinking mode (model card).
THINK_SAMPLING = dict(temperature=0.6, top_p=0.95, top_k=20)
PROBE_SUFFIX = "\n</think>\n\nThe final answer is $\\boxed{"


class VLLMBackend:
    def __init__(self, model: str, revision: str | None, max_model_len: int, gpu_mem: float = 0.90, seed: int = 0):
        from vllm import LLM

        self.llm = LLM(model=model, revision=revision, max_model_len=max_model_len, enable_prefix_caching=True,
                       gpu_memory_utilization=gpu_mem, seed=seed, dtype="bfloat16")
        self.tok = self.llm.get_tokenizer()
        self.think_end_id = self.tok.convert_tokens_to_ids("</think>")

    def chat_prompt_ids(self, user: str) -> list[int]:
        text = self.tok.apply_chat_template([{"role": "user", "content": user}], tokenize=False,
                                            add_generation_prompt=True, enable_thinking=True)
        return self.tok.encode(text, add_special_tokens=False)

    def encode(self, text: str) -> list[int]:
        return self.tok.encode(text, add_special_tokens=False)

    def decode(self, ids: list[int]) -> str:
        return self.tok.decode(ids, skip_special_tokens=False)

    def generate_traces(self, prompts: list[list[int]], n: int, max_tokens: int, seed: int) -> list[list[list[int]]]:
        from vllm import SamplingParams, TokensPrompt

        sp = SamplingParams(n=n, max_tokens=max_tokens, **THINK_SAMPLING)  # engine-level seed; a per-request seed forces a slow per-request sampler
        outs = self.llm.generate([TokensPrompt(prompt_token_ids=p) for p in prompts], sp)
        return [[list(c.token_ids) for c in o.outputs] for o in outs]

    def probe(self, prompts: list[list[int]], max_tokens: int = 40) -> list[tuple[list[int], list[float]]]:
        """Greedy continuation; returns token ids and their log-probabilities."""
        from vllm import SamplingParams, TokensPrompt

        sp = SamplingParams(temperature=0.0, max_tokens=max_tokens, logprobs=0, stop=["}$", "}\n"],
                            include_stop_str_in_output=True)
        outs = self.llm.generate([TokensPrompt(prompt_token_ids=p) for p in prompts], sp, use_tqdm=False)
        res = []
        for o in outs:
            c = o.outputs[0]
            lps = [lp[t].logprob for t, lp in zip(c.token_ids, c.logprobs)] if c.logprobs else [0.0] * len(c.token_ids)
            res.append((list(c.token_ids), lps))
        return res

    def continue_from(self, prompts: list[list[int]], max_tokens, seed: int) -> list[list[int]]:
        """max_tokens: an int, or one int per prompt. No per-request seed (it forces vLLM's slow
        per-request sampler); the engine seed makes runs repeatable."""
        from vllm import SamplingParams, TokensPrompt

        if isinstance(max_tokens, int):
            max_tokens = [max_tokens] * len(prompts)
        sp = [SamplingParams(n=1, max_tokens=m, **THINK_SAMPLING) for m in max_tokens]
        outs = self.llm.generate([TokensPrompt(prompt_token_ids=p) for p in prompts], sp)
        return [list(o.outputs[0].token_ids) for o in outs]


class FakeBackend:
    """Character-level 'model'. Each trace is a scripted thinking text whose intermediate
    answers change at known places, so tests can check the probing logic exactly."""

    def __init__(self, scripts: dict[str, str]):
        # scripts: problem text -> full generated text (thinking + '</think>' + answer)
        self.scripts = scripts
        self.vocab = {}
        self.inv = {}
        self.think_end_id = self._id("</think>")

    def _id(self, piece: str) -> int:
        if piece not in self.vocab:
            self.vocab[piece] = len(self.vocab)
            self.inv[self.vocab[piece]] = piece
        return self.vocab[piece]

    def encode(self, text: str) -> list[int]:
        ids, i = [], 0
        while i < len(text):
            if text.startswith("</think>", i):
                ids.append(self._id("</think>"))
                i += len("</think>")
            else:
                ids.append(self._id(text[i]))
                i += 1
        return ids

    def decode(self, ids: list[int]) -> str:
        return "".join(self.inv[i] for i in ids)

    def chat_prompt_ids(self, user: str) -> list[int]:
        return self.encode("<|user|>" + user + "<|assistant|>")

    def _problem_of(self, prompt: list[int]) -> str:
        text = self.decode(prompt)
        return text.split("<|user|>")[1].split("<|assistant|>")[0]

    def generate_traces(self, prompts, n, max_tokens, seed):
        return [[self.encode(self.scripts[self._problem_of(p)])[:max_tokens] for _ in range(n)] for p in prompts]

    def probe(self, prompts, max_tokens=40):
        res = []
        for p in prompts:
            text = self.decode(p)
            think = text.split("<|assistant|>")[1]
            # the 'model' answers with the last 'ANS=x' it has written so far
            import re

            m = re.findall(r"ANS=(\S+)", think)
            ans = m[-1] if m else "0"
            ids = self.encode(ans + "}$")
            conf = -0.1 if len(m) >= 2 and m[-1] == m[-2] else -1.0
            res.append((ids, [conf] + [0.0] * (len(ids) - 1)))
        return res

    def continue_from(self, prompts, max_tokens, seed):
        rng = random.Random(seed)
        return [self.encode(f" ANS={rng.randint(0, 9)}</think>\\boxed{{{rng.randint(0, 9)}}}") for _ in prompts]
