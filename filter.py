from dataclasses import dataclass

import llama_cpp
import numpy as np
from guidance.models import LlamaCpp as GLlamacpp
from llguidance import LLMatcher, LLTokenizer, TokenizerWrapper


@dataclass
class Particle:
    tokens: list[int]
    log_weight: float
    active: bool
    matcher: LLMatcher


class Filter:
    def __init__(
        self,
        model: llama_cpp.Llama,
        prompt: bytes,
        grammar: str,
        max_new_tokens: int,
        N: int,
        tau: float = 0.5,
        temperature: float = 1.0,
        stratified: bool = False,
    ) -> None:
        tokens = model.tokenize(prompt)
        tokens_len = len(tokens)

        n_len = tokens_len + max_new_tokens
        n_ctx = model.n_ctx()
        if n_len * N > n_ctx:
            raise ValueError(f"n_len * n_parallel ({n_len * N}) > n_ctx ({n_ctx})")

        # Initialize the batch
        batch = llama_cpp.llama_batch_init(max(n_len, N), 0, 1)
        batch.n_tokens = tokens_len
        for i in range(tokens_len):
            batch.token[i] = tokens[i]
            batch.pos[i] = i
            batch.seq_id[i][0] = 0
            batch.n_seq_id[i] = 1
            batch.logits[i] = False
        batch.logits[tokens_len - 1] = True

        # Prefill the batch with the prompt
        if llama_cpp.llama_decode(model.ctx, batch) != 0:
            raise RuntimeError(f"Could not decode the prompt")

        # Copy the kv cache for each particle
        for i in range(N):
            llama_cpp.llama_kv_cache_seq_cp(model.ctx, 0, i, 0, batch.n_tokens)

        self.model = model
        self.prompt = prompt
        self.prompt_tokens = tokens
        self.batch = batch
        self.max_new_tokens = max_new_tokens
        self.N = N
        self.tau = tau
        self.temperature = temperature
        self.stratified = stratified

        g_model = GLlamacpp(model)
        ll_tokenizer = LLTokenizer(
            TokenizerWrapper(g_model._interpreter.engine.tokenizer)
        )  # Oof
        self.particles = [
            Particle(
                tokens=[],
                log_weight=0.0,
                active=True,
                matcher=LLMatcher(tokenizer=ll_tokenizer, grammar=grammar),
            )
            for _ in range(N)
        ]

    def sample(self) -> list[int]:
        log_weights = [particle.log_weight for particle in self.particles]
        log_W = np.logaddexp.reduce(log_weights)
        log_probs = log_weights - log_W
        probs = np.exp(log_probs)
        index = int(np.random.choice(
            range(self.N),
            p=probs,
        ))
        return self.particles[index].tokens

    def resample(self) -> None:
        # Log-sum-exp trick for log(W) and log(sum(w_i^2))
        log_weights = [particle.log_weight for particle in self.particles]
        log_W = np.logaddexp.reduce(log_weights)
        log_W2 = np.logaddexp.reduce(2 * log_weights)
        log_N_eff = 2 * log_W - log_W2
        N_eff = np.exp(log_N_eff)
        # Resample if effective sample size is too small
        if N_eff < self.tau * self.N:
            log_probs = log_weights - log_W
            probs = np.exp(log_probs)
            if self.stratified:
                counts = stratified_resample(probs)
            else:
                counts = np.random.multinomial(self.N, probs)
            particles: list[Particle] = [None] * self.N  # type: ignore
            empty_indices = [i for i, ct in enumerate(counts) if ct == 0]
            for i, ct in enumerate(counts):
                if ct >= 1:
                    # First copy: keep tokens in place
                    particles[i] = Particle(
                        tokens=self.particles[i].tokens,
                        log_weight=log_W-np.log(self.N),
                        active=self.particles[i].active,
                        matcher=self.particles[i].matcher,
                    )
                for _ in range(ct - 1):
                    assert empty_indices, "No empty indices left to fill"
                    j = empty_indices.pop()
                    # !IMPORTANT! Copy innards here
                    particles[j] = Particle(
                        tokens=list(self.particles[i].tokens),
                        log_weight=log_W-np.log(self.N),
                        active=self.particles[i].active,
                        matcher=self.particles[i].matcher.deep_copy(),
                    )
                    # Copy kv cache: i (source) -> j (target)
                    # TODO: just noticed that we're not referencing the batch
                    # directly here... is the cache sitting in the ctx? We're
                    # probably not able to reuse the same model if this is the
                    # case. FIX THIS
                    llama_cpp.llama_kv_cache_seq_cp(
                        self.model.ctx, i, j, 0, self.batch.n_tokens
                    )
            assert not empty_indices, "Not all empty indices were filled"
            self.particles = particles

    def propagate(self) -> None:
        if not any(p.active for p in self.particles):
            raise RuntimeError("No active particles left")
        for particle in self.particles:
            if not particle.active:
                continue
            logits_ = llama_cpp.llama_get_logits(self.model.ctx)
            logits = np.ctypeslib.as_array(
                logits_, shape=(self.model.n_vocab(),)
            ).copy()
            byte_mask = particle.matcher.compute_logit_bias()
            mask = np.frombuffer(byte_mask, dtype=np.uint8)
            masked_logits = np.where(mask > 0, logits, -np.inf)
            q = softmax(masked_logits, self.temperature)
            new_token_id = int(
                np.random.choice(
                    range(self.model.n_vocab()),
                    p=q,
                )
            )
            assert mask[new_token_id] > 0
            assert particle.matcher.consume_token(
                new_token_id
            ), f"Token {new_token_id} not in grammar"
            particle.tokens.append(new_token_id)

            p = softmax(logits, self.temperature)
            # Note: these are equivalent
            # z = np.sum(np.where(mask > 0, p, 0))
            log_z = np.log(p[new_token_id]) - np.log(q[new_token_id])
            particle.log_weight += log_z

            if (
                new_token_id == self.model.token_eos()
                # TODO: make sure we're correctly reweighting in
                # the different "done" conditions
                or len(particle.tokens) == self.max_new_tokens
                or particle.matcher.is_stopped()
            ):
                particle.active = False

        if llama_cpp.llama_decode(self.model.ctx, self.batch) != 0:
            raise RuntimeError("Could not decode the batch")


def softmax(logits: np.ndarray, temperature) -> np.ndarray:
    max_logit = np.max(logits[np.isfinite(logits)], initial=-np.inf)
    stable_scores = (logits - max_logit) / temperature
    exp_scores = np.exp(stable_scores)
    probs = exp_scores / np.sum(exp_scores)
    return probs

def stratified_resample(probs: np.ndarray[float]) -> np.ndarray[int]:
    N = probs.shape[0]
    u = (np.arange(N) + np.random.rand(N)) / N
    bins = np.cumsum(probs)
    indices =  np.digitize(u, bins)
    # histogram of indices
    return np.bincount(indices, minlength=N)

def main():
    repo_id = "unsloth/Qwen3-1.7B-GGUF"
    filename = "Qwen3-1.7B-Q4_K_M.gguf"
    n_ctx = 4096

    model: llama_cpp.Llama = llama_cpp.Llama.from_pretrained(
        repo_id=repo_id,
        filename=filename,
        n_ctx=n_ctx,
    )

    filter = Filter(
        model=model,
        prompt=b"My favorite character from Star Wars is",
        grammar=LLMatcher.grammar_from_regex(
            r" Luke Wilson| Ben Kenobi| Anakin Skywalker| Darth Vader| Leia Organa"
        ),
        max_new_tokens=10,
        N=100,
        tau=0.5,
        temperature=0.7,
        stratified=True,
    )

    while any(p.active for p in filter.particles):
        filter.propagate()
        filter.resample()

    return filter


if __name__ == "__main__":
    filter = main()
    print(
        filter.model.detokenize(filter.prompt_tokens + filter.sample())
    )
