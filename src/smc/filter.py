import llama_cpp
import numpy as np
from dataclasses import dataclass

@dataclass
class Particle:
    tokens: list[int]
    weight: float = 1.0
    active: bool = True


class Filter:
    def __init__(
        self,
        model: llama_cpp.Llama,
        prompt: bytes,
        max_new_tokens: int,
        N: int,
        tau: float = 0.5,
        temperature: float = 1.0,
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
        self.particles = [
            Particle(tokens=[], weight=1.0, active=True) for _ in range(N)
        ]

    def resample(self) -> None:
        W = sum([particle.weight for particle in self.particles])
        N_eff = W**2 / sum([particle.weight**2 for particle in self.particles])
        # Resample if effective sample size is too small
        if N_eff < self.tau * self.N:
            counts = np.random.multinomial(
                n=self.N,
                pvals=[particle.weight / W for particle in self.particles],
            )
            print(counts)
            particles: list[Particle] = [None] * self.N  # type: ignore
            empty_indices = [i for i, ct in enumerate(counts) if ct == 0]
            for i, ct in enumerate(counts):
                if ct >= 1:
                    # First copy: keep tokens in place
                    particles[i] = Particle(
                        tokens=self.particles[i].tokens,
                        weight=1 / self.N,
                        active=self.particles[i].active,
                    )
                for _ in range(ct - 1):
                    assert empty_indices, "No empty indices left to fill"
                    j = empty_indices.pop()
                    print(f"Copying particle {i} to index {j}")
                    particles[j] = Particle(
                        # !IMPORTANT! Copy the tokens here
                        tokens=list(self.particles[i].tokens),
                        weight=1 / self.N,
                        active=self.particles[i].active,
                    )
                    # Copy kv cache: i (source) -> j (target)
                    # TODO: just noticed that we're not referencing the batch
                    # directly here... is the cache sitting in the ctx? We're
                    # probably not able to reuse the same model if this is the
                    # case. FIX THIS
                    llama_cpp.llama_kv_cache_seq_cp(
                        self.model.ctx, i, j, 0, self.batch.n_tokens
                    )
            self.particles = particles

    def propagate(self) -> None:
        if not any(p.active for p in self.particles):
            raise RuntimeError("No active particles left")
        for particle in self.particles:
            if not particle.active:
                continue
            # TODO: use mask and re-weigh the particles
            logits_ = llama_cpp.llama_get_logits(self.model.ctx)
            logits = np.ctypeslib.as_array(logits_, shape=(self.model.n_vocab(),)).copy()
            scores = logits / self.temperature
            probs = np.exp(scores - np.max(scores))
            probs /= np.sum(probs)
            new_token_id = int(
                np.random.choice(
                    range(self.model.n_vocab()),
                    p=probs,
                )
            )

            if new_token_id == self.model.token_eos() or len(particle.tokens) == self.max_new_tokens:
                particle.active = False
                continue

            particle.tokens.append(new_token_id)

        if llama_cpp.llama_decode(self.model.ctx, self.batch) != 0:
            raise RuntimeError("Could not decode the batch")

def main():
    repo_id = "unsloth/Qwen3-1.7B-GGUF"
    filename = "Qwen3-1.7B-Q4_K_M.gguf"
    n_ctx = 4098

    model: llama_cpp.Llama = llama_cpp.Llama.from_pretrained(
        repo_id=repo_id,
        filename=filename,
        n_ctx=n_ctx,
    )

    filter = Filter(
        model=model,
        prompt=b"Hello, my name is",
        max_new_tokens=10,
        N=10,
        tau=1.1,
        temperature=1.0,
    )

    while any(p.active for p in filter.particles):
        filter.propagate()
        filter.resample()

    return filter

if __name__ == "__main__":
    filter = main()
