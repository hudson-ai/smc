import llama_cpp
import numpy as np


def generate_parallel(
    model: llama_cpp.Llama,
    prompt: bytes,
    max_new_tokens: int,
    n_parallel: int,
    temperature: float = 1.0,
):
    tokens = model.tokenize(prompt)
    tokens_len = len(tokens)

    n_vocab = model.n_vocab()
    n_len = tokens_len + max_new_tokens
    n_ctx = model.n_ctx()
    if n_len * n_parallel > n_ctx:
        raise ValueError(f"n_len * n_parallel ({n_len * n_parallel}) > n_ctx ({n_ctx})")

    batch = llama_cpp.llama_batch_init(max(n_len, n_parallel), 0, 1)
    batch.n_tokens = tokens_len
    for i in range(tokens_len):
        batch.token[i] = tokens[i]
        batch.pos[i] = i
        batch.seq_id[i][0] = 0
        batch.n_seq_id[i] = 1
        batch.logits[i] = False

    batch.logits[tokens_len - 1] = True

    if llama_cpp.llama_decode(model.ctx, batch) != 0:
        print("Error decoding")

    for i in range(n_parallel):
        llama_cpp.llama_kv_cache_seq_cp(model.ctx, 0, i, 0, batch.n_tokens)

    streams = [b""] * n_parallel
    i_batch = [batch.n_tokens - 1] * n_parallel

    n_cur = tokens_len
    n_decode = 0

    while n_cur <= n_len:
        batch.n_tokens = 0
        for i in range(n_parallel):
            if i_batch[i] < 0:
                continue

            logits_ = llama_cpp.llama_get_logits(model.ctx, i_batch[i])
            logits = np.ctypeslib.as_array(logits_, shape=(n_vocab,)).copy()
            scores = logits / temperature
            probs = np.exp(scores - np.max(scores))
            probs /= np.sum(probs)
            new_token_id = int(
                np.random.choice(
                    range(n_vocab),
                    p=probs,
                )
            )

            if new_token_id == model.token_eos() or n_cur == n_len:
                i_batch[i] = -1
                continue

            new_bytes = model.detokenize([new_token_id], prev_tokens=tokens[:n_cur])
            streams[i] += new_bytes

            batch.token[batch.n_tokens] = new_token_id
            batch.pos[batch.n_tokens] = n_cur
            batch.seq_id[batch.n_tokens][0] = i
            batch.n_seq_id[batch.n_tokens] = 1
            batch.logits[batch.n_tokens] = True

            i_batch[i] = batch.n_tokens
            batch.n_tokens += 1
            n_decode += 1

        if batch.n_tokens == 0:
            break

        n_cur += 1

        if llama_cpp.llama_decode(model.ctx, batch) != 0:
            print("Error decoding", flush=True)
            break

        print(n_cur)
        print(streams)


def main():
    repo_id = "unsloth/Qwen3-1.7B-GGUF"
    filename = "Qwen3-1.7B-Q4_K_M.gguf"
    n_ctx = 4098

    model: llama_cpp.Llama = llama_cpp.Llama.from_pretrained(
        repo_id=repo_id,
        filename=filename,
        n_ctx=n_ctx,
    )

    generate_parallel(
        model=model,
        prompt=b"The quick brown fox",
        max_new_tokens=32,
        n_parallel=2,
        temperature=0.7,
    )


if __name__ == "__main__":
    main()
