import llama_cpp
import numpy as np

repo_id = "unsloth/Qwen3-1.7B-GGUF"
filename = "Qwen3-1.7B-Q4_K_M.gguf"

n_len = 100
n_parallel = 2
n_ctx = n_len * n_parallel

model_obj: llama_cpp.Llama = llama_cpp.Llama.from_pretrained(
    repo_id=repo_id,
    filename=filename,
    seed=1234,
    n_ctx=n_ctx,
    n_batch=max(n_len, n_parallel),
    n_threads=1,
    n_threads_batch=1
)

model = model_obj.model
tokenizer = model_obj.tokenizer_
ctx = model_obj.ctx

temperature = .7
prompt = b"The quick brown fox"
tokens = tokenizer.tokenize(prompt)
tokens_len = len(tokens)

n_ctx = llama_cpp.llama_n_ctx(ctx)
batch = llama_cpp.llama_batch_init(max(n_len, n_parallel), 0, 1)

batch.n_tokens = tokens_len
for i in range(tokens_len):
    batch.token[i] = tokens[i]
    batch.pos[i] = i
    batch.seq_id[i][0] = 0
    batch.n_seq_id[i] = 1
    batch.logits[i] = False

batch.logits[batch.n_tokens - 1] = True

if llama_cpp.llama_decode(ctx, batch) != 0:
    print("Error decoding")

for i in range(n_parallel):
    llama_cpp.llama_kv_cache_seq_cp(ctx, 0, i, 0, batch.n_tokens)

streams = [b""] * n_parallel
i_batch = [batch.n_tokens - 1] * n_parallel

n_cur = batch.n_tokens
n_decode = 0

while n_cur <= n_len:
    batch.n_tokens = 0
    for i in range(n_parallel):
        if i_batch[i] < 0:
            continue

        logits_ = llama_cpp.llama_get_logits(ctx, i_batch[i])
        logits = np.ctypeslib.as_array(logits_, shape=(model_obj.n_vocab(),)).copy()
        scores = logits / temperature
        probs = np.exp(scores - np.max(scores))
        probs /= np.sum(probs)
        new_token_id = int(np.random.choice(
            range(model_obj.n_vocab()),
            p=probs,
        ))

        if new_token_id == llama_cpp.llama_token_eos(ctx) or n_cur == n_len:
            i_batch[i] = -1
            continue

        new_bytes = tokenizer.detokenize([new_token_id], prev_tokens=tokens[:n_cur])
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

    if llama_cpp.llama_decode(ctx, batch) != 0:
        print("Error decoding", flush=True)
        break
    print(n_cur)
    print(streams)