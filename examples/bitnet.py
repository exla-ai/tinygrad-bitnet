from pathlib import Path
from typing import List
import json, argparse, random, time, os
import tinygrad; tinygrad.DEBUG = 2
import tiktoken
from tiktoken.load import load_tiktoken_bpe
from extra.models.bitnet import BitNetConfig, BitNetForCausalLM, convert_from_huggingface, build_transformer, debug
from extra.models.llama import fix_bf16
from tinygrad.nn.state import safe_load, torch_load, load_state_dict, get_parameters, gguf_load
from tinygrad import Tensor, dtypes, nn, Context, Device, GlobalCounters
from tinygrad.helpers import Profiling, Timing, DEBUG, colored, fetch, tqdm, getenv, CI, JIT

import sys

from tokenizers import Tokenizer as HFTokenizer

# Debug prints for device information
print("[DEBUG-DEVICE] Device.DEFAULT:", Device.DEFAULT)


class Tokenizer:
  pat_str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
  def __init__(self, model_path: str):
    mergeable_ranks = load_tiktoken_bpe(model_path)
    self.num_base_tokens = len(mergeable_ranks)
    special_tokens = [
      "<|begin_of_text|>",
      "<|end_of_text|>",
      "<|reserved_special_token_0|>",
      "<|reserved_special_token_1|>",
      "<|reserved_special_token_2|>",
      "<|reserved_special_token_3|>",
      "<|start_header_id|>",
      "<|end_header_id|>",
      "<|reserved_special_token_4|>",
      "<|eot_id|>",
    ] + [
      f"<|reserved_special_token_{i}|>"
      for i in range(5, 256 - 5)
    ]
    self.special_tokens = {token: len(mergeable_ranks) + i for i, token in enumerate(special_tokens)}

    self.model = tiktoken.Encoding(name=model_path, pat_str=self.pat_str, mergeable_ranks=mergeable_ranks, special_tokens=self.special_tokens)

  @property
  def bos_id(self): return self.special_tokens["<|begin_of_text|>"]

  @property
  def stop_tokens(self): return {self.special_tokens["<|end_of_text|>"], self.special_tokens["<|eot_id|>"]}

  def decode(self, toks): return self.model.decode(toks)
  def encode(self, text, allow_special=False):
    return self.model.encode(text, allowed_special="all" if allow_special else set(), disallowed_special=set())

def load(fn:str):
  if fn.endswith('.index.json'):
    with open(fn) as fp: weight_map = json.load(fp)['weight_map']
    parts = {n: load(str(Path(fn).parent / Path(n).name)) for n in set(weight_map.values())}
    return {k: parts[n][k] for k, n in weight_map.items()}
  elif fn.endswith(".gguf"):
    gguf_tensor = Tensor.empty(os.stat(fn).st_size, dtype=dtypes.uint8, device=f"disk:{fn}").to(Device.DEFAULT)
    return gguf_load(gguf_tensor)[1]
  elif fn.endswith(".safetensors"):
    return safe_load(fn)
  else:
    return torch_load(fn)


# default settings - adjusted for better diversity
TEMPERATURE = 0.8  # Increased from 0.0 to introduce diversity
TOP_K       = 40   # Increased from 1 to consider more tokens
TOP_P       = 0.95 # Using nucleus sampling to filter unlikely tokens
ALPHA_F     = 0.0  # Frequency penalty (unchanged)
ALPHA_P     = 0.0  # Presence penalty (unchanged)


last_seen_toks = []
def prefill(model, prompt_ids: List[int], past=None):
  if not prompt_ids:
    print("[PREFILL] Empty prompt_ids, returning past cache")
    return past
  print(f"[PREFILL] Processing {len(prompt_ids)} tokens: {prompt_ids}")
  prompt_ids = Tensor([prompt_ids], device=Device.DEFAULT)
  print(f"[PREFILL] Prompt tensor shape: {prompt_ids.shape}")
  # model returns (logits, new_cache) when no sample_args are given
  logits, past = model(prompt_ids, past)
  print(f"[PREFILL] Completed with logits shape: {logits.shape if hasattr(logits, 'shape') else 'N/A'}, cache size: {len(past) if past is not None else 'None'}")
  return past


if __name__ == "__main__":
  Tensor.no_grad = True

  parser = argparse.ArgumentParser(description="Run BitNet", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument("--download_model", action="store_true", help="Download the model specified by --model if it doesn't exist")
  parser.add_argument("--model", type=Path, help="Path to the model directory or file")
  parser.add_argument("--size", type=str, default="2B", choices=['2B'], help="Size of model to use")
  parser.add_argument("--shard", type=int, default=1, help="Number of shards to use")
  parser.add_argument("--no_api", action="store_true", help="Do not start the Gradio API")
  parser.add_argument("--host", type=str, default="0.0.0.0", help="Web server bind address")
  parser.add_argument("--port", type=int, default=7776, help="Web server port")
  parser.add_argument("--debug", action="store_true", help="Enable debug mode")
  parser.add_argument("--seed", type=int, help="Random seed")
  parser.add_argument("--temperature", type=float, default=0.85, help="Temperature")
  parser.add_argument("--benchmark", action="store_true", help="Run a benchmark")
  parser.add_argument("--timing", action="store_true", help="Print timing per token")
  parser.add_argument("--profile", action="store_true", help="Output profile data")
  parser.add_argument("--prompt", type=str, default=None, help="Prompt for generation")
  parser.add_argument("--count", type=int, default=1000, help="Max number of tokens to generate")
  parser.add_argument("--device", type=str, default=Device.DEFAULT, help="Device to use (e.g., METAL, CUDA, CPU)")
  args = parser.parse_args()


  # download_model is the default without a model passed in
  if args.download_model or not args.model:
    # bitnet uses the same tokenizer as llama3
    fetch("https://huggingface.co/bofenghuang/Meta-Llama-3-8B/resolve/main/original/tokenizer.model", "tokenizer.model", subdir="bitnet") 
    args.model = fetch("https://huggingface.co/microsoft/bitnet-b1.58-2B-4T/resolve/main/model.safetensors", "model.safetensors", subdir="bitnet")

  assert args.model is not None, "please provide --model option"

  if args.seed is not None: Tensor.manual_seed(args.seed)
  if args.benchmark: Tensor.manual_seed(42)
  print(f"seed = {Tensor._seed}")
  TEMPERATURE = args.temperature

  # Use absolute path for tokenizer model
  model_dir = args.model if args.model.is_dir() else args.model.parent
  tokenizer_path = (model_dir / "tokenizer.model").resolve()
  print(f"Tokenizer model path: {tokenizer_path}")
  print(f"Model path: {model_dir}")
  if not tokenizer_path.is_file():
    raise FileNotFoundError(f"Tokenizer model not found at expected path: {tokenizer_path}")
  
  tokenizer = Tokenizer(str(tokenizer_path))

  def encode_role(role: str):
    # Flatten the list concatenation
    return [tokenizer.special_tokens["<|start_header_id|>"], ] + tokenizer.encode(role) + [tokenizer.special_tokens["<|end_header_id|>"], ] + tokenizer.encode("\n\n")

  def encode_message(role: str, content: str):
    return encode_role(role) + tokenizer.encode(content.strip()) + [tokenizer.special_tokens["<|eot_id|>"], ]

  base_device = args.device
  device = tuple(f"{base_device}:{i}" for i in range(args.shard)) if args.shard > 1 else base_device
  model = build_transformer(args.model)[0]

  param_bytes = sum(x.lazydata.size * x.dtype.itemsize for x in get_parameters(model))
  print(f"ram used: {param_bytes/1e9:.2f} GB")


  if not args.no_api and not args.benchmark:
    from bottle import Bottle, request, response, HTTPResponse, abort, static_file
    app = Bottle()

    cors_headers = {
      "Access-Control-Allow-Origin": "*",
      "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS",
      "Access-Control-Allow-Headers": "Origin, Accept, Content-Type, X-Requested-With, X-CSRF-Token, Authorization",
      "Access-Control-Allow-Credentials": "true",
    }
    @app.hook("before_request")
    def handle_options():
      if request.method == "OPTIONS": raise HTTPResponse(headers=cors_headers)
    @app.hook("after_request")
    def enable_cors():
      for key, value in cors_headers.items(): response.set_header(key, value)

    @app.route("/<filename>")
    def server_static(filename): return static_file(filename, root=(Path(__file__).parent / "tinychat").as_posix())
    @app.route("/assets/<filename:path>")
    def server_assets(filename): return static_file(filename, root=(Path(__file__).parent / "tinychat" / "assets").as_posix())
    @app.route("/")
    def index():
      return static_file("index.html", root=(Path(__file__).parent / "tinychat").as_posix())

    @app.get("/v1/models")
    def models():
      return json.dumps([str(args.model)])

    @app.post("/v1/internal/token-count")
    def token_count():
      rjson = json.loads(request.body.read())
      return json.dumps(len(tokenizer.encode(rjson.get("text", ""))))
    @app.post("/v1/token/encode")
    def token_encode():
      rjson = json.loads(request.body.read())
      return json.dumps(tokenizer.encode(rjson.get("text", "")))

    @app.post("/v1/completions")
    def completions():
      rjson = json.loads(request.body.read())

      # check if we are streaming
      if rjson.get("stream", False):
        response.content_type = "text/event-stream"
        response.set_header("Cache-Control", "no-cache")
      else: abort(400, "streaming required")

      toks = [tokenizer.bos_id] + tokenizer.encode(rjson.get("prompt", ""), allow_special=True)


      start_pos = prefill(model, toks[:-1])

      h = model.norm(model.tok_embeddings(Tensor([tokens], device=device)))
      logits = model.output(h)[0,-1, :]

      topk_vals, topk_idxs = logits.topk(5)
      print("Top-5 candidate token IDs:", topk_idxs.tolist())
      print("Corresponding scores      :", topk_vals.tolist())
      print("Decoded strings          :", [tokenizer.decode([i]) for i in topk_idxs.tolist()])

      last_tok = toks[-1]
      while True:
        GlobalCounters.reset()
        token_tensor = model(Tensor([[last_tok]], device=device), start_pos, TEMPERATURE, TOP_K, TOP_P, ALPHA_F, ALPHA_P)
        token_id = token_tensor if isinstance(token_tensor, int) else token_tensor.item()
        print(f"  Generated token ID: {token_id}", file=sys.stderr) # Print ID to stderr
        if start_pos is None: 
            start_pos = len(toks) - 1
        start_pos += 1
        last_tok = token_id
        if last_tok in tokenizer.stop_tokens: break

        res = {
          "choices": [{
            "text": tokenizer.decode([last_tok]),
          }]
        }
        yield f"data: {json.dumps(res)}\n\n"

    @app.post("/v1/chat/token/encode")
    def chat_token_encode():
      rjson = json.loads(request.body.read())
      if "messages" not in rjson: abort(400, "messages required")
      toks = [tokenizer.bos_id]
      for message in rjson["messages"]:
        toks += encode_message(message["role"], message["content"])
      if len(rjson["messages"]) > 0 and message["role"] == "user":
        toks += encode_role("assistant")
      return json.dumps(toks)

    @app.post("/v1/chat/completions")
    def chat_completions():
      try:
        print("[DEBUG] Starting chat_completions handler")
        global last_seen_toks
        print(f"[DEBUG] stop_tokens = {tokenizer.stop_tokens}")
        rjson = json.loads(request.body.read())
        print(f"[DEBUG] Request JSON: {rjson}")
        if "messages" not in rjson: abort(400, "messages required")

        # check if we are streaming
        if rjson.get("stream", False):
          response.content_type = "text/event-stream"
          response.set_header("Cache-Control", "no-cache")
        else: abort(400, "streaming required")

        print("[DEBUG] Creating token sequence")
        toks = [tokenizer.bos_id]
        print(f"[DEBUG] Starting with BOS token: {tokenizer.bos_id}")
        for message in rjson["messages"]:
          print(f"[DEBUG] Processing message: {message['role']}")
          message_tokens = encode_message(message["role"], message["content"])
          print(f"[DEBUG] Encoded to {len(message_tokens)} tokens")
          toks += message_tokens
        # ensure that the last message was a user message
        if message["role"] != "user": abort(400, "last message must be a user message")
        print("[DEBUG] Adding assistant role")
        toks += encode_role("assistant")

        random_id = random.randbytes(16).hex()
        print(f"[DEBUG] Token sequence length: {len(toks)}")
  
        print("[DEBUG] Starting prefill with token sequence")
        kv_cache = prefill(model, toks[:-1])   
        print(f"[DEBUG] Prefill complete, kv_cache obtained")
        last_tok = toks[-1]
        seq_pos = len(toks) - 1                   # Numeric cursor
        print(f"[DEBUG] Initial last_tok = {last_tok}, initial seq_pos = {seq_pos}")
        last_seen_toks.append(last_tok)
        print("[DEBUG] Beginning generation loop")
        while True:
          try:
            print(f"[DEBUG] Generating next token from last_tok={last_tok} at position {seq_pos}")
            GlobalCounters.reset()
            print(f"[DEBUG] Creating input tensor with shape [[{last_tok}]]")
            input_tensor = Tensor([[last_tok]], device=device)
            print(f"[DEBUG] Calling model with input tensor and kv_cache")
            token_tensor, kv_cache, logits = model(input_tensor, # Pass and receive kv_cache
                                kv_cache,                # K/V cache as 2nd arg
                                TEMPERATURE, TOP_K, TOP_P, ALPHA_F, ALPHA_P)
            print(f"[DEBUG] Got token tensor {token_tensor}")
            print(f"[DEBUG] Got logits tensor with shape {logits.shape}, extracting item")
            token_id = token_tensor if isinstance(token_tensor, int) else token_tensor.item()
            print(f"[DEBUG] Generated token ID: {token_id}")
            seq_pos += 1 # Increment numeric cursor
            last_tok = token_id
            last_seen_toks.append(last_tok)
            if last_tok in tokenizer.stop_tokens: 
              print(f"[DEBUG] Found stop token: {last_tok}, breaking generation loop")
              break

            res = {
              "id": random_id,
              "object": "chat.completion.chunk",
              "created": int(time.time()),
              "model": str(args.model),
              "choices": [{
                "index": 0,
                "delta": {
                  "role": "assistant",
                  "content": tokenizer.decode([last_tok]),
                },
                "finish_reason": None,
              }]
            }
            yield f"data: {json.dumps(res)}\n\n"
          except Exception as e:
            print(f"[DEBUG] Error in generation loop: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            raise

        # Final response after generation is complete
        res = {
          "id": random_id,
          "object": "chat.completion.chunk",
          "created": int(time.time()),
          "model": str(args.model),
          "choices": [{
            "index": 0,
            "delta": {},
            "finish_reason": "stop",
          }]
        }
        yield f"data: {json.dumps(res)}\n\n"
      except Exception as e:
        print(f"[DEBUG] CRITICAL ERROR in chat_completions: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        raise

    app.run(host=args.host, port=args.port, debug=args.debug)
  elif args.benchmark:
    toks = [tokenizer.bos_id] + encode_message("user", "Hello.") + encode_role("assistant")
    print(f"[BENCHMARK] Initial tokens: {toks}")
    print(f"[BENCHMARK] Initial text: '{tokenizer.decode(toks)}'")
    
    # Prefill step
    print("[BENCHMARK] Starting prefill step...")
    kv_cache = prefill(model, toks[:-1])      # Renamed and captures K/V cache
    last_tok = toks[-1]
    seq_pos = len(toks) - 1                  # Numeric cursor
    print(f"[BENCHMARK] Prefill complete, cursor at position {seq_pos}, last token: {last_tok}")

    total_time = 0
    print(f"Benchmarking generation of {args.count} tokens...")
    # Exact number of tokens to generate in the loop is args.count
    for i in range(args.count):
        GlobalCounters.reset()
        st = time.perf_counter()
        
        print(f"\n[BENCHMARK] Token {i+1}/{args.count} generation")
        input_tensor = Tensor([[last_tok]], device=device)
        print(f"[BENCHMARK] Input tensor: shape={input_tensor.shape}, value={last_tok}")
        next_tok_val, kv_cache, _ = model( # Pass and receive kv_cache
            input_tensor,
            kv_cache,                # K/V cache as 2nd arg
            args.temperature, 
            args.top_k,
            args.top_p,
            args.alpha_function,
            args.alpha_presence
        )
        print(f"[BENCHMARK] Model output: {type(next_tok_val)}, value={'tensor' if isinstance(next_tok_val, Tensor) else next_tok_val}")
        next_tok = next_tok_val.item() if isinstance(next_tok_val, Tensor) else int(next_tok_val)
        print(f"[BENCHMARK] Generated token: {next_tok}")

        seq_pos += 1   # Increment numeric cursor

        last_tok = next_tok # Update for the next iteration
        
        # To be fair, only time after the first token if prefill is considered separate
        # However, typical benchmarks include the whole generation loop.
        # Let's time from the first generated token onwards if count > 0. Or all if simple. User's previous scripts timed after 1st iter.
        if i >= 0: # Start timing from the very first token generation in this loop. Or i > 0 if prefill timing is excluded. Let's include all loop iterations.
            total_time += (time.perf_counter() - st)
            
    if args.count > 0:
        print(f"Benchmark: {args.count} tokens in {total_time:.2f}s, {args.count/total_time:.2f} tok/s")
    else:
        print(f"Benchmark: 0 tokens generated.")

  else:
    prompt = [tokenizer.bos_id] + encode_message("system", "You are an helpful assistant.")
    print(f"[INIT] System prompt tokens: {prompt}")
    print(f"[INIT] System prompt text: '{tokenizer.decode(prompt)}'")

    kv_cache = prefill(model, prompt)      # Renamed and captures K/V cache
    seq_pos = len(prompt) - 1                  # Numeric cursor
    print(f"[INIT] Initialized with prompt length: {seq_pos+1} tokens")

    while True:
      user_input = input("Q: ")
      print(f"[CHAT] User input: '{user_input}'")
      toks = encode_message("user", user_input) + encode_role("assistant")
      print(f"[CHAT] Encoded user message tokens: {toks}")
      print(f"[CHAT] Encoded text: '{tokenizer.decode(toks)}'")

      print("[CHAT] Starting prefill with user message...") 
      kv_cache = prefill(model, toks[:-1], kv_cache)      # Pass and receive kv_cache
      last_tok = toks[-1]
      seq_pos = len(toks) - 1                  # Numeric cursor
      print(f"[CHAT] Prefill complete, cursor at position {seq_pos}, last token: {last_tok}")

      while True:
        GlobalCounters.reset()
        if args.timing or args.profile: print("")
        st = GlobalCounters.time_sum_s
        with Profiling(enabled=args.profile):
          with Timing("total ", enabled=args.timing, on_exit=lambda x: f", {1e9/x:.2f} tok/s, {GlobalCounters.global_mem/x:.2f} GB/s, param {param_bytes/x:.2f} GB/s"):
            with Timing("enqueue in ", on_exit=(lambda et: (f", {(GlobalCounters.time_sum_s-st)*1e3:.2f} ms on GPU" if DEBUG>=2 else "")+
                        f", {GlobalCounters.global_ops*1e-9:.2f} GOPS, {GlobalCounters.global_mem*1e-9:.2f} GB"+
                        (f", {GlobalCounters.global_mem*1e-9/(GlobalCounters.time_sum_s-st):.2f} GB/s, param {param_bytes*1e-9/(GlobalCounters.time_sum_s-st):.2f} GB/s" if DEBUG>=2 else "")) if DEBUG else None, enabled=args.timing):

              print(f"\n[GENERATION] Processing token at position {seq_pos}, input token: {last_tok}")
              input_tensor = Tensor([[last_tok]], device=device)
              print(f"[GENERATION] Input tensor shape: {input_tensor.shape}")
              token_tensor, kv_cache, _ = model( # Pass and receive kv_cache
                  input_tensor,
                  kv_cache,                # K/V cache as 2nd arg
                  args.temperature, 
                  args.top_k,
                  args.top_p,
                  args.alpha_function,
                  args.alpha_presence
              )
              print(f"[GENERATION] Raw model output: type={type(token_tensor)}, value={token_tensor if not isinstance(token_tensor, Tensor) else 'tensor'}")
              token_id = token_tensor if isinstance(token_tensor, int) else token_tensor.item()
              print(f"[GENERATION] Generated token ID: {token_id}", file=sys.stderr) # Print ID to stderr
              print(f"[GENERATION] Generated text segment: '{tokenizer.decode([token_id])}'")
            last_tok = token_id
        seq_pos += 1   # Increment numeric cursor
        if last_tok in tokenizer.stop_tokens: break
        print(tokenizer.decode([last_tok]), end="", flush=True)
      print(flush=True)
