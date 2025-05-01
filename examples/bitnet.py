from pathlib import Path
from typing import List
import json, argparse, random, time, os
import tiktoken
from tiktoken.load import load_tiktoken_bpe
from extra.models.bitnet import Transformer, convert_from_gguf, fix_bf16
from tinygrad.nn.state import safe_load, torch_load, load_state_dict, get_parameters, gguf_load
from tinygrad import Tensor, dtypes, nn, Context, Device, GlobalCounters
from tinygrad.helpers import Profiling, Timing, DEBUG, colored, fetch, tqdm, getenv, CI, JIT

import sys

from tokenizers import Tokenizer as HFTokenizer

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

  def decode(self, toks): return self.model.decode([t for t in toks if t < self.num_base_tokens])
  def encode(self, text, allow_special=False):
    return self.model.encode(text, allowed_special="all" if allow_special else set(), disallowed_special=set())

# **** helper functions ****
def concat_weights(models, device=None):
  def convert(name) -> Tensor:
    disk_tensors: List[Tensor] = [model[name] for model in models]
    if len(disk_tensors) == 1 or len(disk_tensors[0].shape) == 1:
      return disk_tensors[0].to(device=device)
    axis = 1 if name.endswith((".attention.wo.weight", ".feed_forward.w2.weight")) else 0
    lazy_tensors = [data.to(device=device) for data in disk_tensors]
    return lazy_tensors[0].cat(*lazy_tensors[1:], dim=axis)
  return {name: convert(name) for name in {name: None for model in models for name in model}}

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

# **** quantized linears ****
class Int8Linear:
  def __init__(self, in_features, out_features, bias=False):
    assert bias == False
    self.weight = Tensor.ones(out_features, in_features, dtype=dtypes.int8)
    self.scale = Tensor.ones(out_features, dtype=dtypes.half)

  def __call__(self, x):
    return x.dot(self.weight.cast(self.scale.dtype).T*self.scale)

  @staticmethod
  def quantize(tensors, device, scale_dtype=dtypes.float16, quantize_embeds=False):
    new_tensors = {}
    for name,v in tensors.items():
      if "feed_forward" in name or "attention.w" in name or (quantize_embeds and "tok_embeddings.weight" in name):
        assert "weight" in name, name
        v = v.cast(scale_dtype)
        scale = v.abs().max(axis=1) / 127.0
        int8_weight = (v.T/scale).T.round().cast(dtype=dtypes.int8) # without round(), cast truncates -34.9 to -34
        new_tensors[name] = int8_weight
        new_tensors[name.replace('weight', 'scale')] = scale
        if isinstance(device, tuple):
          new_tensors[name].shard_(device, axis=-1)
          new_tensors[name.replace('weight', 'scale')].shard_(device, axis=None)
      else:
        new_tensors[name] = v
    if quantize_embeds: new_tensors.update({"output.weight": new_tensors["tok_embeddings.weight"], "output.scale": new_tensors["tok_embeddings.scale"]})
    return new_tensors

class Int8Embedding:
  def __init__(self, vocab_size:int, embed_size:int):
    self.vocab_sz, self.embed_sz = vocab_size, embed_size
    self.weight, self.scale = Tensor.ones(vocab_size, embed_size, dtype=dtypes.int8), Tensor.ones(vocab_size, dtype=dtypes.half)

  def __call__(self, idx:Tensor) -> Tensor:
    if not hasattr(self, 'arange'): self.arange = Tensor.arange(self.vocab_sz, requires_grad=False, device=self.weight.device).unsqueeze(-1)
    big_shp = idx.shape+(self.vocab_sz, self.embed_sz)
    arange, idx, vals = self.arange.expand(big_shp), idx.reshape(idx.shape+(1, 1)).expand(big_shp), (self.weight.cast(self.scale.dtype).T*self.scale).T
    return (arange == idx).mul(vals).sum(-2, dtype=vals.dtype)

def NF4Linear(block_size):
  _CODE = [
    -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453, -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
    0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224, 0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0,
  ]
  CODE = Tensor.stack(*[Tensor(c, dtype=dtypes.float16) for c in _CODE])
  class _NF4Linear:
    def __init__(self, in_features, out_features, bias=False):
      assert not bias, "bias not supported"
      self.in_features, self.out_features = in_features, out_features
      self.weight = Tensor.empty(int(out_features * in_features / 2), dtype=dtypes.uint8)
      self.scale = Tensor.empty(int(out_features * in_features / block_size), 1, dtype=dtypes.float16)

    def __call__(self, x: Tensor) -> Tensor:
      high_bits = self.weight
      low_bits = (self.weight * 2 ** 4).contiguous()
      unpacked = Tensor.stack(high_bits, low_bits, dim=-1).idiv(2 ** 4)
      unscaled = CODE[unpacked].to(x.device).reshape(-1, block_size) * self.scale
      return x.linear(unscaled.reshape(self.out_features, self.in_features).T)

    @staticmethod
    def quantize(state_dict: dict[str, Tensor], device, scale_dtype=dtypes.float16) -> dict[str, Tensor]:
      new_state_dict = {}
      for k, v in state_dict.items():
        if "feed_forward" in k or "attention.w" in k:
          grouped = v.reshape(-1, block_size)
          scale = (grouped.abs().max(axis=1, keepdim=True))
          coded = ((grouped / scale).unsqueeze(-1) - CODE.to(v.device)).abs().argmin(axis=-1).cast(dtypes.uint8).flatten()
          new_state_dict[k] = coded[::2] * 2 ** 4 + coded[1::2]
          new_state_dict[k.replace(".weight", ".scale")] = scale.cast(scale_dtype)
          if isinstance(device, tuple):
            new_state_dict[k].shard_(device, axis=-1)
            new_state_dict[k.replace('weight', 'scale')].shard_(device, axis=None)
        else:
          new_state_dict[k] = v
      return new_state_dict
  return _NF4Linear

MODEL_PARAMS = {
  "2B": {
    "args": {"dim": 2560, "n_heads": 20, "n_kv_heads": 5, "n_layers": 30, "norm_eps": 1e-5, "rope_theta": 500000, "vocab_size": 128256, "hidden_dim": 6912},
    "files": 1
  }
}

def build_transformer(model_path: Path, model_size="2B", quantize=None, scale_dtype=dtypes.float16, device=None, max_context=8192, load_weights=True):
  # build model
  linear, embedding, quantize_embeds = nn.BitLinear, nn.Embedding, False
  model = Transformer(**MODEL_PARAMS[model_size]["args"], linear=linear, embedding=embedding, max_context=max_context, jit=True)

  if not load_weights: return model
  # load weights
  if model_path.is_dir():
    if (model_path / "ggml-model-i2_s.gguf").exists(): weights = load(str(model_path / "ggml-model-i2_s.gguf"))
    else: weights = concat_weights([load(str(model_path / f"consolidated.{i:02d}.pth")) for i in range(MODEL_PARAMS[model_size]["files"])], device[0] if isinstance(device, tuple) else device)
  else:
    weights = load(str(model_path))
  if "token_embd.weight" in weights:
    weights = convert_from_gguf(weights, model)
  else:
    raise ValueError("Currently only support converting from GGUF")
  # weights = fix_bf16(weights)

  with Context(BEAM=0):
    # quantize
    if quantize == "float16": weights = {k:v.cast(quantize).contiguous() for k,v in weights.items()}
    elif quantize is not None:
      weights = linear.quantize(weights, device, scale_dtype, quantize_embeds)
      for _,v in weights.items(): v.realize()

    # shard
    if isinstance(device, tuple):
      for k,v in nn.state.get_state_dict(model).items():
        if 'scale' in k: v.shard_(device, axis=None)  # from quantized
        elif '.attention.' in k: v.shard_(device, axis=-1)
        elif '.feed_forward.w1.' in k: v.shard_(device, axis=0)
        elif '.feed_forward.w3.' in k: v.shard_(device, axis=0)
        elif '.feed_forward.' in k: v.shard_(device, axis=-1)
        elif 'tok_embeddings.weight' in k: v.shard_(device, axis=0)
        elif 'output.weight' in k: v.shard_(device, axis=0)
        else: v.shard_(device, axis=None)

    # Reshape weights that don't match the model's expected dimensions (for GGUF compatibility)
    # This handles the mismatch between expected dimensions and actual GGUF dimensions
    model_dict = nn.state.get_state_dict(model)
    for k, v in list(weights.items()):
      if k.endswith('.attention.wk.weight') or k.endswith('.attention.wv.weight'):
        if v.shape[0] != model_dict[k].shape[0]:
          print(f"Reshaping {k} from {v.shape} to match model's {model_dict[k].shape}")
          # Use padding or truncation to make the weights match
          if v.shape[0] < model_dict[k].shape[0]:
            # Pad with zeros if GGUF dimension is smaller
            padding = model_dict[k].shape[0] - v.shape[0]
            pad_tensor = Tensor.zeros(padding, v.shape[1], dtype=v.dtype)
            weights[k] = Tensor.cat(v, pad_tensor)
          else:
            # Truncate if GGUF dimension is larger
            weights[k] = v[:model_dict[k].shape[0]]
          
          # Also handle the corresponding scale tensor
          scale_key = k.replace('.weight', '.scale')
          if scale_key in weights and scale_key in model_dict:
            print(f"Reshaping {scale_key} from {weights[scale_key].shape} to match model's {model_dict[scale_key].shape}")
            if weights[scale_key].shape[0] < model_dict[scale_key].shape[0]:
              # Pad with ones if GGUF dimension is smaller
              padding = model_dict[scale_key].shape[0] - weights[scale_key].shape[0]
              pad_tensor = Tensor.ones(padding, dtype=weights[scale_key].dtype)
              weights[scale_key] = Tensor.cat(weights[scale_key], pad_tensor)
            else:
              # Truncate if GGUF dimension is larger
              weights[scale_key] = weights[scale_key][:model_dict[scale_key].shape[0]]
      
      # Handle feed-forward weights that need transposing
      if '.feed_forward.w2.weight' in k:
        if v.shape != model_dict[k].shape:
          print(f"Transposing {k} from {v.shape} to match model's {model_dict[k].shape}")
          weights[k] = v.transpose()
          
          # Also handle the corresponding scale tensor
          scale_key = k.replace('.weight', '.scale')
          if scale_key in weights and scale_key in model_dict:
            print(f"Reshaping {scale_key} from {weights[scale_key].shape} to match model's {model_dict[scale_key].shape}")
            # When we transpose the weights, we need to reorient the scales
            if weights[scale_key].shape[0] != model_dict[scale_key].shape[0]:
              weights[scale_key] = weights[scale_key][:model_dict[scale_key].shape[0]]
    
    # replace weights in model
    load_state_dict(model, weights, strict=False, consume=True)
  return model

# default settings
TEMPERATURE = 1.2
TOP_K = 0
TOP_P = 0.0
ALPHA_F = 0.0
ALPHA_P = 0.0

last_seen_toks = []
def prefill(model, toks, start_pos=0):
  global last_seen_toks

  # we can skip part of the prompt if it is the same as last and start_pos=0
  if start_pos == 0:
    for i, (a, b) in enumerate(zip(toks, last_seen_toks)):
      if a != b: break
    else: i = min(len(toks), len(last_seen_toks))
    start_pos += i
    last_seen_toks = toks
    toks = toks[i:]

  # prefill the model
  for tok in tqdm(toks):
    GlobalCounters.reset()
    model(Tensor([[tok]], device=device), start_pos, TEMPERATURE, TOP_K, TOP_P, ALPHA_F, ALPHA_P).realize()
    start_pos += 1
  return start_pos

if __name__ == "__main__":
  Tensor.no_grad = True

  parser = argparse.ArgumentParser(description="Run BitNet", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument("--download_model", action="store_true", help="Download the model specified by --model if it doesn't exist")
  parser.add_argument("--model", type=Path, help="Path to the model directory or file")
  parser.add_argument("--size", type=str, default="2B", choices=['2B'], help="Size of model to use")
  parser.add_argument("--shard", type=int, default=1, help="Number of shards to use")
  parser.add_argument("--quantize", type=str, choices=['int8', 'nf4', 'float16'], default=None, help="Quantize the weights to the specified type")
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
    fetch("https://huggingface.co/bofenghuang/Meta-Llama-3-8B/resolve/main/original/tokenizer.model", "tokenizer.model", subdir="bitnet") # bitnet uses the same tokenizer as llama3
    args.model = fetch("https://huggingface.co/microsoft/bitnet-b1.58-2B-4T-gguf/resolve/main/ggml-model-i2_s.gguf", "ggml-model-i2_s.gguf", subdir="bitnet")

  assert args.model is not None, "please provide --model option"

  if args.seed is not None: Tensor.manual_seed(args.seed)
  if args.benchmark: Tensor.manual_seed(42)
  print(f"seed = {Tensor._seed}")
  TEMPERATURE = args.temperature

  # Use absolute path for tokenizer model
  model_dir = args.model if args.model.is_dir() else args.model.parent
  tokenizer_path = (model_dir / "tokenizer.model").resolve()
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
  model = build_transformer(args.model, model_size=args.size, quantize=args.quantize, device=device)

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
      last_tok = toks[-1]
      while True:
        GlobalCounters.reset()
        token_tensor = model(Tensor([[last_tok]], device=device), start_pos, TEMPERATURE, TOP_K, TOP_P, ALPHA_F, ALPHA_P)
        token_id = token_tensor.item()
        print(f"  Generated token ID: {token_id}", file=sys.stderr) # Print ID to stderr
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
        start_pos = prefill(model, toks[:-1])
        print(f"[DEBUG] Prefill complete, start_pos = {start_pos}")
        last_tok = toks[-1]
        print(f"[DEBUG] Initial last_tok = {last_tok}")
        last_seen_toks.append(last_tok)
        print("[DEBUG] Beginning generation loop")
        while True:
          try:
            print(f"[DEBUG] Generating next token from last_tok={last_tok} at position {start_pos}")
            GlobalCounters.reset()
            print(f"[DEBUG] Creating input tensor with shape [[{last_tok}]]")
            input_tensor = Tensor([[last_tok]], device=device)
            print(f"[DEBUG] Calling model with input tensor")
            token_tensor = model(input_tensor, start_pos, TEMPERATURE, TOP_K, TOP_P, ALPHA_F, ALPHA_P)
            print(f"[DEBUG] Got token tensor with shape {token_tensor.shape}, extracting item")
            token_id = token_tensor.item()
            print(f"[DEBUG] Generated token ID: {token_id}")
            start_pos += 1
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

    start_pos = prefill(model, toks[:-1])
    last_tok = toks[-1]
    generated = ""
    for _ in range(20):
      GlobalCounters.reset()
      st = GlobalCounters.time_sum_s
      with Profiling(enabled=args.profile):
        with Timing("total ", on_exit=lambda x: f", {1e9/x:.2f} tok/s, {GlobalCounters.global_mem/x:.2f} GB/s, param {param_bytes/x:.2f} GB/s"):
          with Timing("enqueue in ", on_exit=(lambda et: (f", {(GlobalCounters.time_sum_s-st)*1e3:.2f} ms on GPU" if DEBUG>=2 else "")+
                      f", {GlobalCounters.global_ops*1e-9:.2f} GOPS, {GlobalCounters.global_mem*1e-9:.2f} GB"+
                      (f", {GlobalCounters.global_mem*1e-9/(GlobalCounters.time_sum_s-st):.2f} GB/s, param {param_bytes*1e-9/(GlobalCounters.time_sum_s-st):.2f} GB/s" if DEBUG>=2 else "")) if DEBUG else None):
            token_tensor = model(Tensor([[last_tok]], device=device), start_pos, TEMPERATURE, TOP_K, TOP_P, ALPHA_F, ALPHA_P)
            token_id = token_tensor.item()
            print(f"  Generated token ID: {token_id}", file=sys.stderr) # Print ID to stderr
          last_tok = token_id
      start_pos += 1
      generated += tokenizer.decode([last_tok])
      print(generated)
  else:
    prompt = [tokenizer.bos_id] + encode_message("system", "You are an helpful assistant.")

    start_pos = prefill(model, prompt)
    while True:
      toks = encode_message("user", input("Q: ")) + encode_role("assistant")

      start_pos = prefill(model, toks[:-1], start_pos=start_pos)
      last_tok = toks[-1]
      while True:
        GlobalCounters.reset()
        if args.timing or args.profile: print("")
        st = GlobalCounters.time_sum_s
        with Profiling(enabled=args.profile):
          with Timing("total ", enabled=args.timing, on_exit=lambda x: f", {1e9/x:.2f} tok/s, {GlobalCounters.global_mem/x:.2f} GB/s, param {param_bytes/x:.2f} GB/s"):
            with Timing("enqueue in ", on_exit=(lambda et: (f", {(GlobalCounters.time_sum_s-st)*1e3:.2f} ms on GPU" if DEBUG>=2 else "")+
                        f", {GlobalCounters.global_ops*1e-9:.2f} GOPS, {GlobalCounters.global_mem*1e-9:.2f} GB"+
                        (f", {GlobalCounters.global_mem*1e-9/(GlobalCounters.time_sum_s-st):.2f} GB/s, param {param_bytes*1e-9/(GlobalCounters.time_sum_s-st):.2f} GB/s" if DEBUG>=2 else "")) if DEBUG else None, enabled=args.timing):

              token_tensor = model(Tensor([[last_tok]], device=device), start_pos, TEMPERATURE, TOP_K, TOP_P, ALPHA_F, ALPHA_P)
              token_id = token_tensor.item()
              print(f"  Generated token ID: {token_id}", file=sys.stderr) # Print ID to stderr
            last_tok = token_id
        start_pos += 1
        if last_tok in tokenizer.stop_tokens: break
        print(tokenizer.decode([last_tok]), end="", flush=True)
      print(flush=True)
