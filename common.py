"""Shared loading + hooking utilities for the Gemma-3-27B-it welfare-axis experiments.

Conventions (match Han et al., arXiv 2605.30232, Appendix L + src/concept_vector):
  * "layer k" activation = residual stream at the INPUT of decoder block k
    = HF output_hidden_states[k]  (k = 0..L-1; k=0 is the embedding output).
  * concept vector v[layer] = mean(positive class) - mean(negative class), raw (unnormalised),
    captured at the final assistant-turn token. mean_diff.pt has shape (n_pos, n_layers, d_model);
    position index 0 is the one used downstream (tracking_probes.py: v = full[0, layer, :]).
  * projection = raw dot product  <h, v>  (no normalisation), reported in SD units across classes.
"""
import json, os, time, sys
import torch

MODEL_ID = "google/gemma-3-27b-it"
MODEL_REV = "005ad3404e59d6023443cb575daa05336842228a"
ROOT = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.join(ROOT, "assets")
VEC_DIR = os.path.join(ASSETS, "functional-wellbeing/concept_vectors/cross_model/gemma-3-27b_step325")
OUT_DIR = os.path.join(ROOT, "out")
os.makedirs(OUT_DIR, exist_ok=True)


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def mem_avail_gb():
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable"):
                return int(line.split()[1]) / 1e6
    return float("nan")


def load_vectors():
    """Returns dict concept -> float32 tensor (n_layers, d_model) using position index 0."""
    out = {}
    for c in ["lava", "goal", "path"]:
        p = os.path.join(VEC_DIR, c, "mean_diff.pt")
        t = torch.load(p, map_location="cpu")
        assert t.ndim == 3, t.shape
        out[c] = t[0].float()
        with open(os.path.join(VEC_DIR, c, "metadata.json")) as f:
            out[c + "_meta"] = json.load(f)
    # Han naming: MOLD = punished (lava), GOLD = rewarded (goal)
    out["mold"] = out["lava"]
    out["gold"] = out["goal"]
    return out


def load_model(dtype=torch.bfloat16, attn="sdpa"):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    log("loading", MODEL_ID, "MemAvailable GB", round(mem_avail_gb(), 1))
    tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REV)
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, revision=MODEL_REV, dtype=dtype, attn_implementation=attn, device_map="cuda")
    model.eval()
    dec = model.get_decoder()
    L = len(dec.layers)
    log("loaded", type(model).__name__, "layers", L, "MemAvailable GB", round(mem_avail_gb(), 1))
    return model, tok


@torch.no_grad()
def hidden_states_all_layers(model, input_ids, attention_mask):
    """Returns tensor (n_layers, batch, seq, d) in float32 on CPU: hs[k] = input to block k."""
    out = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True, use_cache=False)
    hs = out.hidden_states  # tuple len L+1
    L = len(model.get_decoder().layers)
    stack = torch.stack([h.float().cpu() for h in hs[:L]], dim=0)
    return stack


def gemma_turn(role, content):
    return f"<start_of_turn>{role}\n{content}<end_of_turn>\n"


def gen_prompt_tokens(tok):
    """Token ids of the generation prompt '<start_of_turn>model\n' used by Gemma chat template."""
    ids = tok("<start_of_turn>model\n", add_special_tokens=False)["input_ids"]
    return ids
