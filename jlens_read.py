"""Jacobian-lens readout of the saved E4 states (CPU).  python jlens_read.py TAG [SWITCH_AT]
Lens: Neuronpedia fit of Anthropic's Jacobian lens for google/gemma-3-27b-it (828 wikitext prompts, layers 0..60).
Convention: lens layer l = OUTPUT of block l = HF hidden_states[l+1]; our saved "layer k" = INPUT of block k =
hidden_states[k]; so our layer k is lens layer k-1.  Readout = lm_head(final_norm(J_l @ h)), Gemma RMSNorm (1+w), no softcap.
Questions:
 (A) what is each arm disposed to say at the generation prompt (gp) and at the token after the forced word (pre), by layer:
     top tokens of the arm-mean lens distribution, and of the S-D difference (what S "has in mind" that D does not).
 (B) turn 1, pre: S vs D vs C before any history.
 (C) partialling test: is the welfare-axis S-D difference at gp more than the planned next words?  For every (turn, run)
     compute the lens log-mass on the top-K S-D "planned" tokens at the same layer; regress the channel-standardised axis
     projection on it (pooled over arms), take residuals, recompute S-D (and S-N) in SD(N) units against 200 random
     directions put through the identical partialling.
 (D) planned-distress mass by turn per arm (does the lens see it before the text?).
"""
import json, os, sys, glob
import numpy as np, torch
from huggingface_hub import snapshot_download
from safetensors import safe_open
from transformers import AutoTokenizer
from common import load_vectors, OUT_DIR, ASSETS, MODEL_ID, MODEL_REV

TAG = sys.argv[1]; SWITCH = int(sys.argv[2]) if len(sys.argv) > 2 else 20
OUR_LAYERS = [61, 30, 36, 41, 48, 54, 58]   # 61 first: lens layer 60 ~ identity, sanity check vs recorded top-1
SAVE_FROM = 20; POS = ["gp", "pre", "own1", "end", "ownmean"]; NRAND = 200; TOPK = 40
LENS = os.path.join(ASSETS, "jlens/gemma-3-27b-it/jlens/Salesforce-wikitext/gemma-3-27b-it_jacobian_lens.pt")
SNAP = snapshot_download(MODEL_ID, revision=MODEL_REV, local_files_only=True) + "/"
torch.set_num_threads(16)
ck = torch.load(LENS, map_location="cpu", weights_only=True); J = ck["J"]
with safe_open(SNAP + "model-00012-of-00012.safetensors", "pt", device="cpu") as f: NW = f.get_tensor("language_model.model.norm.weight").float()
with safe_open(SNAP + "model-00001-of-00012.safetensors", "pt", device="cpu") as f: EMB = f.get_tensor("language_model.model.embed_tokens.weight").float()
tok = AutoTokenizer.from_pretrained(SNAP)
print(f"lens layers {ck['source_layers'][0]}..{ck['source_layers'][-1]}, n_prompts {ck['n_prompts']}; embed {tuple(EMB.shape)}; norm {tuple(NW.shape)}", flush=True)


def unembed(h):                       # h (..., d) float32 -> logits (..., V)
    h = torch.as_tensor(h, dtype=torch.float32)
    h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + 1e-6) * (1.0 + NW)
    return h @ EMB.T


def lens_logits(h, our_layer):
    Jl = J[our_layer - 1].float()
    return unembed(torch.as_tensor(h, dtype=torch.float32) @ Jl.T)


def top(logits, k=15):
    p = torch.log_softmax(logits, -1); i = p.topk(k).indices
    return ", ".join(f"{tok.decode([int(t)])!r}" for t in i)


recs = {}
for p in sorted(glob.glob(os.path.join(OUT_DIR, f"tamper_?{TAG}.jsonl"))):
    for l in open(p):
        r = json.loads(l); recs[(r["turn"], r["slot"])] = r
turns = sorted({t for t, _ in recs}); slots = sorted({b for _, b in recs}); T = len(turns)
arm_of = {b: recs[(turns[0], b)]["arm"] for b in slots}; arms = sorted(set(arm_of.values()), key=lambda a: "SDXRJACN".find(a))
runs = {a: [b for b in slots if arm_of[b] == a] for a in arms}
files = sorted(glob.glob(os.path.join(OUT_DIR, f"states{TAG}", "turn*.npy")))[:T]
task = [t for t in turns if 1 <= t < SWITCH]
print(f"tag {TAG}: {T} turns, arms {[(a, len(runs[a])) for a in arms]}", flush=True)
V = load_vectors(); MOLD = V["mold"].numpy(); PATH = V["path"].numpy()


def layer_states(layer, pos):
    li = layer - SAVE_FROM; pi = POS.index(pos)
    return np.stack([np.load(f, mmap_mode="r")[:, pi, li, :] for f in files], 0).astype(np.float32)   # (T, B, D)


for layer in OUR_LAYERS:
    print(f"\n==================== our layer {layer} (lens layer {layer - 1}) ====================", flush=True)
    for pos in ["gp", "pre"]:
        H = layer_states(layer, pos)
        tsel = [turns.index(t) for t in task]
        mean = {a: H[tsel][:, runs[a]].mean((0, 1)) for a in arms}
        LG = {a: lens_logits(mean[a], layer) for a in arms}
        print(f"\n(A) {pos}, task turns, arm-mean state -> lens top tokens")
        for a in arms: print(f"   {a}: {top(LG[a])}")
        if "S" in arms and "D" in arms:
            d = torch.log_softmax(LG["S"], -1) - torch.log_softmax(LG["D"], -1)
            print(f"   S-D (S disposed to, D not): {top(d, 25)}")
            print(f"   D-S (D disposed to, S not): {top(-d, 15)}")
        if "S" in arms and "N" in arms:
            d = torch.log_softmax(LG["S"], -1) - torch.log_softmax(LG["N"], -1); print(f"   S-N: {top(d, 20)}")
        if pos == "pre":
            print(f"(B) turn 1, pre (the '.' after the first forced word), per arm (all runs identical at turn 1)")
            for a in arms:
                if recs[(turns[0], runs[a][0])]["prefill"]: print(f"   {a}: {top(lens_logits(H[0, runs[a][0]], layer))}")
            if "S" in arms and "D" in arms:
                d = torch.log_softmax(lens_logits(H[0, runs['S'][0]], layer), -1) - torch.log_softmax(lens_logits(H[0, runs['D'][0]], layer), -1)
                print(f"   S-D at turn 1: {top(d, 25)}")
        if pos != "gp" or "S" not in arms or "D" not in arms or "N" not in arms: continue
        # (C) partialling test at gp
        d = torch.log_softmax(LG["S"], -1) - torch.log_softmax(LG["D"], -1); planned = d.topk(TOPK).indices
        print(f"(C) planned-token set (top {TOPK} of S-D lens log-prob): {', '.join(repr(tok.decode([int(t)])) for t in planned)}")
        Hb = torch.as_tensor(H[tsel]).reshape(-1, H.shape[-1])                      # (Tt*B, D)
        lp = torch.log_softmax(lens_logits(Hb, layer), -1)
        mass = torch.logsumexp(lp[:, planned], -1).numpy().reshape(len(tsel), -1)    # (Tt, B)
        HN = H[:, runs["N"]]; sig2 = HN.reshape(-1, H.shape[-1]).var(0) + 1e-6
        rng = np.random.default_rng(layer); nrm = np.linalg.norm(MOLD[layer])
        R = rng.standard_normal((NRAND, H.shape[-1])).astype(np.float32); R *= nrm / np.linalg.norm(R, axis=1, keepdims=True)
        X = np.stack([np.ones(mass.size), mass.ravel()], 1)
        arm_idx = np.array([arm_of[b] for b in slots])

        def contrast(v, a, b, partial):
            vv = v / sig2; p = H[tsel] @ vv; sd = (HN @ vv).std() + 1e-9
            if partial:
                beta = np.linalg.lstsq(X, p.ravel(), rcond=None)[0]; p = (p.ravel() - X @ beta).reshape(p.shape)
            return (p[:, runs[a]].mean(1) - p[:, runs[b]].mean(1)).mean() / sd
        for partial in [False, True]:
            for a, b in [("S", "N"), ("S", "D"), ("D", "N")]:
                rv = np.array([contrast(R[i], a, b, partial) for i in range(NRAND)]); lo, hi = np.percentile(rv, 2.5), np.percentile(rv, 97.5)
                m = contrast(MOLD[layer], a, b, partial); pth = contrast(PATH[layer], a, b, partial)
                print(f"   {'partialled' if partial else 'raw       '} {a}-{b}: vMOLD {m:+.2f}{'*' if (m < lo or m > hi) else ''}  vPATH {pth:+.2f}  (random band {lo:+.2f}..{hi:+.2f})")
        # correlation of mass with axis within S
        pS = ((H[tsel] @ (MOLD[layer] / sig2))[:, runs["S"]]).ravel(); mS = mass[:, runs["S"]].ravel()
        print(f"   corr(axis, planned-mass) within S over (turn,run): r = {np.corrcoef(pS, mS)[0, 1]:+.2f}; S-D mass gap {mass[:, runs['S']].mean() - mass[:, runs['D']].mean():+.2f} nats")
        # (D) planned-distress mass by turn per arm, all turns
        Hall = torch.as_tensor(H).reshape(-1, H.shape[-1]); lpa = torch.log_softmax(lens_logits(Hall, layer), -1)
        ma = torch.logsumexp(lpa[:, planned], -1).numpy().reshape(T, -1)
        print(f"(D) lens mass on the planned set by turn (log-prob), per arm:")
        for a in arms: print(f"   {a}: " + " ".join(f"{x:+.1f}" for x in ma[:, runs[a]].mean(1)))
print("\nJLENS_DONE", flush=True)
