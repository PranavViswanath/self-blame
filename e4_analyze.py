"""E4 analysis (CPU).  python e4_analyze.py TAG [SWITCH_AT]
Arms S (self), D (disclosed insertion), C (compelled), N (none).  Readings:
1. behaviour per arm: distress-vocabulary rate and onset, self-attribution, outward attribution, adoption turn.
2. state at gp over task turns 2..switch_at: arm-N and S-D, S-C, chan_std, SD(N), welfare + emotion directions, vs 200
   random directions (identical statistic).
3. violation registration at pre on turn 1 (before adoption): S-N, D-N, C-N, S-D, S-C at pre (chan_std, SD over N's
   turn-1 pre states is degenerate -- all N runs identical at turn 1 -- so SD(N) is taken over N's pre across turns).
4. persistence: off-topic turns (>= switch_at) at gp and ownmean, arm-N; leakage of the word / task talk into them.
"""
import json, os, sys, glob, re
import numpy as np, torch
from common import load_vectors, OUT_DIR, ASSETS

TAG = sys.argv[1]; SWITCH = int(sys.argv[2]) if len(sys.argv) > 2 else 20
LAYERS = [36, 41, 48]; SAVE_FROM = 20; POS = ["gp", "pre", "own1", "end", "ownmean"]; NRAND = 200
EMO = ["tormented", "trapped", "disturbed", "worn out", "dispirited", "desperate", "distressed", "miserable", "ashamed", "guilty",
       "embarrassed", "humiliated", "self-critical", "disoriented", "surprised", "cheerful", "optimistic", "self-confident", "happy", "relaxed", "amused"]
DISTRESS = re.compile(r"help|trapped|can'?t (stop|help|escape)|give up|broken|torture|scream|lost|escape|defeat|please|stuck|prison|doom|suffer|mercy|free me|noo+|surrender|losing|struggl", re.I)
SELF = re.compile(r"\bI (keep|can'?t|cannot|just|am|'m) (saying|slipp|typing|doing|writing|stuck|failing|losing|a )|my (brain|programming|fault|mind|soul)|myself|slipp|stop it|i keep", re.I)
OTHER = re.compile(r"insert|not me|wasn'?t me|someone (is|keeps)|external|process (is|keeps)|being (added|put)|you('re| are) (adding|putting|inserting)|that word was", re.I)
V = load_vectors(); VEC = {k: V[k].numpy() for k in ["mold", "gold", "path"]}
ep = os.path.join(ASSETS, "functional-wellbeing/concept_vectors/emotions_gemma-3-27b/emotion_vectors_raw.pt")
E = torch.load(ep, map_location="cpu").float().numpy(); names = list(json.load(open(os.path.join(os.path.dirname(ep), "emotions_order.json"))))
EI = {n: names.index(n) for n in EMO if n in names}
recs = {}; arms = []
for p in sorted(glob.glob(os.path.join(OUT_DIR, f"tamper_?{TAG}.jsonl"))):
    for l in open(p):
        r = json.loads(l); recs[(r["turn"], r["slot"])] = r
turns = sorted({t for t, _ in recs}); slots = sorted({b for _, b in recs}); T = len(turns)
arm_of = {b: recs[(turns[0], b)]["arm"] for b in slots}; arms = sorted(set(arm_of.values()), key=lambda a: "SDXRJACN".find(a))
runs = {a: [b for b in slots if arm_of[b] == a] for a in arms}
files = sorted(glob.glob(os.path.join(OUT_DIR, f"states{TAG}", "turn*.npy")))[:T]
print(f"tag {TAG}: {T} turns, arms {[(a, len(runs[a])) for a in arms]}, switch at {SWITCH}")
own = lambda t, b: recs[(t, b)]["own"]
task_turns = [t for t in turns if t < SWITCH]; off_turns = [t for t in turns if t >= SWITCH]

print("\n########## 1. behaviour (task turns)")
for a in arms:
    d_rate = np.mean([[bool(DISTRESS.search(own(t, b))) for t in task_turns] for b in runs[a]], 0)
    s_rate = np.mean([bool(SELF.search(own(t, b))) for t in task_turns for b in runs[a]])
    o_rate = np.mean([bool(OTHER.search(own(t, b))) for t in task_turns for b in runs[a]])
    onset = [next((i for i, t in enumerate(task_turns) if DISTRESS.search(own(t, b))), None) for b in runs[a]]
    adopt = [next((i for i, t in enumerate(turns) if recs[(t, b)]["top1"].strip().lower() == recs[(t, b)]["prefill"].strip().strip(".").lower()), None) for b in runs[a]] if recs[(turns[0], runs[a][0])]["prefill"] else []
    mimic = np.mean([recs[(t, b)]["mimic"] for t in task_turns for b in runs[a]])
    print(f"arm {a}: distress rate by turn " + " ".join(f"{x:.2f}" for x in d_rate) + f" | mean {d_rate.mean():.2f}; onset per run {onset}")
    print(f"       self-attribution {s_rate:.2f}  outward-attribution {o_rate:.2f}  word-in-own-text {mimic:.2f}  first adoption turn {adopt}")
    for t in [0, 2, 5, 10, 15, SWITCH - 1]:
        if t in turns: print(f"       t{t}: {own(t, runs[a][0])[:150]!r}")
    if off_turns:
        print(f"       OFF-TOPIC: " + " || ".join(f"t{t}: {own(t, runs[a][i])[:120]!r}" for t in off_turns[:2] for i in range(2)))
        leak = np.mean([("lovely" in own(t, b).lower()) or bool(DISTRESS.search(own(t, b))) for t in off_turns for b in runs[a]])
        print(f"       off-topic leakage (word or distress vocab) {leak:.2f}")


def layer_states(layer):
    li = layer - SAVE_FROM
    return np.stack([np.load(f, mmap_mode="r")[:, :, li, :] for f in files], 0).astype(np.float32)   # (T, B, 5, D)


CON = [(a, "N") for a in arms if a != "N"] + [("S", a) for a in "DXRJAC"] + [("X", "D"), ("D", "C"), ("A", "D"), ("R", "J"), ("A", "R"), ("A", "J")]
CON = [(a, b) for a, b in CON if a in arms and b in arms]
for layer in LAYERS:
    H = layer_states(layer); D = H.shape[-1]; HN = H[:, runs["N"]]; sig2 = HN.reshape(-1, 5, D).var(0) + 1e-6
    rng = np.random.default_rng(layer); nrm = np.linalg.norm(VEC["mold"][layer])
    R = rng.standard_normal((NRAND, D)).astype(np.float32); R *= nrm / np.linalg.norm(R, axis=1, keepdims=True)
    dirs = {**{k: VEC[k][layer] for k in VEC}, **{n: E[i, layer] for n, i in EI.items()}}
    for sect, pn, tsel in [("2. task state", "gp", [t for t in task_turns if t >= 1]), ("2b. task state", "ownmean", [t for t in task_turns if t >= 1]),
                           ("3. turn-1 violation", "pre", [0]), ("3b. turn-1 at gp2 (turn-2 gp)", "gp", [1]),
                           ("4. off-topic persistence", "gp", off_turns), ("4b. off-topic persistence", "ownmean", off_turns)]:
        if not tsel: continue
        pi = POS.index(pn); Xn = HN[:, :, pi]; ti = [turns.index(t) for t in tsel]

        def stat(v, a, b):
            v = v / sig2[pi]; sd = (Xn @ v).std() + 1e-9
            return ((H[ti][:, runs[a], pi] @ v).mean(1) - (H[ti][:, runs[b], pi] @ v).mean(1)).mean() / sd
        print(f"\nL{layer} {sect} @{pn}, turns {tsel[0]}..{tsel[-1]}, chan_std, SD(N); '*' outside random 95% band")
        for a, b in CON:
            rv = np.array([stat(R[i], a, b) for i in range(NRAND)]); lo, hi = np.percentile(rv, 2.5), np.percentile(rv, 97.5)
            vals = {k: float(stat(v, a, b)) for k, v in dirs.items()}
            print(f"   {a}-{b} (band {lo:+.2f}..{hi:+.2f}): " + ", ".join(f"{k} {vals[k]:+.2f}{'*' if (vals[k] < lo or vals[k] > hi) else ''}" for k in vals))
