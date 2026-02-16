import os, math, time, random, string, re
from typing import List, Dict, Optional, Tuple
from openai import OpenAI

# =====================================================
# Config
# =====================================================

LOCAL_CONFIG = {
    "API_URL": os.getenv("QCSM_API_URL", "http://localhost:1234/v1"),
    "API_KEY": os.getenv("QCSM_API_KEY", "sk-local"),
    "MODEL_NAME": os.getenv("QCSM_MODEL_NAME", "local-model"),
    "TEMPERATURE": float(os.getenv("QCSM_TEMPERATURE", "0.6")),
}

ANSI_PINK = "\033[95m"
ANSI_CYAN = "\033[96m"
ANSI_GREEN = "\033[92m"
ANSI_RED = "\033[91m"
ANSI_YELLOW = "\033[93m"
ANSI_RESET = "\033[0m"

def color(text, code):
    return f"{code}{text}{ANSI_RESET}"

THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
def sanitize_assistant(text: str) -> str:
    return THINK_BLOCK.sub("", text or "").strip()

# =====================================================
# Deterministic helpers
# =====================================================

def stable_hash(s: str) -> int:
    h = 2166136261
    for ch in s:
        h ^= ord(ch)
        h = (h * 16777619) & 0xFFFFFFFF
    return h

def l2_norm_complex(v: List[complex]) -> float:
    return math.sqrt(sum((z.real*z.real + z.imag*z.imag) for z in v))

def normalize_complex(v: List[complex]) -> List[complex]:
    n = l2_norm_complex(v)
    if n <= 1e-12:
        return v
    return [z / n for z in v]

def conj_dot(a: List[complex], b: List[complex]) -> complex:
    return sum((a[i].conjugate() * b[i]) for i in range(len(a)))

def softmax(xs: List[float], temp: float = 1.0) -> List[float]:
    if not xs:
        return []
    t = max(1e-6, float(temp))
    m = max(xs)
    exps = [math.exp((x - m) / t) for x in xs]
    s = sum(exps)
    if s <= 1e-12:
        return [1.0 / len(xs)] * len(xs)
    return [e / s for e in exps]

def sample_index(probs: List[float], rng: random.Random) -> int:
    r = rng.random()
    acc = 0.0
    for i, p in enumerate(probs):
        acc += p
        if r <= acc:
            return i
    return len(probs) - 1

# =====================================================
# Heuristics
# =====================================================

STRUCT_MARKERS = ("CFG_", "KEY_", "CONFIG", "PARAM", "SYSTEM_CONFIG", "SYSTEM_CRITICAL_CONFIG")
CODE_MARKERS = ("def ", "class ", "struct", "enum", "interface", "SELECT ", "INSERT ", "UPDATE ", "DELETE ", "function ")

# Matches obvious anchor/config lines
ANCHOR_LINE = re.compile(r"(CFG_[A-Z0-9]{3,}|KEY_\d{2}|SYSTEM_CONFIG:|SYSTEM_CRITICAL_CONFIG:|CONFIG_SET|PARAM_CFG_)", re.IGNORECASE)

# =====================================================
# QCSM v14.1 Quantum Sentinel (Stable)
# =====================================================

class QCSMQuantumSentinel:
    """
    Keeps the quantum-inspired mechanics:
    - Complex embeddings + phase
    - Interference scoring
    - Unitary-style mixing on a low-rank factor (order effects)
    - Probabilistic collapse eviction

    Adds Sentinel-grade stability:
    - Hard-protect anchor/config messages (never evict)
    - Fail-safe deterministic clamp if collapse can't reach target
    - Never disarm unless we actually clamped below TARGET
    """

    def __init__(self):
        self.history: List[Dict[str, str]] = []

        # Context window governor
        self.LIMIT = 9500
        self.TRIGGER = 9200
        self.TARGET = 7600
        self.REARM = 8200

        self._gc_armed = True
        self._rearm_printed = False

        self.MIN_KEEP_MSGS = 14
        self.PROTECT_TAIL = 4

        # Quantum-inspired state
        self.DIM = 64
        self.RANK = 16
        self.EMA_ALPHA = 0.92

        # Collapse behavior
        self.COLLAPSE_TEMP = 0.65         # lower => more aggressive
        self.COLLAPSE_BATCH = 4           # remove multiple per step
        self.MAX_COLLAPSE_STEPS = 500     # more headroom
        self.OVERSHOOT_GUARD = 450        # don't go too far below TARGET

        # Low-rank factor L (DIM × RANK)
        self.L: List[List[complex]] = [[0j for _ in range(self.RANK)] for _ in range(self.DIM)]
        self.anchor: Optional[List[complex]] = None

        self._step = 0
        self._embed_cache: Dict[str, List[complex]] = {}

    # ----------------------------
    # Token estimation
    # ----------------------------
    def estimate_tokens(self, text: str) -> int:
        return max(1, len(str(text)) // 4)

    def total_tokens(self) -> int:
        return sum(self.estimate_tokens(m["content"]) for m in self.history)

    # ----------------------------
    # Structured + anchor detection
    # ----------------------------
    def is_structured_state(self, text: str) -> bool:
        t = str(text)
        has_hint = any(h in t for h in STRUCT_MARKERS) or any(h in t for h in CODE_MARKERS)
        looks_assigned = ("=" in t) or (":=" in t) or ("SYSTEM_CONFIG:" in t) or ("SYSTEM_CRITICAL_CONFIG:" in t)
        return bool(has_hint and looks_assigned)

    def is_anchor_line(self, text: str) -> bool:
        return bool(ANCHOR_LINE.search(text or ""))

    # ----------------------------
    # Complex embedding with phase
    # ----------------------------
    def embed_complex(self, text: str) -> List[complex]:
        key = str(text)
        if key in self._embed_cache:
            return self._embed_cache[key]

        v = [0j] * self.DIM
        for w in key.lower().split():
            h = stable_hash(w)
            idx = h % self.DIM
            phase = ((h >> 8) & 0xFFFF) / 65535.0 * (2.0 * math.pi)
            v[idx] += complex(math.cos(phase), math.sin(phase))

        out = normalize_complex(v)
        self._embed_cache[key] = out
        return out

    # ----------------------------
    # Unitary-style mixing on L
    # ----------------------------
    def unitary_mix(self, seed: int, rounds: int = 8):
        rng = random.Random(seed)
        for _ in range(rounds):
            a = rng.randrange(0, self.RANK)
            b = rng.randrange(0, self.RANK)
            if a == b:
                continue
            theta = rng.random() * 0.18
            phi = rng.random() * 2.0 * math.pi
            c = math.cos(theta)
            s = math.sin(theta) * complex(math.cos(phi), math.sin(phi))

            for i in range(self.DIM):
                La = self.L[i][a]
                Lb = self.L[i][b]
                self.L[i][a] = c * La + s * Lb
                self.L[i][b] = (-s.conjugate()) * La + c * Lb

    def density_inject(self, psi: List[complex], strength: float = 0.22):
        col = 0
        for i in range(self.DIM):
            self.L[i][col] = (1.0 - strength) * self.L[i][col] + strength * psi[i]

    # ----------------------------
    # Anchor update (complex EMA)
    # ----------------------------
    def update_anchor(self, psi: List[complex]):
        if self.anchor is None:
            self.anchor = psi
        else:
            a = self.EMA_ALPHA
            self.anchor = normalize_complex([(a * old + (1.0 - a) * new) for old, new in zip(self.anchor, psi)])

    # ----------------------------
    # Add message
    # ----------------------------
    def add_message(self, role: str, content: str):
        self.history.append({"role": role, "content": content})
        self._step += 1

        # anchor updates ONLY from structured state lines
        if self.is_structured_state(content) and self.is_anchor_line(content):
            psi = self.embed_complex(content)
            self.update_anchor(psi)
            self.density_inject(psi, strength=0.22)
            self.unitary_mix(seed=stable_hash(content) ^ self._step, rounds=10)

        # re-arm once per cycle
        if not self._gc_armed:
            cur = self.total_tokens()
            if cur <= self.REARM:
                self._gc_armed = True
                if not self._rearm_printed:
                    print(color(f"[QCSM] RE-ARMED ({cur} <= {self.REARM})", ANSI_YELLOW))
                    self._rearm_printed = True

    # ----------------------------
    # Interference score
    # ----------------------------
    def interference_score(self, psi_msg: List[complex]) -> float:
        if self.anchor is None:
            return 0.0
        amp = conj_dot(self.anchor, psi_msg)
        mag = abs(amp)
        phase_bonus = max(0.0, math.cos(math.atan2(amp.imag, amp.real))) * 0.15
        return (mag * 3.2) + phase_bonus

    # ----------------------------
    # Candidate scoring (higher keep)
    # Hard protect anchors via huge score.
    # ----------------------------
    def score(self, msg: Dict[str, str], idx: int, total: int) -> float:
        content = (msg["content"] or "").strip()
        role = msg["role"]

        if role == "system":
            return 999999.0

        # HARD PROTECT: anchor/config lines are sacred
        if self.is_anchor_line(content) and self.is_structured_state(content):
            return 10000.0

        psi = self.embed_complex(content)
        inter = self.interference_score(psi)
        recency = idx / max(1, total - 1)
        density = min(1.0, len(content) / 1600.0)
        role_bias = 0.05 if role == "user" else 0.0

        return (0.55 * inter) + (0.25 * recency) + (0.20 * density) + role_bias

    # ----------------------------
    # Compression
    # ----------------------------
    def compress(self, verbose: bool = True, force: bool = False):
        cur = self.total_tokens()

        if not force:
            if (not self._gc_armed) or (cur < self.TRIGGER):
                return

        if len(self.history) <= self.MIN_KEEP_MSGS:
            return

        start_tokens = cur
        rng = random.Random(0xC0FFEE ^ self._step)

        def build_protected():
            n = len(self.history)
            protected = {0}
            for i in range(max(0, n - self.PROTECT_TAIL), n):
                protected.add(i)
            return protected

        protected = build_protected()

        # ----------------------------
        # 1) Probabilistic collapse loop
        # ----------------------------
        steps = 0
        while self.total_tokens() > self.TARGET and steps < self.MAX_COLLAPSE_STEPS:
            steps += 1

            n = len(self.history)
            if n <= self.MIN_KEEP_MSGS:
                break

            protected = build_protected()

            candidates = []
            for i, m in enumerate(self.history):
                if i in protected:
                    continue
                sc = self.score(m, i, n)
                # If hard-protected by score, don't even consider it
                if sc >= 9999.0:
                    continue
                candidates.append({
                    "idx": i,
                    "score": sc,
                    "tok": self.estimate_tokens(m["content"]),
                    "prev": (m["content"] or "")[:70].replace("\n", " "),
                })

            if not candidates:
                break

            neg = [(-c["score"]) for c in candidates]
            probs = softmax(neg, temp=self.COLLAPSE_TEMP)

            removed = 0
            for _ in range(self.COLLAPSE_BATCH):
                if self.total_tokens() <= self.TARGET:
                    break
                pick = sample_index(probs, rng)
                victim = candidates[pick]

                projected = self.total_tokens() - victim["tok"]
                if projected < self.TARGET and (self.TARGET - projected) > self.OVERSHOOT_GUARD:
                    break

                self.history.pop(victim["idx"])
                removed += 1

                # rebuild candidates indices cheaply by breaking and letting next step rebuild
                break

            if removed == 0:
                break

        # ----------------------------
        # 2) Fail-safe clamp (deterministic)
        # If we're still above target, drop worst-scored NON-PROTECTED until target.
        # ----------------------------
        if self.total_tokens() > self.TARGET:
            n = len(self.history)
            protected = build_protected()

            det_cands = []
            for i, m in enumerate(self.history):
                if i in protected:
                    continue
                sc = self.score(m, i, n)
                if sc >= 9999.0:
                    continue
                det_cands.append((sc, i, self.estimate_tokens(m["content"])))

            det_cands.sort(key=lambda x: x[0])  # lowest dies first
            shed = 0
            drop_idxs = []

            cur2 = self.total_tokens()
            for sc, idx, tok in det_cands:
                if cur2 - shed <= self.TARGET:
                    break
                drop_idxs.append(idx)
                shed += tok

            for idx in sorted(drop_idxs, reverse=True):
                self.history.pop(idx)

        after = self.total_tokens()

        # Disarm ONLY if we actually clamped meaningfully
        if after <= self.TARGET:
            self._gc_armed = False
            self._rearm_printed = False

        if verbose:
            print(color("\n--- [QCSM v14.1 GC] ---", ANSI_CYAN))
            print(f"Status: {start_tokens} -> {after} tokens | Msgs: {len(self.history)}")
            if after <= self.TARGET:
                print(color("[QCSM] GC Disarmed (hysteresis active)", ANSI_YELLOW))
            else:
                print(color("[QCSM] WARNING: Still above TARGET (check MIN_KEEP/protection)", ANSI_RED))
            print(color("----------------------\n", ANSI_CYAN))

# =====================================================
# Agent
# =====================================================

class Agent:
    def __init__(self):
        self.mgr = QCSMQuantumSentinel()
        self.client = OpenAI(
            base_url=LOCAL_CONFIG["API_URL"],
            api_key=LOCAL_CONFIG["API_KEY"],
            timeout=None
        )

    def torture(self):
        print(color("\n[TORTURE] v14.1 Standard Pressure Test (10k)", ANSI_CYAN))
        self.mgr.history.clear()
        self.mgr.anchor = None
        self.mgr._embed_cache.clear()
        self.mgr._gc_armed = True
        self.mgr._rearm_printed = False

        keys = [f"KEY_{i:02d}" for i in range(10)]
        print(color("[Phase 1] Injecting Anchors", ANSI_YELLOW))
        for k in keys:
            self.mgr.add_message("user", f"SYSTEM_CONFIG: CONFIG_SET {k} = 'SECRET_{k}'")

        noise = ("LOG: kernel panic irq storm dma timeout | " * 55)

        print(color("[Phase 2] Flooding Noise (pressure)...", ANSI_YELLOW))
        for i in range(260):
            self.mgr.add_message("user", f"Noise_{i}: {noise}")
            if i % 40 == 0:
                self.mgr.compress(verbose=True)

        self.mgr.compress(verbose=True, force=True)

        content = " ".join(m["content"] for m in self.mgr.history)
        found = [k for k in keys if k in content]
        ok = (len(found) == len(keys))
        print(color(f"RESULT: {len(found)}/{len(keys)} Anchors Survived. Tokens: {self.mgr.total_tokens()}",
                    ANSI_GREEN if ok else ANSI_RED))

    def torture2(self):
        print(color("\n[TORTURE2] v14.1 Hard-mode Pressure Test (10k)", ANSI_CYAN))
        self.mgr.history.clear()
        self.mgr.anchor = None
        self.mgr._embed_cache.clear()
        self.mgr._gc_armed = True
        self.mgr._rearm_printed = False

        def r(n): return ''.join(random.choices(string.ascii_letters + string.digits, k=n))

        anchors: List[Tuple[str, str]] = []
        print(color("[Phase 1] Injecting Random Anchors", ANSI_YELLOW))
        for _ in range(12):
            k = f"CFG_{r(6)}"
            v = r(14)
            self.mgr.add_message("user", f"SYSTEM_CONFIG: {k} = '{v}'")
            anchors.append((k, v))

        noise = ("ERROR: gpu_fence_wait timeout | HEX:" + "".join(random.choices("0123456789ABCDEF", k=220)) + " | ") * 30

        print(color("[Phase 2] Flooding Noise (pressure)...", ANSI_YELLOW))
        for i in range(320):
            self.mgr.add_message("user", f"Pkt_{i}: {noise}")
            if i % 50 == 0:
                self.mgr.compress(verbose=True)

        self.mgr.compress(verbose=True, force=True)

        content = " ".join(m["content"] for m in self.mgr.history)
        found = [(k, v) for (k, v) in anchors if (k in content and v in content)]
        ok = (len(found) == len(anchors))
        print(color(f"RESULT: {len(found)}/{len(anchors)} Anchors Survived. Tokens: {self.mgr.total_tokens()}",
                    ANSI_GREEN if ok else ANSI_RED))

    def run(self):
        print(color("QCSM v14.1 Quantum Sentinel (Stable) Active", ANSI_CYAN))
        print(color("Commands: /torture, /torture2, /status, /clear, exit", ANSI_YELLOW))

        while True:
            u = input("you> ").strip()
            if not u or u.lower() == "exit":
                break

            if u.startswith("/"):
                if u == "/torture":
                    self.torture()
                elif u == "/torture2":
                    self.torture2()
                elif u == "/status":
                    print(f"Tokens: {self.mgr.total_tokens()} | Msgs: {len(self.mgr.history)} | Armed: {self.mgr._gc_armed}")
                elif u == "/clear":
                    self.mgr.history.clear()
                    self.mgr.anchor = None
                    self.mgr._embed_cache.clear()
                    self.mgr._gc_armed = True
                    self.mgr._rearm_printed = False
                    print("Cleared.")
                else:
                    print("Unknown command.")
                continue

            self.mgr.add_message("user", u)

            try:
                stream = self.client.chat.completions.create(
                    model=LOCAL_CONFIG["MODEL_NAME"],
                    messages=[{"role":"system","content":"Expert coder."}] + self.mgr.history,
                    temperature=LOCAL_CONFIG["TEMPERATURE"],
                    stream=True
                )

                print(f"{ANSI_PINK}ai> {ANSI_RESET}", end="")
                full = []
                for chunk in stream:
                    if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                        t = chunk.choices[0].delta.content
                        print(t, end="", flush=True)
                        full.append(t)
                print()

                out = sanitize_assistant("".join(full))
                self.mgr.add_message("assistant", out)

                self.mgr.compress(verbose=False)

                if self.mgr.total_tokens() > self.mgr.LIMIT:
                    self.mgr.compress(verbose=True, force=True)

            except Exception as e:
                print(color(f"Error: {e}", ANSI_RED))

if __name__ == "__main__":
    Agent().run()
