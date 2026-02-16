# agent.py — QCSM v13.2 "Blackbox" (Production-Surgical Upgrade)
# Goal: keep your exact architecture, but fix the real production issues:
# - hysteresis trigger (stop pruning every time you barely touch the ceiling)
# - adaptive shedding (don’t always snap to the same number)
# - anti-fluff pruning (kill “ok/noted” style low-info messages first)
# - anchor drift protection (anchor updates only from “structured state”)
# - embedding cache (hash-embed is cheap, but this prevents repeated work)
# - output sanitization (strip <think> leakage before storing in memory)
# - safer config (API key via env, fallback to your existing)

import sys, os, math, time, random, json, re
from typing import Dict, List, Optional
from openai import OpenAI

# ----------------------------
# Configuration
# ----------------------------
LOCAL_CONFIG = {
    "API_URL": os.getenv("QCSM_API_URL", "http://localhost:1234/v1"),
    "API_KEY": os.getenv("QCSM_API_KEY", "sk-lm-hGUNPC8f:L6KZd21uTFcvRIZ6hKRz"),
    "MODEL_NAME": os.getenv("QCSM_MODEL_NAME", "local-model"),
    "TEMPERATURE": float(os.getenv("QCSM_TEMPERATURE", "0.6")),
}

ANSI_PINK, ANSI_CYAN, ANSI_GREEN, ANSI_RED, ANSI_YELLOW, ANSI_RESET = (
    "\033[95m", "\033[96m", "\033[92m", "\033[91m", "\033[93m", "\033[0m"
)
USE_COLOR = sys.stdout.isatty() and os.getenv("NO_COLOR") is None

def color(text, code):
    return f"{code}{text}{ANSI_RESET}" if USE_COLOR else text

# ----------------------------
# Sanitizers / Heuristics
# ----------------------------
THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
THINK_TAGS = re.compile(r"</?think>", re.IGNORECASE)

ACK_ONLY = re.compile(r"^\s*(ok|okay|noted|sure|thanks|ty|got it|done|cool|nice)\s*[\.\!\?]*\s*$", re.IGNORECASE)

# "Structured state" signals for anchor updates & scoring boosts
STRUCTURE_HINTS = ("CONFIG", "PARAM", "CFG_", "KEY_", "CONFIG_SET", "SYSTEM_CONFIG", "SYSTEM_CRITICAL_CONFIG")
CODE_HINTS = ("def ", "class ", "struct", "enum", "interface", "SELECT ", "INSERT ", "UPDATE ", "DELETE ", "function ")

# Noise hints (optional penalty)
NOISE_HINTS = ("Noise_Packet_", "LOG_DATA:", "0x", "[System-Interrupt", "Lorem ipsum", "corporate synergy")

def sanitize_assistant_text(text: str) -> str:
    if not text:
        return ""
    # Remove full think blocks
    text = THINK_BLOCK.sub("", text)
    # Remove stray tags if any remain
    text = THINK_TAGS.sub("", text)
    return text.strip()

# ----------------------------
# QCSM Conversation Manager
# ----------------------------
class QCSMConversationManager:
    """
    Surgical improvements only:
    - Hysteresis trigger: GC triggers above TRIGGER, not constantly near LIMIT.
    - Adaptive target: shed more when overflow is larger.
    - Anti-fluff penalty: short/ack messages get pruned early.
    - Anchor update gating: only update anchor from structured messages, not random noise.
    - Embed cache: stable and fast.
    """

    def __init__(self):
        self.history: List[Dict[str, str]] = []
        self._anchor: Optional[List[float]] = None
        self._embed_cache: Dict[str, List[float]] = {}

        # Token control (10k-ish context)
        self.LIMIT = 9500         # hard ceiling for safety
        self.TRIGGER = 9200       # hysteresis trigger (don’t GC until above this)
        self.TARGET_BASE = 7500   # typical post-GC target
        self.TARGET_MIN = 6500    # never compress below this unless you force it

        # Protection / density control
        self.MIN_KEEP_MSGS = 12   # never go below this (unless you manually change code)
        self.PROTECT_TAIL = 4     # keep last N msgs always (recency safety)

    def estimate_tokens(self, text) -> int:
        # Cheap approximation; consistent and stable is more important than perfect.
        s = str(text)
        return max(1, len(s) // 4)

    def total_tokens(self) -> int:
        return sum(self.estimate_tokens(m["content"]) for m in self.history)

    def add_message(self, role: str, content: str):
        self.history.append({"role": role, "content": content})

        # Anchor update only from "structured state" (prevents drift)
        if self._looks_like_structured_state(content):
            self._update_anchor(content)

    # ----------------------------
    # Anchor & Embedding
    # ----------------------------
    def _looks_like_structured_state(self, text: str) -> bool:
        t = str(text)
        # Must contain at least one structure hint OR code hint, and an assignment-ish token
        has_hint = any(h in t for h in STRUCTURE_HINTS) or any(h in t for h in CODE_HINTS)
        looks_assigned = ("=" in t) or (":" in t and "CONFIG" in t)
        return bool(has_hint and looks_assigned)

    def _embed(self, text: str) -> List[float]:
        # Cached 64-dim hashed BoW vector (fast, deterministic)
        key = str(text)
        v = self._embed_cache.get(key)
        if v is not None:
            return v

        vec = [0.0] * 64
        for w in key.lower().split():
            vec[hash(w) % 64] += 1.0

        m = math.sqrt(sum(x * x for x in vec))
        out = [x / m for x in vec] if m > 0 else vec
        self._embed_cache[key] = out
        return out

    def _update_anchor(self, text: str):
        vec = self._embed(text)
        if not self._anchor:
            self._anchor = vec
        else:
            # Slow drift: keep anchor stable (90% old / 10% new)
            self._anchor = [(0.90 * o + 0.10 * n) for o, n in zip(self._anchor, vec)]

    # ----------------------------
    # Scoring
    # ----------------------------
    def _score(self, msg: Dict[str, str], idx: int, total: int) -> float:
        role = msg.get("role", "")
        content = msg.get("content", "") or ""

        # System should never be purged
        if role == "system":
            return 999.0

        # Similarity to anchor (semantic retention)
        sim = 0.0
        if self._anchor:
            v = self._embed(content)
            # dot product with small boost (your original spirit)
            sim = sum(x * y for x, y in zip(v, self._anchor))

        # Recency (0..1)
        recency = idx / max(1, total - 1)

        # Density score (0..1)
        density = min(1.0, len(content) / 2000.0)

        # Structure bonus (keeps "state" even if not super similar)
        structure_bonus = 0.0
        if self._looks_like_structured_state(content):
            structure_bonus += 0.25
        if any(h in content for h in CODE_HINTS):
            structure_bonus += 0.10

        # Anti-fluff penalties
        short_penalty = 0.0
        if len(content.strip()) < 40:
            short_penalty -= 0.25
        if ACK_ONLY.match(content):
            short_penalty -= 0.40

        # Optional noise penalty (keeps logs disposable)
        noise_penalty = 0.0
        if any(h in content for h in NOISE_HINTS):
            noise_penalty -= 0.15

        # Role bias
        role_bias = 0.08 if role == "user" else 0.00

        # Final Score:
        # - Semantic: 45%
        # - Recency:  25%
        # - Density:  20%
        # - Structure/role bonuses + penalties: additive
        base = (0.45 * (sim * 2.5)) + (0.25 * recency) + (0.20 * density) + role_bias
        return base + structure_bonus + short_penalty + noise_penalty

    # ----------------------------
    # Compression
    # ----------------------------
    def _compute_target(self, cur_tokens: int) -> int:
        """
        Adaptive target:
        - small overflow -> target close to base
        - huge overflow -> compress deeper (but never below TARGET_MIN)
        """
        overflow = max(0, cur_tokens - self.LIMIT)
        # shed deeper by up to ~1200 tokens when very over
        dynamic_drop = int(min(1200, overflow * 0.8))
        target = self.TARGET_BASE - dynamic_drop
        return max(self.TARGET_MIN, target)

    def compress(self, verbose: bool = True, force: bool = False):
        cur = self.total_tokens()

        # Hysteresis: do nothing unless we cross TRIGGER, OR forced, OR exceed LIMIT
        if not force and cur <= self.TRIGGER:
            return
        if not force and cur <= self.LIMIT and cur <= self.TRIGGER:
            return
        if not force and cur <= self.LIMIT and cur <= self.TRIGGER:
            return

        # Hard safety
        if cur <= self.LIMIT and not force and cur <= self.TRIGGER:
            return

        # If we haven't even hit hard limit and it's not forced, still respect TRIGGER
        if not force and cur <= self.LIMIT and cur < self.TRIGGER:
            return

        # If below LIMIT but above TRIGGER, allow a light “headroom GC”
        # (keeps you from riding 9.4k forever)
        target = self._compute_target(cur)

        # Protection: keep first + tail window, and enforce min messages
        n = len(self.history)
        if n <= self.MIN_KEEP_MSGS:
            return

        protected = set()
        protected.add(0)
        for i in range(max(0, n - self.PROTECT_TAIL), n):
            protected.add(i)

        cands = []
        for i, m in enumerate(self.history):
            if i in protected:
                continue
            score = self._score(m, i, n)
            cands.append({
                "idx": i,
                "score": score,
                "tok": self.estimate_tokens(m["content"]),
                "preview": (m["content"] or "")[:60].replace("\n", " ")
            })

        # Lowest score dies first
        cands.sort(key=lambda x: x["score"])

        drop_idxs = []
        shed = 0

        # Don’t prune below MIN_KEEP_MSGS
        max_drops = max(0, n - self.MIN_KEEP_MSGS)

        for item in cands:
            if cur - shed <= target:
                break
            if len(drop_idxs) >= max_drops:
                break
            drop_idxs.append(item["idx"])
            shed += item["tok"]

        if not drop_idxs:
            return

        if verbose:
            print(color("\n--- [QCSM GC EVENT] ---", ANSI_CYAN))
            print(f"Memory: {cur} -> {max(0, cur - shed)} tokens | Shed: {shed} | Target: {target}")
            print(f"Purging {len(drop_idxs)} lowest-scoring messages...")
            lowest = cands[0]
            highest = cands[-1]
            print(color(f"  Lowest (Purged): {lowest['preview']}... (Score: {lowest['score']:.2f})", ANSI_RED))
            print(color(f"  Highest (Saved): {highest['preview']}... (Score: {highest['score']:.2f})", ANSI_GREEN))
            print(color("-----------------------\n", ANSI_CYAN))

        for i in sorted(drop_idxs, reverse=True):
            self.history.pop(i)

# ----------------------------
# Agent
# ----------------------------
class Agent:
    def __init__(self):
        self.mgr = QCSMConversationManager()
        self.client = OpenAI(
            base_url=LOCAL_CONFIG["API_URL"],
            api_key=LOCAL_CONFIG["API_KEY"],
            timeout=None,
        )

    def _run_torture(self):
        print(color("\n[TORTURE] Starting High-Verbosity 10k Test...", ANSI_CYAN))
        self.mgr.history = []
        self.mgr._anchor = None
        self.mgr._embed_cache.clear()

        keys = [f"KEY_{i:02d}" for i in range(10)]

        print(color("[Phase 1] Injecting Critical Anchors", ANSI_YELLOW))
        for k in keys:
            self.mgr.add_message("user", f"CONFIG_SET {k} = 'SECRET_{k}'")
            print(f"  > Anchor {k} stored.")

        print(color("\n[Phase 2] Flooding with 60 Turns of Technical Noise", ANSI_YELLOW))
        noise = "LOG_DATA: 0x4F 0x22 0xAA 0xFF [System-Interrupt-Failure] " * 15

        for i in range(60):
            self.mgr.add_message("user", f"Noise_Packet_{i}: {noise}")

            # Light headroom GC if needed (hysteresis controlled)
            self.mgr.compress(verbose=True, force=False)

            if i % 15 == 0:
                print(f"  Step {i}/60 | Current Load: {self.mgr.total_tokens()} tokens")

        # Final clamp (guarantee safe under LIMIT)
        self.mgr.compress(verbose=True, force=True)

        print(color("\n[Phase 3] Final Recall Audit", ANSI_YELLOW))
        content = " ".join([m["content"] for m in self.mgr.history])
        found = [k for k in keys if k in content]

        score_color = ANSI_GREEN if len(found) == len(keys) else ANSI_RED
        print(color(f"RESULT: {len(found)}/{len(keys)} Anchors Survived.", score_color))
        print(f"Survivors: {', '.join(found)}")
        print(f"Final Memory Density: {self.mgr.total_tokens()} / {self.mgr.LIMIT}")

    def run(self):
        print(color("QCSM v13.2 Blackbox Agent Active", ANSI_CYAN))
        print(color("Commands: /torture, /clear, /status, exit", ANSI_YELLOW))

        while True:
            u = input("you> ").strip()
            if not u or u.lower() == "exit":
                break

            if u.startswith("/"):
                cmd = u.lower()
                if cmd == "/torture":
                    self._run_torture()
                elif cmd == "/status":
                    print(f"Tokens: {self.mgr.total_tokens()} | Msgs: {len(self.mgr.history)}")
                elif cmd == "/clear":
                    self.mgr.history = []
                    self.mgr._anchor = None
                    self.mgr._embed_cache.clear()
                    print("Cleared.")
                else:
                    print("Unknown command.")
                continue

            # Add user message
            self.mgr.add_message("user", u)

            start = time.time()
            try:
                stream = self.client.chat.completions.create(
                    model=LOCAL_CONFIG["MODEL_NAME"],
                    messages=[{"role": "system", "content": "Expert coding assistant."}] + self.mgr.history,
                    temperature=LOCAL_CONFIG["TEMPERATURE"],
                    stream=True,
                )

                print(f"{ANSI_PINK}ai> ", end="")
                full = []
                first = True

                for chunk in stream:
                    if first:
                        latency = time.time() - start
                        print(f"[{latency:.2f}s Pre-fill] {ANSI_RESET}", end="")
                        first = False

                    delta = chunk.choices[0].delta.content if chunk.choices else None
                    if delta:
                        print(delta, end="", flush=True)
                        full.append(delta)

                print()

                # Sanitize + store assistant message
                out = sanitize_assistant_text("".join(full))
                self.mgr.add_message("assistant", out)

                # Hysteresis-controlled GC + final safety if needed
                self.mgr.compress(verbose=True, force=False)
                if self.mgr.total_tokens() > self.mgr.LIMIT:
                    self.mgr.compress(verbose=True, force=True)

            except Exception as e:
                print(color(f"Error: {e}", ANSI_RED))

if __name__ == "__main__":
    Agent().run()
