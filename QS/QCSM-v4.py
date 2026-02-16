import os, math, time, random, string, re
from typing import List, Dict, Optional, Tuple
from openai import OpenAI

# =====================================================
# Configuration
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

# =====================================================
# Output Sanitization
# =====================================================

THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)

def sanitize_assistant(text: str) -> str:
    return THINK_BLOCK.sub("", text or "").strip()

# =====================================================
# Memory Governor
# =====================================================

STRUCT_MARKERS = ("CFG_", "KEY_", "CONFIG", "PARAM")
ACK_ONLY = re.compile(r"^\s*(ok|okay|sure|thanks|noted|done|cool)\s*$", re.IGNORECASE)

class QCSMConversationManager:

    def __init__(self):
        self.history: List[Dict[str, str]] = []
        self._anchor: Optional[List[float]] = None
        self._embed_cache: Dict[str, List[float]] = {}

        # Context window
        self.LIMIT = 9500
        self.TRIGGER = 9200
        self.TARGET = 7500
        self.REARM = 8200

        self._gc_armed = True
        self._rearm_printed = False

        self.MIN_KEEP_MSGS = 14
        self.PROTECT_TAIL = 4

    # -----------------------------
    # Token estimation
    # -----------------------------

    def estimate_tokens(self, text: str) -> int:
        return max(1, len(str(text)) // 4)

    def total_tokens(self) -> int:
        return sum(self.estimate_tokens(m["content"]) for m in self.history)

    # -----------------------------
    # Structured detection
    # -----------------------------

    def _is_structured(self, text: str) -> bool:
        t = str(text)
        return any(k in t for k in STRUCT_MARKERS) and "=" in t

    # -----------------------------
    # Embedding
    # -----------------------------

    def _embed(self, text: str) -> List[float]:
        key = str(text)
        if key in self._embed_cache:
            return self._embed_cache[key]

        v = [0.0] * 64
        for w in key.lower().split():
            v[hash(w) % 64] += 1.0

        mag = math.sqrt(sum(x*x for x in v))
        out = [x/mag for x in v] if mag > 0 else v
        self._embed_cache[key] = out
        return out

    def _update_anchor(self, text: str):
        vec = self._embed(text)
        if self._anchor is None:
            self._anchor = vec
        else:
            self._anchor = [(0.9*o + 0.1*n) for o, n in zip(self._anchor, vec)]

    # -----------------------------
    # Add message
    # -----------------------------

    def add_message(self, role: str, content: str):
        self.history.append({"role": role, "content": content})

        if self._is_structured(content):
            self._update_anchor(content)

        if not self._gc_armed:
            cur = self.total_tokens()
            if cur <= self.REARM:
                self._gc_armed = True
                if not self._rearm_printed:
                    print(color(f"[QCSM] RE-ARMED ({cur} <= {self.REARM})", ANSI_YELLOW))
                    self._rearm_printed = True

    # -----------------------------
    # Scoring
    # -----------------------------

    def _score(self, msg: Dict[str,str], idx: int, total: int) -> float:

        if msg["role"] == "system":
            return 999.0

        content = msg["content"].strip()
        is_struct = self._is_structured(content)

        if not is_struct and ACK_ONLY.match(content):
            return 0.01

        sim = 0.0
        if self._anchor:
            sim = sum(x*y for x,y in zip(self._embed(content), self._anchor))

        recency = idx / max(1,total-1)
        density = min(1.0, len(content)/1500)

        structure_bonus = 0.8 if is_struct else 0.0

        return (0.5*(sim*3.0)) + (0.25*recency) + (0.2*density) + structure_bonus

    # -----------------------------
    # Compression
    # -----------------------------

    def compress(self, verbose: bool=True, force: bool=False):

        cur = self.total_tokens()

        if not force:
            if (not self._gc_armed) or (cur < self.TRIGGER):
                return

        if len(self.history) <= self.MIN_KEEP_MSGS:
            return

        n = len(self.history)
        protected = {0}
        for i in range(max(0, n-self.PROTECT_TAIL), n):
            protected.add(i)

        candidates = []
        for i,m in enumerate(self.history):
            if i in protected:
                continue
            candidates.append({
                "idx": i,
                "score": self._score(m,i,n),
                "tok": self.estimate_tokens(m["content"]),
                "prev": m["content"][:60]
            })

        candidates.sort(key=lambda x: x["score"])

        drop_idxs = []
        shed = 0

        for c in candidates:
            projected = cur - (shed + c["tok"])
            if projected <= self.TARGET:
                break
            drop_idxs.append(c["idx"])
            shed += c["tok"]

        if not drop_idxs:
            return

        for i in sorted(drop_idxs, reverse=True):
            self.history.pop(i)

        after = self.total_tokens()
        self._gc_armed = False
        self._rearm_printed = False

        if verbose:
            print(color("\n--- [SENTINEL GC EVENT] ---", ANSI_CYAN))
            print(f"Status: {cur} -> {after} | Dropped: {len(drop_idxs)}")
            print(color("[QCSM] GC Disarmed", ANSI_YELLOW))
            print(color("--------------------------\n", ANSI_CYAN))

# =====================================================
# Agent Interface
# =====================================================

class Agent:

    def __init__(self):
        self.mgr = QCSMConversationManager()
        self.client = OpenAI(
            base_url=LOCAL_CONFIG["API_URL"],
            api_key=LOCAL_CONFIG["API_KEY"]
        )

    # -----------------------------
    # Torture (Standard)
    # -----------------------------

    def torture(self):
        print(color("\n[TORTURE] Standard 10k Pressure Test", ANSI_CYAN))
        self.mgr.history.clear()
        self.mgr._anchor = None
        self.mgr._gc_armed = True

        keys = [f"KEY_{i:02d}" for i in range(10)]

        for k in keys:
            self.mgr.add_message("user", f"CONFIG_SET {k} = 'SECRET_{k}'")

        noise = ("ERROR_LOG: segfault kernel panic trace | ") * 40

        for i in range(150):
            self.mgr.add_message("user", f"Noise_{i}: {noise}")
            self.mgr.compress(verbose=(i%25==0))

        self.mgr.compress(force=True)

        content = " ".join(m["content"] for m in self.mgr.history)
        found = [k for k in keys if k in content]
        print(color(f"RESULT: {len(found)}/10 Anchors Survived. Tokens: {self.mgr.total_tokens()}",
                    ANSI_GREEN if len(found)==10 else ANSI_RED))

    # -----------------------------
    # Hard Mode
    # -----------------------------

    def torture2(self):
        print(color("\n[TORTURE2] Hard-mode 10k Pressure Test", ANSI_CYAN))
        self.mgr.history.clear()
        self.mgr._anchor = None
        self.mgr._gc_armed = True

        def r(n): return ''.join(random.choices(string.ascii_letters+string.digits,k=n))

        anchors=[]
        for _ in range(10):
            k=f"CFG_{r(6)}"
            v=r(12)
            self.mgr.add_message("user", f"Boot object {k} with signature={v}.")
            anchors.append((k,v))

        noise = ("IRQ_FAIL dma timeout gpu stall panic | ") * 50

        for i in range(200):
            self.mgr.add_message("user", f"Pkt_{i}: {noise}")
            self.mgr.compress(verbose=(i%30==0))

        self.mgr.compress(force=True)

        content=" ".join(m["content"] for m in self.mgr.history)
        found=[k for k,v in anchors if k in content and v in content]

        print(color(f"RESULT: {len(found)}/10 Anchors Survived. Tokens: {self.mgr.total_tokens()}",
                    ANSI_GREEN if len(found)==10 else ANSI_RED))

    # -----------------------------

    def run(self):
        print(color("QCSM v13.8 Sentinel Agent Active", ANSI_CYAN))
        print("Commands: /torture, /torture2, /status, /clear, exit")

        while True:
            u=input("you> ").strip()
            if not u or u=="exit":
                break

            if u=="/torture":
                self.torture()
                continue
            if u=="/torture2":
                self.torture2()
                continue
            if u=="/status":
                print(f"Tokens: {self.mgr.total_tokens()} | Msgs: {len(self.mgr.history)}")
                continue
            if u=="/clear":
                self.mgr.history.clear()
                print("Cleared.")
                continue

            self.mgr.add_message("user", u)

            try:
                stream=self.client.chat.completions.create(
                    model=LOCAL_CONFIG["MODEL_NAME"],
                    messages=[{"role":"system","content":"Expert coder."}] + self.mgr.history,
                    temperature=LOCAL_CONFIG["TEMPERATURE"],
                    stream=True
                )

                print(f"{ANSI_PINK}ai> {ANSI_RESET}", end="")
                full=[]
                for chunk in stream:
                    if chunk.choices and chunk.choices[0].delta.content:
                        t=chunk.choices[0].delta.content
                        print(t,end="",flush=True)
                        full.append(t)
                print()

                out=sanitize_assistant("".join(full))
                self.mgr.add_message("assistant", out)
                self.mgr.compress(verbose=False)

            except Exception as e:
                print(color(f"Error: {e}", ANSI_RED))

if __name__=="__main__":
    Agent().run()
