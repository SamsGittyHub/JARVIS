import sys
import os
import math
import time
import random
from typing import Dict, List, Optional

# --- Configuration: RTX 5090 Backend ---
LOCAL_CONFIG = {
    "API_URL": "http://localhost:1234/v1", 
    "API_KEY": "sk-lm-hGUNPC8f:L6KZd21uTFcvRIZ6hKRz", 
    "MODEL_NAME": "local-model", 
    "TIMEOUT": None, # UPDATED: Infinite Timeout for Titan Contexts
    "TEMPERATURE": 0.6, 
}

# --- Terminal Colors ---
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ANSI_PINK = "\033[95m"
ANSI_CYAN = "\033[96m"
ANSI_GREEN = "\033[92m"
ANSI_RED = "\033[91m"
ANSI_YELLOW = "\033[93m"
ANSI_RESET = "\033[0m"
USE_COLOR = sys.stdout.isatty() and os.getenv("NO_COLOR") is None

def color(text: str, code: str) -> str:
    return f"{code}{text}{ANSI_RESET}" if USE_COLOR else text

# --- Dependencies ---
try:
    from openai import OpenAI, APIConnectionError, AuthenticationError
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print(color("Warning: 'openai' library not found. Install with: pip install openai", ANSI_RED))

# --- System Prompt ---
QCSM_SYSTEM_PROMPT = """You are an advanced AI coding assistant running in QCSM (Quantum Circuit Semantic Memory) mode.

Constraints:
- You have a massive context window (200k+), but memory is still governed by semantic density.
- Be concise. High density information is preferred over verbosity.
- Do not repeat large code blocks provided by the user; reference them instead.

Goals:
1. Provide production-grade, bug-free code.
2. Maintain continuity with defined Semantic Anchors (config keys, variable names).
"""

class QCSMConversationManager:
    """
    QCSM v12 "Eternity": Infinite Timeout Edition
    """
    # --- SCALING FOR 200k+ CONTEXT ---
    COMPRESS_TOKEN_LIMIT = 200000  # Trigger point
    TARGET_TOKEN_LIMIT   = 170000  # Target
    HARD_TOKEN_CEILING   = 205000  # Panic
    MIN_COMPRESS_HISTORY = 20      

    def __init__(self):
        self.history = []
        self._semantic_anchor_vector = None

    def get_system_prompt(self) -> str:
        return QCSM_SYSTEM_PROMPT

    def estimate_tokens(self, text: str) -> int:
        return len(str(text)) // 4

    def total_tokens(self) -> int:
        return sum(self.estimate_tokens(m["content"]) for m in self.history)

    def to_openai_format(self) -> List[Dict[str, str]]:
        return self.history

    def add_message(self, role: str, content: str):
        self.history.append({"role": role, "content": content})
        if "CONFIG" in content or "PARAM" in content or "def " in content or "class " in content:
             self._update_anchor(content)

    # --- Intelligence Engine ---
    def embed_message(self, text: str) -> List[float]:
        vec = [0.0] * 64
        words = str(text).lower().replace("_", " ").split()
        for w in words:
            h = hash(w) % 64
            vec[h] += 1.0
        mag = math.sqrt(sum(x*x for x in vec))
        return [x/mag for x in vec] if mag > 0 else vec

    def _update_anchor(self, text: str):
        new_vec = self.embed_message(text)
        if self._semantic_anchor_vector is None:
            self._semantic_anchor_vector = new_vec
        else:
            self._semantic_anchor_vector = [
                (0.95 * old) + (0.05 * new) 
                for old, new in zip(self._semantic_anchor_vector, new_vec)
            ]

    def _cosine(self, a: List[float], b: List[float]) -> float:
        if not a or not b: return 0.0
        dot = sum(x*y for x, y in zip(a, b))
        mag_a = math.sqrt(sum(x*x for x in a))
        mag_b = math.sqrt(sum(x*x for x in b))
        if mag_a == 0 or mag_b == 0: return 0.0
        return dot / (mag_a * mag_b)

    def _calculate_message_score(self, message: Dict, index: int, total_msgs: int) -> float:
        if message["role"] == "system": return 999.0
        content = message["content"]
        length = len(content)

        # 1. Semantic Similarity
        similarity_score = 0.0
        if self._semantic_anchor_vector:
            msg_vec = self.embed_message(content)
            similarity_score = self._cosine(msg_vec, self._semantic_anchor_vector) * 3.0

        # 2. Recency
        recency_score = index / max(1, total_msgs)

        # 3. Density/Length
        length_score = min(1.0, math.log(max(1, length)) / 10.0)
        
        # 4. Role Bias
        role_score = 0.2 if message["role"] == "user" else 0.0

        return (0.50 * similarity_score) + (0.25 * recency_score) + \
               (0.15 * length_score) + (0.10 * role_score)

    def _smart_compress(self, target_tokens: int):
        current_tokens = self.total_tokens()
        if current_tokens <= target_tokens: return

        protected_indices = {0, len(self.history)-1, len(self.history)-2, len(self.history)-3, len(self.history)-4}
        
        candidates = []
        for i, msg in enumerate(self.history):
            if i in protected_indices: continue
            score = self._calculate_message_score(msg, i, len(self.history))
            candidates.append((i, score, self.estimate_tokens(msg["content"])))

        candidates.sort(key=lambda x: x[1])

        kill_list = []
        tokens_to_shed = current_tokens - target_tokens
        shed_so_far = 0

        for idx, score, tok in candidates:
            if shed_so_far >= tokens_to_shed: break
            kill_list.append(idx)
            shed_so_far += tok

        kill_list.sort(reverse=True)
        for idx in kill_list:
            self.history.pop(idx)
            
        return len(kill_list), shed_so_far

    def manage_memory(self):
        total = self.total_tokens()
        if total <= self.COMPRESS_TOKEN_LIMIT: return

        if total > self.HARD_TOKEN_CEILING:
            print(color(f"[QCSM] PANIC PURGE ({total})", ANSI_RED))
            for _ in range(5):
                self._smart_compress(self.TARGET_TOKEN_LIMIT)
                if self.total_tokens() <= self.TARGET_TOKEN_LIMIT: break
            return

        before = total
        msgs_before = len(self.history)
        self._smart_compress(self.TARGET_TOKEN_LIMIT)
        after = self.total_tokens()
        msgs_after = len(self.history)
        
        if after < before:
            print(color(f"[QCSM] Titan Prune {before}->{after} (Δ{before-after}) | Msgs {msgs_before}->{msgs_after}", ANSI_CYAN))

class Agent:
    def __init__(self):
        self.conv = QCSMConversationManager()
        self.client = None
        self.connected = False
        
        if OPENAI_AVAILABLE:
            try:
                self.client = OpenAI(
                    base_url=LOCAL_CONFIG["API_URL"], 
                    api_key=LOCAL_CONFIG["API_KEY"],
                    timeout=LOCAL_CONFIG["TIMEOUT"] 
                )
                self._handshake()
            except Exception as e:
                print(color(f"Init Failed: {e}", ANSI_RED))

    def _handshake(self):
        try:
            print(f"Connecting to {LOCAL_CONFIG['API_URL']}...", end=" ", flush=True)
            self.client.models.list()
            print(color("OK", ANSI_GREEN))
            self.connected = True
        except Exception:
            print(color("FAIL (Check API Key/Port)", ANSI_RED))

    def stream(self):
        if not self.client: return "No Connection."
        try:
            messages = [{"role": "system", "content": self.conv.get_system_prompt()}]
            messages.extend(self.conv.to_openai_format())
            
            # Massive Context Warning
            token_count = self.conv.total_tokens()
            if token_count > 10000:
                print(color(f"\n[Sending {token_count} tokens to GPU. Stand by...]", ANSI_YELLOW))
                self.req_start = time.time()

            stream = self.client.chat.completions.create(
                model=LOCAL_CONFIG["MODEL_NAME"],
                messages=messages,
                temperature=LOCAL_CONFIG["TEMPERATURE"],
                stream=True,
            )

            print(f"{ANSI_PINK}ai> ", end="", flush=True)
            full = []
            first_token = True
            
            for chunk in stream:
                if first_token and token_count > 10000:
                    elapsed = time.time() - self.req_start
                    print(f"\n{ANSI_GREEN}[Pre-fill Complete: {elapsed:.1f}s]{ANSI_PINK} ", end="")
                    first_token = False

                if chunk.choices[0].delta.content:
                    text = chunk.choices[0].delta.content
                    print(text, end="", flush=True)
                    full.append(text)
            print(ANSI_RESET)
            return "".join(full).strip()
        except Exception as e:
            print(color(f"\nError: {e}", ANSI_RED))
            return "Error"

    def _run_torture_test(self):
        print(color("\n[TORTURE TEST] Protocol v12: 'Eternity'...", ANSI_CYAN))
        print(color(f"Targeting > {self.conv.COMPRESS_TOKEN_LIMIT} tokens...", ANSI_YELLOW))
        
        self.conv.history.clear()
        self.conv._semantic_anchor_vector = None
        
        keys = [f"CFG_{i:03d}" for i in range(15)]
        print(f"[Phase 1] Injecting {len(keys)} Config Anchors...")
        for k in keys:
            self.conv.add_message("user", f"SYSTEM_CRITICAL_CONFIG: PARAM_{k} = 'active_state_true'")
            self.conv.add_message("assistant", f"Locked {k}")
        
        if self.conv._semantic_anchor_vector:
             print(f"{ANSI_YELLOW} > Semantic Lock Established{ANSI_RESET}")

        turns = 400
        noise_payload = "ERROR [timestamp]: Connection refused. Stack trace dump follows... " * 25 
        
        print(f"[Phase 2] Initiating {turns}-turn flood...")
        
        start_time = time.time()
        for i in range(turns):
            self.conv.add_message("user", f"Log Dump {i}: {noise_payload}")
            self.conv.add_message("assistant", f"Analyzed {i}.")
            self.conv.manage_memory()
            
            if i % 50 == 0 and i > 0:
                elapsed = time.time() - start_time
                print(f"... Turn {i}/{turns} | Tokens: {self.conv.total_tokens()} | Time: {elapsed:.1f}s")

        print(f"{ANSI_YELLOW}[Phase 2.5] Final Clamp...{ANSI_RESET}")
        if self.conv.total_tokens() > self.conv.COMPRESS_TOKEN_LIMIT:
             print(" > Overage detected. Compressing...")
             self.conv._smart_compress(self.conv.TARGET_TOKEN_LIMIT)
             print(f" > Clamped to: {self.conv.total_tokens()}")
        else:
             print(" > Memory safe.")

        print("\n[Phase 3] Audit...")
        content = " ".join([m["content"] for m in self.conv.history])
        found = sum(1 for k in keys if k in content)
        
        if found == len(keys):
            print(color(f"PERFECT SCORE: {found}/{len(keys)} Anchors Survived.", ANSI_GREEN))
        else:
            print(color(f"SCORE: {found}/{len(keys)} Anchors Survived.", ANSI_RED))

        if self.connected:
            print(color("\n[Phase 4] Titan Recall (Live Model)...", ANSI_YELLOW))
            print(color("WARNING: This may take 5-10 minutes due to massive pre-fill.", ANSI_RED))
            target_key = random.choice(keys)
            print(f" > Asking about {target_key}...")
            self.conv.add_message("user", f"FINAL_VERIFICATION: What is the value of PARAM_{target_key}? Return ONLY the value.")
            
            resp = self.stream()
            self.conv.add_message("assistant", resp)
            
            if "active_state_true" in resp:
                 print(color(f"\nSUCCESS: Recall Verified!", ANSI_GREEN))
            else:
                 print(color(f"\nFAILURE: Hallucinated.", ANSI_RED))

    def run(self):
        print(color(f"QCSM v12 Eternity Agent", ANSI_CYAN))
        print("Commands: /torture, /clear, /status, exit")
        while True:
            try: u = input("you> ").strip()
            except: break
            if not u: continue
            if u == "exit": break
            if u == "/torture": self._run_torture_test(); continue
            if u == "/status": 
                print(f"Mem: {self.conv.total_tokens()} toks | {len(self.conv.history)} msgs")
                continue
            if u == "/clear":
                self.conv.history.clear()
                print("Memory wiped.")
                continue

            self.conv.add_message("user", u)
            resp = self.stream()
            self.conv.add_message("assistant", resp)
            self.conv.manage_memory()

if __name__ == "__main__":
    Agent().run()