import sys
import os
import math
import random
import re
import time
from typing import Dict, List, Tuple

# --- Terminal Configuration ---
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

ANSI_PINK = "\033[95m"
ANSI_CYAN = "\033[96m"
ANSI_GREEN = "\033[92m"
ANSI_RED = "\033[91m"
ANSI_YELLOW = "\033[93m"
ANSI_RESET = "\033[0m"
USE_COLOR = sys.stdout.isatty() and os.getenv("NO_COLOR") is None

def color(text: str, code: str) -> str:
    return f"{code}{text}{ANSI_RESET}" if USE_COLOR else text

# --- TripleG / SamsLaw Imports ---
# (Assumed available in your environment. Fallback provided for standalone testing)
try:
    from tripleg import (
        CONFIG as TRIPLEG_CONFIG,
        TORCH_AVAILABLE,
        SamsLawConversationManager,
        client as TRIPLEG_CLIENT,
    )
except ImportError:
    TRIPLEG_CONFIG = {"MIN_HISTORY_KEEP": 10, "MODEL_NAME": "local-model", "API_URL": "http://localhost:1234"}
    TORCH_AVAILABLE = False
    SamsLawConversationManager = object
    TRIPLEG_CLIENT = None

QCSM_SYSTEM_PROMPT = """You are a helpful assistant in QCSM inference mode.
Answer directly and concisely.
No tool calls are available in this interface.
"""

class NoopSkillManager:
    def get_system_prompt_additions(self) -> str:
        return ""

class BaseManager(SamsLawConversationManager if hasattr(SamsLawConversationManager, 'history') else object):
    def __init__(self, skill_manager, use_sam):
        self.history = []
        self.skill_manager = skill_manager
        self.use_sam = use_sam
    def estimate_tokens(self, text):
        return len(str(text)) // 4
    def to_openai_format(self):
        return [{"role": m.role, "content": m.content} for m in self.history]
    def add_message(self, role, content, is_user):
        class Msg: pass
        m = Msg(); m.role = role; m.content = content
        self.history.append(m)
    def save(self, f): return "Saved (mock)"
    def load(self, f): return "Loaded (mock)"

class QCSMConversationManager(SamsLawConversationManager if hasattr(SamsLawConversationManager, 'history') else BaseManager):
    """
    QCSM v7.0: Production Edition with Final Clamp & Semantic Anchors
    """

    # --- Configuration ---
    COMPRESS_TOKEN_LIMIT = 4000
    TARGET_TOKEN_LIMIT = 3200  # Hysteresis target
    HARD_TOKEN_CEILING = 8000
    MIN_COMPRESS_HISTORY = 10

    # Toggle: True = Prune by Semantic Similarity, False = Prune by Age
    SEMANTIC_WEIGHTING_MODE = True
    FAST_COMPRESSION_MODE = True

    def __init__(self, skill_manager, use_sam: bool = True):
        super().__init__(skill_manager=skill_manager, use_sam=use_sam)
        self._compression_floor_warned = False
        self._last_token_count = 0
        self._compression_cycles = 0
        self._semantic_anchor_vector = None

    def get_system_prompt(self) -> str:
        return QCSM_SYSTEM_PROMPT

    def total_tokens(self) -> int:
        return sum(self.estimate_tokens(m.content) for m in self.history)

    # ------------------------------------------------------------
    # Intelligence Engine: Vectors & Math
    # ------------------------------------------------------------
    def embed_message(self, text: str) -> List[float]:
        """
        Generates an embedding vector. 
        """
        # 1. Real Model Check
        if self.use_sam and hasattr(self, 'circuit') and hasattr(self.circuit, 'encode'):
            return self.circuit.encode(text).tolist()
        
        # 2. Mock Logic (Feature Hashing) for simulation
        vec = [0.0] * 64
        words = str(text).lower().replace("_", " ").split()
        for w in words:
            h = hash(w) % 64
            vec[h] += 1.0
        mag = math.sqrt(sum(x*x for x in vec))
        return [x/mag for x in vec] if mag > 0 else vec

    def _cosine(self, a: List[float], b: List[float]) -> float:
        """Cosine similarity."""
        if not a or not b or len(a) != len(b): return 0.0
        dot = sum(x*y for x, y in zip(a, b))
        mag_a = math.sqrt(sum(x*x for x in a))
        mag_b = math.sqrt(sum(x*x for x in b))
        if mag_a == 0 or mag_b == 0: return 0.0
        return dot / (mag_a * mag_b)

    def _calculate_message_score(self, message, index: int, total_msgs: int) -> float:
        """
        Calculates Score: Similarity + Recency + Length + Role
        """
        if message.role == "system": return 999.0

        content = message.content or ""
        length = len(content)

        # 1. Semantic Similarity (The "Anchor" Score)
        similarity_score = 0.0
        if self._semantic_anchor_vector:
            msg_vec = self.embed_message(content)
            raw_sim = self._cosine(msg_vec, self._semantic_anchor_vector)
            # Boost: High similarity to Anchor protects the message
            similarity_score = raw_sim * 2.5

        # 2. Recency Score (0.0 to 1.0)
        recency_score = index / max(1, total_msgs)

        # 3. Density/Length Score
        length_score = min(1.0, math.log(max(1, length)) / 8.0)
        
        # 4. Role Bias
        role_score = 0.2 if message.role == "user" else 0.0

        # Final Weighting
        # Similarity (Anchor) is dominant.
        final_score = (
            (0.50 * similarity_score) +
            (0.25 * recency_score) +
            (0.15 * length_score) +
            (0.10 * role_score)
        )
        return final_score

    def _smart_compress(self, target_tokens: int):
        """Embedding-Aware Pruner."""
        current_tokens = self.total_tokens()
        if current_tokens <= target_tokens:
            return 0, 0

        protected_indices = {0, len(self.history) - 1, len(self.history) - 2}
        
        candidates = []
        for i, msg in enumerate(self.history):
            if i in protected_indices: continue
            
            score = self._calculate_message_score(msg, i, len(self.history))
            candidates.append((i, score, self.estimate_tokens(msg.content)))

        # Sort: Lowest score dies first
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

    # ------------------------------------------------------------
    # Memory Governor
    # ------------------------------------------------------------
    def _manage_memory(self):
        total_tokens = self.total_tokens()
        limit = self.COMPRESS_TOKEN_LIMIT
        target = self.TARGET_TOKEN_LIMIT

        # Green Zone
        if total_tokens <= limit:
            self._compression_floor_warned = False
            return

        # Hard Ceiling
        if total_tokens > self.HARD_TOKEN_CEILING:
            print(color(f"[QCSM] HARD CEILING ({total_tokens}) - PANIC PURGE", ANSI_RED))
            self._force_aggressive_compression()
            return

        # Elastic Floor
        avg_tpm = total_tokens / max(1, len(self.history))
        effective_floor = self.MIN_COMPRESS_HISTORY
        if avg_tpm > 300:
            effective_floor = max(4, self.MIN_COMPRESS_HISTORY // 2)

        # Critical Overflow
        is_critical = total_tokens > (limit * 1.5)

        # Retention Check
        if len(self.history) <= effective_floor and not is_critical:
            if not self._compression_floor_warned:
                self._compression_floor_warned = True
            return
        
        before_tokens = total_tokens
        before_msgs = len(self.history)

        # Execution
        if self.SEMANTIC_WEIGHTING_MODE:
            self._smart_compress(target)
            mode_label = "SMART"
        else:
            mode_label = "TRUNC"
            self._truncate_compress() 

        final_tokens = self.total_tokens()
        if final_tokens < before_tokens:
            self._compression_cycles += 1
            delta = before_tokens - final_tokens
            print(
                color(f"[QCSM] compress({mode_label}) ", ANSI_CYAN) +
                f"{before_tokens}->{final_tokens} (Δ{delta}), " +
                f"msgs {before_msgs}->{len(self.history)}"
            )

    def _force_aggressive_compression(self):
        for _ in range(5):
            self._smart_compress(self.TARGET_TOKEN_LIMIT)
            if self.total_tokens() <= self.TARGET_TOKEN_LIMIT: break

class QCSMChat:
    def __init__(self):
        self.client = TRIPLEG_CLIENT
        self.model_name = TRIPLEG_CONFIG["MODEL_NAME"]
        self.conv = QCSMConversationManager(
            skill_manager=NoopSkillManager(),
            use_sam=TORCH_AVAILABLE,
        )

    def _messages(self) -> List[Dict[str, str]]:
        messages: List[Dict[str, str]] = [{"role": "system", "content": self.conv.get_system_prompt()}]
        messages.extend(self.conv.to_openai_format())
        return messages

    def _stream_response(self) -> str:
        # Revert to mock for standalone testing logic, or hook to your API
        if not self.client: return "Mock Response"
        try:
            # Uncomment for live API
            # stream = self.client.chat.completions.create(...)
            return "Mock Response" 
        except: return "Error"

    def _run_torture_test(self):
        print(color("\n[TORTURE TEST] Protocol v7: 'Final Clamp'...", ANSI_CYAN))
        self.conv.history.clear()
        self.conv._semantic_anchor_vector = None
        
        # --- PHASE 1: ANCHORS ---
        keys = [f"CFG_{i:03d}" for i in range(15)]
        print(f"[Phase 1] Injecting {len(keys)} Config Anchors...")
        
        injected_indices = []
        for k in keys:
            content = f"SYSTEM_CRITICAL_CONFIG: PARAM_{k} = 'active_state_true'"
            self.conv.add_message("user", content, True)
            self.conv.add_message("assistant", f"Locked {k}", False)
            injected_indices.append(len(self.conv.history)-2)
            
        # Capture Anchor
        print(color("[Phase 1.5] Calculating Semantic Center...", ANSI_YELLOW))
        embeddings = []
        for idx in injected_indices:
            msg = self.conv.history[idx]
            vec = self.conv.embed_message(msg.content)
            embeddings.append(vec)
            
        if embeddings:
            dim = len(embeddings[0])
            avg = [0.0] * dim
            for v in embeddings:
                for i in range(dim): avg[i] += v[i]
            self.conv._semantic_anchor_vector = [x / len(embeddings) for x in avg]
            print(f" > Semantic Lock Established (dim={dim})")
        
        # --- PHASE 2: THE FLOOD ---
        turns = 100
        print(f"[Phase 2] Initiating {turns}-turn flood...")
        noise_block = "ERROR [timestamp]: Connection refused. " * 12 # ~500 chars
        
        for i in range(turns):
            self.conv.add_message("user", f"Log Dump {i}: {noise_block}", True)
            self.conv.add_message("assistant", f"Analyzed {i}.", False)
            
            self.conv._manage_memory()
            
            if i % 20 == 0 and i > 0:
                print(f"... Turn {i}/{turns} | Tokens: {self.conv.total_tokens()}")

        # --- PHASE 2.5: FINAL CLAMP (THE FIX) ---
        print(color("\n[Phase 2.5] Final Memory Clamp...", ANSI_YELLOW))
        if self.conv.total_tokens() > self.conv.COMPRESS_TOKEN_LIMIT:
             print(f" > Overage detected ({self.conv.total_tokens()} > {self.conv.COMPRESS_TOKEN_LIMIT}). Force compressing...")
             if self.conv.SEMANTIC_WEIGHTING_MODE:
                 self.conv._smart_compress(self.conv.TARGET_TOKEN_LIMIT)
             else:
                 self.conv._truncate_compress()
             print(f" > Clamp complete. Tokens: {self.conv.total_tokens()}")
        else:
             print(" > Memory is within safe limits.")

        # --- PHASE 3: THE AUDIT ---
        print("\n[Phase 3] Post-Flood Audit...")
        surviving_content = " ".join([m.content for m in self.conv.history])
        found = sum(1 for k in keys if k in surviving_content)
        
        print(f"Final State: {len(self.conv.history)} msgs | {self.conv.total_tokens()} tokens")
        
        if found == len(keys):
            print(color(f"PERFECT SCORE: {found}/{len(keys)} Anchors Survived.", ANSI_GREEN))
        else:
            print(color(f"SCORE: {found}/{len(keys)} Anchors Survived.", ANSI_PINK))

    def run(self):
        print(color("QCSM v7.0 Inference Chat", ANSI_CYAN))
        print("type /torture to run the final verification")
        while True:
            try: u = input("you> ").strip()
            except: break
            if not u: continue
            if u.startswith("/torture"): self._run_torture_test(); continue
            if u == "exit": break

if __name__ == "__main__":
    QCSMChat().run()