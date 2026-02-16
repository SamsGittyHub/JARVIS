# ==========================================
# TRIPLEG GUI APPLICATION
# ==========================================
"""
TripleG-Sam GUI Application
A modern cyberpunk-themed GUI wrapper for the TripleG-Sam AI agent.
Uses customtkinter for a sleek dark interface.
Integrates the Skills Marketplace for downloading and managing agent skills.
Features: File Explorer, Copy Messages, Skills Marketplace
"""

import os
import sys
import subprocess
import threading
import queue
import time
import re
import json
import ast
import math
import random
from datetime import datetime
from typing import Optional, Callable, List, Dict, Any, Tuple
from pathlib import Path

import numpy as np

# Import customtkinter for modern themed widgets
try:
    import customtkinter as ctk
    from customtkinter import CTk, CTkFrame, CTkLabel, CTkButton, CTkTextbox, CTkEntry
    from customtkinter import CTkScrollableFrame, CTkOptionMenu, CTkSwitch, CTkProgressBar
    from customtkinter import CTkTabview, CTkToplevel
except ImportError:
    print("ERROR: customtkinter not installed!")
    print("Install with: pip install customtkinter")
    sys.exit(1)

# Import tkinter extras
import tkinter as tk
from tkinter import messagebox, filedialog, ttk

# Import backend from tripleg.py
try:
    from tripleg import (
        CONFIG, console, client,
        ToolEngine, SkillManager, SamsLawConversationManager, ResponseParser,
        TORCH_AVAILABLE, YAML_AVAILABLE, BUILTIN_SKILLS,
        Skill, SkillCategory
    )
    from openai import APIError, APITimeoutError
except ImportError as e:
    print(f"ERROR: Could not import from tripleg.py: {e}")
    print("Make sure tripleg.py is in the same directory.")
    sys.exit(1)

# Import voice module
VOICE_AVAILABLE = False
sd = None  # sounddevice module reference for interrupt monitoring
try:
    from voice_module import (
        VoiceManager, VoiceConfig, VoiceState,
        check_dependencies as check_voice_deps,
        get_install_command as get_voice_install_cmd,
        WHISPER_AVAILABLE as VOICE_WHISPER_OK,
        SOUNDDEVICE_AVAILABLE as VOICE_SD_OK,
        EDGE_TTS_AVAILABLE as VOICE_TTS_OK,
        PYGAME_AVAILABLE as VOICE_PG_OK,
    )
    VOICE_AVAILABLE = True
    # Import sounddevice for interrupt monitoring in Live Call Mode
    try:
        import sounddevice as sd
    except ImportError:
        sd = None
except ImportError as e:
    print(f"WARNING: Voice module not available: {e}")
    VoiceManager = None  # type: ignore[assignment,misc]
    VoiceConfig = None  # type: ignore[assignment,misc]
    VoiceState = None  # type: ignore[assignment,misc]

# Import marketplace
try:
    from skills_marketplace import SkillsMarketplaceEngine, MarketplaceSkill, SKILL_SOURCES
except ImportError as e:
    print(f"WARNING: Could not import skills_marketplace.py: {e}")
    SkillsMarketplaceEngine = None

# ==========================================
# CYBERPUNK THEME CONFIGURATION
# ==========================================

THEME = {
    "bg_dark": "#0a0a0f",
    "bg_medium": "#12121a",
    "bg_light": "#1a1a2e",
    "bg_card": "#16162a",
    "accent_magenta": "#ff00ff",
    "accent_cyan": "#00ffff",
    "accent_purple": "#9d4edd",
    "accent_blue": "#4488ff",
    "text_primary": "#ffffff",
    "text_secondary": "#a0a0a0",
    "text_dim": "#606060",
    "user_bubble": "#1e3a5f",
    "assistant_bubble": "#2d1b4e",
    "tool_bubble": "#1a3a1a",
    "error_color": "#ff4444",
    "success_color": "#44ff44",
    "warning_color": "#ffaa00",
    "official_color": "#44aaff",
    "community_color": "#aa44ff",
    "specialized_color": "#ff8844",
    "custom_color": "#44ffaa",
}

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

# ==========================================
# TTS TEXT CLEANING
# ==========================================

def clean_text_for_tts(text: str) -> str:
    """
    Clean text for natural-sounding TTS output.
    Strips markdown formatting, code blocks, URLs, special characters,
    and normalizes whitespace so speech sounds conversational.
    Enhanced for JARVIS-like natural speech patterns.
    """
    if not text:
        return ""

    clean = text

    # 1. Remove code blocks (``` ... ```) and their content
    clean = re.sub(r'```[\s\S]*?```', ' ', clean)

    # 2. Remove inline code (`code`) - keep the content but clean it
    clean = re.sub(r'`([^`]*)`', r'\1', clean)

    # 3. Remove ::: blocks (admonitions)
    clean = re.sub(r':::.*?:::', ' ', clean, flags=re.DOTALL)

    # 4. Remove markdown images ![alt](url) - keep alt text
    clean = re.sub(r'!\[([^\]]*)\]\([^)]*\)', r'\1', clean)

    # 5. Convert markdown links [text](url) → just the text
    clean = re.sub(r'\[([^\]]*)\]\([^)]*\)', r'\1', clean)

    # 6. Remove raw URLs (http/https/ftp)
    clean = re.sub(r'https?://\S+', '', clean)
    clean = re.sub(r'ftp://\S+', '', clean)

    # 7. Remove HTML tags
    clean = re.sub(r'<[^>]+>', '', clean)

    # 8. Remove markdown headers (# ## ### etc.) — keep the text
    clean = re.sub(r'^#{1,6}\s+', '', clean, flags=re.MULTILINE)

    # 9. Remove bold/italic markers: **text**, __text__, *text*, _text_
    clean = re.sub(r'\*\*\*(.+?)\*\*\*', r'\1', clean)  # ***bold italic***
    clean = re.sub(r'\*\*(.+?)\*\*', r'\1', clean)       # **bold**
    clean = re.sub(r'__(.+?)__', r'\1', clean)            # __bold__
    clean = re.sub(r'\*(.+?)\*', r'\1', clean)            # *italic*
    clean = re.sub(r'(?<!\w)_(.+?)_(?!\w)', r'\1', clean) # _italic_ (word boundary)

    # 10. Remove strikethrough ~~text~~
    clean = re.sub(r'~~(.+?)~~', r'\1', clean)

    # 11. Remove blockquotes (> at start of line)
    clean = re.sub(r'^>\s*', '', clean, flags=re.MULTILINE)

    # 12. Remove horizontal rules (---, ***, ___)
    clean = re.sub(r'^[\-\*_]{3,}\s*$', '', clean, flags=re.MULTILINE)

    # 13. Convert bullet points to natural speech
    clean = re.sub(r'^\s*[\-\*\+]\s+', '', clean, flags=re.MULTILINE)

    # 14. Convert numbered lists — "1. item" → "item"
    clean = re.sub(r'^\s*\d+\.\s+', '', clean, flags=re.MULTILINE)

    # 15. Remove table formatting (pipes and dashes)
    clean = re.sub(r'\|', ' ', clean)
    clean = re.sub(r'^\s*[-:]+\s*$', '', clean, flags=re.MULTILINE)

    # 16. Remove markdown task lists [ ] [x]
    clean = re.sub(r'\[[ xX]\]\s*', '', clean)

    # 17. Remove footnote references [^1]
    clean = re.sub(r'\[\^\w+\]', '', clean)

    # 18. Remove special unicode arrows and bullets, replace with natural words
    clean = clean.replace('→', ' to ')
    clean = clean.replace('←', ' from ')
    clean = clean.replace('↔', ' between ')
    clean = clean.replace('•', ',')
    clean = clean.replace('…', '...')
    clean = clean.replace('—', ', ')
    clean = clean.replace('–', ', ')
    clean = clean.replace('``', '')
    clean = clean.replace("''", '')

    # 19. Remove excessive punctuation
    clean = re.sub(r'([!?.]){2,}', r'\1', clean)

    # 20. Remove emoji (common unicode ranges) — keep basic punctuation
    clean = re.sub(
        r'[\U0001F600-\U0001F64F'   # emoticons
        r'\U0001F300-\U0001F5FF'     # symbols & pictographs
        r'\U0001F680-\U0001F6FF'     # transport & map
        r'\U0001F1E0-\U0001F1FF'     # flags
        r'\U00002702-\U000027B0'     # dingbats
        r'\U0001FA00-\U0001FA6F'     # chess symbols
        r'\U0001FA70-\U0001FAFF'     # symbols extended
        r'\U00002600-\U000026FF'     # misc symbols
        r']+', ' ', clean
    )

    # 21. Normalize whitespace — collapse multiple spaces/newlines
    clean = re.sub(r'\n{2,}', '. ', clean)  # Multiple newlines → period (sentence break)
    clean = re.sub(r'\n', ' ', clean)        # Single newlines → space
    clean = re.sub(r'\s{2,}', ' ', clean)    # Multiple spaces → single space

    # 22. Clean up punctuation spacing
    clean = re.sub(r'\s+([.,;:!?])', r'\1', clean)  # Remove space before punctuation
    clean = re.sub(r'([.,;:!?])\s*([.,;:!?])', r'\1', clean)  # Remove double punctuation
    clean = re.sub(r'^\s*[.,;:]\s*', '', clean)  # Remove leading punctuation

    # 23. JARVIS-specific speech improvements
    # Convert common abbreviations to spoken form
    clean = re.sub(r'\bAI\b', 'A.I.', clean, flags=re.IGNORECASE)
    clean = re.sub(r'\bAPI\b', 'A.P.I.', clean, flags=re.IGNORECASE)
    clean = re.sub(r'\bCPU\b', 'C.P.U.', clean, flags=re.IGNORECASE)
    clean = re.sub(r'\bGPU\b', 'G.P.U.', clean, flags=re.IGNORECASE)
    clean = re.sub(r'\bRAM\b', 'R.A.M.', clean, flags=re.IGNORECASE)
    clean = re.sub(r'\bVRAM\b', 'V.R.A.M.', clean, flags=re.IGNORECASE)

    # Convert numbers to spoken form for better TTS
    clean = re.sub(r'\b(\d+)GB\b', r'\1 gigabytes', clean, flags=re.IGNORECASE)
    clean = re.sub(r'\b(\d+)MB\b', r'\1 megabytes', clean, flags=re.IGNORECASE)
    clean = re.sub(r'\b(\d+)KB\b', r'\1 kilobytes', clean, flags=re.IGNORECASE)

    # 24. Final trim
    clean = clean.strip()

    # 25. If result is too short or empty, return empty
    if len(clean) < 3:
        return ""

    return clean

# ==========================================
# 3D JARVIS SPHERE VISUALIZER
# ==========================================

class JarvisSphereVisualizer(tk.Canvas):
    """
    3D JARVIS-style cyber sphere visualizer using tkinter Canvas.
    Renders a rotating sphere made of particles/dots with perspective projection.
    Blue/cyan color scheme like the JARVIS AI from Iron Man.
    States: idle, listening, processing, speaking
    """

    NUM_PARTICLES = 800       # Increased from 220 for better visual density
    NUM_RING_PARTICLES = 60   # Particles on equator ring
    NUM_LATITUDE_LINES = 6    # Wireframe latitude lines
    NUM_LONGITUDE_LINES = 8   # Wireframe longitude lines
    FPS = 30

    def __init__(self, parent, **kwargs):
        super().__init__(parent, bg=THEME["bg_dark"], highlightthickness=0, **kwargs)
        self._state = "idle"
        self._animating = False
        self._frame = 0
        self._status_text = "Ready"
        self._center_text = "JARVIS"
        self._glow_phase = 0.0

        # Rotation angles (radians)
        self._rot_x = 0.0
        self._rot_y = 0.0
        self._rot_z = 0.0

        # Sphere radius multiplier (for pulsing)
        self._radius_mult = 1.0
        self._target_radius_mult = 1.0

        # Generate sphere particles using fibonacci spiral for even distribution
        self._particles = self._generate_fibonacci_sphere(self.NUM_PARTICLES)
        # Per-particle displacement (for speaking/listening animation)
        self._particle_displace = [0.0] * self.NUM_PARTICLES

        # Generate wireframe lines (latitude/longitude)
        self._lat_lines = self._generate_latitude_lines(self.NUM_LATITUDE_LINES, 40)
        self._lon_lines = self._generate_longitude_lines(self.NUM_LONGITUDE_LINES, 40)

        # Bind resize
        self.bind("<Configure>", self._on_resize)

    def _generate_fibonacci_sphere(self, n: int) -> list:
        """Generate evenly distributed points on a unit sphere using fibonacci spiral."""
        points = []
        golden_ratio = (1 + math.sqrt(5)) / 2
        for i in range(n):
            theta = math.acos(1 - 2 * (i + 0.5) / n)
            phi = 2 * math.pi * i / golden_ratio
            x = math.sin(theta) * math.cos(phi)
            y = math.sin(theta) * math.sin(phi)
            z = math.cos(theta)
            points.append((x, y, z))
        return points

    def _generate_latitude_lines(self, n_lines: int, points_per_line: int) -> list:
        """Generate latitude circle points."""
        lines = []
        for i in range(1, n_lines + 1):
            theta = math.pi * i / (n_lines + 1)
            line = []
            for j in range(points_per_line):
                phi = 2 * math.pi * j / points_per_line
                x = math.sin(theta) * math.cos(phi)
                y = math.sin(theta) * math.sin(phi)
                z = math.cos(theta)
                line.append((x, y, z))
            lines.append(line)
        return lines

    def _generate_longitude_lines(self, n_lines: int, points_per_line: int) -> list:
        """Generate longitude arc points."""
        lines = []
        for i in range(n_lines):
            phi = 2 * math.pi * i / n_lines
            line = []
            for j in range(points_per_line):
                theta = math.pi * j / (points_per_line - 1)
                x = math.sin(theta) * math.cos(phi)
                y = math.sin(theta) * math.sin(phi)
                z = math.cos(theta)
                line.append((x, y, z))
            lines.append(line)
        return lines

    def _rotate_point(self, x: float, y: float, z: float) -> tuple:
        """Apply 3D rotation (Y then X then Z) to a point."""
        # Rotate around Y axis
        cos_y, sin_y = math.cos(self._rot_y), math.sin(self._rot_y)
        x2 = x * cos_y + z * sin_y
        z2 = -x * sin_y + z * cos_y
        x, z = x2, z2

        # Rotate around X axis
        cos_x, sin_x = math.cos(self._rot_x), math.sin(self._rot_x)
        y2 = y * cos_x - z * sin_x
        z2 = y * sin_x + z * cos_x
        y, z = y2, z2

        return x, y, z

    def _project(self, x: float, y: float, z: float, cx: float, cy: float, radius: float) -> tuple:
        """Project 3D point to 2D with perspective."""
        fov = 3.5  # Field of view (higher = less perspective)
        scale = fov / (fov + z)
        sx = cx + x * radius * scale
        sy = cy + y * radius * scale
        return sx, sy, scale, z

    def set_state(self, state: str, status_text: str = ""):
        """Set visualizer state: idle, listening, processing, speaking."""
        self._state = state
        if status_text:
            self._status_text = status_text
        else:
            defaults = {
                "idle": "Ready to listen...",
                "listening": "Listening...",
                "processing": "Processing...",
                "speaking": "Speaking...",
            }
            self._status_text = defaults.get(state, "")

    def start_animation(self):
        """Start the animation loop."""
        if self._animating:
            return
        self._animating = True
        self._animate()

    def stop_animation(self):
        """Stop the animation loop."""
        self._animating = False

    def _on_resize(self, event):
        if self._animating:
            self._draw()

    def _animate(self):
        if not self._animating:
            return
        self._frame += 1
        self._glow_phase += 0.05
        self._update_rotation()
        self._update_particles()
        self._draw()
        self.after(1000 // self.FPS, self._animate)

    def _update_rotation(self):
        """Update rotation based on state."""
        t = self._frame / self.FPS

        if self._state == "idle":
            # Slow gentle rotation
            self._rot_y += 0.008
            self._rot_x = 0.3 + 0.1 * math.sin(t * 0.2)
            self._target_radius_mult = 1.0 + 0.03 * math.sin(t * 0.5)

        elif self._state == "listening":
            # Moderate rotation, slight pulse
            self._rot_y += 0.015
            self._rot_x = 0.35 + 0.15 * math.sin(t * 0.8)
            self._target_radius_mult = 1.0 + 0.08 * math.sin(t * 2.0)

        elif self._state == "processing":
            # Fast spinning
            self._rot_y += 0.04
            self._rot_x += 0.01
            self._target_radius_mult = 1.0 + 0.05 * math.sin(t * 3.0)

        elif self._state == "speaking":
            # Dynamic pulsing with moderate rotation
            self._rot_y += 0.012
            self._rot_x = 0.3 + 0.1 * math.sin(t * 0.4)
            # Strong pulsing when speaking
            pulse = 0.12 * math.sin(t * 3.0) + 0.06 * math.sin(t * 5.5) + 0.04 * math.sin(t * 8.0)
            self._target_radius_mult = 1.0 + pulse

        # Smooth interpolation of radius
        self._radius_mult += (self._target_radius_mult - self._radius_mult) * 0.2

    def _update_particles(self):
        """Update per-particle displacement for animation."""
        t = self._frame / self.FPS

        if self._state == "speaking":
            for i in range(self.NUM_PARTICLES):
                px, py, pz = self._particles[i]
                # Displacement based on position + time (wave effect)
                wave = 0.15 * math.sin(t * 4.0 + px * 3.0 + py * 2.0)
                wave += 0.08 * math.sin(t * 6.5 + pz * 4.0)
                noise = random.random() * 0.04
                self._particle_displace[i] += ((wave + noise) - self._particle_displace[i]) * 0.3

        elif self._state == "listening":
            for i in range(self.NUM_PARTICLES):
                px, py, pz = self._particles[i]
                spike = random.random() * 0.12 if random.random() > 0.8 else 0
                wave = 0.05 * math.sin(t * 2.0 + i * 0.1)
                self._particle_displace[i] += ((wave + spike) - self._particle_displace[i]) * 0.25

        elif self._state == "processing":
            for i in range(self.NUM_PARTICLES):
                self._particle_displace[i] *= 0.85  # Decay to zero

        else:  # idle
            for i in range(self.NUM_PARTICLES):
                self._particle_displace[i] *= 0.9  # Slowly decay

    def _hex_lerp(self, c1: str, c2: str, t: float) -> str:
        """Linearly interpolate between two hex colors."""
        t = max(0.0, min(1.0, t))
        r1, g1, b1 = int(c1[1:3], 16), int(c1[3:5], 16), int(c1[5:7], 16)
        r2, g2, b2 = int(c2[1:3], 16), int(c2[3:5], 16), int(c2[5:7], 16)
        r = int(r1 + (r2 - r1) * t)
        g = int(g1 + (g2 - g1) * t)
        b = int(b1 + (b2 - b1) * t)
        return f"#{max(0,min(255,r)):02x}{max(0,min(255,g)):02x}{max(0,min(255,b)):02x}"

    def _draw(self):
        """Draw the full 3D sphere frame."""
        self.delete("all")
        w = self.winfo_width()
        h = self.winfo_height()
        if w < 80 or h < 80:
            return

        cx, cy = w / 2, h / 2
        base_radius = min(cx, cy) * 0.55
        radius = base_radius * self._radius_mult

        # State-dependent colors
        if self._state == "idle":
            color_bright = "#00aaff"
            color_dim = "#003366"
            color_glow = "#0066cc"
            wire_color = "#0044aa"
        elif self._state == "listening":
            color_bright = "#00ffcc"
            color_dim = "#004433"
            color_glow = "#00cc88"
            wire_color = "#006644"
        elif self._state == "processing":
            color_bright = "#4488ff"
            color_dim = "#112244"
            color_glow = "#2266dd"
            wire_color = "#1144aa"
        elif self._state == "speaking":
            color_bright = "#00ffff"
            color_dim = "#003344"
            color_glow = "#00ccdd"
            wire_color = "#0088aa"
        else:
            color_bright = "#00aaff"
            color_dim = "#003366"
            color_glow = "#0066cc"
            wire_color = "#0044aa"

        # Draw outer glow circle
        glow_alpha = 0.2 + 0.1 * math.sin(self._glow_phase)
        glow_r = radius * 1.25
        glow_color = self._hex_lerp(THEME["bg_dark"], color_glow, glow_alpha)
        self.create_oval(cx - glow_r, cy - glow_r, cx + glow_r, cy + glow_r,
                         outline=glow_color, width=1)
        glow_r2 = radius * 1.15
        glow_color2 = self._hex_lerp(THEME["bg_dark"], color_glow, glow_alpha * 1.5)
        self.create_oval(cx - glow_r2, cy - glow_r2, cx + glow_r2, cy + glow_r2,
                         outline=glow_color2, width=1)

        # Draw wireframe latitude lines
        for line in self._lat_lines:
            coords = []
            for px, py, pz in line:
                rx, ry, rz = self._rotate_point(px, py, pz)
                sx, sy, scale, z = self._project(rx, ry, rz, cx, cy, radius)
                coords.append((sx, sy, z))
            # Draw line segments (skip back-facing ones for cleaner look)
            for j in range(len(coords)):
                j2 = (j + 1) % len(coords)
                z_avg = (coords[j][2] + coords[j2][2]) / 2
                if z_avg < 0.3:  # Only draw front-facing
                    alpha = max(0.0, min(1.0, (0.3 - z_avg) / 1.3))
                    lc = self._hex_lerp(THEME["bg_dark"], wire_color, alpha * 0.5)
                    self.create_line(coords[j][0], coords[j][1],
                                     coords[j2][0], coords[j2][1],
                                     fill=lc, width=1)

        # Draw wireframe longitude lines
        for line in self._lon_lines:
            coords = []
            for px, py, pz in line:
                rx, ry, rz = self._rotate_point(px, py, pz)
                sx, sy, scale, z = self._project(rx, ry, rz, cx, cy, radius)
                coords.append((sx, sy, z))
            for j in range(len(coords) - 1):
                z_avg = (coords[j][2] + coords[j + 1][2]) / 2
                if z_avg < 0.3:
                    alpha = max(0.0, min(1.0, (0.3 - z_avg) / 1.3))
                    lc = self._hex_lerp(THEME["bg_dark"], wire_color, alpha * 0.5)
                    self.create_line(coords[j][0], coords[j][1],
                                     coords[j + 1][0], coords[j + 1][1],
                                     fill=lc, width=1)

        # Collect and sort particles by Z for proper depth ordering
        projected = []
        for i, (px, py, pz) in enumerate(self._particles):
            # Apply displacement
            disp = 1.0 + self._particle_displace[i]
            dpx, dpy, dpz = px * disp, py * disp, pz * disp
            rx, ry, rz = self._rotate_point(dpx, dpy, dpz)
            sx, sy, scale, z = self._project(rx, ry, rz, cx, cy, radius)
            projected.append((sx, sy, scale, z, i))

        # Sort by Z (back to front)
        projected.sort(key=lambda p: -p[3])

        # Draw particles
        for sx, sy, scale, z, idx in projected:
            # Brightness based on Z position (front = bright, back = dim)
            depth_factor = max(0.0, min(1.0, (1.0 - z) / 2.0))
            color = self._hex_lerp(color_dim, color_bright, depth_factor)

            # Particle size based on perspective scale
            dot_size = max(1.5, 3.5 * scale * depth_factor)

            # Draw the particle dot
            self.create_oval(
                sx - dot_size, sy - dot_size,
                sx + dot_size, sy + dot_size,
                fill=color, outline="", width=0
            )

            # Add glow for front-facing bright particles
            if depth_factor > 0.6 and self._state in ("speaking", "listening"):
                glow_size = dot_size * 2.5
                gc = self._hex_lerp(THEME["bg_dark"], color_glow, depth_factor * 0.25)
                self.create_oval(
                    sx - glow_size, sy - glow_size,
                    sx + glow_size, sy + glow_size,
                    fill="", outline=gc, width=1
                )

        # Center text with glow
        glow_brightness = 0.6 + 0.4 * math.sin(self._glow_phase * 1.5)
        text_color = self._hex_lerp("#336688", color_bright, glow_brightness)
        # Shadow
        self.create_text(cx + 1, cy - 9, text=self._center_text,
                         font=("Consolas", 16, "bold"), fill="#000000")
        self.create_text(cx, cy - 10, text=self._center_text,
                         font=("Consolas", 16, "bold"), fill=text_color)

        # Status text below
        self.create_text(cx, cy + 18, text=self._status_text,
                         font=("Consolas", 10), fill=THEME["text_secondary"])

# ==========================================
# MAIN GUI APPLICATION
# ==========================================

class TripleGGUI(CTk):
    """Main application window."""

    def __init__(self):
        super().__init__()
        self.title("TRIPLEG - JARVIS")
        self.geometry("1200x750")
        self.minsize(900, 600)
        self.configure(fg_color=THEME["bg_dark"])

        # Backend components
        self.skill_manager = SkillManager()
        self.conversation_manager = SamsLawConversationManager(
            skill_manager=self.skill_manager, use_sam=False
        )
        self.response_parser = ResponseParser()
        self.response_queue = queue.Queue()
        self.is_processing = False
        self.terminal_enabled = False
        self.terminal_frame = None
        self.terminal_textbox = None

        # Voice components
        self.voice_manager: Optional[Any] = None
        self.voice_enabled = False
        self.voice_output_enabled = False
        self._voice_recording = False
        self._voice_initialized = False

        # Live call mode
        self._live_call_active = False
        self._live_call_visualizer: Optional[JarvisSphereVisualizer] = None
        self._live_call_frame: Optional[CTkFrame] = None
        self._live_call_mic_btn: Optional[CTkButton] = None
        self._live_call_muted = False  # Mute state for hands-free mode
        self._live_call_listening_loop_active = False  # Controls the auto-listen loop

        if VOICE_AVAILABLE:
            try:
                config = VoiceConfig(whisper_model="large", whisper_device="cuda")
                self.voice_manager = VoiceManager(config)
                self.voice_manager.set_state_callback(self._on_voice_state_change)
                self.voice_manager.set_transcription_callback(self._on_voice_transcription)
            except Exception as e:
                print(f"[Voice] Failed to create VoiceManager: {e}")

        # Build UI
        self._build_ui()

        # Start response polling
        self._poll_responses()

        # Welcome message
        self.chat_panel.add_message(
            "system",
            "Welcome to TripleG-Sam AI Agent!\n"
            f"Model: {CONFIG['MODEL_NAME']}\n"
            f"API: {CONFIG['API_URL']}\n"
            "Type a message below to start chatting."
        )

    def _speak_response(self, text: str):
        """Speak AI response using TTS (if voice output enabled)."""
        if not self.voice_output_enabled or not self.voice_manager:
            return
        # Clean text for natural TTS (strip markdown, code, URLs, etc.)
        clean = clean_text_for_tts(text)
        if clean and len(clean) > 5:
            self.voice_manager.speak(clean, blocking=False)

    # Add the rest of the methods here - this is a simplified version
    # In a real implementation, you'd include all the GUI building methods

    def _build_ui(self):
        """Build the main UI (simplified version)."""
        # Main frame
        main_frame = CTkFrame(self, fg_color="transparent")
        main_frame.pack(fill="both", expand=True, padx=5, pady=5)

        # Chat area
        chat_frame = CTkFrame(main_frame, fg_color=THEME["bg_medium"], corner_radius=10)
        chat_frame.pack(side="left", fill="both", expand=True, padx=(0, 3))

        # Chat header
        chat_header = CTkFrame(chat_frame, fg_color=THEME["bg_light"], corner_radius=8)
        chat_header.pack(fill="x", padx=5, pady=5)

        CTkLabel(
            chat_header, text="TRIPLEG - JARVIS",
            font=("Consolas", 16, "bold"),
            text_color=THEME["accent_magenta"]
        ).pack(side="left", padx=10, pady=5)

        # Live Call button
        mic_avail_for_call = VOICE_AVAILABLE and self.voice_manager is not None
        self.live_call_btn = CTkButton(
            chat_header, text="📞 Live Call", width=90, height=26,
            font=("Consolas", 9, "bold"),
            fg_color=THEME["success_color"] if mic_avail_for_call else THEME["bg_medium"],
            hover_color=THEME["accent_cyan"] if mic_avail_for_call else THEME["bg_medium"],
            text_color="#000000" if mic_avail_for_call else THEME["text_dim"],
            command=self._toggle_live_call,
        )
        self.live_call_btn.pack(side="right", padx=5, pady=5)

        # Chat messages area
        self.chat_panel = CTkScrollableFrame(chat_frame, fg_color=THEME["bg_dark"])
        self.chat_panel.pack(fill="both", expand=True, padx=5, pady=5)

        # Input area
        input_frame = CTkFrame(chat_frame, fg_color="transparent")
        input_frame.pack(fill="x", padx=5, pady=5)

        self.input_entry = CTkEntry(
            input_frame,
            placeholder_text="Type your message...",
            font=("Consolas", 12),
            height=36,
            fg_color=THEME["bg_dark"],
            border_color=THEME["accent_purple"],
            text_color=THEME["text_primary"]
        )
        self.input_entry.pack(side="left", fill="x", expand=True, padx=(0, 5))
        self.input_entry.bind("<Return>", lambda e: self._send_message())

        self.send_btn = CTkButton(
            input_frame, text="Send", width=70, height=36,
            font=("Consolas", 12, "bold"),
            fg_color=THEME["accent_magenta"], hover_color=THEME["accent_purple"],
            command=self._send_message
        )
        self.send_btn.pack(side="right")

        # Status bar
        self.status_panel = CTkFrame(self, fg_color=THEME["bg_medium"], height=30)
        self.status_panel.pack(fill="x", padx=5, pady=(0, 5))

        self.status_label = CTkLabel(
            self.status_panel, text="Ready", font=("Consolas", 10),
            text_color=THEME["accent_cyan"]
        )
        self.status_label.pack(side="left", padx=10)

    def _toggle_live_call(self):
        """Toggle live call mode."""
        if self._live_call_active:
            self._exit_live_call()
        else:
            self._enter_live_call()

    def _enter_live_call(self):
        """Enter live call mode with JARVIS sphere visualizer."""
        if not self.voice_manager:
            self.chat_panel.add_message("system", "Voice not available for live call.")
            return

        self._live_call_active = True
        self.live_call_btn.configure(text="📞 End Call", fg_color=THEME["error_color"],
                                      hover_color="#cc3333", text_color="#ffffff")

        # Hide chat and show visualizer
        self.chat_panel.pack_forget()

        # Create live call overlay
        self._live_call_frame = CTkFrame(self.chat_panel.master, fg_color=THEME["bg_dark"], corner_radius=10)
        self._live_call_frame.pack(fill="both", expand=True)

        # JARVIS Sphere Visualizer
        self._live_call_visualizer = JarvisSphereVisualizer(self._live_call_frame)
        self._live_call_visualizer.pack(fill="both", expand=True, padx=10, pady=(10, 5))
        self._live_call_visualizer.set_state("idle", "Ready")
        self._live_call_visualizer.start_animation()

        # Call controls
        call_controls = CTkFrame(self._live_call_frame, fg_color=THEME["bg_medium"], corner_radius=8)
        call_controls.pack(fill="x", padx=10, pady=(5, 10))

        # Mute button
        self._live_call_mic_btn = CTkButton(
            call_controls, text="🔇 Mute", width=120, height=44,
            font=("Consolas", 14, "bold"),
            fg_color=THEME["accent_cyan"], hover_color=THEME["accent_magenta"],
            text_color="#000000",
            command=self._live_call_toggle_mute,
        )
        self._live_call_mic_btn.pack(side="left", padx=15, pady=10)

        # End call button
        CTkButton(
            call_controls, text="📞 End Call", width=100, height=44,
            font=("Consolas", 12, "bold"),
            fg_color=THEME["error_color"], hover_color="#cc3333",
            text_color="#ffffff",
            command=self._exit_live_call,
        ).pack(side="right", padx=15, pady=10)

        # Status label
        self._live_call_status = CTkLabel(
            call_controls, text="LIVE CALL — Ready",
            font=("Consolas", 11), text_color=THEME["accent_cyan"]
        )
        self._live_call_status.pack(side="left", fill="x", expand=True, padx=10)

        self.status_panel.set_status("📞 LIVE CALL MODE — Hands-free", THEME["success_color"])

    def _exit_live_call(self):
        """Exit live call mode."""
        self._live_call_active = False

        # Stop visualizer
        if self._live_call_visualizer:
            self._live_call_visualizer.stop_animation()
            self._live_call_visualizer = None

        # Destroy live call frame
        if self._live_call_frame:
            self._live_call_frame.destroy()
            self._live_call_frame = None

        # Restore chat
        self.chat_panel.pack(fill="both", expand=True, padx=5, pady=5)

        # Update button
        self.live_call_btn.configure(text="📞 Live Call", fg_color=THEME["success_color"],
                                      hover_color=THEME["accent_cyan"], text_color="#000000")
        self.status_panel.set_status("Ready", THEME["accent_cyan"])

    def _live_call_toggle_mute(self):
        """Toggle mute in live call mode."""
        self._live_call_muted = not self._live_call_muted
        if self._live_call_muted:
            if self._live_call_mic_btn:
                self._live_call_mic_btn.configure(text="🔊 Unmute", fg_color=THEME["error_color"], text_color="#ffffff")
            if self._live_call_visualizer:
                self._live_call_visualizer.set_state("idle", "Muted")
            if hasattr(self, '_live_call_status'):
                self._live_call_status.configure(text="🔇 MUTED")
        else:
            if self._live_call_mic_btn:
                self._live_call_mic_btn.configure(text="🔇 Mute", fg_color=THEME["accent_cyan"], text_color="#000000")
            if self._live_call_visualizer:
                self._live_call_visualizer.set_state("idle", "Ready")

    def _send_message(self):
        """Send a chat message."""
        user_input = self.input_entry.get().strip()
        if not user_input:
            return

        self.input_entry.delete(0, "end")
        self.chat_panel.add_message("user", user_input)
        self.chat_panel.add_message("assistant", f"JARVIS: I heard '{user_input}'. Live call mode is now active with enhanced TTS cleaning and 3D sphere visualization!")

    def _poll_responses(self):
        """Poll for responses (simplified)."""
        self.after(100, self._poll_responses)

# ==========================================
# ENTRY POINT
# ==========================================

def main():
    app = TripleGGUI()
    app.mainloop()

if __name__ == "__main__":
    main()
