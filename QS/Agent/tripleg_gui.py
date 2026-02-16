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
# CHAT MESSAGE WIDGET
# ==========================================

class ChatMessage(CTkFrame):
    """A single chat message bubble with copy support."""

    def __init__(self, parent, role: str, content: str, timestamp: str = "", **kwargs):
        super().__init__(parent, **kwargs)
        self._content = content

        if role == "user":
            bg_color = THEME["user_bubble"]
            label_text = "You"
            label_color = THEME["accent_cyan"]
        elif role == "assistant":
            bg_color = THEME["assistant_bubble"]
            label_text = "JARVIS"
            label_color = THEME["accent_magenta"]
        elif role == "tool":
            bg_color = THEME["tool_bubble"]
            label_text = "Tool"
            label_color = THEME["warning_color"]
        elif role == "system":
            bg_color = THEME["bg_light"]
            label_text = "System"
            label_color = THEME["text_dim"]
        else:
            bg_color = THEME["bg_medium"]
            label_text = role.title()
            label_color = THEME["text_secondary"]

        self.configure(fg_color=bg_color, corner_radius=10)

        header_frame = CTkFrame(self, fg_color="transparent")
        header_frame.pack(fill="x", padx=10, pady=(8, 2))

        role_label = CTkLabel(
            header_frame,
            text=label_text,
            font=("Consolas", 11, "bold"),
            text_color=label_color
        )
        role_label.pack(side="left")

        # Copy button
        self._copy_btn = CTkButton(
            header_frame,
            text="📋",
            width=28, height=20,
            font=("Consolas", 10),
            fg_color="transparent",
            hover_color=THEME["bg_light"],
            text_color=THEME["text_dim"],
            command=self._copy_to_clipboard
        )
        self._copy_btn.pack(side="right", padx=(4, 0))

        if timestamp:
            time_label = CTkLabel(
                header_frame,
                text=timestamp,
                font=("Consolas", 9),
                text_color=THEME["text_dim"]
            )
            time_label.pack(side="right")

        content_label = CTkLabel(
            self,
            text=content,
            font=("Consolas", 12),
            text_color=THEME["text_primary"],
            wraplength=500,
            justify="left",
            anchor="w"
        )
        content_label.pack(fill="x", padx=10, pady=(2, 10))

        # Right-click context menu
        self._ctx_menu = tk.Menu(self, tearoff=0, bg=THEME["bg_light"],
                                  fg=THEME["text_primary"], activebackground=THEME["accent_purple"],
                                  activeforeground="#ffffff", font=("Consolas", 10))
        self._ctx_menu.add_command(label="Copy Message", command=self._copy_to_clipboard)
        self._ctx_menu.add_command(label="Copy as Code Block", command=self._copy_as_code)

        # Bind right-click to self and all children
        self.bind("<Button-3>", self._show_ctx_menu)
        content_label.bind("<Button-3>", self._show_ctx_menu)
        header_frame.bind("<Button-3>", self._show_ctx_menu)
        role_label.bind("<Button-3>", self._show_ctx_menu)

    def _copy_to_clipboard(self):
        self.clipboard_clear()
        self.clipboard_append(self._content)
        # Flash the copy button to confirm
        self._copy_btn.configure(text="✓", text_color=THEME["success_color"])
        self.after(1000, lambda: self._copy_btn.configure(text="📋", text_color=THEME["text_dim"]))

    def _copy_as_code(self):
        self.clipboard_clear()
        self.clipboard_append(f"```\n{self._content}\n```")
        self._copy_btn.configure(text="✓", text_color=THEME["success_color"])
        self.after(1000, lambda: self._copy_btn.configure(text="📋", text_color=THEME["text_dim"]))

    def _show_ctx_menu(self, event):
        try:
            self._ctx_menu.tk_popup(event.x_root, event.y_root)
        finally:
            self._ctx_menu.grab_release()


# ==========================================
# FILE EXPLORER PANEL
# ==========================================

# File extension to icon/color mapping
FILE_ICONS = {
    ".py": ("🐍", "#3572A5"),
    ".js": ("📜", "#f1e05a"),
    ".ts": ("📘", "#2b7489"),
    ".html": ("🌐", "#e34c26"),
    ".css": ("🎨", "#563d7c"),
    ".json": ("📋", "#40d47e"),
    ".md": ("📝", "#083fa1"),
    ".txt": ("📄", "#a0a0a0"),
    ".yaml": ("⚙️", "#cb171e"),
    ".yml": ("⚙️", "#cb171e"),
    ".toml": ("⚙️", "#9c4221"),
    ".cfg": ("⚙️", "#9c4221"),
    ".ini": ("⚙️", "#9c4221"),
    ".sh": ("🖥️", "#89e051"),
    ".bat": ("🖥️", "#C1F12E"),
    ".ps1": ("🖥️", "#012456"),
    ".cpp": ("⚡", "#f34b7d"),
    ".c": ("⚡", "#555555"),
    ".h": ("⚡", "#a0a0a0"),
    ".java": ("☕", "#b07219"),
    ".rs": ("🦀", "#dea584"),
    ".go": ("🐹", "#00ADD8"),
    ".rb": ("💎", "#701516"),
    ".php": ("🐘", "#4F5D95"),
    ".sql": ("🗃️", "#e38c00"),
    ".xml": ("📰", "#0060ac"),
    ".svg": ("🖼️", "#ff9900"),
    ".png": ("🖼️", "#a0a0a0"),
    ".jpg": ("🖼️", "#a0a0a0"),
    ".gif": ("🖼️", "#a0a0a0"),
    ".ico": ("🖼️", "#a0a0a0"),
    ".zip": ("📦", "#a0a0a0"),
    ".gz": ("📦", "#a0a0a0"),
    ".tar": ("📦", "#a0a0a0"),
    ".exe": ("⚙️", "#a0a0a0"),
    ".dll": ("⚙️", "#a0a0a0"),
    ".gitignore": ("🚫", "#f05032"),
    ".env": ("🔒", "#ecd53f"),
    ".log": ("📊", "#a0a0a0"),
}


class FileViewerPopup(CTkToplevel):
    """Popup window for viewing file contents."""

    def __init__(self, parent, filepath: str, **kwargs):
        super().__init__(parent, **kwargs)
        self.filepath = filepath
        fname = os.path.basename(filepath)
        self.title(f"File: {fname}")
        self.geometry("700x550")
        self.configure(fg_color=THEME["bg_dark"])
        self.transient(parent)

        # Header
        header = CTkFrame(self, fg_color=THEME["bg_medium"], corner_radius=8)
        header.pack(fill="x", padx=8, pady=(8, 4))

        ext = os.path.splitext(fname)[1].lower()
        icon, color = FILE_ICONS.get(ext, ("📄", THEME["text_dim"]))

        CTkLabel(header, text=f"{icon} {fname}", font=("Consolas", 14, "bold"),
                 text_color=color).pack(side="left", padx=10, pady=6)

        # Path label
        CTkLabel(header, text=filepath, font=("Consolas", 9),
                 text_color=THEME["text_dim"]).pack(side="right", padx=10, pady=6)

        # Button row
        btn_row = CTkFrame(self, fg_color="transparent")
        btn_row.pack(fill="x", padx=8, pady=4)

        CTkButton(btn_row, text="📋 Copy All", width=90, height=26,
                  font=("Consolas", 9), fg_color=THEME["accent_purple"],
                  hover_color=THEME["accent_magenta"],
                  command=self._copy_all).pack(side="left", padx=3)

        CTkButton(btn_row, text="📋 Copy Path", width=90, height=26,
                  font=("Consolas", 9), fg_color=THEME["bg_light"],
                  hover_color=THEME["accent_blue"],
                  command=self._copy_path).pack(side="left", padx=3)

        # File size label
        try:
            size = os.path.getsize(filepath)
            if size < 1024:
                size_str = f"{size} B"
            elif size < 1024 * 1024:
                size_str = f"{size / 1024:.1f} KB"
            else:
                size_str = f"{size / (1024 * 1024):.1f} MB"
        except OSError:
            size_str = "?"
        CTkLabel(btn_row, text=size_str, font=("Consolas", 9),
                 text_color=THEME["text_dim"]).pack(side="right", padx=10)

        # Content textbox
        self.content_box = CTkTextbox(
            self, font=("Consolas", 11),
            fg_color=THEME["bg_light"],
            text_color=THEME["text_primary"],
            wrap="none"
        )
        self.content_box.pack(fill="both", expand=True, padx=8, pady=(0, 4))

        # Load content
        self._load_file()

        # Close button
        CTkButton(self, text="Close", command=self.destroy, width=80, height=28,
                  fg_color=THEME["accent_purple"], hover_color=THEME["accent_magenta"],
                  font=("Consolas", 10)).pack(pady=(0, 8))

    def _load_file(self):
        try:
            # Check file size - don't load huge files
            size = os.path.getsize(self.filepath)
            if size > 2 * 1024 * 1024:  # 2MB limit
                self.content_box.insert("1.0", f"[File too large to display: {size / (1024*1024):.1f} MB]")
                self.content_box.configure(state="disabled")
                return

            # Try reading as text
            with open(self.filepath, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
            self.content_box.insert("1.0", content)
            self.content_box.configure(state="disabled")
        except Exception as e:
            self.content_box.insert("1.0", f"[Error reading file: {e}]")
            self.content_box.configure(state="disabled")

    def _copy_all(self):
        content = self.content_box.get("1.0", "end").strip()
        self.clipboard_clear()
        self.clipboard_append(content)

    def _copy_path(self):
        self.clipboard_clear()
        self.clipboard_append(self.filepath)


class FileExplorer(CTkFrame):
    """VS Code-style file explorer panel with tree view."""

    # Folders/files to hide
    HIDDEN_PATTERNS = {
        "__pycache__", ".git", ".venv", "venv", "node_modules",
        ".mypy_cache", ".pytest_cache", ".tox", "dist", "build",
        "*.pyc", "*.pyo", ".DS_Store", "Thumbs.db", ".env",
    }

    def __init__(self, parent, chat_input_callback=None, **kwargs):
        super().__init__(parent, **kwargs)
        self.configure(fg_color=THEME["bg_medium"])
        self.chat_input_callback = chat_input_callback
        self.current_root = os.path.expanduser("~")


        # Title bar
        title_frame = CTkFrame(self, fg_color="transparent")
        title_frame.pack(fill="x", padx=4, pady=(4, 0))

        CTkLabel(title_frame, text="📁 EXPLORER",
                 font=("Consolas", 11, "bold"),
                 text_color=THEME["accent_cyan"]).pack(side="left", padx=4)

        CTkButton(title_frame, text="⟳", width=24, height=22,
                  font=("Consolas", 11), fg_color="transparent",
                  hover_color=THEME["bg_light"], text_color=THEME["text_dim"],
                  command=self._refresh_tree).pack(side="right", padx=2)

        CTkButton(title_frame, text="📂", width=24, height=22,
                  font=("Consolas", 11), fg_color="transparent",
                  hover_color=THEME["bg_light"], text_color=THEME["text_dim"],
                  command=self._open_folder).pack(side="right", padx=2)

        # Current folder label
        self.folder_label = CTkLabel(
            self, text=self._short_path(self.current_root),
            font=("Consolas", 9), text_color=THEME["text_dim"],
            anchor="w"
        )
        self.folder_label.pack(fill="x", padx=8, pady=(2, 4))

        # Style the ttk Treeview to match cyberpunk theme
        style = ttk.Style()
        style.theme_use("clam")

        style.configure("Cyber.Treeview",
                         background=THEME["bg_dark"],
                         foreground=THEME["text_primary"],
                         fieldbackground=THEME["bg_dark"],
                         borderwidth=0,
                         font=("Consolas", 10),
                         rowheight=22)
        style.configure("Cyber.Treeview.Heading",
                         background=THEME["bg_medium"],
                         foreground=THEME["accent_cyan"],
                         font=("Consolas", 9, "bold"),
                         borderwidth=0)
        style.map("Cyber.Treeview",
                   background=[("selected", THEME["accent_purple"])],
                   foreground=[("selected", "#ffffff")])

        # Tree container
        tree_frame = CTkFrame(self, fg_color=THEME["bg_dark"], corner_radius=6)
        tree_frame.pack(fill="both", expand=True, padx=4, pady=(0, 4))

        # Treeview widget
        self.tree = ttk.Treeview(tree_frame, style="Cyber.Treeview",
                                  show="tree", selectmode="browse")
        self.tree.pack(fill="both", expand=True, padx=2, pady=2)

        # Scrollbar
        scrollbar = ttk.Scrollbar(tree_frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")

        # Bindings
        self.tree.bind("<Double-1>", self._on_double_click)
        self.tree.bind("<<TreeviewOpen>>", self._on_expand)

        # Right-click context menu
        self._ctx_menu = tk.Menu(self, tearoff=0, bg=THEME["bg_light"],
                                  fg=THEME["text_primary"],
                                  activebackground=THEME["accent_purple"],
                                  activeforeground="#ffffff",
                                  font=("Consolas", 10))
        self._ctx_menu.add_command(label="📋 Copy Path", command=self._copy_selected_path)
        self._ctx_menu.add_command(label="💬 Send to Chat", command=self._send_to_chat)
        self._ctx_menu.add_separator()
        self._ctx_menu.add_command(label="👁️ View File", command=self._view_selected)
        self._ctx_menu.add_command(label="📂 Open in Explorer", command=self._open_in_system)
        self.tree.bind("<Button-3>", self._show_ctx_menu)

        # Action buttons
        btn_frame = CTkFrame(self, fg_color="transparent")
        btn_frame.pack(fill="x", padx=4, pady=(0, 4))

        CTkButton(btn_frame, text="💬 Send Path", height=24,
                  font=("Consolas", 9), fg_color=THEME["accent_purple"],
                  hover_color=THEME["accent_magenta"],
                  command=self._send_to_chat).pack(side="left", fill="x", expand=True, padx=2)

        CTkButton(btn_frame, text="👁️ View", height=24,
                  font=("Consolas", 9), fg_color=THEME["bg_light"],
                  hover_color=THEME["accent_blue"],
                  command=self._view_selected).pack(side="left", fill="x", expand=True, padx=2)

        # Populate tree
        self._populate_root()

    def _short_path(self, path: str, max_len: int = 30) -> str:
        if len(path) <= max_len:
            return path
        parts = Path(path).parts
        if len(parts) <= 2:
            return path
        return str(Path(parts[0]) / "..." / parts[-1])

    def _should_hide(self, name: str) -> bool:
        if name.startswith(".") and name not in (".env",):
            return True
        for pattern in self.HIDDEN_PATTERNS:
            if pattern.startswith("*"):
                if name.endswith(pattern[1:]):
                    return True
            elif name == pattern:
                return True
        return False

    def _open_folder(self):
        folder = filedialog.askdirectory(
            title="Open Project Folder",
            initialdir=self.current_root
        )
        if folder:
            self.current_root = folder
            self.folder_label.configure(text=self._short_path(folder))
            self._populate_root()

    def _refresh_tree(self):
        self._populate_root()

    def _populate_root(self):
        # Clear existing
        for item in self.tree.get_children():
            self.tree.delete(item)

        root_name = os.path.basename(self.current_root) or self.current_root
        root_node = self.tree.insert("", "end", text=f"📁 {root_name}",
                                      values=(self.current_root,), open=True)
        self._populate_children(root_node, self.current_root)

    def _populate_children(self, parent_node, dir_path: str):
        # Clear placeholder children
        for child in self.tree.get_children(parent_node):
            self.tree.delete(child)

        try:
            entries = sorted(os.listdir(dir_path))
        except PermissionError:
            self.tree.insert(parent_node, "end", text="⚠️ Permission denied", values=("",))
            return
        except OSError:
            return

        # Separate dirs and files, sort dirs first
        dirs = []
        files = []
        for entry in entries:
            if self._should_hide(entry):
                continue
            full_path = os.path.join(dir_path, entry)
            if os.path.isdir(full_path):
                dirs.append((entry, full_path))
            else:
                files.append((entry, full_path))

        # Add directories
        for name, full_path in sorted(dirs, key=lambda x: x[0].lower()):
            node = self.tree.insert(parent_node, "end", text=f"📁 {name}",
                                     values=(full_path,))
            # Add placeholder so expand arrow shows
            self.tree.insert(node, "end", text="...", values=("__placeholder__",))

        # Add files
        for name, full_path in sorted(files, key=lambda x: x[0].lower()):
            ext = os.path.splitext(name)[1].lower()
            icon, _ = FILE_ICONS.get(ext, ("📄", THEME["text_dim"]))
            self.tree.insert(parent_node, "end", text=f"{icon} {name}",
                              values=(full_path,))

    def _on_expand(self, event):
        node = self.tree.focus()
        children = self.tree.get_children(node)
        # Check if this has the placeholder
        if len(children) == 1:
            vals = self.tree.item(children[0], "values")
            if vals and vals[0] == "__placeholder__":
                path = self.tree.item(node, "values")[0]
                self._populate_children(node, path)

    def _on_double_click(self, event):
        node = self.tree.focus()
        if not node:
            return
        vals = self.tree.item(node, "values")
        if not vals or not vals[0] or vals[0] == "__placeholder__":
            return
        path = vals[0]
        if os.path.isfile(path):
            FileViewerPopup(self.winfo_toplevel(), path)

    def _get_selected_path(self) -> Optional[str]:
        node = self.tree.focus()
        if not node:
            return None
        vals = self.tree.item(node, "values")
        if not vals or not vals[0] or vals[0] == "__placeholder__":
            return None
        return vals[0]

    def _copy_selected_path(self):
        path = self._get_selected_path()
        if path:
            self.clipboard_clear()
            self.clipboard_append(path)

    def _send_to_chat(self):
        path = self._get_selected_path()
        if path and self.chat_input_callback:
            self.chat_input_callback(path)

    def _view_selected(self):
        path = self._get_selected_path()
        if path and os.path.isfile(path):
            FileViewerPopup(self.winfo_toplevel(), path)

    def _open_in_system(self):
        path = self._get_selected_path()
        if path:
            target = path if os.path.isdir(path) else os.path.dirname(path)
            if sys.platform == "win32":
                os.startfile(target)  # type: ignore[attr-defined]
            elif sys.platform == "darwin":
                subprocess.Popen(["open", target])
            else:
                subprocess.Popen(["xdg-open", target])

    def _show_ctx_menu(self, event):
        # Select the item under cursor
        item = self.tree.identify_row(event.y)
        if item:
            self.tree.selection_set(item)
            self.tree.focus(item)
        try:
            self._ctx_menu.tk_popup(event.x_root, event.y_root)
        finally:
            self._ctx_menu.grab_release()


# ==========================================
# CHAT PANEL
# ==========================================

class ChatPanel(CTkScrollableFrame):
    """Scrollable chat message panel."""

    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        self.configure(fg_color=THEME["bg_dark"])
        self.messages = []

    def add_message(self, role: str, content: str, timestamp: str = ""):
        if not timestamp:
            timestamp = datetime.now().strftime("%H:%M:%S")

        msg = ChatMessage(self, role, content, timestamp)
        msg.pack(fill="x", padx=5, pady=5, anchor="w" if role != "user" else "e")
        self.messages.append(msg)

        self.after(50, lambda: self._parent_canvas.yview_moveto(1.0))

    def clear(self):
        for msg in self.messages:
            msg.destroy()
        self.messages = []


# ==========================================
# SKILL CARD WIDGET
# ==========================================

class SkillCard(CTkFrame):
    """A card widget displaying a single marketplace skill."""

    def __init__(self, parent, skill, on_action=None, **kwargs):
        super().__init__(parent, **kwargs)
        self.skill = skill
        self.on_action = on_action

        self.configure(fg_color=THEME["bg_card"], corner_radius=8)

        # Determine if this is a MarketplaceSkill or builtin Skill
        is_marketplace = hasattr(skill, 'source_repo')

        # Header row
        header = CTkFrame(self, fg_color="transparent")
        header.pack(fill="x", padx=8, pady=(6, 2))

        # Category badge color
        if is_marketplace:
            cat = getattr(skill, 'category', 'community')
        else:
            cat = 'builtin'

        badge_colors = {
            "official": THEME["official_color"],
            "community": THEME["community_color"],
            "specialized": THEME["specialized_color"],
            "custom": THEME["custom_color"],
            "builtin": THEME["accent_cyan"],
        }
        badge_color = badge_colors.get(cat, THEME["text_dim"])

        # Verified badge
        name_text = skill.name
        if is_marketplace and getattr(skill, 'verified', False):
            name_text = f"✓ {name_text}"

        name_label = CTkLabel(
            header,
            text=name_text,
            font=("Consolas", 12, "bold"),
            text_color=THEME["text_primary"],
            anchor="w"
        )
        name_label.pack(side="left", fill="x", expand=True)

        # Category badge
        cat_label = CTkLabel(
            header,
            text=cat.upper(),
            font=("Consolas", 8),
            text_color=badge_color,
        )
        cat_label.pack(side="right")

        # Description
        desc = getattr(skill, 'description', '')[:120]
        if desc:
            desc_label = CTkLabel(
                self,
                text=desc,
                font=("Consolas", 10),
                text_color=THEME["text_secondary"],
                wraplength=280,
                justify="left",
                anchor="w"
            )
            desc_label.pack(fill="x", padx=8, pady=(0, 2))

        # Tags row
        if is_marketplace and getattr(skill, 'tags', []):
            tags_text = " ".join(f"#{t}" for t in skill.tags[:4])
            tags_label = CTkLabel(
                self,
                text=tags_text,
                font=("Consolas", 9),
                text_color=THEME["accent_purple"],
                anchor="w"
            )
            tags_label.pack(fill="x", padx=8, pady=(0, 2))

        # Source
        if is_marketplace:
            source_text = f"from {skill.source_repo}"
        else:
            source_text = f"builtin v{skill.version}"

        source_label = CTkLabel(
            self,
            text=source_text,
            font=("Consolas", 9),
            text_color=THEME["text_dim"],
            anchor="w"
        )
        source_label.pack(fill="x", padx=8, pady=(0, 2))

        # Action buttons row
        btn_frame = CTkFrame(self, fg_color="transparent")
        btn_frame.pack(fill="x", padx=8, pady=(2, 6))

        # Determine state and buttons
        if is_marketplace:
            is_installed = getattr(skill, 'installed', False)
            is_active = getattr(skill, 'active', False)
        else:
            # Builtin skill - check via skill_manager if available in parent chain
            is_installed = True  # builtins are always "installed"
            # Check if skill is active by looking at the skill object's installed flag
            is_active = getattr(skill, 'installed', False) and skill.id in getattr(skill, '_active_ids', set())
            # Fallback: builtins show as installed but not active by default
            # The actual active state is managed by SkillManager and reflected on refresh
            if hasattr(parent, 'master') and hasattr(parent.master, 'skill_manager'):
                sm = parent.master.skill_manager
                is_active = skill.id in getattr(sm, 'active_skills', {})

        if is_active:
            # Show deactivate button
            deact_btn = CTkButton(
                btn_frame, text="Deactivate", width=80, height=24,
                font=("Consolas", 9),
                fg_color=THEME["error_color"], hover_color="#cc3333",
                command=lambda: self._do_action("deactivate")
            )
            deact_btn.pack(side="left", padx=2)

            status_label = CTkLabel(
                btn_frame, text="ACTIVE", font=("Consolas", 9, "bold"),
                text_color=THEME["success_color"]
            )
            status_label.pack(side="right", padx=4)

        elif is_installed:
            # Show activate button
            act_btn = CTkButton(
                btn_frame, text="Activate", width=70, height=24,
                font=("Consolas", 9),
                fg_color=THEME["success_color"], hover_color="#33cc33",
                text_color="#000000",
                command=lambda: self._do_action("activate")
            )
            act_btn.pack(side="left", padx=2)

            # Uninstall button (not for builtins)
            if is_marketplace:
                uninst_btn = CTkButton(
                    btn_frame, text="Uninstall", width=70, height=24,
                    font=("Consolas", 9),
                    fg_color=THEME["bg_light"], hover_color=THEME["error_color"],
                    command=lambda: self._do_action("uninstall")
                )
                uninst_btn.pack(side="left", padx=2)

            status_label = CTkLabel(
                btn_frame, text="INSTALLED", font=("Consolas", 9),
                text_color=THEME["accent_cyan"]
            )
            status_label.pack(side="right", padx=4)

        else:
            # Show install button
            inst_btn = CTkButton(
                btn_frame, text="Install", width=70, height=24,
                font=("Consolas", 9),
                fg_color=THEME["accent_purple"], hover_color=THEME["accent_magenta"],
                command=lambda: self._do_action("install")
            )
            inst_btn.pack(side="left", padx=2)

            # Detail button
            detail_btn = CTkButton(
                btn_frame, text="Details", width=60, height=24,
                font=("Consolas", 9),
                fg_color=THEME["bg_light"], hover_color=THEME["accent_blue"],
                command=lambda: self._do_action("details")
            )
            detail_btn.pack(side="left", padx=2)

    def _do_action(self, action: str):
        if self.on_action:
            skill_id = self.skill.id if hasattr(self.skill, 'id') else getattr(self.skill, 'id', '')
            self.on_action(action, skill_id, self.skill)


# ==========================================
# SKILL DETAIL POPUP
# ==========================================

class SkillDetailPopup(CTkToplevel):
    """Popup window showing full skill details."""

    def __init__(self, parent, skill, marketplace_engine=None, **kwargs):
        super().__init__(parent, **kwargs)
        self.title(f"Skill: {skill.name}")
        self.geometry("600x500")
        self.configure(fg_color=THEME["bg_dark"])

        # Make modal
        self.transient(parent)
        self.grab_set()

        # Header
        header = CTkFrame(self, fg_color=THEME["bg_medium"], corner_radius=10)
        header.pack(fill="x", padx=10, pady=10)

        CTkLabel(
            header, text=skill.name,
            font=("Consolas", 18, "bold"),
            text_color=THEME["accent_magenta"]
        ).pack(padx=10, pady=(10, 2))

        CTkLabel(
            header, text=skill.description,
            font=("Consolas", 11),
            text_color=THEME["text_secondary"],
            wraplength=550
        ).pack(padx=10, pady=(0, 10))

        # Info grid
        info_frame = CTkFrame(self, fg_color=THEME["bg_medium"], corner_radius=10)
        info_frame.pack(fill="x", padx=10, pady=5)

        is_mp = hasattr(skill, 'source_repo')
        info_items = [
            ("Source", getattr(skill, 'source_repo', 'builtin')),
            ("Category", getattr(skill, 'category', 'N/A')),
            ("Version", getattr(skill, 'version', '1.0.0')),
            ("Author", getattr(skill, 'author', 'Unknown')),
            ("Verified", "Yes" if getattr(skill, 'verified', False) else "No"),
            ("Tags", ", ".join(getattr(skill, 'tags', [])) or "None"),
        ]

        for label, value in info_items:
            row = CTkFrame(info_frame, fg_color="transparent")
            row.pack(fill="x", padx=10, pady=1)
            CTkLabel(row, text=f"{label}:", font=("Consolas", 10, "bold"),
                     text_color=THEME["accent_cyan"], width=80, anchor="w").pack(side="left")
            CTkLabel(row, text=str(value), font=("Consolas", 10),
                     text_color=THEME["text_primary"], anchor="w").pack(side="left", fill="x", expand=True)

        # Content
        CTkLabel(
            self, text="SKILL.md Content:",
            font=("Consolas", 11, "bold"),
            text_color=THEME["accent_cyan"]
        ).pack(padx=10, pady=(10, 2), anchor="w")

        content_box = CTkTextbox(
            self, font=("Consolas", 10),
            fg_color=THEME["bg_light"],
            text_color=THEME["text_primary"],
            wrap="word"
        )
        content_box.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        # Get content
        if marketplace_engine and is_mp:
            content = marketplace_engine.get_skill_content(skill.id)
        else:
            content = getattr(skill, 'content', '') or getattr(skill, 'system_prompt_addition', 'No content available.')

        content_box.insert("1.0", content)
        content_box.configure(state="disabled")

        # Close button
        CTkButton(
            self, text="Close", command=self.destroy,
            fg_color=THEME["accent_purple"], hover_color=THEME["accent_magenta"],
            width=100
        ).pack(pady=10)


# ==========================================
# CREATE SKILL POPUP
# ==========================================

class CreateSkillPopup(CTkToplevel):
    """Popup for creating a custom skill."""

    def __init__(self, parent, marketplace_engine, on_created=None, **kwargs):
        super().__init__(parent, **kwargs)
        self.title("Create Custom Skill")
        self.geometry("600x550")
        self.configure(fg_color=THEME["bg_dark"])
        self.marketplace_engine = marketplace_engine
        self.on_created = on_created

        self.transient(parent)
        self.grab_set()

        CTkLabel(
            self, text="Create Custom Skill",
            font=("Consolas", 18, "bold"),
            text_color=THEME["accent_magenta"]
        ).pack(pady=10)

        # Name
        CTkLabel(self, text="Name:", font=("Consolas", 11),
                 text_color=THEME["accent_cyan"]).pack(padx=20, anchor="w")
        self.name_entry = CTkEntry(self, font=("Consolas", 12),
                                   fg_color=THEME["bg_light"], text_color=THEME["text_primary"])
        self.name_entry.pack(fill="x", padx=20, pady=(0, 10))

        # Description
        CTkLabel(self, text="Description:", font=("Consolas", 11),
                 text_color=THEME["accent_cyan"]).pack(padx=20, anchor="w")
        self.desc_entry = CTkEntry(self, font=("Consolas", 12),
                                   fg_color=THEME["bg_light"], text_color=THEME["text_primary"])
        self.desc_entry.pack(fill="x", padx=20, pady=(0, 10))

        # Tags
        CTkLabel(self, text="Tags (comma-separated):", font=("Consolas", 11),
                 text_color=THEME["accent_cyan"]).pack(padx=20, anchor="w")
        self.tags_entry = CTkEntry(self, font=("Consolas", 12),
                                   fg_color=THEME["bg_light"], text_color=THEME["text_primary"])
        self.tags_entry.pack(fill="x", padx=20, pady=(0, 10))

        # Content
        CTkLabel(self, text="Skill Instructions (SKILL.md content):", font=("Consolas", 11),
                 text_color=THEME["accent_cyan"]).pack(padx=20, anchor="w")
        self.content_box = CTkTextbox(
            self, font=("Consolas", 11),
            fg_color=THEME["bg_light"], text_color=THEME["text_primary"],
            wrap="word", height=200
        )
        self.content_box.pack(fill="both", expand=True, padx=20, pady=(0, 10))
        self.content_box.insert("1.0", "# My Custom Skill\n\n## Instructions\n\nDescribe what this skill teaches the agent to do...\n")

        # Buttons
        btn_frame = CTkFrame(self, fg_color="transparent")
        btn_frame.pack(fill="x", padx=20, pady=10)

        CTkButton(
            btn_frame, text="Create Skill", command=self._create,
            fg_color=THEME["success_color"], hover_color="#33cc33",
            text_color="#000000", font=("Consolas", 12, "bold")
        ).pack(side="left", padx=5)

        CTkButton(
            btn_frame, text="Cancel", command=self.destroy,
            fg_color=THEME["bg_light"], hover_color=THEME["error_color"],
            font=("Consolas", 12)
        ).pack(side="right", padx=5)

    def _create(self):
        name = self.name_entry.get().strip()
        desc = self.desc_entry.get().strip()
        tags_str = self.tags_entry.get().strip()
        content = self.content_box.get("1.0", "end").strip()

        if not name:
            messagebox.showwarning("Missing Name", "Please enter a skill name.")
            return
        if not content:
            messagebox.showwarning("Missing Content", "Please enter skill instructions.")
            return

        tags = [t.strip() for t in tags_str.split(",") if t.strip()] if tags_str else []

        result = self.marketplace_engine.create_custom_skill(name, desc or name, content, tags)

        if self.on_created:
            self.on_created(result)

        self.destroy()


# ==========================================
# ENHANCED SKILLS MARKETPLACE PANEL
# ==========================================

class MarketplacePanel(CTkFrame):
    """Full marketplace panel with tabs, search, and skill cards."""

    def __init__(self, parent, skill_manager, chat_callback=None, **kwargs):
        super().__init__(parent, **kwargs)
        self.skill_manager = skill_manager
        self.chat_callback = chat_callback
        self.configure(fg_color=THEME["bg_medium"])

        # Initialize marketplace engine
        self.marketplace_engine = None
        if SkillsMarketplaceEngine:
            self.marketplace_engine = SkillsMarketplaceEngine()

        # Title
        title_frame = CTkFrame(self, fg_color="transparent")
        title_frame.pack(fill="x", padx=5, pady=(5, 0))

        CTkLabel(
            title_frame, text="SKILLS MARKETPLACE",
            font=("Consolas", 13, "bold"),
            text_color=THEME["accent_cyan"]
        ).pack(side="left", padx=5)

        # Stats label
        self.stats_label = CTkLabel(
            title_frame, text="",
            font=("Consolas", 9),
            text_color=THEME["text_dim"]
        )
        self.stats_label.pack(side="right", padx=5)

        # Search bar
        search_frame = CTkFrame(self, fg_color="transparent")
        search_frame.pack(fill="x", padx=5, pady=5)

        self.search_entry = CTkEntry(
            search_frame,
            placeholder_text="Search skills...",
            font=("Consolas", 10),
            height=28,
            fg_color=THEME["bg_dark"],
            border_color=THEME["accent_purple"],
            text_color=THEME["text_primary"]
        )
        self.search_entry.pack(side="left", fill="x", expand=True, padx=(0, 3))
        self.search_entry.bind("<Return>", lambda e: self._do_search())

        CTkButton(
            search_frame, text="Go", width=35, height=28,
            font=("Consolas", 9),
            fg_color=THEME["accent_purple"], hover_color=THEME["accent_magenta"],
            command=self._do_search
        ).pack(side="right")

        # Tab view
        self.tabview = CTkTabview(
            self, fg_color=THEME["bg_dark"],
            segmented_button_fg_color=THEME["bg_light"],
            segmented_button_selected_color=THEME["accent_purple"],
            segmented_button_unselected_color=THEME["bg_medium"],
        )
        self.tabview.pack(fill="both", expand=True, padx=5, pady=5)

        # Create tabs
        self.tab_browse = self.tabview.add("Browse")
        self.tab_installed = self.tabview.add("Installed")
        self.tab_active = self.tabview.add("Active")
        self.tab_sources = self.tabview.add("Sources")

        # Browse tab content
        self._build_browse_tab()

        # Installed tab content
        self._build_installed_tab()

        # Active tab content
        self._build_active_tab()

        # Sources tab content
        self._build_sources_tab()

        # Action buttons at bottom
        self._build_action_buttons()

        # Initial populate
        self._refresh_all_tabs()

    def _build_browse_tab(self):
        """Build the browse/marketplace tab."""
        # Category filter
        filter_frame = CTkFrame(self.tab_browse, fg_color="transparent")
        filter_frame.pack(fill="x", padx=2, pady=2)

        self.category_var = ctk.StringVar(value="All")
        categories = ["All", "Official", "Community", "Specialized", "Custom", "Builtin"]

        CTkOptionMenu(
            filter_frame,
            values=categories,
            variable=self.category_var,
            command=lambda _: self._populate_browse(),
            font=("Consolas", 9),
            fg_color=THEME["bg_light"],
            button_color=THEME["accent_purple"],
            button_hover_color=THEME["accent_magenta"],
            width=100, height=24
        ).pack(side="left", padx=2)

        # Skill list
        self.browse_list = CTkScrollableFrame(
            self.tab_browse, fg_color=THEME["bg_dark"]
        )
        self.browse_list.pack(fill="both", expand=True, padx=2, pady=2)

    def _build_installed_tab(self):
        self.installed_list = CTkScrollableFrame(
            self.tab_installed, fg_color=THEME["bg_dark"]
        )
        self.installed_list.pack(fill="both", expand=True, padx=2, pady=2)

    def _build_active_tab(self):
        self.active_list = CTkScrollableFrame(
            self.tab_active, fg_color=THEME["bg_dark"]
        )
        self.active_list.pack(fill="both", expand=True, padx=2, pady=2)

    def _build_sources_tab(self):
        self.sources_list = CTkScrollableFrame(
            self.tab_sources, fg_color=THEME["bg_dark"]
        )
        self.sources_list.pack(fill="both", expand=True, padx=2, pady=2)

        # Add source button
        add_frame = CTkFrame(self.tab_sources, fg_color="transparent")
        add_frame.pack(fill="x", padx=2, pady=5)

        self.source_entry = CTkEntry(
            add_frame,
            placeholder_text="owner/repo or GitHub URL",
            font=("Consolas", 9), height=26,
            fg_color=THEME["bg_dark"],
            text_color=THEME["text_primary"]
        )
        self.source_entry.pack(side="left", fill="x", expand=True, padx=(0, 3))

        CTkButton(
            add_frame, text="Add", width=40, height=26,
            font=("Consolas", 9),
            fg_color=THEME["accent_purple"], hover_color=THEME["accent_magenta"],
            command=self._add_source
        ).pack(side="right")

    def _build_action_buttons(self):
        btn_frame = CTkFrame(self, fg_color="transparent")
        btn_frame.pack(fill="x", padx=5, pady=5)

        CTkButton(
            btn_frame, text="Fetch All", height=26,
            font=("Consolas", 9, "bold"),
            fg_color=THEME["accent_magenta"], hover_color=THEME["accent_purple"],
            command=self._fetch_all_sources
        ).pack(side="left", fill="x", expand=True, padx=2)

        CTkButton(
            btn_frame, text="Create", height=26,
            font=("Consolas", 9),
            fg_color=THEME["success_color"], hover_color="#33cc33",
            text_color="#000000",
            command=self._create_skill
        ).pack(side="left", fill="x", expand=True, padx=2)

        CTkButton(
            btn_frame, text="GitHub", height=26,
            font=("Consolas", 9),
            fg_color=THEME["accent_blue"], hover_color="#3377dd",
            command=self._install_from_url
        ).pack(side="left", fill="x", expand=True, padx=2)

    # ------------------------------------------
    # POPULATE TABS
    # ------------------------------------------

    def _refresh_all_tabs(self):
        self._populate_browse()
        self._populate_installed()
        self._populate_active()
        self._populate_sources()
        self._update_stats()

    def _clear_frame(self, frame):
        for widget in frame.winfo_children():
            widget.destroy()

    def _populate_browse(self):
        self._clear_frame(self.browse_list)
        category_filter = self.category_var.get().lower()

        skills = []

        # Add builtin skills
        for skill_id, skill in BUILTIN_SKILLS.items():
            if category_filter in ("all", "builtin"):
                skills.append(("builtin", skill))

        # Add marketplace skills
        if self.marketplace_engine:
            for skill in self.marketplace_engine.registry.values():
                if category_filter == "all" or category_filter == skill.category:
                    skills.append(("marketplace", skill))

        if not skills:
            CTkLabel(
                self.browse_list,
                text="No skills found.\nClick 'Fetch All' to download from sources.",
                font=("Consolas", 10),
                text_color=THEME["text_dim"]
            ).pack(pady=20)
            return

        for source_type, skill in skills:
            card = SkillCard(
                self.browse_list, skill,
                on_action=self._handle_skill_action
            )
            card.pack(fill="x", padx=2, pady=2)

    def _populate_installed(self):
        self._clear_frame(self.installed_list)

        skills = []
        if self.marketplace_engine:
            skills = self.marketplace_engine.get_installed_skills()

        if not skills:
            CTkLabel(
                self.installed_list,
                text="No installed skills yet.",
                font=("Consolas", 10),
                text_color=THEME["text_dim"]
            ).pack(pady=20)
            return

        for skill in skills:
            card = SkillCard(self.installed_list, skill, on_action=self._handle_skill_action)
            card.pack(fill="x", padx=2, pady=2)

    def _populate_active(self):
        self._clear_frame(self.active_list)
        skills = []
        if self.marketplace_engine:
            skills = self.marketplace_engine.get_active_skills()

        if not skills:
            CTkLabel(
                self.active_list,
                text="No active skills.\nActivate installed skills to use them.",
                font=("Consolas", 10),
                text_color=THEME["text_dim"]
            ).pack(pady=20)
            return

        for skill in skills:
            card = SkillCard(self.active_list, skill, on_action=self._handle_skill_action)
            card.pack(fill="x", padx=2, pady=2)

    def _populate_sources(self):
        self._clear_frame(self.sources_list)
        if not self.marketplace_engine:
            return
        all_sources = self.marketplace_engine.get_all_sources()
        for repo_id, info in all_sources.items():
            row = CTkFrame(self.sources_list, fg_color=THEME["bg_card"], corner_radius=6)
            row.pack(fill="x", padx=2, pady=2)
            is_custom = repo_id in self.marketplace_engine.custom_sources
            badge = "CUSTOM" if is_custom else info.get("category", "").upper()
            badge_color = THEME["custom_color"] if is_custom else THEME["official_color"]
            CTkLabel(row, text=badge, font=("Consolas", 8), text_color=badge_color).pack(side="left", padx=4)
            CTkLabel(row, text=repo_id, font=("Consolas", 10, "bold"),
                     text_color=THEME["text_primary"]).pack(side="left", padx=4)
            CTkLabel(row, text=info.get("description", "")[:50], font=("Consolas", 9),
                     text_color=THEME["text_secondary"]).pack(side="left", fill="x", expand=True, padx=4)
            CTkButton(
                row, text="Fetch", width=45, height=22, font=("Consolas", 8),
                fg_color=THEME["accent_purple"], hover_color=THEME["accent_magenta"],
                command=lambda rid=repo_id: self._fetch_single_source(rid)
            ).pack(side="right", padx=2, pady=2)
            if is_custom:
                CTkButton(
                    row, text="X", width=22, height=22, font=("Consolas", 8),
                    fg_color=THEME["error_color"], hover_color="#cc3333",
                    command=lambda rid=repo_id: self._remove_source(rid)
                ).pack(side="right", padx=2, pady=2)

    def _update_stats(self):
        if not self.marketplace_engine:
            self.stats_label.configure(text="No engine")
            return
        stats = self.marketplace_engine.get_stats()
        self.stats_label.configure(
            text=f"{stats['total_skills']} skills | {stats['installed']} installed | {stats['active']} active"
        )

    # ------------------------------------------
    # ACTIONS
    # ------------------------------------------

    def _handle_skill_action(self, action: str, skill_id: str, skill):
        msg = ""
        
        # Check if this is a builtin skill (from BUILTIN_SKILLS)
        is_builtin = skill_id in BUILTIN_SKILLS
        
        if action == "details":
            SkillDetailPopup(self.winfo_toplevel(), skill, self.marketplace_engine)
            return
        
        if is_builtin:
            # Handle builtin skills via skill_manager
            if action == "install":
                msg = self.skill_manager.install_skill(skill_id)
            elif action == "uninstall":
                msg = f"Cannot uninstall builtin skill: {skill_id}"
            elif action == "activate":
                # First install if not installed
                if skill_id not in self.skill_manager.installed_skills:
                    install_msg = self.skill_manager.install_skill(skill_id)
                    if self.chat_callback:
                        self.chat_callback("system", install_msg)
                msg = self.skill_manager.activate_skill(skill_id)
            elif action == "deactivate":
                msg = self.skill_manager.deactivate_skill(skill_id)
        else:
            # Handle marketplace skills via marketplace_engine
            if not self.marketplace_engine:
                msg = "Marketplace engine not available"
            elif action == "install":
                msg = self.marketplace_engine.install_skill(skill_id)
            elif action == "uninstall":
                msg = self.marketplace_engine.uninstall_skill(skill_id)
            elif action == "activate":
                msg = self.marketplace_engine.activate_skill(skill_id)
            elif action == "deactivate":
                msg = self.marketplace_engine.deactivate_skill(skill_id)

        if msg and self.chat_callback:
            self.chat_callback("system", msg)
        self._refresh_all_tabs()

    def _do_search(self):
        query = self.search_entry.get().strip()
        if not query or not self.marketplace_engine:
            self._populate_browse()
            return
        self._clear_frame(self.browse_list)
        results = self.marketplace_engine.search_skills(query)
        if not results:
            CTkLabel(self.browse_list, text=f"No results for '{query}'",
                     font=("Consolas", 10), text_color=THEME["text_dim"]).pack(pady=20)
            return
        for skill in results:
            card = SkillCard(self.browse_list, skill, on_action=self._handle_skill_action)
            card.pack(fill="x", padx=2, pady=2)

    def _fetch_all_sources(self):
        engine = self.marketplace_engine
        if not engine:
            return
        if self.chat_callback:
            self.chat_callback("system", "Fetching skills from all sources...")

        def _do_fetch():
            total, messages = engine.fetch_all_sources()
            self.after(0, lambda: self._on_fetch_done(messages))

        threading.Thread(target=_do_fetch, daemon=True).start()

    def _on_fetch_done(self, messages):
        self._refresh_all_tabs()
        if self.chat_callback:
            self.chat_callback("system", "\n".join(messages))

    def _fetch_single_source(self, repo_id):
        engine = self.marketplace_engine
        if not engine:
            return
        if self.chat_callback:
            self.chat_callback("system", f"Fetching from {repo_id}...")

        def _do():
            count, msgs = engine.fetch_skills_from_source(repo_id)
            self.after(0, lambda: self._on_fetch_done(msgs))

        threading.Thread(target=_do, daemon=True).start()

    def _add_source(self):
        url = self.source_entry.get().strip()
        if not url or not self.marketplace_engine:
            return
        result = self.marketplace_engine.add_source(url)
        self.source_entry.delete(0, "end")
        self._populate_sources()
        if self.chat_callback:
            self.chat_callback("system", str(result or "Source added"))

    def _remove_source(self, repo_id):
        if not self.marketplace_engine:
            return
        result = self.marketplace_engine.remove_source(repo_id)
        self._populate_sources()
        if self.chat_callback:
            self.chat_callback("system", str(result or "Source removed"))

    def _create_skill(self):
        if not self.marketplace_engine:
            return
        def on_created(result):
            self._refresh_all_tabs()
            if self.chat_callback:
                self.chat_callback("system", str(result or "Skill created"))
        CreateSkillPopup(self.winfo_toplevel(), self.marketplace_engine, on_created)

    def _install_from_url(self):
        engine = self.marketplace_engine
        if not engine:
            return
        dialog = CTkToplevel(self.winfo_toplevel())
        dialog.title("Install from GitHub")
        dialog.geometry("450x150")
        dialog.configure(fg_color=THEME["bg_dark"])
        dialog.transient(self.winfo_toplevel())
        dialog.grab_set()

        CTkLabel(dialog, text="Enter GitHub repo (owner/repo or URL):",
                 font=("Consolas", 11), text_color=THEME["accent_cyan"]).pack(padx=15, pady=(15, 5))
        entry = CTkEntry(dialog, font=("Consolas", 12), fg_color=THEME["bg_light"],
                         text_color=THEME["text_primary"])
        entry.pack(fill="x", padx=15, pady=5)

        def do_install():
            url = entry.get().strip()
            if url:
                dialog.destroy()
                if self.chat_callback:
                    self.chat_callback("system", f"Installing from {url}...")
                def _do():
                    result = engine.install_from_github_url(url)
                    self.after(0, lambda: self._on_fetch_done([str(result or "Done")]))
                threading.Thread(target=_do, daemon=True).start()

        CTkButton(dialog, text="Install", command=do_install,
                  fg_color=THEME["accent_purple"], hover_color=THEME["accent_magenta"],
                  font=("Consolas", 12, "bold")).pack(pady=10)


# ==========================================
# TTS TEXT CLEANING
# ==========================================

def clean_text_for_tts(text: str) -> str:
    """
    Clean text for natural-sounding TTS output.
    Strips markdown formatting, code blocks, URLs, special characters,
    and normalizes whitespace so speech sounds conversational.
    """
    if not text:
        return ""

    clean = text

    # 1. Remove code blocks (``` ... ```) and their content
    clean = re.sub(r'```[\s\S]*?```', ' ', clean)

    # 2. Remove inline code (`code`)
    clean = re.sub(r'`([^`]*)`', r'\1', clean)

    # 3. Remove ::: blocks (admonitions)
    clean = re.sub(r':::.*?:::', ' ', clean, flags=re.DOTALL)

    # 4. Remove markdown images ![alt](url)
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

    # 23. Final trim
    clean = clean.strip()

    # 24. If result is too short or empty, return empty
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

    Audio-reactive: When speaking, particles move based on real FFT frequency
    analysis of the TTS audio, making the sphere appear alive.
    """

    NUM_PARTICLES = 900       # Total sphere particles (increased for dense, alive look)
    NUM_RING_PARTICLES = 80   # Particles on equator ring
    NUM_LATITUDE_LINES = 8    # Wireframe latitude lines
    NUM_LONGITUDE_LINES = 10  # Wireframe longitude lines
    FPS = 30

    # Audio analysis constants
    NUM_FREQ_BANDS = 32       # Number of FFT frequency bands (increased for precise tone mapping)
    AUDIO_FPS = 60            # FFT frames per second (doubled for smoother audio response)

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

        # Pre-compute each particle's latitude band index and angular position
        # for efficient audio-reactive mapping
        self._particle_band_idx = []
        self._particle_phi = []
        for px, py, pz in self._particles:
            # Map py (-1..1) to band index (0..NUM_FREQ_BANDS-1)
            band = int((py + 1.0) / 2.0 * (self.NUM_FREQ_BANDS - 1))
            band = max(0, min(self.NUM_FREQ_BANDS - 1, band))
            self._particle_band_idx.append(band)
            # Angular position around Y axis (for wave propagation)
            self._particle_phi.append(math.atan2(pz, px))

        # Audio analysis state
        self._audio_fft_frames: list = []       # Pre-computed FFT frames (list of np arrays)
        self._speaking_start_time: float = 0.0  # When speaking playback started
        self._audio_energy: float = 0.0         # Overall audio energy (for pulsing)
        self._current_bands = np.zeros(self.NUM_FREQ_BANDS)  # Current frequency band values
        self._prev_bands = np.zeros(self.NUM_FREQ_BANDS)     # Previous frame bands (smoothing)

        # Generate wireframe lines (latitude/longitude)
        self._lat_lines = self._generate_latitude_lines(self.NUM_LATITUDE_LINES, 50)
        self._lon_lines = self._generate_longitude_lines(self.NUM_LONGITUDE_LINES, 50)

        # Bind resize
        self.bind("<Configure>", self._on_resize)

    # ------------------------------------------
    # AUDIO ANALYSIS METHODS
    # ------------------------------------------

    def load_audio_for_visualization(self, filepath: str) -> bool:
        """
        Load an audio file and pre-compute FFT frames for visualization.
        Call this before starting TTS playback.
        
        Args:
            filepath: Path to the audio file (MP3, WAV, etc.)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            import pygame
            
            # Ensure pygame mixer is initialized
            if not pygame.mixer.get_init():
                pygame.mixer.init()
            
            # Load audio as Sound object to get raw samples
            sound = pygame.mixer.Sound(filepath)
            
            # Get raw audio data as numpy array
            try:
                import pygame.sndarray
                raw = pygame.sndarray.array(sound)
            except Exception as e:
                print(f"[Visualizer] pygame.sndarray failed: {e}")
                self._audio_fft_frames = []
                return False
            
            # Convert to mono float32
            if raw.ndim > 1:
                # Stereo -> mono by averaging channels
                samples = raw.mean(axis=1).astype(np.float32)
            else:
                samples = raw.astype(np.float32)
            
            # Normalize to -1..1 range
            max_val = np.max(np.abs(samples))
            if max_val > 0:
                samples = samples / max_val
            
            # Get sample rate from mixer
            sample_rate = pygame.mixer.get_init()[0]
            
            # Compute FFT frames
            self._compute_fft_frames(samples, sample_rate)
            
            print(f"[Visualizer] Loaded audio: {len(samples)} samples, {len(self._audio_fft_frames)} FFT frames")
            return True
            
        except Exception as e:
            print(f"[Visualizer] Failed to load audio for visualization: {e}")
            self._audio_fft_frames = []
            return False

    def _compute_fft_frames(self, samples: np.ndarray, sample_rate: int):
        """
        Pre-compute FFT frequency bands for each animation frame.
        
        Args:
            samples: Audio samples as float32 numpy array (-1..1 range)
            sample_rate: Sample rate in Hz
        """
        frame_samples = sample_rate // self.AUDIO_FPS  # Samples per animation frame
        n_frames = len(samples) // frame_samples
        
        self._audio_fft_frames = []
        
        for i in range(n_frames):
            start = i * frame_samples
            end = start + frame_samples
            if end > len(samples):
                break
            
            chunk = samples[start:end]
            
            # Apply Hanning window to reduce spectral leakage
            window = np.hanning(len(chunk))
            windowed = chunk * window
            
            # Compute FFT (only positive frequencies)
            fft = np.abs(np.fft.rfft(windowed))
            n_fft = len(fft)
            
            # Group FFT bins into frequency bands using logarithmic spacing
            # This gives more resolution to lower frequencies (more perceptually important)
            bands = np.zeros(self.NUM_FREQ_BANDS)
            
            # Logarithmic band edges (skip DC component at index 0)
            min_bin = 1
            max_bin = n_fft - 1
            if max_bin <= min_bin:
                self._audio_fft_frames.append(bands)
                continue
                
            freq_edges = np.logspace(
                np.log10(min_bin), 
                np.log10(max_bin), 
                self.NUM_FREQ_BANDS + 1
            ).astype(int)
            freq_edges = np.clip(freq_edges, min_bin, max_bin)
            
            for b in range(self.NUM_FREQ_BANDS):
                lo = freq_edges[b]
                hi = max(freq_edges[b + 1], lo + 1)
                if hi > lo:
                    bands[b] = np.mean(fft[lo:hi])
            
            # Normalize bands to 0..1 range
            max_band = np.max(bands)
            if max_band > 0:
                bands = bands / max_band
            
            # Apply slight smoothing with previous frame for natural transitions
            if self._audio_fft_frames:
                prev = self._audio_fft_frames[-1]
                bands = 0.7 * bands + 0.3 * prev
            
            self._audio_fft_frames.append(bands)

    def start_speaking_visualization(self):
        """
        Mark the start of speaking playback.
        Call this right before starting audio playback.
        """
        self._speaking_start_time = time.time()
        self._current_bands = np.zeros(self.NUM_FREQ_BANDS)
        self._prev_bands = np.zeros(self.NUM_FREQ_BANDS)
        self._audio_energy = 0.0

    def _get_current_audio_bands(self) -> np.ndarray:
        """
        Get the current frequency band values based on elapsed time since speaking started.
        
        Returns:
            Numpy array of frequency band values (0..1 range), shape (NUM_FREQ_BANDS,)
        """
        if not self._audio_fft_frames:
            return np.zeros(self.NUM_FREQ_BANDS)
        
        elapsed = time.time() - self._speaking_start_time
        frame_idx = int(elapsed * self.AUDIO_FPS)
        
        if frame_idx < 0:
            return np.zeros(self.NUM_FREQ_BANDS)
        
        if frame_idx >= len(self._audio_fft_frames):
            # Audio finished - decay to zero
            self._current_bands *= 0.85
            return self._current_bands
        
        # Get the current frame's bands
        target_bands = self._audio_fft_frames[frame_idx]
        
        # Smooth interpolation — fast response for alive feel (75% target, 25% previous)
        self._prev_bands = self._current_bands.copy()
        self._current_bands = 0.75 * target_bands + 0.25 * self._prev_bands
        
        # Update overall energy (for radius pulsing)
        self._audio_energy = float(np.mean(self._current_bands))
        
        return self._current_bands

    def clear_audio_data(self):
        """Clear loaded audio data."""
        self._audio_fft_frames = []
        self._current_bands = np.zeros(self.NUM_FREQ_BANDS)
        self._prev_bands = np.zeros(self.NUM_FREQ_BANDS)
        self._audio_energy = 0.0

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
            # Audio-reactive rotation and pulsing
            self._rot_y += 0.012
            self._rot_x = 0.3 + 0.1 * math.sin(t * 0.4)

            if self._audio_fft_frames:
                # Use real audio energy for breathing/pulsing
                energy = self._audio_energy
                # Bass bands (0-7) drive the main pulse, treble adds shimmer
                bands = self._current_bands
                bass_energy = float(np.mean(bands[:8])) if len(bands) >= 8 else energy
                mid_energy = float(np.mean(bands[8:20])) if len(bands) >= 20 else energy * 0.7
                treble_energy = float(np.mean(bands[20:])) if len(bands) >= 21 else energy * 0.5
                # Breathing: bass drives large pulse, mid adds body, treble adds fast shimmer
                pulse = 0.25 * bass_energy + 0.12 * mid_energy + 0.08 * treble_energy * math.sin(t * 10.0)
                # Add rotation speed variation based on energy (more dramatic)
                self._rot_y += 0.015 * energy
                self._target_radius_mult = 1.0 + pulse
            else:
                # Fallback: sine-wave pulsing when no audio data
                pulse = 0.12 * math.sin(t * 3.0) + 0.06 * math.sin(t * 5.5) + 0.04 * math.sin(t * 8.0)
                self._target_radius_mult = 1.0 + pulse

        # Smooth interpolation of radius
        self._radius_mult += (self._target_radius_mult - self._radius_mult) * 0.2

    def _update_particles(self):
        """Update per-particle displacement for animation."""
        t = self._frame / self.FPS

        if self._state == "speaking":
            if self._audio_fft_frames:
                # === AUDIO-REACTIVE MODE ===
                # Get current frequency bands from pre-computed FFT data
                bands = self._get_current_audio_bands()
                overall_energy = float(np.mean(bands))
                bass_energy = float(np.mean(bands[:8])) if len(bands) >= 8 else overall_energy
                treble_energy = float(np.mean(bands[20:])) if len(bands) >= 21 else overall_energy * 0.5

                for i in range(self.NUM_PARTICLES):
                    px, py, pz = self._particles[i]
                    band_idx = self._particle_band_idx[i]
                    phi = self._particle_phi[i]

                    # Get this particle's frequency band intensity (0..1)
                    band_val = float(bands[band_idx])

                    # Base displacement from frequency band — STRONG response
                    # Higher band value = more outward displacement
                    freq_disp = band_val * 0.45

                    # Harmonic resonance: particles vibrate at their band's natural frequency
                    # Lower bands = slower oscillation, higher bands = faster shimmer
                    harmonic_freq = 3.0 + band_idx * 0.8  # 3Hz for bass, ~28Hz for treble
                    harmonic_amp = band_val * 0.12
                    harmonic = harmonic_amp * math.sin(t * harmonic_freq + phi * 1.5)

                    # Wave propagation: ripple travels around the sphere
                    # based on angular position, creating a "living" effect
                    wave_phase = t * 6.0 - phi * 3.0 + band_idx * 0.4
                    wave_ripple = band_val * 0.14 * math.sin(wave_phase)

                    # Cross-band influence: neighboring bands bleed (wider for 32 bands)
                    neighbor_val = 0.0
                    for offset in [-2, -1, 1, 2]:
                        nb = band_idx + offset
                        if 0 <= nb < self.NUM_FREQ_BANDS:
                            weight = 0.12 if abs(offset) == 1 else 0.06
                            neighbor_val += float(bands[nb]) * weight
                    cross_disp = neighbor_val * 0.10

                    # Bass-driven global pulse: all particles breathe with the bass
                    bass_pulse = bass_energy * 0.08 * math.sin(t * 2.5 + py * 1.5)

                    # Treble shimmer: high-frequency particles get extra jitter
                    treble_shimmer = 0.0
                    if band_idx >= 20:
                        treble_shimmer = treble_energy * 0.06 * math.sin(t * 15.0 + phi * 5.0)

                    # Micro-noise for organic feel (scaled by band intensity)
                    noise = band_val * random.random() * 0.04

                    # Total target displacement
                    target = freq_disp + harmonic + wave_ripple + cross_disp + bass_pulse + treble_shimmer + noise

                    # Smooth interpolation (fast attack, moderate decay for snappy alive feel)
                    if target > self._particle_displace[i]:
                        lerp_speed = 0.6   # Very fast attack — snap to peaks
                    else:
                        lerp_speed = 0.25  # Moderate decay — natural falloff
                    self._particle_displace[i] += (target - self._particle_displace[i]) * lerp_speed
            else:
                # Fallback: sine-wave animation when no audio data loaded
                for i in range(self.NUM_PARTICLES):
                    px, py, pz = self._particles[i]
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

            # Particle size based on perspective scale (smaller for 900 particles)
            dot_size = max(1.0, 2.2 * scale * depth_factor)

            # Draw the particle dot
            self.create_oval(
                sx - dot_size, sy - dot_size,
                sx + dot_size, sy + dot_size,
                fill=color, outline="", width=0
            )

            # Add glow for front-facing bright particles (limited for performance)
            if depth_factor > 0.75 and self._state in ("speaking", "listening"):
                glow_size = dot_size * 2.0
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
# STATUS PANEL
# ==========================================

class StatusPanel(CTkFrame):
    """Bottom status bar."""

    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        self.configure(fg_color=THEME["bg_medium"], height=30)

        self.status_label = CTkLabel(
            self, text="Ready", font=("Consolas", 10),
            text_color=THEME["accent_cyan"]
        )
        self.status_label.pack(side="left", padx=10)

        self.model_label = CTkLabel(
            self, text=f"Model: {CONFIG['MODEL_NAME']}",
            font=("Consolas", 9), text_color=THEME["text_dim"]
        )
        self.model_label.pack(side="right", padx=10)

        self.api_label = CTkLabel(
            self, text=f"API: {CONFIG['API_URL']}",
            font=("Consolas", 9), text_color=THEME["text_dim"]
        )
        self.api_label.pack(side="right", padx=10)

    def set_status(self, text: str, color: Optional[str] = None):
        self.status_label.configure(text=text, text_color=color if color else THEME["accent_cyan"])


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

    def _build_ui(self):
        # Main horizontal layout: [FileExplorer | Chat | Marketplace]
        main_frame = CTkFrame(self, fg_color="transparent")
        main_frame.pack(fill="both", expand=True, padx=5, pady=5)

        # Left: File Explorer sidebar (280px)
        self.file_explorer = FileExplorer(
            main_frame,
            chat_input_callback=self._insert_path_to_chat,
            width=280
        )
        self.file_explorer.pack(side="left", fill="y", padx=(0, 3))

        # Center: Chat area (takes most space)
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

        CTkButton(
            chat_header, text="Clear Chat", width=80, height=26,
            font=("Consolas", 9),
            fg_color=THEME["bg_medium"], hover_color=THEME["error_color"],
            command=self._clear_chat
        ).pack(side="right", padx=5, pady=5)

        CTkButton(
            chat_header, text="New Session", width=90, height=26,
            font=("Consolas", 9),
            fg_color=THEME["accent_purple"], hover_color=THEME["accent_magenta"],
            command=self._new_session
        ).pack(side="right", padx=5, pady=5)

        # 📞 Live Call button
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
        if not mic_avail_for_call:
            self.live_call_btn.configure(state="disabled")

        # Terminal toggle switch
        self.terminal_switch_var = ctk.BooleanVar(value=False)
        self.terminal_switch = CTkSwitch(
            chat_header,
            text="Terminal",
            font=("Consolas", 9),
            text_color=THEME["accent_cyan"],
            variable=self.terminal_switch_var,
            command=self._toggle_terminal,
            onvalue=True,
            offvalue=False,
            width=40,
            progress_color=THEME["accent_cyan"],
            button_color=THEME["accent_magenta"],
            button_hover_color=THEME["accent_purple"],
        )
        self.terminal_switch.pack(side="right", padx=10, pady=5)

        # Chat + Terminal container (uses a PanedWindow-like approach)
        self.chat_terminal_container = CTkFrame(chat_frame, fg_color="transparent")
        self.chat_terminal_container.pack(fill="both", expand=True, padx=5, pady=5)

        # Chat messages
        self.chat_panel = ChatPanel(self.chat_terminal_container)
        self.chat_panel.pack(fill="both", expand=True)

        # Terminal frame (hidden by default)
        self.terminal_frame = CTkFrame(self.chat_terminal_container, fg_color=THEME["bg_dark"], corner_radius=8, height=180)
        # Terminal header
        self.terminal_header = CTkFrame(self.terminal_frame, fg_color=THEME["bg_light"], corner_radius=6)
        self.terminal_header.pack(fill="x", padx=4, pady=(4, 2))
        CTkLabel(
            self.terminal_header, text="TERMINAL OUTPUT",
            font=("Consolas", 10, "bold"),
            text_color=THEME["accent_cyan"]
        ).pack(side="left", padx=8, pady=3)
        CTkButton(
            self.terminal_header, text="Clear", width=50, height=20,
            font=("Consolas", 8), fg_color=THEME["bg_medium"],
            hover_color=THEME["error_color"], text_color=THEME["text_dim"],
            command=self._clear_terminal
        ).pack(side="right", padx=4, pady=3)
        # Terminal textbox
        self.terminal_textbox = CTkTextbox(
            self.terminal_frame,
            font=("Consolas", 10),
            fg_color="#050510",
            text_color=THEME["accent_cyan"],
            wrap="word",
            height=140,
        )
        self.terminal_textbox.pack(fill="both", expand=True, padx=4, pady=(0, 4))
        self.terminal_textbox.configure(state="disabled")
        # Terminal is hidden by default (not packed)

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

        # 🎤 Microphone button (hold-to-talk)
        mic_available = VOICE_AVAILABLE and self.voice_manager is not None
        self.mic_btn = CTkButton(
            input_frame, text="🎤", width=42, height=36,
            font=("Consolas", 16),
            fg_color=THEME["bg_light"] if mic_available else THEME["bg_dark"],
            hover_color=THEME["error_color"] if mic_available else THEME["bg_dark"],
            text_color=THEME["accent_cyan"] if mic_available else THEME["text_dim"],
            command=self._toggle_voice_recording,
        )
        self.mic_btn.pack(side="right", padx=(0, 3))
        if not mic_available:
            self.mic_btn.configure(state="disabled")

        # 🔊 Voice output toggle
        self.voice_output_var = ctk.BooleanVar(value=False)
        self.voice_output_switch = CTkSwitch(
            input_frame,
            text="🔊",
            font=("Consolas", 12),
            text_color=THEME["accent_cyan"],
            variable=self.voice_output_var,
            command=self._toggle_voice_output,
            onvalue=True,
            offvalue=False,
            width=36,
            progress_color=THEME["accent_magenta"],
            button_color=THEME["accent_purple"],
            button_hover_color=THEME["accent_magenta"],
        )
        if mic_available:
            self.voice_output_switch.pack(side="right", padx=(0, 5))

        # Right: Skills Marketplace sidebar
        self.marketplace_panel = MarketplacePanel(
            main_frame,
            skill_manager=self.skill_manager,
            chat_callback=self._add_system_message,
            width=340
        )
        self.marketplace_panel.pack(side="right", fill="y", padx=(3, 0))

        # Bottom status bar
        self.status_panel = StatusPanel(self)
        self.status_panel.pack(fill="x", padx=5, pady=(0, 5))

    # ------------------------------------------
    # CHAT LOGIC
    # ------------------------------------------

    # ------------------------------------------
    # TOOL ARGUMENT DECODING (ported from CLI)
    # ------------------------------------------

    def _decode_tool_args(self, tool_args: Optional[str]) -> Tuple[List[Any], Dict[str, Any]]:
        """Decode tool args from protocol text into *args/**kwargs with safe fallbacks."""
        if tool_args is None:
            return [], {}

        raw = tool_args.strip()
        if not raw:
            return [], {}

        # Accept fenced payloads (```json ...``` / ``` ... ```).
        fence = re.match(r"^```(?:json|python)?\s*(.*?)\s*```$", raw, re.DOTALL | re.IGNORECASE)
        if fence:
            raw = fence.group(1).strip()

        # Accept common wrappers the model may emit.
        for prefix in ("args=", "args:", "arguments:", "payload:", "input:"):
            if raw.lower().startswith(prefix):
                raw = raw[len(prefix):].strip()
                break

        def _parse_structured(candidate: str) -> Any:
            """Parse JSON/Python-literal payloads, including nested quoted payloads."""
            current: Any = candidate
            for _ in range(3):
                if not isinstance(current, str):
                    return current
                text = current.strip()
                if not text:
                    return text

                parsed = None
                try:
                    parsed = json.loads(text)
                except Exception:
                    try:
                        parsed = ast.literal_eval(text)
                    except Exception:
                        return text

                # Some model outputs are double-encoded strings; unwrap repeatedly.
                if isinstance(parsed, str):
                    if parsed.strip() == text:
                        return parsed
                    current = parsed
                    continue

                return parsed
            return current

        def _extract_balanced(text: str, open_ch: str, close_ch: str) -> Optional[str]:
            start = text.find(open_ch)
            if start < 0:
                return None
            depth = 0
            for i in range(start, len(text)):
                ch = text[i]
                if ch == open_ch:
                    depth += 1
                elif ch == close_ch:
                    depth -= 1
                    if depth == 0:
                        return text[start:i + 1]
            return None

        parsed = _parse_structured(raw)
        if isinstance(parsed, dict):
            # Unwrap nested wrapper keys used by some model responses.
            if len(parsed) == 1:
                wrapper_key = next(iter(parsed.keys()))
                if wrapper_key in {"args", "arguments", "payload", "input"}:
                    inner = _parse_structured(str(parsed[wrapper_key]))
                    if isinstance(inner, dict):
                        return [], inner
                    if isinstance(inner, list):
                        return inner, {}
                    if isinstance(inner, str):
                        raw = inner.strip()
                    else:
                        return [inner], {}
                else:
                    return [], parsed
            else:
                return [], parsed
        elif isinstance(parsed, list):
            return parsed, {}
        elif not isinstance(parsed, str):
            return [parsed], {}
        else:
            raw = parsed.strip()

        # If structured payload is embedded in prose, extract first balanced block.
        embedded = _extract_balanced(raw, "{", "}") or _extract_balanced(raw, "[", "]")
        if embedded:
            parsed_embedded = _parse_structured(embedded)
            if isinstance(parsed_embedded, dict):
                return [], parsed_embedded
            if isinstance(parsed_embedded, list):
                return parsed_embedded, {}
            if not isinstance(parsed_embedded, str):
                return [parsed_embedded], {}

        # Fallback: key=value or key: value lines.
        lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
        kv: Dict[str, Any] = {}
        if lines:
            has_explicit_kv = any(re.match(r"^[A-Za-z_]\w*\s*=", ln) for ln in lines) or any(
                re.match(r"^[A-Za-z_]\w*\s*:\s+.+$", ln) for ln in lines
            )
            valid_kv = True
            if has_explicit_kv:
                for ln in lines:
                    if re.match(r"^[A-Za-z_]\w*\s*=", ln):
                        k, v = ln.split("=", 1)
                    elif re.match(r"^[A-Za-z_]\w*\s*:\s+.+$", ln):
                        k, v = ln.split(":", 1)
                    else:
                        valid_kv = False
                        break
                    k = k.strip()
                    v = v.strip()
                    if not k or not re.match(r"^[A-Za-z_]\w*$", k):
                        valid_kv = False
                        break
                    if (v.startswith('"') and v.endswith('"')) or (v.startswith("'") and v.endswith("'")):
                        v = v[1:-1]
                    kv[k] = v
                if valid_kv and kv:
                    if len(kv) == 1 and next(iter(kv.keys())) in {"args", "arguments", "payload", "input"}:
                        inner = _parse_structured(next(iter(kv.values())))
                        if isinstance(inner, dict):
                            return [], inner
                        if isinstance(inner, list):
                            return inner, {}
                        if not isinstance(inner, str):
                            return [inner], {}
                    return [], kv

        # Legacy fallback: single raw argument string.
        return [raw], {}

    # ------------------------------------------
    # CHAT LOGIC
    # ------------------------------------------

    def _send_message(self):
        user_input = self.input_entry.get().strip()
        if not user_input or self.is_processing:
            return

        self.input_entry.delete(0, "end")
        self.chat_panel.add_message("user", user_input)
        self.is_processing = True
        self.send_btn.configure(state="disabled", text="...")
        self.status_panel.set_status("Processing...", THEME["warning_color"])

        # Reset tool loop guard state
        self._last_tool_signature: Optional[str] = None
        self._last_tool_result: Optional[str] = None
        self._repeat_tool_hits: int = 0
        self._tool_error_streak: int = 0
        self._tool_loop_guard_active: bool = False

        # Add to conversation
        self.conversation_manager.add_message("user", user_input)

        # Process in background thread
        threading.Thread(target=self._process_message, args=(user_input,), daemon=True).start()

    def _process_message(self, user_input: str):
        """Process user message with multi-turn tool execution (up to MAX_ITERATIONS)."""
        MAX_ITERATIONS = 6
        
        self.response_queue.put(("_terminal", f"[START] Processing user message: {user_input[:100]}..."))
        
        try:
            for iteration in range(MAX_ITERATIONS):
                # Check tool loop guard
                if self._tool_loop_guard_active:
                    self.response_queue.put(("_terminal", "[GUARD] Tool loop guard active, forcing final answer"))
                    self.response_queue.put(("system", "⚠ Tool loop guard active: forcing final answer."))
                    break

                # Build messages: system prompt + conversation history
                system_prompt = self.conversation_manager.get_system_prompt()

                # Add active marketplace skills to system prompt
                if hasattr(self.marketplace_panel, 'marketplace_engine') and self.marketplace_panel.marketplace_engine:
                    skill_additions = self.marketplace_panel.marketplace_engine.get_active_system_prompt_additions()
                    if skill_additions:
                        system_prompt += "\n\n" + skill_additions

                messages: list = [{"role": "system", "content": system_prompt}]
                messages.extend(self.conversation_manager.to_openai_format())

                # Update status for multi-turn
                if iteration > 0:
                    self.response_queue.put(("system", f"--- Turn {iteration + 1} ---"))

                self.response_queue.put(("_terminal", f"[API] Sending request to {CONFIG['MODEL_NAME']} (turn {iteration + 1}/{MAX_ITERATIONS}, {len(messages)} messages)"))

                response = client.chat.completions.create(
                    model=CONFIG["MODEL_NAME"],
                    messages=messages,  # type: ignore[arg-type]
                    temperature=0.1,
                    max_tokens=16384,
                )

                assistant_msg = response.choices[0].message.content or ""
                self.response_queue.put(("_terminal", f"[API] Response received ({len(assistant_msg)} chars)"))
                self.conversation_manager.add_message("assistant", assistant_msg)

                # Parse for tool calls - returns (text, tool_name, tool_args)
                text, tool_name, tool_args = self.response_parser.parse(assistant_msg)
                self.response_queue.put(("_terminal", f"[PARSE] text={len(text or '')} chars, tool={tool_name or 'None'}, args={len(tool_args or '')} chars"))

                # Display the text portion of the response
                if text:
                    self.response_queue.put(("assistant", text))

                if tool_name:
                    available_tools = self.skill_manager.get_active_tools()
                    self.response_queue.put(("_terminal", f"[TOOL] Calling: {tool_name} | Available: {', '.join(available_tools.keys())}"))
                    self.response_queue.put(("tool", f"⚡ Executing: {tool_name}"))
                    
                    if tool_name in available_tools:
                        tool_fn = available_tools[tool_name]
                        
                        # Decode arguments properly
                        args, kwargs = self._decode_tool_args(tool_args)
                        self.response_queue.put(("_terminal", f"[TOOL] Decoded args={args}, kwargs={list(kwargs.keys()) if kwargs else []}"))
                        
                        try:
                            # Try with decoded *args, **kwargs first
                            result = tool_fn(*args, **kwargs)
                        except TypeError as te:
                            self.response_queue.put(("_terminal", f"[TOOL] TypeError, trying raw fallback: {te}"))
                            # Fallback: some tools expect raw string (backward compat)
                            if tool_args is not None:
                                try:
                                    result = tool_fn(tool_args)
                                except Exception:
                                    result = f"💥 Tool argument error: {te}"
                            else:
                                result = f"💥 Tool argument error: {te}"
                        except Exception as e:
                            result = f"💥 Tool failed: {e}"
                        
                        result_str = str(result)[:2500]  # Increased limit for better context
                        self.response_queue.put(("_terminal", f"[TOOL] Result ({len(result_str)} chars): {result_str[:120]}..."))
                        self.response_queue.put(("tool", f"Result:\n{result_str}"))
                        
                        # Tool loop guard: detect repeated calls with same signature/result
                        signature = f"{tool_name}|{repr(args)}|{repr(sorted(kwargs.items()) if kwargs else [])}"
                        if signature == self._last_tool_signature and result_str == self._last_tool_result:
                            self._repeat_tool_hits += 1
                        else:
                            self._repeat_tool_hits = 0
                        self._last_tool_signature = signature
                        self._last_tool_result = result_str
                        
                        # Track error streak
                        if result_str.startswith("💥"):
                            self._tool_error_streak += 1
                        else:
                            self._tool_error_streak = 0
                        
                        # Activate loop guard if needed
                        if self._repeat_tool_hits >= 1 or self._tool_error_streak >= 2:
                            self._tool_loop_guard_active = True
                            self.conversation_manager.add_message(
                                "user",
                                f"RESULT [{tool_name}]: {result_str}\n"
                                "Stop tool repetition. Provide the best direct answer now, "
                                "or ask the user for one missing input."
                            )
                        else:
                            # Add tool result to conversation for next iteration
                            self.conversation_manager.add_message(
                                "user", f"RESULT [{tool_name}]: {result_str}\nProceed."
                            )
                        # Continue to next iteration to let AI process the result
                        continue
                    else:
                        self.response_queue.put(("tool", f"Unknown tool: {tool_name}. Available: {', '.join(available_tools.keys())}"))
                        # Don't continue loop for unknown tools
                        break
                else:
                    # No tool call - AI gave final response, we're done
                    if not text:
                        # If we didn't display text above, display the full message
                        self.response_queue.put(("assistant", assistant_msg))
                    break

        except APITimeoutError:
            self.response_queue.put(("error", f"API Timeout: Server at {CONFIG['API_URL']} is not responding"))
        except APIError as e:
            self.response_queue.put(("error", f"API Error: {e}"))
        except Exception as e:
            self.response_queue.put(("error", f"Error: {str(e)}"))
        finally:
            self.response_queue.put(("_done", ""))

    def _toggle_terminal(self):
        """Toggle the terminal panel visibility."""
        self.terminal_enabled = self.terminal_switch_var.get()
        if self.terminal_enabled:
            self.terminal_frame.pack(fill="x", padx=0, pady=(4, 0))
            self._log_terminal("[TERMINAL] Terminal view enabled. AI processing steps will appear here.")
        else:
            self.terminal_frame.pack_forget()

    def _log_terminal(self, text: str):
        """Append a timestamped line to the terminal textbox."""
        if not self.terminal_enabled or self.terminal_textbox is None:
            return
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        self.terminal_textbox.configure(state="normal")
        self.terminal_textbox.insert("end", f"[{timestamp}] {text}\n")
        self.terminal_textbox.see("end")
        self.terminal_textbox.configure(state="disabled")

    def _clear_terminal(self):
        """Clear the terminal textbox."""
        if self.terminal_textbox is not None:
            self.terminal_textbox.configure(state="normal")
            self.terminal_textbox.delete("1.0", "end")
            self.terminal_textbox.configure(state="disabled")

    def _poll_responses(self):
        try:
            while True:
                role, content = self.response_queue.get_nowait()
                if role == "_done":
                    self.is_processing = False
                    self.send_btn.configure(state="normal", text="Send")
                    self.status_panel.set_status("Ready", THEME["accent_cyan"])
                    self._log_terminal("[DONE] Processing complete.")
                elif role == "error":
                    self.chat_panel.add_message("system", content)
                    self.status_panel.set_status("Error", THEME["error_color"])
                    self._log_terminal(f"[ERROR] {content}")
                elif role == "_terminal":
                    # Terminal-only messages (not shown in chat)
                    self._log_terminal(content)
                else:
                    self.chat_panel.add_message(role, content)
                    # Speak assistant responses if voice output enabled
                    if role == "assistant" and self.voice_output_enabled and self.voice_manager:
                        self._speak_response(content)
                    # Also log to terminal
                    short = content[:200].replace("\n", " ")
                    self._log_terminal(f"[{role.upper()}] {short}")
        except queue.Empty:
            pass
        self.after(100, self._poll_responses)

    def _add_system_message(self, role: str, content: str):
        self.chat_panel.add_message(role, content)

    def _clear_chat(self):
        self.chat_panel.clear()
        self.chat_panel.add_message("system", "Chat cleared.")

    def _new_session(self):
        self.conversation_manager = SamsLawConversationManager(
            skill_manager=self.skill_manager, use_sam=False
        )
        self.chat_panel.clear()
        self.chat_panel.add_message("system", "New session started.")
        self.status_panel.set_status("New session", THEME["success_color"])

    # ------------------------------------------
    # VOICE METHODS
    # ------------------------------------------

    def _toggle_voice_recording(self):
        """Toggle voice recording on/off (push-to-talk style)."""
        if not self.voice_manager:
            self.chat_panel.add_message("system", "Voice not available. Install: pip install openai-whisper sounddevice numpy edge-tts pygame")
            return

        if self.is_processing:
            return

        if not self._voice_recording:
            # Start recording
            if not self._voice_initialized:
                # First time: load Whisper model in background
                self.mic_btn.configure(text="⏳", fg_color=THEME["warning_color"], text_color="#000000")
                self.status_panel.set_status("Loading Whisper large model (first time)...", THEME["warning_color"])
                self.chat_panel.add_message("system", "Loading Whisper large model... This may take a moment on first run (~1.5GB download).")

                def _init_and_record():
                    success = self.voice_manager.initialize(
                        progress_callback=lambda msg: self.response_queue.put(("_terminal", f"[VOICE] {msg}"))
                    )
                    if success:
                        self._voice_initialized = True
                        # Start recording after init
                        self.after(0, self._start_recording)
                    else:
                        self.response_queue.put(("system", "❌ Failed to load Whisper model. Check console for errors."))
                        self.after(0, lambda: self.mic_btn.configure(
                            text="🎤", fg_color=THEME["bg_light"], text_color=THEME["accent_cyan"]
                        ))
                        self.after(0, lambda: self.status_panel.set_status("Ready", THEME["accent_cyan"]))

                threading.Thread(target=_init_and_record, daemon=True).start()
                return

            self._start_recording()
        else:
            # Stop recording and transcribe
            self._stop_recording()

    def _start_recording(self):
        """Start microphone recording."""
        if not self.voice_manager:
            return
        self._voice_recording = True
        self.mic_btn.configure(text="⏹", fg_color=THEME["error_color"], text_color="#ffffff")
        self.status_panel.set_status("🎤 Recording... Click mic to stop", THEME["error_color"])
        self._log_terminal("[VOICE] Recording started")

        if self.voice_manager.start_recording():
            # Pulse animation
            self._voice_pulse()
        else:
            self._voice_recording = False
            self.mic_btn.configure(text="🎤", fg_color=THEME["bg_light"], text_color=THEME["accent_cyan"])
            self.status_panel.set_status("Ready", THEME["accent_cyan"])
            self.chat_panel.add_message("system", "❌ Failed to start recording. Check microphone.")

    def _stop_recording(self):
        """Stop recording and transcribe."""
        if not self.voice_manager:
            return
        self._voice_recording = False
        self.mic_btn.configure(text="⏳", fg_color=THEME["warning_color"], text_color="#000000")
        self.status_panel.set_status("Transcribing...", THEME["warning_color"])
        self._log_terminal("[VOICE] Recording stopped, transcribing...")

        def _transcribe():
            text = self.voice_manager.stop_recording_and_transcribe()
            if text:
                self.after(0, lambda: self._on_voice_transcription(text))
            else:
                self.after(0, lambda: self.chat_panel.add_message("system", "No speech detected. Try again."))
            self.after(0, lambda: self.mic_btn.configure(
                text="🎤", fg_color=THEME["bg_light"], text_color=THEME["accent_cyan"]
            ))
            self.after(0, lambda: self.status_panel.set_status("Ready", THEME["accent_cyan"]))

        threading.Thread(target=_transcribe, daemon=True).start()

    def _voice_pulse(self):
        """Animate the mic button while recording."""
        if not self._voice_recording:
            return
        current = self.mic_btn.cget("fg_color")
        next_color = THEME["accent_magenta"] if current == THEME["error_color"] else THEME["error_color"]
        self.mic_btn.configure(fg_color=next_color)
        self.after(500, self._voice_pulse)

    def _on_voice_state_change(self, state, message: str):
        """Callback from VoiceManager for state changes."""
        self._log_terminal(f"[VOICE] State: {state} - {message}")

    def _on_voice_transcription(self, text: str):
        """Callback when voice transcription is complete — send as chat message."""
        if not text.strip():
            return
        # Insert transcribed text into input and send
        self.input_entry.delete(0, "end")
        self.input_entry.insert(0, text)
        self._send_message()

    def _toggle_voice_output(self):
        """Toggle TTS voice output for AI responses."""
        self.voice_output_enabled = self.voice_output_var.get()
        status = "enabled" if self.voice_output_enabled else "disabled"
        self._log_terminal(f"[VOICE] Voice output {status}")

    def _speak_response(self, text: str):
        """Speak AI response using TTS with audio-reactive visualization.
        
        Pipeline: clean text → synthesize MP3 → load FFT for visualizer → play audio.
        """
        if not self.voice_output_enabled or not self.voice_manager:
            return
        # Clean text for natural TTS (strip markdown, code, URLs, etc.)
        clean = clean_text_for_tts(text)
        if not clean or len(clean) <= 5:
            return

        def _synth_and_play():
            try:
                import pygame

                # Step 1: Synthesize to MP3 file (once only)
                audio_path = self.voice_manager.synthesizer.synthesize(clean)
                if not audio_path:
                    # Fallback: speak without visualization
                    self.voice_manager.speak(clean, blocking=False)
                    return

                # Step 2: Load audio for FFT visualization (if live call visualizer exists)
                visualizer = self._live_call_visualizer
                if visualizer:
                    visualizer.load_audio_for_visualization(audio_path)
                    visualizer.start_speaking_visualization()

                # Step 3: Play the already-synthesized file directly (no re-synthesis)
                if not self.voice_manager.synthesizer._pygame_initialized:
                    self.voice_manager.synthesizer._init_pygame()
                pygame.mixer.music.load(audio_path)
                pygame.mixer.music.play()
                self.voice_manager.synthesizer._current_audio_path = audio_path

                # Step 4: Wait for playback to finish, then clean up
                def _wait_and_cleanup():
                    while self.voice_manager.synthesizer.is_playing():
                        time.sleep(0.1)
                    if visualizer:
                        visualizer.clear_audio_data()
                    # Clean up temp audio file
                    try:
                        if audio_path and os.path.exists(audio_path):
                            os.remove(audio_path)
                    except Exception:
                        pass

                threading.Thread(target=_wait_and_cleanup, daemon=True).start()

            except Exception as e:
                print(f"[TTS] Audio-reactive speak failed: {e}")
                # Fallback: speak without visualization
                try:
                    self.voice_manager.speak(clean, blocking=False)
                except Exception:
                    pass

        threading.Thread(target=_synth_and_play, daemon=True).start()

    # ------------------------------------------
    # LIVE CALL MODE
    # ------------------------------------------

    def _toggle_live_call(self):
        """Toggle live call mode on/off."""
        if self._live_call_active:
            self._exit_live_call()
        else:
            self._enter_live_call()

    def _enter_live_call(self):
        """Enter live call mode — hide sidebars/chat, show visualizer + terminal.
        Hands-free: automatically listens, transcribes, sends to AI, speaks back, repeats."""
        if not self.voice_manager:
            self.chat_panel.add_message("system", "Voice not available for live call.")
            return

        self._live_call_active = True
        self._live_call_muted = False
        self._live_call_listening_loop_active = False
        self.voice_output_enabled = True  # Auto-enable TTS in call mode
        self.live_call_btn.configure(text="📞 End Call", fg_color=THEME["error_color"],
                                      hover_color="#cc3333", text_color="#ffffff")
        self._log_terminal("[LIVE CALL] Entering hands-free live call mode")

        # Hide sidebars and chat panel
        self.file_explorer.pack_forget()
        self.marketplace_panel.pack_forget()
        self.chat_panel.pack_forget()

        # Create live call overlay in the chat_terminal_container
        self._live_call_frame = CTkFrame(self.chat_terminal_container, fg_color=THEME["bg_dark"], corner_radius=10)
        self._live_call_frame.pack(fill="both", expand=True)

        # Visualizer (fills most of the space)
        self._live_call_visualizer = JarvisSphereVisualizer(self._live_call_frame)
        self._live_call_visualizer.pack(fill="both", expand=True, padx=10, pady=(10, 5))
        self._live_call_visualizer.set_state("idle", "Connecting...")
        self._live_call_visualizer.start_animation()

        # Call controls bar at bottom of visualizer
        call_controls = CTkFrame(self._live_call_frame, fg_color=THEME["bg_medium"], corner_radius=8)
        call_controls.pack(fill="x", padx=10, pady=(5, 10))

        # Mute/Unmute button (replaces Talk button for hands-free)
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

        # Call status label
        self._live_call_status = CTkLabel(
            call_controls, text="LIVE CALL — Connecting...",
            font=("Consolas", 11), text_color=THEME["accent_cyan"]
        )
        self._live_call_status.pack(side="left", fill="x", expand=True, padx=10)

        # Enable terminal if not already
        if not self.terminal_enabled:
            self.terminal_switch_var.set(True)
            self._toggle_terminal()

        # Initialize voice if needed, then start listening loop
        if not self._voice_initialized:
            self._live_call_visualizer.set_state("processing", "Loading Whisper model...")
            self._live_call_status.configure(text="Loading Whisper large model...")

            def _init_voice():
                success = self.voice_manager.initialize(
                    progress_callback=lambda msg: self.response_queue.put(("_terminal", f"[VOICE] {msg}"))
                )
                if success:
                    self._voice_initialized = True
                    self.after(0, self._live_call_start_listening_loop)
                else:
                    self.after(0, lambda: self._live_call_visualizer.set_state("idle", "Model load failed") if self._live_call_visualizer else None)
                    self.response_queue.put(("system", "❌ Failed to load Whisper model."))

            threading.Thread(target=_init_voice, daemon=True).start()
        else:
            # Already initialized — start listening immediately
            self.after(100, self._live_call_start_listening_loop)

        self.status_panel.set_status("📞 LIVE CALL MODE — Hands-free", THEME["success_color"])

    def _exit_live_call(self):
        """Exit live call mode — restore normal GUI."""
        self._live_call_active = False
        self._live_call_listening_loop_active = False  # Stop the listening loop
        self._log_terminal("[LIVE CALL] Exiting live call mode")

        # Stop any ongoing recording
        if self._voice_recording and self.voice_manager:
            self._voice_recording = False
            try:
                self.voice_manager.recorder.is_recording = False  # Force stop
                self.voice_manager.recorder.stop_recording()
            except Exception:
                pass
        
        # Stop any ongoing speech
        if self.voice_manager:
            try:
                self.voice_manager.stop_speaking()
            except Exception:
                pass

        # Stop visualizer
        if self._live_call_visualizer:
            self._live_call_visualizer.stop_animation()
            self._live_call_visualizer = None

        # Destroy live call frame
        if self._live_call_frame:
            self._live_call_frame.destroy()
            self._live_call_frame = None

        self._live_call_mic_btn = None

        # Restore normal layout
        self.chat_panel.pack(fill="both", expand=True)
        self.file_explorer.pack(side="left", fill="y", padx=(0, 3), before=self.chat_panel.master.master)
        self.marketplace_panel.pack(side="right", fill="y", padx=(3, 0))

        # Update button
        self.live_call_btn.configure(text="📞 Live Call", fg_color=THEME["success_color"],
                                      hover_color=THEME["accent_cyan"], text_color="#000000")
        self.status_panel.set_status("Ready", THEME["accent_cyan"])

    def _live_call_toggle_mute(self):
        """Toggle mute in live call mode (hands-free)."""
        if not self.voice_manager or not self._voice_initialized:
            return

        self._live_call_muted = not self._live_call_muted

        if self._live_call_muted:
            # Mute: stop listening loop
            self._live_call_listening_loop_active = False
            # Force stop any active recording
            if self._voice_recording:
                self._voice_recording = False
                self.voice_manager.recorder.is_recording = False
                try:
                    self.voice_manager.recorder.stop_recording()
                except Exception:
                    pass
                self.voice_manager._update_state(VoiceState.IDLE, "Muted")
            if self._live_call_mic_btn:
                self._live_call_mic_btn.configure(text="🔊 Unmute", fg_color=THEME["error_color"], text_color="#ffffff")
            if self._live_call_visualizer:
                self._live_call_visualizer.set_state("idle", "Muted")
            if hasattr(self, '_live_call_status'):
                self._live_call_status.configure(text="🔇 MUTED — Click Unmute to resume")
            self._log_terminal("[LIVE CALL] Muted")
        else:
            # Unmute: restart listening loop
            if self._live_call_mic_btn:
                self._live_call_mic_btn.configure(text="🔇 Mute", fg_color=THEME["accent_cyan"], text_color="#000000")
            self._log_terminal("[LIVE CALL] Unmuted — resuming listening")
            self._live_call_start_listening_loop()

    def _live_call_start_listening_loop(self):
        """Start the hands-free continuous listening loop."""
        if not self._live_call_active or self._live_call_muted:
            return
        if self._live_call_listening_loop_active:
            return  # Already running

        self._live_call_listening_loop_active = True
        self._log_terminal("[LIVE CALL] Starting hands-free listening loop")

        def _listen_loop():
            """Background thread: continuously listen → transcribe → send → speak → repeat."""
            while self._live_call_active and self._live_call_listening_loop_active and not self._live_call_muted:
                try:
                    # Update UI: listening state
                    self.after(0, lambda: self._live_call_update_ui("listening", "🎤 Listening... speak now"))

                    self._voice_recording = True

                    # Use record_until_silence — blocks until user stops speaking
                    self.voice_manager._update_state(VoiceState.RECORDING, "Listening...")
                    audio = self.voice_manager.recorder.record_until_silence(
                        timeout=self.voice_manager.config.max_recording_duration
                    )

                    self._voice_recording = False

                    # Check if we should still be running
                    if not self._live_call_active or not self._live_call_listening_loop_active or self._live_call_muted:
                        break

                    # Check if we got valid audio
                    if audio is None or len(audio) < self.voice_manager.config.sample_rate * 0.5:
                        self._log_terminal("[LIVE CALL] No speech detected, re-listening...")
                        time.sleep(0.2)  # Brief pause before re-listening
                        continue

                    # Update UI: transcribing
                    self.after(0, lambda: self._live_call_update_ui("processing", "Transcribing..."))
                    self.voice_manager._update_state(VoiceState.TRANSCRIBING, "Transcribing...")

                    # Transcribe
                    text, metadata = self.voice_manager.transcriber.transcribe(audio)

                    if not text or not text.strip():
                        self._log_terminal("[LIVE CALL] Empty transcription, re-listening...")
                        self.voice_manager._update_state(VoiceState.IDLE, "Ready")
                        time.sleep(0.2)
                        continue

                    self._log_terminal(f"[LIVE CALL] You said: {text}")

                    # Check if still active
                    if not self._live_call_active or not self._live_call_listening_loop_active or self._live_call_muted:
                        break

                    # Send to AI and get response (blocking)
                    self.after(0, lambda t=text: self._live_call_show_user_message(t))
                    self.after(0, lambda: self._live_call_update_ui("processing", "Thinking..."))

                    # Add to conversation
                    self.conversation_manager.add_message("user", text)

                    # Call AI
                    response_text = self._live_call_get_ai_response()

                    if not response_text:
                        self._log_terminal("[LIVE CALL] No AI response, re-listening...")
                        time.sleep(0.3)
                        continue

                    # Check if still active
                    if not self._live_call_active or not self._live_call_listening_loop_active or self._live_call_muted:
                        break

                    # Show AI response
                    self.after(0, lambda r=response_text: self.chat_panel.add_message("assistant", r))
                    self._log_terminal(f"[LIVE CALL] JARVIS: {response_text[:100]}...")

                    # Speak the response — clean for natural TTS
                    clean = clean_text_for_tts(response_text)

                    if clean and len(clean) > 5:
                        self.after(0, lambda c=clean: self._live_call_update_ui("speaking", f"JARVIS: {c[:60]}..."))
                        self.voice_manager._update_state(VoiceState.SPEAKING, "Speaking...")
                        
                        # Synthesize audio file first for FFT visualization (once only)
                        audio_path = None
                        try:
                            audio_path = self.voice_manager.synthesizer.synthesize(clean)
                        except Exception as synth_err:
                            self._log_terminal(f"[LIVE CALL] Synth error: {synth_err}")
                        
                        # Load FFT data into visualizer for audio-reactive particles
                        if audio_path and self._live_call_visualizer:
                            self._live_call_visualizer.load_audio_for_visualization(audio_path)
                            self._live_call_visualizer.start_speaking_visualization()
                        
                        # Play the already-synthesized file directly (no re-synthesis)
                        tts_started = False
                        if audio_path:
                            try:
                                import pygame
                                if not self.voice_manager.synthesizer._pygame_initialized:
                                    self.voice_manager.synthesizer._init_pygame()
                                pygame.mixer.music.load(audio_path)
                                pygame.mixer.music.play()
                                self.voice_manager.synthesizer._current_audio_path = audio_path
                                tts_started = True
                            except Exception as play_err:
                                self._log_terminal(f"[LIVE CALL] Play error: {play_err}")
                                tts_started = False
                        
                        if not tts_started:
                            # Fallback: use speak_nonblocking which re-synthesizes
                            tts_started = self.voice_manager.synthesizer.speak_nonblocking(clean)
                        interrupted = False
                        
                        if tts_started:
                            # Monitor mic while JARVIS speaks — detect user interruption
                            interrupt_stream = None
                            interrupt_energy_history = []
                            interrupt_threshold = self.voice_manager.config.silence_threshold * 3  # Higher threshold to avoid self-trigger
                            interrupt_sustained_frames = 0  # Count consecutive loud frames
                            INTERRUPT_FRAMES_NEEDED = 4  # ~400ms of sustained speech to interrupt (at 100ms polling)
                            
                            try:
                                if VOICE_AVAILABLE and hasattr(sd, 'InputStream'):
                                    # Open a separate mic stream to monitor for speech during TTS
                                    interrupt_buffer = []
                                    
                                    def _interrupt_callback(indata, frames, time_info, status):
                                        interrupt_buffer.append(indata.copy())
                                    
                                    interrupt_stream = sd.InputStream(
                                        samplerate=self.voice_manager.config.sample_rate,
                                        channels=self.voice_manager.config.channels,
                                        dtype=self.voice_manager.config.dtype,
                                        callback=_interrupt_callback
                                    )
                                    interrupt_stream.start()
                            except Exception as mic_err:
                                self._log_terminal(f"[LIVE CALL] Mic monitor setup failed: {mic_err}")
                                interrupt_stream = None
                            
                            # Wait for TTS to finish OR user interruption
                            while self.voice_manager.synthesizer.is_playing():
                                if not self._live_call_active or not self._live_call_listening_loop_active or self._live_call_muted:
                                    self.voice_manager.synthesizer.stop()
                                    interrupted = True
                                    break
                                
                                # Check mic energy for interruption
                                if interrupt_stream and interrupt_buffer:
                                    try:
                                        recent = interrupt_buffer[-1]
                                        rms = float(np.sqrt(np.mean(np.array(recent)**2)))
                                        interrupt_energy_history.append(rms)
                                        
                                        if rms > interrupt_threshold:
                                            interrupt_sustained_frames += 1
                                            if interrupt_sustained_frames >= INTERRUPT_FRAMES_NEEDED:
                                                # User is speaking! Stop TTS and capture their speech
                                                self._log_terminal(f"[LIVE CALL] 🗣️ User interrupted JARVIS (energy: {rms:.4f})")
                                                self.voice_manager.synthesizer.stop()
                                                self.after(0, lambda: self._live_call_update_ui("listening", "🎤 Interrupted — listening..."))
                                                interrupted = True
                                                break
                                        else:
                                            interrupt_sustained_frames = 0  # Reset if quiet frame
                                    except Exception:
                                        pass
                                
                                time.sleep(0.1)
                            
                            # Clean up interrupt monitor stream
                            if interrupt_stream:
                                try:
                                    interrupt_stream.stop()
                                    interrupt_stream.close()
                                except Exception:
                                    pass
                            
                            # If interrupted, we already captured some audio in interrupt_buffer
                            # The next loop iteration will pick up the user's speech naturally
                            if interrupted and self._live_call_active and not self._live_call_muted:
                                self._log_terminal("[LIVE CALL] TTS interrupted — resuming listening immediately")
                                # Small delay to let the user finish their thought
                                time.sleep(0.1)
                        
                        # Clean up audio visualization data and temp file
                        if self._live_call_visualizer:
                            self._live_call_visualizer.clear_audio_data()
                        if audio_path:
                            try:
                                if os.path.exists(audio_path):
                                    os.remove(audio_path)
                            except Exception:
                                pass
                        
                        self.voice_manager._update_state(VoiceState.IDLE, "Ready")

                    # Check if still active before looping
                    if not self._live_call_active or not self._live_call_listening_loop_active or self._live_call_muted:
                        break

                    # Brief pause before re-listening (natural conversation gap)
                    if not interrupted:
                        time.sleep(0.3)

                except Exception as e:
                    self._log_terminal(f"[LIVE CALL] Error in listening loop: {e}")
                    self._voice_recording = False
                    time.sleep(1)  # Wait before retrying on error

            # Loop ended
            self._live_call_listening_loop_active = False
            self._voice_recording = False
            self._log_terminal("[LIVE CALL] Listening loop stopped")
            if self._live_call_active and not self._live_call_muted:
                self.after(0, lambda: self._live_call_update_ui("idle", "Ready to listen..."))

        threading.Thread(target=_listen_loop, daemon=True).start()

    def _live_call_update_ui(self, state: str, status_text: str):
        """Update visualizer and status label from any thread (must be called via self.after)."""
        if not self._live_call_active:
            return
        if self._live_call_visualizer:
            self._live_call_visualizer.set_state(state, status_text)
        if hasattr(self, '_live_call_status'):
            self._live_call_status.configure(text=status_text)

    def _live_call_show_user_message(self, text: str):
        """Show user's transcribed message in chat panel."""
        if self._live_call_active:
            self.chat_panel.add_message("user", text)

    def _live_call_get_ai_response(self) -> Optional[str]:
        """Get AI response synchronously (called from background thread).
        
        Includes multi-turn tool execution loop so that tool calls are
        actually executed and the AI returns a natural-language answer
        instead of raw :::TOOL:...::: protocol text.
        """
        MAX_ITERATIONS = 6
        last_tool_signature: Optional[str] = None
        last_tool_result: Optional[str] = None
        repeat_tool_hits: int = 0
        tool_error_streak: int = 0
        tool_loop_guard_active: bool = False

        try:
            for iteration in range(MAX_ITERATIONS):
                # Build messages
                system_prompt = self.conversation_manager.get_system_prompt()
                if hasattr(self.marketplace_panel, 'marketplace_engine') and self.marketplace_panel.marketplace_engine:
                    skill_additions = self.marketplace_panel.marketplace_engine.get_active_system_prompt_additions()
                    if skill_additions:
                        system_prompt += "\n\n" + skill_additions

                messages = [{"role": "system", "content": system_prompt}]
                messages.extend(self.conversation_manager.to_openai_format())

                self.response_queue.put(("_terminal", f"[LIVE CALL] Sending to AI ({len(messages)} messages, turn {iteration + 1}/{MAX_ITERATIONS})..."))

                response = client.chat.completions.create(
                    model=CONFIG["MODEL_NAME"],
                    messages=messages,
                    temperature=0.1,
                    max_tokens=16384,
                )

                assistant_msg = response.choices[0].message.content or ""
                self.response_queue.put(("_terminal", f"[LIVE CALL] Response ({len(assistant_msg)} chars)"))
                self.conversation_manager.add_message("assistant", assistant_msg)

                # Parse response for tool calls
                parsed_text, tool_name, tool_args = self.response_parser.parse(assistant_msg)

                # If loop guard is active, stop calling tools and return whatever text we have
                if tool_loop_guard_active:
                    self.response_queue.put(("_terminal", "[LIVE CALL] Tool loop guard active, returning text"))
                    return parsed_text or "I encountered an issue processing that request."

                if tool_name:
                    available_tools = self.skill_manager.get_active_tools()
                    self.response_queue.put(("_terminal", f"[LIVE CALL] Tool call: {tool_name}"))

                    if tool_name not in available_tools:
                        self.response_queue.put(("_terminal", f"[LIVE CALL] Unknown tool: {tool_name}"))
                        # Add error to conversation so AI can recover
                        self.conversation_manager.add_message(
                            "user",
                            f"RESULT [{tool_name}]: Unknown tool. Available tools: {', '.join(available_tools.keys())}. "
                            "Please provide a direct answer instead."
                        )
                        continue

                    # Execute the tool
                    tool_fn = available_tools[tool_name]
                    args, kwargs = self._decode_tool_args(tool_args)
                    self.response_queue.put(("_terminal", f"[LIVE CALL] Executing {tool_name}({args}, {kwargs})"))

                    try:
                        result = tool_fn(*args, **kwargs)
                    except TypeError as te:
                        if tool_args is not None:
                            try:
                                result = tool_fn(tool_args)
                            except Exception:
                                result = f"💥 Tool argument error: {te}"
                        else:
                            result = f"💥 Tool argument error: {te}"
                    except Exception as e:
                        result = f"💥 Tool failed: {e}"

                    result_str = str(result)[:2500]
                    self.response_queue.put(("_terminal", f"[LIVE CALL] Tool result ({len(result_str)} chars): {result_str[:120]}..."))

                    # Show tool execution in chat
                    self.after(0, lambda tn=tool_name, rs=result_str: self.chat_panel.add_message("tool", f"⚡ {tn}\n{rs}"))

                    # Tool loop guard: detect repeated calls
                    signature = f"{tool_name}|{repr(args)}|{repr(sorted(kwargs.items()) if kwargs else [])}"
                    if signature == last_tool_signature and result_str == last_tool_result:
                        repeat_tool_hits += 1
                    else:
                        repeat_tool_hits = 0
                    last_tool_signature = signature
                    last_tool_result = result_str

                    if result_str.startswith("💥"):
                        tool_error_streak += 1
                    else:
                        tool_error_streak = 0

                    if repeat_tool_hits >= 1 or tool_error_streak >= 2:
                        tool_loop_guard_active = True
                        self.conversation_manager.add_message(
                            "user",
                            f"RESULT [{tool_name}]: {result_str}\n"
                            "Stop tool repetition. Provide the best direct answer now."
                        )
                    else:
                        self.conversation_manager.add_message(
                            "user", f"RESULT [{tool_name}]: {result_str}\nProceed."
                        )
                    # Continue loop to let AI process the tool result
                    continue
                else:
                    # No tool call — AI gave a final natural-language response
                    final_text = parsed_text or assistant_msg
                    # Safety: strip any accidental tool protocol remnants
                    if ":::TOOL:" in final_text:
                        # Shouldn't happen, but just in case — remove it
                        final_text = re.sub(r':::TOOL:.*?:::END::(?::)?', '', final_text, flags=re.DOTALL).strip()
                    return final_text if final_text else None

            # Exhausted all iterations — return last parsed text or a fallback
            self.response_queue.put(("_terminal", "[LIVE CALL] Max iterations reached"))
            return parsed_text if parsed_text else "I've completed the task. Let me know if you need anything else."

        except Exception as e:
            self.response_queue.put(("_terminal", f"[LIVE CALL] AI error: {e}"))
            return None

    def _live_call_reset_mic(self, status_text: str = ""):
        """Reset mic button to ready state in live call mode."""
        if not self._live_call_active:
            return
        if self._live_call_mic_btn and not self._live_call_muted:
            self._live_call_mic_btn.configure(text="🔇 Mute", fg_color=THEME["accent_cyan"], text_color="#000000")
        if self._live_call_visualizer:
            self._live_call_visualizer.set_state("idle", status_text or "Ready to listen...")
        if hasattr(self, '_live_call_status'):
            self._live_call_status.configure(text=status_text or "LIVE CALL — Listening...")

    def _insert_path_to_chat(self, path: str):
        """Insert a file path into the chat input (callback from FileExplorer)."""
        current = self.input_entry.get()
        if current:
            self.input_entry.delete(0, "end")
            self.input_entry.insert(0, f"{current} {path}")
        else:
            self.input_entry.insert(0, path)
        self.input_entry.focus()


# ==========================================
# ENTRY POINT
# ==========================================

def main():
    app = TripleGGUI()
    app.mainloop()


if __name__ == "__main__":
    main()
