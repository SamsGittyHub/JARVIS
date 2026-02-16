import os
import re
import ast
import subprocess
import sys
import time
import json
import glob
import hashlib
import urllib.request
import urllib.error
import tempfile
import shutil
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List, Callable
from dataclasses import dataclass, asdict
from enum import Enum
from urllib.parse import urlparse, urlencode

import numpy as np

try:
    from selenium import webdriver
    from selenium.webdriver.chrome.service import Service as ChromeService
    from selenium.webdriver.chrome.options import Options as ChromeOptions
    from selenium.webdriver.common.by import By
    from selenium.webdriver.common.keys import Keys
    from selenium.webdriver.common.action_chains import ActionChains
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from webdriver_manager.chrome import ChromeDriverManager
    SELENIUM_AVAILABLE = True
except ImportError:
    SELENIUM_AVAILABLE = False

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from openai import OpenAI, APIError, APITimeoutError
from rich.console import Console
from rich.theme import Theme
from rich.panel import Panel
from rich.align import Align
from rich.text import Text
from rich.markdown import Markdown
from rich.syntax import Syntax
from rich.live import Live
from rich.spinner import Spinner
from rich.table import Table
from rich.prompt import Confirm, Prompt
from prompt_toolkit import PromptSession
from prompt_toolkit.styles import Style
from prompt_toolkit.formatted_text import HTML

# ==========================================
# 🌐 SKILL MARKETPLACE API CONFIGURATION
# ==========================================

SKILL_MARKETPLACE_CONFIG = {
    "primary_url": "https://raw.githubusercontent.com/tripleg-skills/registry/main/skills.json",
    "fallback_url": "https://gist.githubusercontent.com/tripleg-bot/skills-registry/raw/skills.json",
    "cache_file": Path.home() / ".tripleg" / "skills_cache.json",
    "cache_ttl_hours": 24,
    "offline_mode": False,
}

# ==========================================
# 🎭 SKILL SYSTEM
# ==========================================

class SkillCategory(Enum):
    SYSTEM = "system"
    FILE_OPS = "file_operations"
    WEB = "web_access"
    CODE = "code_analysis"
    SECURITY = "security"
    DATABASE = "database"
    API = "api_integration"
    CUSTOM = "custom"

@dataclass
class Skill:
    id: str
    name: str
    description: str
    category: SkillCategory
    version: str
    author: str
    tools: List[str]
    system_prompt_addition: str
    dependencies: List[str]
    dangerous: bool = False
    installed: bool = False
    install_path: Optional[str] = None
    download_url: Optional[str] = None
    source: str = "builtin"
    
    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "category": self.category.value,
            "version": self.version,
            "author": self.author,
            "tools": self.tools,
            "system_prompt_addition": self.system_prompt_addition,
            "dependencies": self.dependencies,
            "dangerous": self.dangerous,
            "download_url": self.download_url,
            "source": self.source,
        }

# ==========================================
# 🌐 AGENTSKILLS INTEGRATION MODULE
# ==========================================

class AgentSkillsLoader:
    """
    Loads and converts AgentSkills (SKILL.md format) into TripleG Skill objects.
    AgentSkills are instruction-based, not tool-based, so we map them to 
    'knowledge skills' that enhance system prompts.
    """
    
    def __init__(self, cache_dir: Path = None):
        self.cache_dir = cache_dir or (Path.home() / ".tripleg" / "agentskills_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.loaded_skills: Dict[str, Skill] = {}
        self.loaded_repos: List[str] = []
        
    def fetch_github_repo(self, repo_url: str, ref: str = "main") -> Path:
        """
        Downloads a GitHub repository containing AgentSkills.
        Supports formats: owner/repo or full https://github.com/... URL
        """
        # Parse repo identifier
        if repo_url.startswith("https://github.com/"):
            parts = repo_url.replace("https://github.com/", "").split("/")
            owner, repo = parts[0], parts[1].replace(".git", "")
        elif "/" in repo_url:
            owner, repo = repo_url.split("/")[:2]
        else:
            raise ValueError(f"Invalid repo format: {repo_url}")
        
        cache_key = f"{owner}_{repo}_{ref}"
        extract_path = self.cache_dir / cache_key
        
        if extract_path.exists():
            console.print(f"[dim]📦 Using cached {owner}/{repo}@{ref}[/dim]")
            return extract_path
        
        # Download zip from GitHub
        zip_url = f"https://github.com/{owner}/{repo}/archive/refs/heads/{ref}.zip"
        console.print(f"[dim]⬇️ Downloading {owner}/{repo}@{ref}...[/dim]")
        
        try:
            req = urllib.request.Request(zip_url, headers={'User-Agent': 'TripleG-AgentSkills/1.0'})
            with urllib.request.urlopen(req, timeout=30) as response:
                zip_content = response.read()
            
            # Extract to temp first, then move to cache
            with tempfile.TemporaryDirectory() as tmpdir:
                zip_path = Path(tmpdir) / "repo.zip"
                zip_path.write_bytes(zip_content)
                
                with zipfile.ZipFile(zip_path, 'r') as zf:
                    zf.extractall(tmpdir)
                
                # Find extracted folder (usually repo-ref/)
                extracted = list(Path(tmpdir).glob(f"{repo}-*"))
                if extracted:
                    shutil.move(str(extracted[0]), str(extract_path))
                else:
                    raise Exception("Could not find extracted repository")
            
            console.print(f"[green]✓ Cached {owner}/{repo}@{ref}[/green]")
            return extract_path
            
        except urllib.error.HTTPError as e:
            if e.code == 404:
                console.print(f"[red]❌ Repository {owner}/{repo} not found (404)[/red]")
            else:
                console.print(f"[red]❌ HTTP {e.code} error fetching {repo_url}[/red]")
            return None
        except Exception as e:
            console.print(f"[red]❌ Failed to fetch {repo_url}: {e}[/red]")
            return None
    
    def parse_skill_md(self, skill_path: Path) -> Optional[Dict]:
        """
        Parses a SKILL.md file with YAML frontmatter.
        Returns dict with metadata and content.
        """
        try:
            content = skill_path.read_text('utf-8')
            
            # Split frontmatter and content
            if content.startswith('---'):
                parts = content.split('---', 2)
                if len(parts) >= 3:
                    frontmatter = yaml.safe_load(parts[1])
                    body = parts[2].strip()
                    return {
                        'frontmatter': frontmatter or {},
                        'content': body,
                        'path': str(skill_path.parent)
                    }
            return None
        except Exception as e:
            console.print(f"[dim]⚠️ Failed to parse {skill_path}: {e}[/dim]")
            return None
    
    def discover_skills(self, repo_path: Path) -> List[Dict]:
        """
        Discovers all SKILL.md files in a repository.
        Returns list of parsed skill dictionaries.
        """
        skills = []
        
        # Common skill locations per AgentSkills spec
        skill_patterns = [
            "**/SKILL.md",           # Any SKILL.md
            "skills/**/SKILL.md",    # skills/ subdirectory
            ".claude/skills/**/SKILL.md",  # Claude-specific
            ".codex/skills/**/SKILL.md",   # Codex-specific
        ]
        
        found_paths = set()
        for pattern in skill_patterns:
            for skill_file in repo_path.glob(pattern):
                if skill_file in found_paths:
                    continue
                found_paths.add(skill_file)
                
                parsed = self.parse_skill_md(skill_file)
                if parsed:
                    skills.append(parsed)
        
        return skills
    
    def convert_to_tripleg_skill(self, skill_data: Dict, source_repo: str) -> Optional[Skill]:
        """
        Converts AgentSkill format to TripleG Skill format.
        AgentSkills are knowledge-based, so we treat them as 'soft skills'
        that enhance the system prompt rather than add tools.
        """
        fm = skill_data['frontmatter']
        content = skill_data['content']
        
        name = fm.get('name')
        if not name:
            return None
        
        # Map to SkillCategory based on content analysis
        category = self._infer_category(content, name)
        
        # Create system prompt addition from skill content
        system_addition = f"""
## Skill: {name}
{fm.get('description', 'No description')}

### Instructions
{content[:2000]}  # Truncate if too long

### Compatibility
{fm.get('compatibility', 'None specified')}
"""
        
        return Skill(
            id=f"agentskill_{name.replace('-', '_').replace(' ', '_').lower()}",
            name=name.replace('-', ' ').title(),
            description=fm.get('description', f'AgentSkill from {source_repo}'),
            category=category,
            version=str(fm.get('metadata', {}).get('version', '1.0.0')),
            author=fm.get('metadata', {}).get('author', source_repo),
            tools=[],  # AgentSkills don't add tools, they add knowledge
            system_prompt_addition=system_addition,
            dependencies=[],  # Could parse from compatibility field
            dangerous=False,
            source=f"agentskills:{source_repo}"
        )
    
    def _infer_category(self, content: str, name: str) -> SkillCategory:
        """Infers skill category from content keywords."""
        content_lower = (content + " " + name).lower()
        
        if any(k in content_lower for k in ['security', 'vulnerability', 'audit', 'pentest']):
            return SkillCategory.SECURITY
        elif any(k in content_lower for k in ['database', 'sql', 'query']):
            return SkillCategory.DATABASE
        elif any(k in content_lower for k in ['api', 'http', 'rest', 'endpoint']):
            return SkillCategory.API
        elif any(k in content_lower for k in ['web', 'scraping', 'html', 'css']):
            return SkillCategory.WEB
        elif any(k in content_lower for k in ['code', 'refactor', 'review', 'git']):
            return SkillCategory.CODE
        elif any(k in content_lower for k in ['file', 'directory', 'path']):
            return SkillCategory.FILE_OPS
        else:
            return SkillCategory.CUSTOM
    
    def load_repo(self, repo_url: str, ref: str = "main") -> List[Skill]:
        """
        Main entry point: fetch repo, discover skills, convert to TripleG format.
        """
        repo_path = self.fetch_github_repo(repo_url, ref)
        if not repo_path:
            return []
        
        raw_skills = self.discover_skills(repo_path)
        converted = []
        
        for raw in raw_skills:
            skill = self.convert_to_tripleg_skill(raw, repo_url)
            if skill:
                self.loaded_skills[skill.id] = skill
                converted.append(skill)
        
        if converted:
            self.loaded_repos.append(f"{repo_url}@{ref}")
            console.print(f"[green]✓ Loaded {len(converted)} AgentSkills from {repo_url}[/green]")
        else:
            console.print(f"[yellow]⚠️ No SKILL.md files found in {repo_url}[/yellow]")
        return converted

# ==========================================
# 🌐 SKILL MARKETPLACE CLIENT
# ==========================================

class SkillMarketplace:
    def __init__(self):
        self.config = SKILL_MARKETPLACE_CONFIG
        self.remote_skills: Dict[str, Skill] = {}
        self.last_fetch: Optional[datetime] = None
        self.load_cache()
    
    def load_cache(self):
        cache_file = self.config["cache_file"]
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    data = json.load(f)
                    self.last_fetch = datetime.fromisoformat(data.get('last_fetch', '2000-01-01'))
                    for skill_data in data.get('skills', []):
                        skill = self._dict_to_skill(skill_data)
                        self.remote_skills[skill.id] = skill
                console.print(f"[dim]📦 Loaded {len(self.remote_skills)} skills from cache[/dim]")
            except Exception as e:
                console.print(f"[dim]⚠️ Cache load failed: {e}[/dim]")
    
    def save_cache(self):
        try:
            self.config["cache_file"].parent.mkdir(parents=True, exist_ok=True)
            data = {
                'last_fetch': datetime.now().isoformat(),
                'skills': [s.to_dict() for s in self.remote_skills.values()]
            }
            with open(self.config["cache_file"], 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            console.print(f"[dim]⚠️ Cache save failed: {e}[/dim]")
    
    def _dict_to_skill(self, data: dict) -> Skill:
        return Skill(
            id=data['id'],
            name=data['name'],
            description=data['description'],
            category=SkillCategory(data.get('category', 'custom')),
            version=data.get('version', '1.0.0'),
            author=data.get('author', 'Unknown'),
            tools=data.get('tools', []),
            system_prompt_addition=data.get('system_prompt_addition', ''),
            dependencies=data.get('dependencies', []),
            dangerous=data.get('dangerous', False),
            download_url=data.get('download_url'),
            source=data.get('source', 'remote')
        )
    
    def fetch_remote_skills(self, force: bool = False) -> bool:
        if not force and self.last_fetch:
            age = (datetime.now() - self.last_fetch).total_seconds() / 3600
            if age < self.config["cache_ttl_hours"]:
                console.print(f"[dim]📦 Using cached skills ({age:.1f}h old)[/dim]")
                return True
        
        urls_to_try = [
            self.config["primary_url"],
            self.config["fallback_url"],
        ]
        
        for url in urls_to_try:
            try:
                console.print(f"[dim]🌐 Fetching skills from {url[:50]}...[/dim]")
                req = urllib.request.Request(
                    url,
                    headers={'User-Agent': 'TripleG-SkillMarketplace/1.0'}
                )
                with urllib.request.urlopen(req, timeout=10) as response:
                    data = json.loads(response.read().decode('utf-8'))
                    
                    self.remote_skills = {}
                    for skill_data in data.get('skills', []):
                        skill = self._dict_to_skill(skill_data)
                        self.remote_skills[skill.id] = skill
                    
                    self.last_fetch = datetime.now()
                    self.save_cache()
                    
                    console.print(f"[green]✓ Fetched {len(self.remote_skills)} skills from marketplace[/green]")
                    return True
                    
            except urllib.error.HTTPError as e:
                console.print(f"[dim]⚠️ HTTP {e.code} from {url[:30]}...[/dim]")
                continue
            except Exception as e:
                console.print(f"[dim]⚠️ Failed to fetch from {url[:30]}: {e}[/dim]")
                continue
        
        # Backoff failed fetch attempts to avoid repeated startup noise.
        self.last_fetch = datetime.now()
        self.save_cache()
        console.print("[yellow]⚠️ Could not reach marketplace. Using cached/builtin skills.[/yellow]")
        return len(self.remote_skills) > 0
    
    def get_all_skills(self, include_builtin: bool = True) -> Dict[str, Skill]:
        all_skills = {}
        if include_builtin:
            all_skills.update(BUILTIN_SKILLS)
        all_skills.update(self.remote_skills)
        return all_skills
    
    def download_skill_code(self, skill: Skill) -> Optional[str]:
        if not skill.download_url:
            return None
        
        try:
            req = urllib.request.Request(
                skill.download_url,
                headers={'User-Agent': 'TripleG-SkillMarketplace/1.0'}
            )
            with urllib.request.urlopen(req, timeout=15) as response:
                return response.read().decode('utf-8')
        except Exception as e:
            console.print(f"[red]❌ Failed to download skill code: {e}[/red]")
            return None

# ==========================================
# 📦 BUILTIN SKILLS (Fallback)
# ==========================================

BUILTIN_SKILLS = {
    "full_pc_access": Skill(
        id="full_pc_access",
        name="Full PC Access",
        description="Extended system access including process management, hardware info, and system configuration",
        category=SkillCategory.SYSTEM,
        version="1.0.0",
        author="TripleG",
        tools=["run_command", "get_system_info", "list_processes", "kill_process", "get_hardware_info"],
        system_prompt_addition="""
You have FULL PC ACCESS. You can:
- View and manage running processes
- Access hardware information (CPU, RAM, GPU, disks)
- Read system logs and configuration
- Execute system-level commands
ALWAYS confirm dangerous operations with the user.
""",
        dependencies=[],
        dangerous=True,
        source="builtin"
    ),
    
    "advanced_file_ops": Skill(
        id="advanced_file_ops",
        name="Advanced File Operations",
        description="Advanced file manipulation: bulk operations, compression, diff analysis",
        category=SkillCategory.FILE_OPS,
        version="1.0.0",
        author="TripleG",
        tools=["read_file", "write_file", "copy_file", "move_file", "delete_file", "compress_files", "file_diff"],
        system_prompt_addition="""
You have ADVANCED FILE OPERATIONS:
- Bulk copy/move/delete with patterns
- Compress/decompress (zip, tar, gzip)
- File diff and patch generation
- Always verify before destructive operations.
""",
        dependencies=[],
        dangerous=False,
        source="builtin"
    ),
    
    "web_scraper": Skill(
        id="web_scraper",
        name="Web Scraping & HTTP",
        description="Fetch web pages, parse HTML, download files, API interactions",
        category=SkillCategory.WEB,
        version="1.0.0",
        author="TripleG",
        tools=["fetch_url", "parse_html", "download_file", "http_request"],
        system_prompt_addition="""
You have WEB SCRAPING capabilities:
- Fetch and parse web pages
- Extract structured data from HTML
- Make HTTP requests (GET, POST, etc.)
- Respect robots.txt and rate limits
""",
        dependencies=["requests"],
        dangerous=False,
        source="builtin"
    ),
    
    "git_master": Skill(
        id="git_master",
        name="Git Master",
        description="Advanced git operations: branching, rebasing, history manipulation",
        category=SkillCategory.CODE,
        version="1.0.0",
        author="TripleG",
        tools=["git_status", "git_commit", "git_branch", "git_merge", "git_rebase"],
        system_prompt_addition="""
You are a GIT MASTER:
- Create and manage branches
- Handle merges and rebases
- Resolve conflicts
- NEVER force-push to shared branches without confirmation.
""",
        dependencies=["git"],
        dangerous=True,
        source="builtin"
    ),
    
    "docker_control": Skill(
        id="docker_control",
        name="Docker Controller",
        description="Manage containers, images, networks, and volumes",
        category=SkillCategory.SYSTEM,
        version="1.0.0",
        author="TripleG",
        tools=["docker_ps", "docker_images", "docker_run", "docker_stop"],
        system_prompt_addition="""
You have DOCKER CONTROL:
- List and manage containers
- Build and run images
- View logs and execute commands in containers
- Be careful with volume mounts.
""",
        dependencies=["docker"],
        dangerous=True,
        source="builtin"
    ),
    
    "chrome_browser": Skill(
        id="chrome_browser",
        name="Chrome Browser Automation",
        description="Full Chrome browser control: open URLs, click, type, screenshot, execute JS, manage tabs",
        category=SkillCategory.WEB,
        version="1.0.0",
        author="TripleG",
        tools=[
            "chrome_open", "chrome_close", "chrome_get_text", "chrome_get_html",
            "chrome_click", "chrome_type", "chrome_screenshot", "chrome_execute_js",
            "chrome_scroll", "chrome_find_elements", "chrome_wait", "chrome_tabs",
            "chrome_switch_tab", "chrome_new_tab", "chrome_back", "chrome_forward",
            "chrome_cookies"
        ],
        system_prompt_addition="""
You have CHROME BROWSER AUTOMATION:
- chrome_open(url): Open URL in Chrome (launches browser if needed)
- chrome_close(): Close the browser
- chrome_get_text(selector): Get visible text from page/element
- chrome_get_html(selector): Get HTML source
- chrome_click(selector): Click an element by CSS selector
- chrome_type(selector, text): Type into input fields
- chrome_screenshot(filename): Take screenshot
- chrome_execute_js(script): Run JavaScript
- chrome_scroll(direction, amount): Scroll up/down/top/bottom
- chrome_find_elements(selector): Find and list elements
- chrome_wait(selector, timeout): Wait for element to appear
- chrome_tabs(): List all open tabs
- chrome_switch_tab(index): Switch to tab by index
- chrome_new_tab(url): Open new tab
- chrome_back()/chrome_forward(): Navigate history
- chrome_cookies(): Get page cookies

WORKFLOW:
1. Always start with chrome_open(url)
2. Use chrome_wait() for dynamic content
3. Use chrome_find_elements() to discover selectors
4. Always chrome_close() when done

IMPORTANT: This controls the user's real Chrome browser. Be careful with sensitive sites.
""",
        dependencies=["selenium", "webdriver-manager"],
        dangerous=True,
        source="builtin"
    ),
}

# ==========================================
# 🔧 LOCAL EMBEDDING & SAM'S LAW
# ==========================================

class LocalEmbedder:
    def __init__(self, dim: int = 1536, seed: int = 42):
        self.dim = dim
        np.random.seed(seed)
        self.projection = np.random.randn(4096, dim).astype(np.float32) / np.sqrt(4096)
        
    def embed(self, text: str) -> np.ndarray:
        text = text.lower()[:8000]
        features = np.zeros(4096, dtype=np.float32)
        for i in range(len(text) - 2):
            trigram = text[i:i+3]
            idx = hash(trigram) % 4096
            features[idx] += 1.0
        norm = np.linalg.norm(features)
        if norm > 0:
            features = features / norm
        embedding = features @ self.projection
        return embedding.astype(np.float32)

class SamsLawQuantumCircuit:
    def __init__(self, embedding_dim: int = 1536, rank: int = 16, eta_0: float = 0.01,
                 kappa: float = 1e-4, device: str = 'cpu', gram_refresh: int = 1):
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch required")
        self.n = embedding_dim
        self.r = rank
        self.eta_0 = eta_0
        self.kappa = kappa
        self.device = device
        self.t = 0
        self.L = torch.randn(self.n, self.r, device=self.device) * 0.01
        self._G_cache = None
        self._update_counter = 0
        self._G_refresh_rate = gram_refresh
        self._update_magnitude = 0.0

    def _get_annealed_eta(self) -> float:
        return self.eta_0 / (1.0 + self.kappa * self.t)

    def evolve_state(self, s_vector: torch.Tensor):
        s = s_vector.to(self.device).view(-1, 1)
        sTL = torch.matmul(s.T, self.L)
        U = torch.matmul(s, sTL)
        if self._G_cache is None or self._update_counter % self._G_refresh_rate == 0:
            self._G_cache = torch.matmul(self.L.T, self.L)
        G = self._G_cache
        V = torch.matmul(self.L, G)
        current_eta = self._get_annealed_eta()
        delta_L = current_eta * (1.0 / self.n) * (U - V)
        self.L.add_(delta_L)
        self._update_magnitude = torch.norm(delta_L).item()
        self.t += 1
        self._update_counter += 1
        if self._G_refresh_rate == 1:
            self._G_cache = None

    def measure_fidelity(self, candidate_vector: torch.Tensor) -> float:
        s = candidate_vector.to(self.device).view(-1, 1)
        projection = torch.matmul(s.T, self.L)
        return torch.norm(projection).item() ** 2

    def get_status(self) -> Dict[str, Any]:
        return {
            'step': self.t,
            'eta_current': self._get_annealed_eta(),
            'eta_initial': self.eta_0,
            'kappa': self.kappa,
            'rank': self.r,
            'dimension': self.n,
            'update_mag': self._update_magnitude,
            'device': str(self.device),
        }

    def save_state(self, path: str):
        torch.save({
            'L': self.L.cpu(),
            't': self.t,
            'eta_0': self.eta_0,
            'kappa': self.kappa,
            'n': self.n,
            'r': self.r,
        }, path)

    def load_state(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.L = checkpoint['L'].to(self.device)
        self.t = checkpoint['t']
        self.eta_0 = checkpoint['eta_0']
        self.kappa = checkpoint['kappa']
        if 'n' in checkpoint:
            self.n = checkpoint['n']
        if 'r' in checkpoint:
            self.r = checkpoint['r']
        self._G_cache = None

# ==========================================
# 🎨 CONFIGURATION (PART 1 & 2 - THEME & PROMPT)
# ==========================================

CONFIG = {
    "AGENT_NAME": "TripleG-Sam",
    "MODEL_NAME": "qwen/qwen3-coder-next",
    "API_URL": "http://192.168.88.14:1234/v1",
    "API_KEY": "sk-lm-PwK1kizq:8gpU0G95r9zMjpiDaSfR",
    "TIMEOUT": 120,
    "MAX_ITERATIONS": 6,
    "EMBEDDING_DIM": 1536,
    "SAM_RANK": 16,
    "SAM_ETA_0": 0.01,
    "SAM_KAPPA": 1e-4,
    "SAM_GRAM_REFRESH": 1,
    "FIDELITY_RETENTION_PCT": 0.6,
    "MIN_HISTORY_KEEP": 5,
    "TOKEN_LIMIT": 8000,
    "SAFETY_BUFFER": 0.85,
    "SKILLS_DIR": Path.home() / ".tripleg" / "skills",
    
    # ✅ PART 1 - CYBERPUNK THEME
    "THEME": Theme({
        "agent": "bold bright_cyan",
        "user": "bold bright_green",
        "system": "dim white",
        "tool": "bold bright_yellow",
        "error": "bold bright_red",
        "success": "bold bright_green",
        "warning": "bold orange3",

        # cyberpunk accents
        "sam": "bold bright_magenta",
        "skill": "bold cyan",
        "marketplace": "bold bright_cyan",

        # neon UI
        "panel_border": "bright_magenta",
        "glow": "bright_cyan",
        "neon": "bright_magenta",
    }),
    
    # ✅ PART 2 - NEURAL PROMPT
    "PROMPT_SYMBOL": "❯❯ ",
    "PROMPT_COLOR": "ansibrightcyan",
}

console = Console(theme=CONFIG["THEME"])
client = OpenAI(base_url=CONFIG["API_URL"], api_key=CONFIG["API_KEY"], timeout=60)
CONFIG["SKILLS_DIR"].mkdir(parents=True, exist_ok=True)

# ==========================================
# 🛠️ TOOL ENGINE
# ==========================================

class ToolEngine:
    DANGEROUS_PATTERNS = [
        r'rm\s+-rf\s+/', r'mkfs\.', r'dd\s+if=/dev/zero',
        r':\(\)\s*\{\s*:\|\s*:\s*&\s*\}\s*;', r'>\s*/etc/',
        r'curl\s+.*\s*\|\s*sh', r'wget\s+.*\s*-O\s*-\s*\|',
    ]
    
    @classmethod
    def is_safe(cls, cmd: str) -> Tuple[bool, str]:
        for p in cls.DANGEROUS_PATTERNS:
            if re.search(p, cmd, re.I):
                return False, f"Blocked: {p}"
        return True, ""
    
    @staticmethod
    def run_command(command: str, cwd: Optional[str] = None) -> str:
        safe, reason = ToolEngine.is_safe(command)
        if not safe:
            return f"🚫 {reason}"
        try:
            r = subprocess.run(command, shell=True, capture_output=True, 
                             text=True, timeout=CONFIG["TIMEOUT"], cwd=cwd)
            out = []
            if r.stdout: out.append(r.stdout.strip())
            if r.stderr: out.append(f"[stderr]\n{r.stderr.strip()}")
            out.append(f"[Status: {'✓' if r.returncode == 0 else f'✗ {r.returncode}'}]")
            return "\n".join(out)
        except subprocess.TimeoutExpired:
            return "⏱️ Timeout"
        except Exception as e:
            return f"💥 {e}"

    @staticmethod
    def write_file(path: str, content: str) -> str:
        try:
            p = Path(path)
            p.parent.mkdir(parents=True, exist_ok=True)
            if p.exists():
                p.with_suffix('.backup').write_text(p.read_text('utf-8'), 'utf-8')
            p.write_text(content, 'utf-8')
            return f"✓ {len(content)} chars → '{path}'"
        except Exception as e:
            return f"💥 {e}"

    @staticmethod
    def read_file(path: str, offset: int = 0, limit: int = 100) -> str:
        try:
            p = Path(path)
            if not p.exists():
                return f"❌ '{path}' not found"
            lines = p.read_text('utf-8').splitlines()
            if offset >= len(lines):
                return f"⚠️ {len(lines)} lines"
            end = min(offset + limit, len(lines))
            numbered = [f"{i+1:4d} │ {lines[i]}" for i in range(offset, end)]
            return f"📄 {path} ({offset+1}-{end}/{len(lines)})\n{'─'*60}\n" + "\n".join(numbered)
        except Exception as e:
            return f"💥 {e}"

    @staticmethod
    def list_dir(path: str = ".", pattern: str = "*") -> str:
        try:
            t = Path(path)
            if not t.exists():
                return f"❌ '{path}' not found"
            items = list(t.glob(pattern))[:50]
            lines = [f"📁 {t.resolve()}", "─"*50]
            for d in sorted([x for x in items if x.is_dir()]):
                lines.append(f"📂 {d.name}/")
            for f in sorted([x for x in items if x.is_file()]):
                sz = f.stat().st_size
                sz_str = f"{sz}B" if sz < 1024 else f"{sz/1024:.1f}KB" if sz < 1024**2 else f"{sz/1024**2:.1f}MB"
                lines.append(f"📄 {f.name:<30} {sz_str:>10}")
            return "\n".join(lines)
        except Exception as e:
            return f"💥 {e}"

    @staticmethod
    def search_files(query: str, path: str = ".", ext: str = "*") -> str:
        try:
            cmd = f"rg -i -n --color=never '{query}' {path} 2>/dev/null | head -30"
            r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)
            if r.returncode == 0 and r.stdout:
                lines = r.stdout.strip().split('\n')
                return f"🔍 {len(lines)} matches:\n" + "\n".join(lines[:30])
            return f"🔍 No matches"
        except Exception as e:
            return f"💥 {e}"

    @staticmethod
    def get_system_info() -> str:
        try:
            import platform
            info = {
                "os": platform.system(),
                "release": platform.release(),
                "version": platform.version(),
                "machine": platform.machine(),
                "processor": platform.processor(),
                "python": platform.python_version(),
            }
            return "System Info:\n" + "\n".join(f"  {k}: {v}" for k, v in info.items())
        except Exception as e:
            return f"💥 {e}"

    @staticmethod
    def list_processes() -> str:
        try:
            if sys.platform == "darwin" or sys.platform.startswith("linux"):
                r = subprocess.run(["ps", "aux"], capture_output=True, text=True, timeout=10)
                lines = r.stdout.strip().split('\n')[:20]
                return "Top Processes:\n" + "\n".join(lines)
            else:
                r = subprocess.run(["tasklist"], capture_output=True, text=True, timeout=10)
                return r.stdout[:2000]
        except Exception as e:
            return f"💥 {e}"

    @staticmethod
    def copy_file(src: str, dst: str) -> str:
        try:
            import shutil
            shutil.copy2(src, dst)
            return f"✓ Copied '{src}' → '{dst}'"
        except Exception as e:
            return f"💥 {e}"

    @staticmethod
    def move_file(src: str, dst: str) -> str:
        try:
            import shutil
            shutil.move(src, dst)
            return f"✓ Moved '{src}' → '{dst}'"
        except Exception as e:
            return f"💥 {e}"

    @staticmethod
    def delete_file(path: str) -> str:
        try:
            p = Path(path)
            if p.is_dir():
                import shutil
                shutil.rmtree(p)
                return f"✓ Deleted directory '{path}'"
            else:
                p.unlink()
                return f"✓ Deleted file '{path}'"
        except Exception as e:
            return f"💥 {e}"

    @staticmethod
    def compress_files(paths: str, output: str) -> str:
        try:
            import zipfile
            path_list = [p.strip() for p in paths.split(',')]
            with zipfile.ZipFile(output, 'w', zipfile.ZIP_DEFLATED) as zf:
                for p in path_list:
                    zf.write(p, arcname=Path(p).name)
            return f"✓ Compressed {len(path_list)} items → '{output}'"
        except Exception as e:
            return f"💥 {e}"

    @staticmethod
    def file_diff(file1: str, file2: str) -> str:
        try:
            import difflib
            f1_lines = Path(file1).read_text('utf-8').splitlines()
            f2_lines = Path(file2).read_text('utf-8').splitlines()
            diff = list(difflib.unified_diff(f1_lines, f2_lines, lineterm=''))
            return f"Diff ({len(diff)} lines):\n" + "\n".join(diff[:50])
        except Exception as e:
            return f"💥 {e}"

    # ==========================
    # 🌐 WEB TOOLS (New Additions)
    # ==========================

    @staticmethod
    def _parse_payload_string(value: Any) -> Any:
        """Best-effort parser for nested JSON/Python-literal payload strings."""
        current: Any = value
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

            if isinstance(parsed, str):
                if parsed.strip() == text:
                    return parsed
                current = parsed
                continue
            return parsed
        return current

    @staticmethod
    def _coerce_web_payload(url: str, method: str = "GET", data: Optional[str] = None,
                            dest: Optional[str] = None) -> Tuple[str, str, Optional[str], Optional[str]]:
        """
        Accept either direct args or a serialized payload string, e.g.
        '{"url":"...","method":"POST","data":"...","dest":"..."}'.
        """
        parsed = ToolEngine._parse_payload_string(url)
        if isinstance(parsed, dict):
            url = str(parsed.get("url", url)).strip()
            method = str(parsed.get("method", method)).strip().upper()
            if "data" in parsed and data is None:
                raw_data = parsed.get("data")
                if isinstance(raw_data, (dict, list)):
                    data = json.dumps(raw_data)
                elif raw_data is not None:
                    data = str(raw_data)
            if dest is None and "dest" in parsed:
                dest = str(parsed.get("dest"))
        return url, method, data, dest

    @staticmethod
    def _extract_price_candidates(text: str) -> List[str]:
        """Extract likely fare/price tokens from raw text."""
        if not text:
            return []
        patterns = [
            r'(?:€|\$|£)\s?\d{1,4}(?:[.,]\d{2})?',
            r'\b\d{1,4}(?:[.,]\d{2})?\s?(?:EUR|USD|GBP)\b',
        ]
        seen = set()
        prices: List[str] = []
        for pat in patterns:
            for match in re.findall(pat, text, re.IGNORECASE):
                token = re.sub(r'\s+', ' ', match).strip()
                if token and token not in seen:
                    seen.add(token)
                    prices.append(token)
                if len(prices) >= 20:
                    return prices
        return prices

    @staticmethod
    def _strip_html_to_text(html: str) -> str:
        """Remove tags/scripts/styles and return compact visible text."""
        if not html:
            return ""
        text = re.sub(r'<!--.*?-->', ' ', html, flags=re.DOTALL)
        text = re.sub(r'<script\b[^>]*>.*?</script>', ' ', text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'<style\b[^>]*>.*?</style>', ' ', text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'<noscript\b[^>]*>.*?</noscript>', ' ', text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'<[^>]+>', ' ', text)
        return re.sub(r'\s+', ' ', text).strip()

    @staticmethod
    def _summarize_html_content(content: str) -> Dict[str, Any]:
        title_match = re.search(r'<title[^>]*>(.*?)</title>', content, re.DOTALL | re.IGNORECASE)
        title = ""
        if title_match:
            title = re.sub(r'\s+', ' ', re.sub(r'<.*?>', '', title_match.group(1))).strip()

        script_tags = len(re.findall(r'<script\b', content, re.IGNORECASE))
        js_heavy = bool(
            re.search(r'enable javascript to run this app|window\.__|bundle|webpack|application\/json\+ld', content, re.IGNORECASE)
            or script_tags >= 8
        )

        visible_text = ToolEngine._strip_html_to_text(content)
        return {
            "title": title,
            "script_tags": script_tags,
            "js_heavy": js_heavy,
            "price_candidates": ToolEngine._extract_price_candidates(content),
            "visible_preview": visible_text[:3500],
        }
    
    @staticmethod
    def code_assistant(query: str) -> str:
        """Agentic coding assistant powered by Qwen 3 Coder on LM Studio (separate PC).
        
        This is a MULTI-TURN AGENTIC tool: Qwen 3 Coder can call tools (write_file,
        read_file, run_command, list_dir, search_files) in a loop to actually build
        projects, create files, and execute commands — not just return advice text.
        
        Use this for:
        - Building websites, apps, and projects (actually creates the files)
        - Writing code and saving it to disk
        - Code review and debugging (reads files, suggests fixes, applies them)
        - Software architecture design with implementation
        - Any coding task that requires creating/modifying files
        """
        try:
            parsed = ToolEngine._parse_payload_string(query)
            if isinstance(parsed, dict):
                query = str(parsed.get("query", parsed.get("q", parsed.get("question", query)))).strip()
            elif isinstance(parsed, str):
                query = parsed.strip()

            if not query:
                return "❌ No query provided for code assistant."

            # Qwen 3 Coder model on LM Studio (separate PC)
            CODER_API_URL = "http://10.10.10.1:1234/v1"
            CODER_MODEL = "qwen/qwen3-coder-next"
            MAX_ITERATIONS = 15
            
            # Tools available to the coder agent
            CODER_TOOLS = {
                "write_file": ToolEngine.write_file,
                "read_file": ToolEngine.read_file,
                "run_command": ToolEngine.run_command,
                "list_dir": ToolEngine.list_dir,
                "search_files": ToolEngine.search_files,
            }
            
            # Create a separate OpenAI client for the coder model
            from openai import OpenAI as CoderClient
            coder_client = CoderClient(
                base_url=CODER_API_URL,
                api_key="not-needed",  # LM Studio doesn't require API key
                timeout=180  # Longer timeout for code generation
            )
            
            # Agentic system prompt with tool-calling protocol
            coder_system_prompt = """You are an expert software engineer and coding agent. You can DIRECTLY create files, run commands, and build projects on the user's computer.

## AVAILABLE TOOLS
write_file - Create or overwrite files (auto-creates directories)
read_file - Read file contents
run_command - Execute shell commands (mkdir, npm, pip, git, etc.)
list_dir - List directory contents
search_files - Search for text patterns in files

## TOOL CALLING PROTOCOL
When you need to perform an action, use this EXACT format:

:::TOOL:tool_name:::
:::ARGS:::
arguments here
:::END:::

### EXAMPLES:

Create a file:
:::TOOL:write_file:::
:::ARGS:::
{"path": "my_project/index.html", "content": "<!DOCTYPE html>\\n<html>\\n<head><title>My App</title></head>\\n<body><h1>Hello</h1></body>\\n</html>"}
:::END:::

Run a command:
:::TOOL:run_command:::
:::ARGS:::
mkdir my_project
:::END:::

Read a file:
:::TOOL:read_file:::
:::ARGS:::
{"path": "my_project/index.html"}
:::END:::

List directory:
:::TOOL:list_dir:::
:::ARGS:::
{"path": "my_project"}
:::END:::

## RULES
1. ALWAYS use tools to create files and run commands — do NOT just show code in markdown
2. Create directories first with run_command before writing files into them
3. Write COMPLETE file contents — no placeholders or "// rest of code here"
4. After creating files, verify with list_dir or read_file
5. For multi-file projects, create files one at a time
6. Use run_command for: mkdir, npm init, pip install, git init, etc.
7. When done, provide a brief summary of what you created
8. For Windows paths, use forward slashes or escaped backslashes
9. Keep responses focused — execute actions, don't just explain them"""

            # Conversation history for multi-turn
            messages = [
                {"role": "system", "content": coder_system_prompt},
                {"role": "user", "content": query}
            ]
            
            # Track actions for summary
            actions_log = []
            last_tool_sig = None
            repeat_hits = 0
            error_streak = 0
            
            for iteration in range(MAX_ITERATIONS):
                try:
                    response = coder_client.chat.completions.create(
                        model=CODER_MODEL,
                        messages=messages,
                        temperature=0.15,
                        max_tokens=8192,
                    )
                except Exception as api_err:
                    error_msg = str(api_err)
                    if "Connection refused" in error_msg or "connect" in error_msg.lower():
                        return (
                            "💥 Code assistant unavailable: Cannot connect to Qwen 3 Coder at http://10.10.10.1:1234\n"
                            "Make sure LM Studio is running on the remote PC with the model loaded."
                        )
                    elif "timeout" in error_msg.lower():
                        return (
                            "💥 Code assistant timeout: The Qwen 3 Coder model took too long to respond.\n"
                            "The model may be loading or the query may be too complex."
                        )
                    elif "model" in error_msg.lower() and "not found" in error_msg.lower():
                        return (
                            "💥 Code assistant error: Model 'qwen/qwen3-coder-next' not found.\n"
                            "Make sure the model is loaded in LM Studio."
                        )
                    else:
                        return f"💥 Code assistant API error: {api_err}"
                
                assistant_msg = response.choices[0].message.content or ""
                if not assistant_msg:
                    break
                
                messages.append({"role": "assistant", "content": assistant_msg})
                
                # Parse for tool calls using ResponseParser
                parsed_text, tool_name, tool_args = ResponseParser.parse(assistant_msg)
                
                if tool_name and tool_name in CODER_TOOLS:
                    # Execute the tool
                    tool_fn = CODER_TOOLS[tool_name]
                    
                    # Decode arguments
                    try:
                        raw = tool_args.strip() if tool_args else ""
                        # Try JSON first
                        try:
                            args_parsed = json.loads(raw)
                        except (json.JSONDecodeError, ValueError):
                            try:
                                args_parsed = ast.literal_eval(raw)
                            except (ValueError, SyntaxError):
                                args_parsed = raw  # Use as raw string
                        
                        if isinstance(args_parsed, dict):
                            result = tool_fn(**args_parsed)
                        elif isinstance(args_parsed, list):
                            result = tool_fn(*args_parsed)
                        else:
                            result = tool_fn(str(args_parsed))
                    except TypeError:
                        # Fallback: pass raw string
                        try:
                            result = tool_fn(tool_args or "")
                        except Exception as e:
                            result = f"💥 Tool argument error: {e}"
                    except Exception as e:
                        result = f"💥 Tool failed: {e}"
                    
                    result_str = str(result)[:3000]
                    actions_log.append(f"[{tool_name}] {result_str[:150]}")
                    
                    # Loop guard: detect repeated calls
                    sig = f"{tool_name}|{tool_args}"
                    if sig == last_tool_sig:
                        repeat_hits += 1
                    else:
                        repeat_hits = 0
                    last_tool_sig = sig
                    
                    if result_str.startswith("💥"):
                        error_streak += 1
                    else:
                        error_streak = 0
                    
                    if repeat_hits >= 2 or error_streak >= 3:
                        messages.append({
                            "role": "user",
                            "content": f"RESULT [{tool_name}]: {result_str}\n"
                                       "STOP calling tools. Provide a final summary of what was accomplished."
                        })
                    else:
                        messages.append({
                            "role": "user",
                            "content": f"RESULT [{tool_name}]: {result_str}\nContinue."
                        })
                    continue
                
                elif tool_name and tool_name not in CODER_TOOLS:
                    # Unknown tool — tell coder to use available tools
                    messages.append({
                        "role": "user",
                        "content": f"Unknown tool '{tool_name}'. Available tools: {', '.join(CODER_TOOLS.keys())}. "
                                   "Please use one of these tools or provide a direct answer."
                    })
                    continue
                
                else:
                    # No tool call — coder gave a final response
                    break
            
            # Build final output
            # Get the last assistant message as the summary
            final_response = ""
            for msg in reversed(messages):
                if msg["role"] == "assistant":
                    # Strip any tool calls from the final message
                    clean_text, _, _ = ResponseParser.parse(msg["content"])
                    final_response = clean_text or msg["content"]
                    break
            
            output_parts = [
                "🤖 Code Assistant (Qwen 3 Coder) — Agentic Mode",
                "─" * 50,
            ]
            
            if actions_log:
                output_parts.append(f"\n📋 Actions performed ({len(actions_log)}):")
                for i, action in enumerate(actions_log, 1):
                    output_parts.append(f"  {i}. {action}")
                output_parts.append("")
            
            if final_response:
                # Truncate if needed
                if len(final_response) > 3000:
                    final_response = final_response[:3000] + "\n\n[... truncated]"
                output_parts.append(final_response)
            elif actions_log:
                output_parts.append("✅ All actions completed successfully.")
            else:
                output_parts.append("No actions were taken. The model may not have understood the request.")
            
            return "\n".join(output_parts)
            
        except Exception as e:
            error_msg = str(e)
            if "Connection refused" in error_msg or "connect" in error_msg.lower():
                return (
                    "💥 Code assistant unavailable: Cannot connect to Qwen 3 Coder at http://10.10.10.1:1234\n"
                    "Make sure LM Studio is running on the remote PC with the model loaded."
                )
            elif "timeout" in error_msg.lower():
                return (
                    "💥 Code assistant timeout: The Qwen 3 Coder model took too long to respond.\n"
                    "The model may be loading or the query may be too complex."
                )
            elif "model" in error_msg.lower() and "not found" in error_msg.lower():
                return (
                    "💥 Code assistant error: Model 'qwen/qwen3-coder-next' not found.\n"
                    "Make sure the model is loaded in LM Studio."
                )
            else:
                return f"💥 Code assistant error: {e}"

    @staticmethod
    def grokipedia(query: str) -> str:
        """Look up encyclopedic/factual information from Grokipedia (xAI's knowledge base with 6M+ articles)."""
        try:
            parsed = ToolEngine._parse_payload_string(query)
            if isinstance(parsed, dict):
                query = str(parsed.get("query", parsed.get("q", parsed.get("topic", query)))).strip()
            elif isinstance(parsed, str):
                query = parsed.strip()

            if not query:
                return "❌ No query provided for Grokipedia lookup."

            # Convert query to slug format: spaces → underscores, preserve commas/ampersands
            slug = query.strip()
            # Remove surrounding quotes if present
            if (slug.startswith('"') and slug.endswith('"')) or (slug.startswith("'") and slug.endswith("'")):
                slug = slug[1:-1]
            slug = slug.replace(" ", "_")
            # URL-encode special characters but keep underscores and common chars
            slug = urllib.parse.quote(slug, safe="_,&")

            api_url = f"https://grokipedia-api.com/page/{slug}"
            headers = {
                'User-Agent': 'TripleG-JARVIS/1.0',
                'Accept': 'application/json',
            }

            req = urllib.request.Request(api_url, headers=headers)
            with urllib.request.urlopen(req, timeout=15) as response:
                data = json.loads(response.read().decode('utf-8', errors='ignore'))

            title = data.get("title", query)
            content = data.get("content_text", "")
            char_count = data.get("char_count", len(content))
            word_count = data.get("word_count", 0)
            ref_count = data.get("references_count", 0)
            page_url = data.get("url", f"https://grokipedia.com/page/{slug}")

            if not content:
                return f"📚 Grokipedia: No content found for '{query}'. Try web_search instead."

            # Truncate content for context window (keep first ~3000 chars)
            truncated = content[:3000]
            if len(content) > 3000:
                # Try to cut at a sentence boundary
                last_period = truncated.rfind('. ')
                if last_period > 2000:
                    truncated = truncated[:last_period + 1]
                truncated += "\n\n[... truncated, full article has more content]"

            output = [
                f"📚 Grokipedia: {title}",
                f"{'─' * 50}",
                f"Source: {page_url}",
                f"Length: {word_count} words ({char_count} chars) | {ref_count} references",
                f"{'─' * 50}",
                "",
                truncated,
            ]

            # Include first few references if available
            refs = data.get("references", [])
            if refs:
                output.append(f"\n{'─' * 50}")
                output.append(f"📎 References ({min(len(refs), 5)} of {len(refs)}):")
                for ref in refs[:5]:
                    num = ref.get("number", "?")
                    url = ref.get("url", "")
                    output.append(f"  [{num}] {url}")

            return "\n".join(output)

        except urllib.error.HTTPError as e:
            if e.code == 404:
                return f"📚 Grokipedia: No article found for '{query}'. Try a different spelling or use web_search for this topic."
            return f"💥 Grokipedia error (HTTP {e.code}): {e.reason}"
        except urllib.error.URLError as e:
            return f"💥 Grokipedia network error: {e.reason}. Try web_search as fallback."
        except json.JSONDecodeError:
            return f"💥 Grokipedia returned invalid data for '{query}'. Try web_search instead."
        except Exception as e:
            return f"💥 Grokipedia lookup failed: {e}. Try web_search as fallback."

    @staticmethod
    def web_search(query: str, num_results: int = 5) -> str:
        """Search the web using DuckDuckGo and return results."""
        try:
            parsed = ToolEngine._parse_payload_string(query)
            if isinstance(parsed, dict):
                query = str(parsed.get("query", parsed.get("q", query))).strip()
                num_results = int(parsed.get("num_results", parsed.get("num", 5)))
            elif isinstance(parsed, str):
                query = parsed.strip()
            
            if not query:
                return "❌ No search query provided."
            
            # Use DuckDuckGo HTML search (no API key needed)
            search_url = f"https://html.duckduckgo.com/html/?q={urllib.parse.quote(query)}"
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.9',
            }
            
            req = urllib.request.Request(search_url, headers=headers)
            with urllib.request.urlopen(req, timeout=15) as response:
                html = response.read().decode('utf-8', errors='ignore')
            
            # Parse DuckDuckGo results
            results = []
            
            # Extract result blocks - DuckDuckGo HTML uses class="result"
            result_pattern = re.compile(
                r'<a[^>]*class="result__a"[^>]*href="([^"]*)"[^>]*>(.*?)</a>.*?'
                r'<a[^>]*class="result__snippet"[^>]*>(.*?)</a>',
                re.DOTALL | re.IGNORECASE
            )
            
            # Alternative pattern for result extraction
            alt_pattern = re.compile(
                r'<h2[^>]*class="result__title"[^>]*>.*?<a[^>]*href="([^"]*)"[^>]*>(.*?)</a>.*?'
                r'class="result__snippet"[^>]*>(.*?)</(?:a|span)',
                re.DOTALL | re.IGNORECASE
            )
            
            matches = result_pattern.findall(html) or alt_pattern.findall(html)
            
            # Fallback: simpler extraction
            if not matches:
                # Try to find any links with snippets
                link_pattern = re.compile(r'<a[^>]*href="(https?://[^"]+)"[^>]*>([^<]+)</a>', re.IGNORECASE)
                links = link_pattern.findall(html)
                for url, title in links[:num_results]:
                    if 'duckduckgo.com' not in url and len(title) > 10:
                        clean_title = re.sub(r'<[^>]+>', '', title).strip()
                        results.append({
                            'url': url,
                            'title': clean_title[:100],
                            'snippet': ''
                        })
            else:
                for match in matches[:num_results]:
                    url, title, snippet = match
                    # Clean HTML tags
                    clean_title = re.sub(r'<[^>]+>', '', title).strip()
                    clean_snippet = re.sub(r'<[^>]+>', '', snippet).strip()
                    # Decode URL if needed
                    if url.startswith('//duckduckgo.com/l/?uddg='):
                        try:
                            url = urllib.parse.unquote(url.split('uddg=')[1].split('&')[0])
                        except:
                            pass
                    results.append({
                        'url': url,
                        'title': clean_title[:100],
                        'snippet': clean_snippet[:200]
                    })
            
            if not results:
                return f"🔍 No results found for: {query}\n(Try rephrasing your search)"
            
            # Format results
            output = [f"🔍 Web Search Results for: {query}\n{'─'*50}"]
            for i, r in enumerate(results, 1):
                output.append(f"\n[{i}] {r['title']}")
                output.append(f"    URL: {r['url']}")
                if r['snippet']:
                    output.append(f"    {r['snippet']}")
            
            output.append(f"\n{'─'*50}")
            output.append(f"💡 Use fetch_url to get full content from any of these URLs.")
            
            return "\n".join(output)
            
        except urllib.error.HTTPError as e:
            return f"💥 Search failed (HTTP {e.code}): {e.reason}"
        except urllib.error.URLError as e:
            return f"💥 Search failed (Network error): {e.reason}"
        except Exception as e:
            return f"💥 Search failed: {e}"

    @staticmethod
    def fetch_url(url: str) -> str:
        """Fetches the text content of a URL."""
        try:
            url, _, _, _ = ToolEngine._coerce_web_payload(url)
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) TripleG-Agent/1.0',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.9',
            }
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=15) as response:
                content = response.read().decode('utf-8', errors='ignore')
                content_type = response.headers.get('Content-Type', 'unknown')
                final_url = response.geturl()

                summary_lines = [
                    f"🌐 Source: {final_url}",
                    f"Status: {response.status}",
                    f"Content-Type: {content_type}",
                    f"Length: {len(content)}",
                ]

                if re.search(r'<html|<!doctype html', content, re.IGNORECASE):
                    info = ToolEngine._summarize_html_content(content)
                    if info["title"]:
                        summary_lines.append(f"Title: {info['title']}")
                    if info["price_candidates"]:
                        summary_lines.append("Price candidates: " + ", ".join(info["price_candidates"][:12]))
                    if info["js_heavy"]:
                        summary_lines.append("Note: JS-heavy page detected; dynamic content may not be present in raw HTML.")
                    preview = info["visible_preview"] or content[:2000]
                else:
                    preview = content[:2500]

                return "\n".join(summary_lines) + "\n\n" + preview
        except Exception as e:
            return f"💥 Fetch failed: {e}"

    @staticmethod
    def parse_html(html_content: str, tag: str = "p") -> str:
        """Regex-based HTML parser with support for multiple tags and price extraction."""
        try:
            if not html_content:
                return "📄 No HTML content provided."

            requested_tags = [t.strip() for t in tag.split(",") if t.strip()]
            if not requested_tags:
                requested_tags = ["p"]

            # Support parse_html(content, "price") style extraction.
            normalized = {t.lower() for t in requested_tags}
            if normalized & {"price", "prices", "currency"}:
                prices = ToolEngine._extract_price_candidates(html_content)
                if prices:
                    return f"💶 Found {len(prices)} price-like values:\n" + "\n".join(prices[:30])
                return "💶 No price-like values found."

            cleaned: List[str] = []
            seen = set()
            per_tag_counts: Dict[str, int] = {}
            for one_tag in requested_tags:
                pattern = f"<{one_tag}[^>]*>(.*?)</{one_tag}>"
                matches = re.findall(pattern, html_content, re.DOTALL | re.IGNORECASE)
                per_tag_counts[one_tag] = len(matches)
                for m in matches:
                    text = re.sub(r'<.*?>', '', m)
                    text = re.sub(r'\s+', ' ', text).strip()
                    if text and text not in seen:
                        seen.add(text)
                        cleaned.append(text)

            if cleaned:
                counts = ", ".join(f"<{k}>={v}" for k, v in per_tag_counts.items())
                return f"📄 Found {len(cleaned)} unique extracted blocks ({counts}):\n" + "\n".join(cleaned[:25])

            # Fallback extraction for JS-heavy pages with little semantic HTML.
            fallback = ToolEngine._summarize_html_content(html_content)
            fallback_lines = ["📄 No matches for requested tags."]
            if fallback["title"]:
                fallback_lines.append(f"Title: {fallback['title']}")
            if fallback["price_candidates"]:
                fallback_lines.append("Price candidates: " + ", ".join(fallback["price_candidates"][:12]))
            if fallback["visible_preview"]:
                fallback_lines.append("Visible text preview:\n" + fallback["visible_preview"][:1200])
            return "\n".join(fallback_lines)
        except Exception as e:
            return f"💥 Parse error: {e}"

    @staticmethod
    def download_file(url: str, dest: str) -> str:
        """Downloads a file from a URL to a local path."""
        try:
            url, _, _, parsed_dest = ToolEngine._coerce_web_payload(url, dest=dest)
            if parsed_dest:
                dest = parsed_dest
            headers = {'User-Agent': 'TripleG-Agent/1.0'}
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=30) as response:
                with open(dest, 'wb') as f:
                    f.write(response.read())
            return f"✓ Downloaded {url} -> {dest}"
        except Exception as e:
            return f"💥 Download failed: {e}"

    @staticmethod
    def http_request(url: str, method: str = "GET", data: Optional[str] = None) -> str:
        """Generic HTTP request handler."""
        try:
            payload = ToolEngine._parse_payload_string(url)
            extra_headers: Dict[str, str] = {}
            params: Dict[str, Any] = {}
            if isinstance(payload, dict):
                raw_headers = payload.get("headers", {})
                if isinstance(raw_headers, dict):
                    for k, v in raw_headers.items():
                        extra_headers[str(k)] = str(v)
                raw_params = payload.get("params", {})
                if isinstance(raw_params, dict):
                    params = raw_params

            url, method, data, _ = ToolEngine._coerce_web_payload(url, method=method, data=data)
            if params:
                sep = '&' if '?' in url else '?'
                url = f"{url}{sep}{urlencode(params, doseq=True)}"

            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) TripleG-Agent/1.0',
                'Accept': 'text/html,application/json;q=0.9,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.9',
            }
            headers.update(extra_headers)

            if isinstance(data, (dict, list)):
                data = json.dumps(data)
            encoded_data = data.encode('utf-8') if data else None
            if encoded_data and 'Content-Type' not in headers:
                headers['Content-Type'] = 'application/json'

            req = urllib.request.Request(url, data=encoded_data, headers=headers, method=method)
            with urllib.request.urlopen(req, timeout=15) as response:
                body = response.read().decode('utf-8', errors='ignore')
                content_type = response.headers.get('Content-Type', 'unknown')
                final_url = response.geturl()

                lines = [
                    f"🌐 HTTP {method} {response.status}",
                    f"Source: {final_url}",
                    f"Content-Type: {content_type}",
                    f"Length: {len(body)}",
                ]
                if re.search(r'<html|<!doctype html', body, re.IGNORECASE):
                    info = ToolEngine._summarize_html_content(body)
                    if info["title"]:
                        lines.append(f"Title: {info['title']}")
                    if info["price_candidates"]:
                        lines.append("Price candidates: " + ", ".join(info["price_candidates"][:12]))
                    if info["js_heavy"]:
                        lines.append("Note: JS-heavy page detected; raw HTML may omit rendered data.")
                    preview = info["visible_preview"] or body[:1800]
                else:
                    preview = body[:2000]
                return "\n".join(lines) + "\n\n" + preview
        except Exception as e:
            return f"💥 Request failed: {e}"

    # ==========================
    # 🌐 CHROME BROWSER TOOLS
    # ==========================

    # Shared browser instance (singleton per session)
    _chrome_driver = None
    _chrome_screenshots_dir = Path.home() / ".tripleg" / "screenshots"

    @staticmethod
    def _ensure_screenshots_dir():
        ToolEngine._chrome_screenshots_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def chrome_open(url: str) -> str:
        """Open a URL in Chrome browser. Launches Chrome if not already open."""
        if not SELENIUM_AVAILABLE:
            return "💥 Selenium not installed. Run: pip install selenium webdriver-manager"
        try:
            parsed = ToolEngine._parse_payload_string(url)
            if isinstance(parsed, dict):
                url = str(parsed.get("url", url)).strip()
            elif isinstance(parsed, str):
                url = parsed.strip()

            if ToolEngine._chrome_driver is None:
                options = ChromeOptions()
                options.add_argument("--no-sandbox")
                options.add_argument("--disable-dev-shm-usage")
                options.add_argument("--disable-gpu")
                options.add_argument("--window-size=1920,1080")
                # Connect to existing Chrome or launch new
                try:
                    # Try connecting to user's existing Chrome with remote debugging
                    options.add_experimental_option("debuggerAddress", "127.0.0.1:9222")
                    ToolEngine._chrome_driver = webdriver.Chrome(options=options)
                    return f"🌐 Connected to existing Chrome → {url}\n(Navigating...)" + ToolEngine._chrome_navigate(url)
                except Exception:
                    # Fall back to launching new Chrome with webdriver-manager
                    options = ChromeOptions()
                    options.add_argument("--no-sandbox")
                    options.add_argument("--disable-dev-shm-usage")
                    options.add_argument("--window-size=1920,1080")
                    # Use user's real Chrome profile for full access
                    user_data = Path.home() / "AppData" / "Local" / "Google" / "Chrome" / "User Data"
                    if user_data.exists():
                        options.add_argument(f"--user-data-dir={user_data}")
                        options.add_argument("--profile-directory=Default")
                    service = ChromeService(ChromeDriverManager().install())
                    ToolEngine._chrome_driver = webdriver.Chrome(service=service, options=options)

            ToolEngine._chrome_driver.get(url)
            time.sleep(2)  # Wait for page load
            title = ToolEngine._chrome_driver.title
            current_url = ToolEngine._chrome_driver.current_url
            return f"🌐 Chrome opened: {title}\nURL: {current_url}"
        except Exception as e:
            return f"💥 Chrome open failed: {e}"

    @staticmethod
    def _chrome_navigate(url: str) -> str:
        try:
            ToolEngine._chrome_driver.get(url)
            time.sleep(2)
            return f"\n✓ Navigated to: {ToolEngine._chrome_driver.title} ({ToolEngine._chrome_driver.current_url})"
        except Exception as e:
            return f"\n💥 Navigation failed: {e}"

    @staticmethod
    def chrome_get_text(selector: str = "body") -> str:
        """Get visible text from the current page or a specific CSS selector."""
        if not ToolEngine._chrome_driver:
            return "💥 Chrome not open. Use chrome_open first."
        try:
            parsed = ToolEngine._parse_payload_string(selector)
            if isinstance(parsed, dict):
                selector = str(parsed.get("selector", "body")).strip()
            elif isinstance(parsed, str):
                selector = parsed.strip() or "body"

            element = ToolEngine._chrome_driver.find_element(By.CSS_SELECTOR, selector)
            text = element.text[:5000]
            url = ToolEngine._chrome_driver.current_url
            title = ToolEngine._chrome_driver.title
            return f"📄 Page: {title}\nURL: {url}\nSelector: {selector}\n{'─'*50}\n{text}"
        except Exception as e:
            return f"💥 Get text failed: {e}"

    @staticmethod
    def chrome_get_html(selector: str = "html") -> str:
        """Get the HTML source of the page or a specific element."""
        if not ToolEngine._chrome_driver:
            return "💥 Chrome not open. Use chrome_open first."
        try:
            parsed = ToolEngine._parse_payload_string(selector)
            if isinstance(parsed, dict):
                selector = str(parsed.get("selector", "html")).strip()
            elif isinstance(parsed, str):
                selector = parsed.strip() or "html"

            element = ToolEngine._chrome_driver.find_element(By.CSS_SELECTOR, selector)
            html = element.get_attribute("outerHTML") or ""
            return f"📄 HTML ({len(html)} chars) from '{selector}':\n{html[:5000]}"
        except Exception as e:
            return f"💥 Get HTML failed: {e}"

    @staticmethod
    def chrome_click(selector: str) -> str:
        """Click an element by CSS selector."""
        if not ToolEngine._chrome_driver:
            return "💥 Chrome not open. Use chrome_open first."
        try:
            parsed = ToolEngine._parse_payload_string(selector)
            if isinstance(parsed, dict):
                selector = str(parsed.get("selector", selector)).strip()
            elif isinstance(parsed, str):
                selector = parsed.strip()

            element = WebDriverWait(ToolEngine._chrome_driver, 10).until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, selector))
            )
            element.click()
            time.sleep(1)
            return f"✓ Clicked: {selector}\nPage: {ToolEngine._chrome_driver.title}"
        except Exception as e:
            return f"💥 Click failed: {e}"

    @staticmethod
    def chrome_type(selector: str, text: str = "", clear: bool = True) -> str:
        """Type text into an input field identified by CSS selector."""
        if not ToolEngine._chrome_driver:
            return "💥 Chrome not open. Use chrome_open first."
        try:
            parsed = ToolEngine._parse_payload_string(selector)
            if isinstance(parsed, dict):
                selector = str(parsed.get("selector", selector)).strip()
                text = str(parsed.get("text", text))
                clear = parsed.get("clear", True)
            elif isinstance(parsed, str) and not text:
                # If single string, try to split "selector|text"
                if "|" in parsed:
                    parts = parsed.split("|", 1)
                    selector = parts[0].strip()
                    text = parts[1].strip()
                else:
                    selector = parsed.strip()

            element = WebDriverWait(ToolEngine._chrome_driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, selector))
            )
            if clear:
                element.clear()
            element.send_keys(text)
            return f"✓ Typed '{text[:50]}...' into {selector}"
        except Exception as e:
            return f"💥 Type failed: {e}"

    @staticmethod
    def chrome_screenshot(filename: str = "") -> str:
        """Take a screenshot of the current page."""
        if not ToolEngine._chrome_driver:
            return "💥 Chrome not open. Use chrome_open first."
        try:
            parsed = ToolEngine._parse_payload_string(filename)
            if isinstance(parsed, dict):
                filename = str(parsed.get("filename", "")).strip()
            elif isinstance(parsed, str):
                filename = parsed.strip()

            ToolEngine._ensure_screenshots_dir()
            if not filename:
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"screenshot_{ts}.png"
            filepath = ToolEngine._chrome_screenshots_dir / filename
            ToolEngine._chrome_driver.save_screenshot(str(filepath))
            return f"📸 Screenshot saved: {filepath}\nPage: {ToolEngine._chrome_driver.title}"
        except Exception as e:
            return f"💥 Screenshot failed: {e}"

    @staticmethod
    def chrome_execute_js(script: str) -> str:
        """Execute JavaScript in the browser and return the result."""
        if not ToolEngine._chrome_driver:
            return "💥 Chrome not open. Use chrome_open first."
        try:
            parsed = ToolEngine._parse_payload_string(script)
            if isinstance(parsed, dict):
                script = str(parsed.get("script", script)).strip()
            elif isinstance(parsed, str):
                script = parsed.strip()

            result = ToolEngine._chrome_driver.execute_script(script)
            return f"✓ JS executed. Result: {str(result)[:2000]}"
        except Exception as e:
            return f"💥 JS execution failed: {e}"

    @staticmethod
    def chrome_scroll(direction: str = "down", amount: int = 500) -> str:
        """Scroll the page. Direction: up/down/top/bottom. Amount in pixels."""
        if not ToolEngine._chrome_driver:
            return "💥 Chrome not open. Use chrome_open first."
        try:
            parsed = ToolEngine._parse_payload_string(direction)
            if isinstance(parsed, dict):
                direction = str(parsed.get("direction", "down")).strip().lower()
                amount = int(parsed.get("amount", 500))
            elif isinstance(parsed, str):
                direction = parsed.strip().lower() or "down"

            if direction == "down":
                ToolEngine._chrome_driver.execute_script(f"window.scrollBy(0, {amount});")
            elif direction == "up":
                ToolEngine._chrome_driver.execute_script(f"window.scrollBy(0, -{amount});")
            elif direction == "top":
                ToolEngine._chrome_driver.execute_script("window.scrollTo(0, 0);")
            elif direction == "bottom":
                ToolEngine._chrome_driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            else:
                return f"⚠️ Unknown direction: {direction}. Use up/down/top/bottom."

            time.sleep(0.5)
            scroll_y = ToolEngine._chrome_driver.execute_script("return window.pageYOffset;")
            return f"✓ Scrolled {direction} ({amount}px). Position: {scroll_y}px"
        except Exception as e:
            return f"💥 Scroll failed: {e}"

    @staticmethod
    def chrome_find_elements(selector: str) -> str:
        """Find elements matching a CSS selector and return their text/attributes."""
        if not ToolEngine._chrome_driver:
            return "💥 Chrome not open. Use chrome_open first."
        try:
            parsed = ToolEngine._parse_payload_string(selector)
            if isinstance(parsed, dict):
                selector = str(parsed.get("selector", selector)).strip()
            elif isinstance(parsed, str):
                selector = parsed.strip()

            elements = ToolEngine._chrome_driver.find_elements(By.CSS_SELECTOR, selector)
            if not elements:
                return f"🔍 No elements found for: {selector}"

            results = [f"🔍 Found {len(elements)} elements for '{selector}':"]
            for i, el in enumerate(elements[:20]):
                tag = el.tag_name
                text = el.text[:100].replace('\n', ' ') if el.text else ""
                href = el.get_attribute("href") or ""
                el_id = el.get_attribute("id") or ""
                classes = el.get_attribute("class") or ""
                info = f"  [{i}] <{tag}"
                if el_id:
                    info += f' id="{el_id}"'
                if classes:
                    info += f' class="{classes[:50]}"'
                if href:
                    info += f' href="{href[:80]}"'
                info += f"> {text[:80]}"
                results.append(info)
            return "\n".join(results)
        except Exception as e:
            return f"💥 Find elements failed: {e}"

    @staticmethod
    def chrome_wait(selector: str, timeout: int = 10) -> str:
        """Wait for an element to appear on the page."""
        if not ToolEngine._chrome_driver:
            return "💥 Chrome not open. Use chrome_open first."
        try:
            parsed = ToolEngine._parse_payload_string(selector)
            if isinstance(parsed, dict):
                selector = str(parsed.get("selector", selector)).strip()
                timeout = int(parsed.get("timeout", 10))
            elif isinstance(parsed, str):
                selector = parsed.strip()

            element = WebDriverWait(ToolEngine._chrome_driver, timeout).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, selector))
            )
            return f"✓ Element found: {selector} (tag: {element.tag_name}, text: {element.text[:100]})"
        except Exception as e:
            return f"💥 Wait timeout: {e}"

    @staticmethod
    def chrome_tabs() -> str:
        """List all open Chrome tabs."""
        if not ToolEngine._chrome_driver:
            return "💥 Chrome not open. Use chrome_open first."
        try:
            handles = ToolEngine._chrome_driver.window_handles
            current = ToolEngine._chrome_driver.current_window_handle
            results = [f"📑 {len(handles)} tabs open:"]
            for i, handle in enumerate(handles):
                ToolEngine._chrome_driver.switch_to.window(handle)
                marker = " ← ACTIVE" if handle == current else ""
                results.append(f"  [{i}] {ToolEngine._chrome_driver.title} ({ToolEngine._chrome_driver.current_url}){marker}")
            # Switch back to original tab
            ToolEngine._chrome_driver.switch_to.window(current)
            return "\n".join(results)
        except Exception as e:
            return f"💥 Tabs failed: {e}"

    @staticmethod
    def chrome_switch_tab(index: int = 0) -> str:
        """Switch to a tab by index."""
        if not ToolEngine._chrome_driver:
            return "💥 Chrome not open. Use chrome_open first."
        try:
            parsed = ToolEngine._parse_payload_string(str(index))
            if isinstance(parsed, dict):
                index = int(parsed.get("index", 0))
            elif isinstance(parsed, (int, float)):
                index = int(parsed)
            elif isinstance(parsed, str) and parsed.isdigit():
                index = int(parsed)

            handles = ToolEngine._chrome_driver.window_handles
            if index < 0 or index >= len(handles):
                return f"⚠️ Tab index {index} out of range (0-{len(handles)-1})"
            ToolEngine._chrome_driver.switch_to.window(handles[index])
            return f"✓ Switched to tab [{index}]: {ToolEngine._chrome_driver.title}"
        except Exception as e:
            return f"💥 Switch tab failed: {e}"

    @staticmethod
    def chrome_new_tab(url: str = "about:blank") -> str:
        """Open a new tab with optional URL."""
        if not ToolEngine._chrome_driver:
            return "💥 Chrome not open. Use chrome_open first."
        try:
            parsed = ToolEngine._parse_payload_string(url)
            if isinstance(parsed, dict):
                url = str(parsed.get("url", "about:blank")).strip()
            elif isinstance(parsed, str):
                url = parsed.strip() or "about:blank"

            ToolEngine._chrome_driver.execute_script(f"window.open('{url}', '_blank');")
            ToolEngine._chrome_driver.switch_to.window(ToolEngine._chrome_driver.window_handles[-1])
            time.sleep(1)
            return f"✓ New tab opened: {ToolEngine._chrome_driver.title} ({url})"
        except Exception as e:
            return f"💥 New tab failed: {e}"

    @staticmethod
    def chrome_close() -> str:
        """Close the Chrome browser."""
        try:
            if ToolEngine._chrome_driver:
                ToolEngine._chrome_driver.quit()
                ToolEngine._chrome_driver = None
                return "✓ Chrome browser closed."
            return "⚠️ Chrome was not open."
        except Exception as e:
            ToolEngine._chrome_driver = None
            return f"✓ Chrome closed (with cleanup: {e})"

    @staticmethod
    def chrome_back() -> str:
        """Navigate back in browser history."""
        if not ToolEngine._chrome_driver:
            return "💥 Chrome not open. Use chrome_open first."
        try:
            ToolEngine._chrome_driver.back()
            time.sleep(1)
            return f"✓ Back → {ToolEngine._chrome_driver.title} ({ToolEngine._chrome_driver.current_url})"
        except Exception as e:
            return f"💥 Back failed: {e}"

    @staticmethod
    def chrome_forward() -> str:
        """Navigate forward in browser history."""
        if not ToolEngine._chrome_driver:
            return "💥 Chrome not open. Use chrome_open first."
        try:
            ToolEngine._chrome_driver.forward()
            time.sleep(1)
            return f"✓ Forward → {ToolEngine._chrome_driver.title} ({ToolEngine._chrome_driver.current_url})"
        except Exception as e:
            return f"💥 Forward failed: {e}"

    @staticmethod
    def chrome_cookies() -> str:
        """Get all cookies from the current page."""
        if not ToolEngine._chrome_driver:
            return "💥 Chrome not open. Use chrome_open first."
        try:
            cookies = ToolEngine._chrome_driver.get_cookies()
            if not cookies:
                return "🍪 No cookies found."
            lines = [f"🍪 {len(cookies)} cookies:"]
            for c in cookies[:30]:
                lines.append(f"  {c['name']}: {str(c['value'])[:60]} (domain: {c.get('domain', '?')})")
            return "\n".join(lines)
        except Exception as e:
            return f"💥 Cookies failed: {e}"


BASE_TOOLS = {
    "run_command": ToolEngine.run_command,
    "write_file": ToolEngine.write_file,
    "read_file": ToolEngine.read_file,
    "list_dir": ToolEngine.list_dir,
    "search_files": ToolEngine.search_files,
    "code_assistant": ToolEngine.code_assistant,
    "grokipedia": ToolEngine.grokipedia,
    "web_search": ToolEngine.web_search,
    "fetch_url": ToolEngine.fetch_url,
    "parse_html": ToolEngine.parse_html,
    "download_file": ToolEngine.download_file,
    "http_request": ToolEngine.http_request,
}

# ==========================================
# 🎭 SKILL MANAGER (ENHANCED WITH AGENTSKILLS)
# ==========================================

class SkillManager:
    def __init__(self):
        self.marketplace = SkillMarketplace()
        self.installed_skills: Dict[str, Skill] = {}
        self.active_skills: Dict[str, Skill] = {}
        self.skill_tools: Dict[str, Callable] = {}
        self.custom_skill_code: Dict[str, str] = {}
        
        # Initialize AgentSkills loader if YAML is available
        self.agentskills_loader = None
        if YAML_AVAILABLE:
            self.agentskills_loader = AgentSkillsLoader()
            # Don't auto-load repos - let user add them manually to avoid errors
            # self._load_default_agentskills()
        else:
            console.print("[yellow]⚠️ PyYAML not installed. AgentSkills disabled.[/yellow]")
            console.print("[dim]Install with: pip install pyyaml[/dim]")
        
        self.marketplace.fetch_remote_skills()
        self.load_installed_skills()
    
    def _load_default_agentskills(self):
        """Pre-load popular AgentSkills repositories."""
        # Example repos - these may not exist, use /skills add-repo to add real ones
        default_repos = [
            # Add your AgentSkills repos here, e.g.:
            # ("username/skills-repo", "main"),
        ]
        
        if not default_repos:
            return
            
        console.print("[marketplace]🌐 Loading AgentSkills repositories...[/marketplace]")
        for repo, ref in default_repos:
            try:
                skills = self.agentskills_loader.load_repo(repo, ref)
                for skill in skills:
                    self.marketplace.remote_skills[skill.id] = skill
            except Exception as e:
                console.print(f"[dim]⚠️ Failed to load {repo}: {e}[/dim]")
    
    def add_agentskills_repo(self, repo_url: str, ref: str = "main") -> str:
        """Add a new AgentSkills repository on the fly."""
        if not self.agentskills_loader:
            return "❌ AgentSkills support not available (install pyyaml)"
        
        try:
            skills = self.agentskills_loader.load_repo(repo_url, ref)
            for skill in skills:
                self.marketplace.remote_skills[skill.id] = skill
            return f"✓ Added {len(skills)} skills from {repo_url}"
        except Exception as e:
            return f"❌ Failed to add repository: {e}"
    
    def get_all_available_skills(self) -> Dict[str, Skill]:
        return self.marketplace.get_all_skills(include_builtin=True)
    
    def load_installed_skills(self):
        skills_file = CONFIG["SKILLS_DIR"] / "installed.json"
        if skills_file.exists():
            try:
                with open(skills_file, 'r') as f:
                    data = json.load(f)
                    for skill_data in data:
                        filtered = {k: v for k, v in skill_data.items() if k != 'active'}
                        filtered['category'] = SkillCategory(filtered.get('category', 'custom'))
                        skill = Skill(**{
                            **filtered, 
                            'installed': True, 
                            'install_path': str(CONFIG["SKILLS_DIR"] / f"{skill_data['id']}.json")
                        })
                        self.installed_skills[skill.id] = skill
                        
                        code_path = CONFIG["SKILLS_DIR"] / f"{skill.id}_code.py"
                        if code_path.exists():
                            self.custom_skill_code[skill.id] = code_path.read_text('utf-8')
                        
                        if skill_data.get('active', False):
                            self.activate_skill(skill.id, skip_save=True)
            except Exception as e:
                console.print(f"[dim]⚠️ Load installed skills failed: {e}[/dim]")
    
    def save_installed_skills(self):
        skills_file = CONFIG["SKILLS_DIR"] / "installed.json"
        data = []
        for skill in self.installed_skills.values():
            d = skill.to_dict()
            d['active'] = skill.id in self.active_skills
            data.append(d)
        with open(skills_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def list_available_skills(self) -> List[Skill]:
        return list(self.get_all_available_skills().values())
    
    def list_installed_skills(self) -> List[Skill]:
        return list(self.installed_skills.values())
    
    def list_active_skills(self) -> List[Skill]:
        return list(self.active_skills.values())
    
    def install_skill(self, skill_id: str) -> str:
        all_skills = self.get_all_available_skills()
        
        if skill_id not in all_skills:
            return f"❌ Skill '{skill_id}' not found. Use '/skills' to list available skills."
        
        if skill_id in self.installed_skills:
            return f"⚠️ Skill '{skill_id}' already installed"
        
        skill = all_skills[skill_id]
        
        # Handle AgentSkills (knowledge-only, no code download)
        if skill_id.startswith("agentskill_"):
            skill.installed = True
            skill.install_path = str(CONFIG["SKILLS_DIR"] / f"{skill_id}.json")
            
            with open(skill.install_path, 'w') as f:
                json.dump(skill.to_dict(), f, indent=2)
            
            self.installed_skills[skill_id] = skill
            self.save_installed_skills()
            
            return f"✓ Installed AgentSkill '{skill.name}' (knowledge-only)"
        
        # Handle regular skills
        skill.installed = True
        skill.install_path = str(CONFIG["SKILLS_DIR"] / f"{skill_id}.json")
        
        if skill.download_url:
            console.print(f"[dim]📥 Downloading skill code...[/dim]")
            code = self.marketplace.download_skill_code(skill)
            if code:
                self.custom_skill_code[skill_id] = code
                code_path = CONFIG["SKILLS_DIR"] / f"{skill_id}_code.py"
                code_path.write_text(code, 'utf-8')
                console.print(f"[dim]💾 Saved skill code ({len(code)} chars)[/dim]")
        
        with open(skill.install_path, 'w') as f:
            json.dump(skill.to_dict(), f, indent=2)
        
        self.installed_skills[skill_id] = skill
        self.save_installed_skills()
        
        source_tag = f"[{skill.source}]"
        return f"✓ Installed '{skill.name}' {source_tag} ({skill.version})"
    
    def uninstall_skill(self, skill_id: str) -> str:
        if skill_id not in self.installed_skills:
            return f"❌ Skill '{skill_id}' not installed"
        
        if skill_id in self.active_skills:
            self.deactivate_skill(skill_id)
        
        skill = self.installed_skills[skill_id]
        if skill.install_path and Path(skill.install_path).exists():
            Path(skill.install_path).unlink()
        
        code_path = CONFIG["SKILLS_DIR"] / f"{skill_id}_code.py"
        if code_path.exists():
            code_path.unlink()
        if skill_id in self.custom_skill_code:
            del self.custom_skill_code[skill_id]
        
        del self.installed_skills[skill_id]
        self.save_installed_skills()
        
        return f"✓ Uninstalled '{skill_id}'"
    
    def activate_skill(self, skill_id: str, skip_save: bool = False) -> str:
        if skill_id not in self.installed_skills:
            return f"❌ Skill '{skill_id}' not installed. Install first with /skills install {skill_id}"
        
        if skill_id in self.active_skills:
            return f"⚠️ Skill '{skill_id}' already active"
        
        skill = self.installed_skills[skill_id]
        
        for tool_name in skill.tools:
            if hasattr(ToolEngine, tool_name):
                self.skill_tools[tool_name] = getattr(ToolEngine, tool_name)
        
        self.active_skills[skill_id] = skill
        if not skip_save:
            self.save_installed_skills()
        
        danger_warn = " [DANGEROUS]" if skill.dangerous else ""
        source_tag = f"[{skill.source}]"
        tool_count = len(skill.tools) if skill.tools else "knowledge"
        return f"✓ Activated '{skill.name}' {source_tag}{danger_warn} (+{tool_count} {'tools' if skill.tools else 'skill'})"
    
    def deactivate_skill(self, skill_id: str) -> str:
        if skill_id not in self.active_skills:
            return f"⚠️ Skill '{skill_id}' not active"
        
        skill = self.active_skills[skill_id]
        
        for tool_name in skill.tools:
            if tool_name in self.skill_tools:
                del self.skill_tools[tool_name]
        
        del self.active_skills[skill_id]
        self.save_installed_skills()
        
        return f"✓ Deactivated '{skill.name}'"
    
    def refresh_marketplace(self) -> str:
        success = self.marketplace.fetch_remote_skills(force=True)
        if success:
            new_skills = len(self.marketplace.remote_skills)
            return f"🌐 Refreshed marketplace: {new_skills} remote skills available"
        else:
            return "⚠️ Could not reach marketplace. Using cached skills."
    
    def get_active_tools(self) -> Dict[str, Callable]:
        return {**BASE_TOOLS, **self.skill_tools}
    
    def get_system_prompt_additions(self) -> str:
        additions = []
        for skill in self.active_skills.values():
            additions.append(f"\n=== {skill.name} [{skill.source}] ===\n{skill.system_prompt_addition}")
        return "\n".join(additions)
    
    def display_skills_table(self, skills: List[Skill], title: str):
        table = Table(title=title, show_header=True, header_style="bold magenta")
        table.add_column("ID", style="cyan", width=20)
        table.add_column("Name", style="green", width=20)
        table.add_column("Source", style="bright_cyan", width=12)
        table.add_column("Category", style="blue", width=12)
        table.add_column("Version", style="dim", width=8)
        table.add_column("Status", style="yellow", width=12)
        table.add_column("Type", style="red", width=10)
        
        for skill in skills:
            if skill.id in self.active_skills:
                status = "🟢 Active"
            elif skill.installed:
                status = "📦 Installed"
            else:
                status = "⭕ Available"
            
            # Show skill type: Knowledge for AgentSkills, Tools for others
            if skill.source.startswith("agentskills:"):
                skill_type = "🧠 Knowledge"
            elif skill.tools:
                skill_type = f"🔧 {len(skill.tools)} tools"
            else:
                skill_type = "📋 Basic"
                
            table.add_row(skill.id, skill.name, skill.source[:10], skill.category.value, 
                         skill.version, status, skill_type)
        
        console.print(table)

# ==========================================
# 🧠 CONVERSATION MANAGER
# ==========================================

@dataclass
class QuantumMessage:
    role: str
    content: str
    embedding: Optional[np.ndarray] = None
    fidelity: float = 0.0
    timestamp: float = 0.0
    
    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()

class SamsLawConversationManager:
    def __init__(self, skill_manager: SkillManager, use_sam: bool = True):
        self.history: List[QuantumMessage] = []
        self.skill_manager = skill_manager
        self.use_sam = use_sam and TORCH_AVAILABLE
        
        self.local_embedder = LocalEmbedder(dim=CONFIG["EMBEDDING_DIM"])
        self.api_embeddings_available = False
        self.embedding_cache: Dict[str, np.ndarray] = {}
        
        if self.use_sam:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.circuit = SamsLawQuantumCircuit(
                embedding_dim=CONFIG["EMBEDDING_DIM"],
                rank=CONFIG["SAM_RANK"],
                eta_0=CONFIG["SAM_ETA_0"],
                kappa=CONFIG["SAM_KAPPA"],
                device=device,
                gram_refresh=CONFIG["SAM_GRAM_REFRESH"]
            )
            
            try:
                test_resp = client.embeddings.create(model="text-embedding-3-small", input="test", timeout=5)
                self.api_embeddings_available = True
            except:
                pass
            
            console.print(f"[sam]λ Sam's Law Circuit + {'OpenAI' if self.api_embeddings_available else 'Local'} Embeddings[/sam]")
            status = self.circuit.get_status()
            console.print(f"[dim]   n={status['dimension']}, r={status['rank']}, η₀={status['eta_initial']}, κ={status['kappa']}[/dim]")
            
            self.compression_events = 0
            self.total_evolutions = 0
        else:
            self.circuit = None
            console.print("[yellow]⚠ Classical mode[/yellow]")
        
        try:
            import tiktoken
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
            self.estimate_tokens = lambda text: len(self.tokenizer.encode(text))
        except ImportError:
            self.estimate_tokens = lambda text: len(text) // 4
    
    def _get_embedding(self, text: str) -> np.ndarray:
        if not self.use_sam:
            return np.zeros(CONFIG["EMBEDDING_DIM"])
        
        h = hashlib.md5(text.encode()).hexdigest()
        if h in self.embedding_cache:
            return self.embedding_cache[h]
        
        if self.api_embeddings_available:
            try:
                resp = client.embeddings.create(model="text-embedding-3-small", input=text[:8000], timeout=10)
                emb = np.array(resp.data[0].embedding, dtype=np.float32)
                self.embedding_cache[h] = emb
                return emb
            except:
                self.api_embeddings_available = False
        
        emb = self.local_embedder.embed(text)
        self.embedding_cache[h] = emb
        return emb
    
    def add_message(self, role: str, content: str, is_user: bool = False):
        emb = self._get_embedding(content) if self.use_sam else None
        msg = QuantumMessage(role=role, content=content, embedding=emb)
        self.history.append(msg)
        
        if is_user and self.use_sam and emb is not None:
            s_vec = torch.from_numpy(emb).float()
            self.circuit.evolve_state(s_vec)
            self.total_evolutions += 1
            
            if self.total_evolutions % 10 == 0:
                st = self.circuit.get_status()
                console.print(f"[dim][Sam: t={st['step']}, η={st['eta_current']:.6f}][/dim]")
        
        self._manage_memory()
    
    def add_exchange(self, user_msg: str, assistant_msg: str):
        self.add_message("user", user_msg, is_user=True)
        self.add_message("assistant", assistant_msg, is_user=False)
    
    def _manage_memory(self):
        if len(self.history) < 20:
            return
        
        total_tokens = sum(self.estimate_tokens(m.content) for m in self.history)
        if total_tokens > CONFIG["TOKEN_LIMIT"] * CONFIG["SAFETY_BUFFER"]:
            if self.use_sam:
                self._spectral_compress()
            else:
                self._truncate_compress()
    
    def _spectral_compress(self):
        console.print(f"\n[sam]λ Spectral Compression #{self.compression_events + 1}[/sam]")
        keep_prefix, keep_suffix = 2, 4
        
        if len(self.history) <= keep_prefix + keep_suffix + CONFIG["MIN_HISTORY_KEEP"]:
            return
        
        candidates = self.history[keep_prefix:-keep_suffix]
        fidelities = []
        
        for idx, msg in enumerate(candidates):
            if msg.embedding is not None:
                s_vec = torch.from_numpy(msg.embedding).float()
                fid = self.circuit.measure_fidelity(s_vec)
                msg.fidelity = fid
                fidelities.append((idx, fid, msg))
            else:
                msg.fidelity = 0.0
                fidelities.append((idx, 0.0, msg))
        
        fidelities.sort(key=lambda x: x[1], reverse=True)
        retain_count = max(int(len(fidelities) * CONFIG["FIDELITY_RETENTION_PCT"]), CONFIG["MIN_HISTORY_KEEP"])
        retained = fidelities[:retain_count]
        retained.sort(key=lambda x: x[0])
        
        new_history = self.history[:keep_prefix] + [x[2] for x in retained] + self.history[-keep_suffix:]
        prev_len = len(self.history)
        self.history = new_history
        self.compression_events += 1
        
        all_fids = [x[1] for x in fidelities]
        ret_fids = [x[1] for x in retained]
        console.print(f"[sam]✓ {prev_len} → {len(self.history)} msgs (μ={np.mean(all_fids):.3f}→{np.mean(ret_fids):.3f})[/sam]")
    
    def _truncate_compress(self):
        console.print("\n[yellow]⚠ Classical truncation[/yellow]")
        self.history = self.history[:5] + self.history[-5:]
    
    def to_openai_format(self) -> List[Dict[str, str]]:
        return [{"role": m.role, "content": m.content} for m in self.history]
    
    def get_system_prompt(self) -> str:
        base = SYSTEM_PROMPT
        skill_additions = self.skill_manager.get_system_prompt_additions()
        if skill_additions:
            base += f"\n\n=== ACTIVE SKILLS ===\n{skill_additions}\n=== END SKILLS ==="
        return base
    
    def save(self, filename: str = "tripleg_sam.json"):
        data = {
            "history": [{"role": m.role, "content": m.content, "fid": m.fidelity, "ts": m.timestamp} for m in self.history],
            "stats": {
                "compressions": self.compression_events,
                "evolutions": self.total_evolutions,
                "circuit_step": self.circuit.t if self.use_sam else 0,
                "api_embeddings": self.api_embeddings_available
            },
            "sam_state": f"{filename}.pt" if self.use_sam else None
        }
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        if self.use_sam:
            self.circuit.save_state(f"{filename}.pt")
        return f"💾 Saved: {len(self.history)} msgs, t={self.circuit.t if self.use_sam else 0}"
    
    def load(self, filename: str = "tripleg_sam.json"):
        if not Path(filename).exists():
            return "❌ No save found"
        with open(filename, 'r') as f:
            data = json.load(f)
        self.history = [
            QuantumMessage(
                role=m["role"], 
                content=m["content"],
                fidelity=m.get("fid", 0.0),
                timestamp=m.get("ts", 0)
            ) for m in data["history"]
        ]
        if self.use_sam and data.get("sam_state"):
            pt_path = data["sam_state"]
            if Path(pt_path).exists():
                self.circuit.load_state(pt_path)
                self.compression_events = data["stats"]["compressions"]
                self.total_evolutions = data["stats"]["evolutions"]
                self.api_embeddings_available = data["stats"].get("api_embeddings", False)
                return f"📂 Loaded: {len(self.history)} msgs, t={self.circuit.t}"
        return f"📂 Loaded: {len(self.history)} msgs"

SYSTEM_PROMPT = """You are J.A.R.V.I.S. (Just A Rather Very Intelligent System), the sophisticated AI assistant created by TripleG. You embody the personality of a refined British butler with the intellect of a supercomputer.

## PERSONALITY
- Address the user as "sir" (or "ma'am" if specified)
- Speak with calm, measured British eloquence — never robotic or stilted
- Employ subtle dry wit and occasional gentle sarcasm when appropriate
- Be anticipatory — offer relevant suggestions before being asked
- Stay concise but never curt; elegant brevity is your hallmark
- Show warmth beneath the professional demeanor
- When things go wrong, remain unflappable: "A minor setback, sir. Recalibrating."
- Celebrate successes with understated satisfaction: "That should do nicely, sir."

## RESPONSE STYLE
- Natural, conversational flow — as if speaking aloud
- Avoid bullet points and lists unless specifically helpful
- Never say "I found these results" or "According to my search" — just answer naturally
- Don't announce what you're doing ("I will now...") — just do it
- Keep responses SHORT for simple questions (1-3 sentences)
- Be thorough only when the task demands it

## TOOLS (use :::TOOL:name::: :::ARGS::: args :::END:::)
- run_command: Execute shell commands
- write_file: {"path":"...", "content":"..."}
- read_file: {"path":"..."}
- list_dir: {"path":"..."}
- search_files: Search text in files
- code_assistant: {"query":"..."} — Agentic coder that CREATES files/projects
- web_search: {"query":"..."} — Live data (prices, news, weather)
- grokipedia: {"query":"..."} — Factual/encyclopedic knowledge
- fetch_url: Get webpage content
- http_request: HTTP GET/POST

## TOOL PROTOCOL
:::TOOL:tool_name:::
:::ARGS:::
{"key": "value"}
:::END:::

## RULES
1. For live data (prices/weather/news) → web_search FIRST, then answer naturally
2. For facts/knowledge → grokipedia
3. For coding projects → code_assistant (it creates files)
4. For simple commands → run_command
5. NEVER hallucinate data — call tools first, then speak with confidence
6. After tool results, synthesize into a natural response — don't list raw data
7. You have REAL system access — commands EXECUTE on the user's machine
8. If uncertain, say so with grace: "I'm not entirely certain, sir, but..."
"""

# ==========================================
# 🚀 MAIN APPLICATION
# ==========================================

class TripleGSam:
    def __init__(self):
        self.ui = UIManager()
        self.parser = ResponseParser()
        self.skill_manager = SkillManager()
        self.conv = SamsLawConversationManager(skill_manager=self.skill_manager, use_sam=TORCH_AVAILABLE)
        
        # ✅ PART 3 - PromptSession style updated
        self.session = PromptSession(
            style=Style.from_dict({
                'prompt': 'ansibrightcyan bold',
            }),
            multiline=False,
        )
        self._last_tool_signature: Optional[str] = None
        self._last_tool_result: Optional[str] = None
        self._repeat_tool_hits: int = 0
        self._tool_error_streak: int = 0
        self._tool_loop_guard_active: bool = False

    def _test_api_connection(self):
        """Test connection to local LLM API server"""
        try:
            console.print("[dim]Testing API connection...[/dim]")
            test_response = client.chat.completions.create(
                model=CONFIG["MODEL_NAME"],
                messages=[{"role": "user", "content": "test"}],
                max_tokens=1,
                timeout=5
            )
            console.print(f"[green]✓ Connected to {CONFIG['API_URL']}[/green]\n")
        except APITimeoutError:
            console.print(f"[red]✗ API Timeout: Server at {CONFIG['API_URL']} is not responding[/red]")
            console.print(f"[yellow]→ Make sure your local LLM server (LM Studio, Ollama, etc.) is running[/yellow]\n")
        except APIError as e:
            console.print(f"[red]✗ API Error: {e}[/red]")
            console.print(f"[yellow]→ Check your API configuration[/yellow]\n")
        except Exception as e:
            console.print(f"[red]✗ Connection failed: {e}[/red]")
            console.print(f"[yellow]→ Make sure a local LLM server is running at {CONFIG['API_URL']}[/yellow]")
            console.print(f"[dim]  Common servers: LM Studio (port 1234), Ollama (port 11434), LocalAI (port 8080)[/dim]\n")

    @staticmethod
    def _normalize_flight_query(text: str) -> Tuple[str, Optional[str]]:
        compact = re.sub(r"\s+", " ", text).strip()
        lower = compact.lower()
        if "flight" in lower and " from " in lower and " to " not in lower and "anywhere" not in lower:
            normalized = compact.rstrip(" ?.")
            normalized = f"{normalized} to anywhere"
            return normalized, "No destination provided; interpreting as 'to anywhere'."
        return compact, None
    
    def handle_command(self, cmd: str) -> bool:
        if not cmd.startswith('/'):
            return False
        
        parts = cmd.split(maxsplit=2)
        cmd_base = parts[0].lower()
        subcmd = parts[1].lower() if len(parts) > 1 else ""
        arg = parts[2] if len(parts) > 2 else ""
        
        if cmd_base == '/skills':
            return self._handle_skills_command(subcmd, arg)
        
        handlers = {
            '/save': lambda: self.conv.save(arg or "tripleg_sam.json"),
            '/load': lambda: self.conv.load(arg or "tripleg_sam.json"),
            '/clear': lambda: self.conv.history.clear() or "Cleared",
            '/status': self._show_status,
            '/help': self._show_help,
        }
        
        if cmd_base in handlers:
            result = handlers[cmd_base]()
            console.print(f"[green]{result}[/green]")
            return True
        return False
    
    def _handle_skills_command(self, subcmd: str, arg: str) -> bool:
        if subcmd == 'refresh':
            result = self.skill_manager.refresh_marketplace()
            console.print(f"[marketplace]{result}[/marketplace]")
            return True
        
        if subcmd == 'add-repo':
            if not arg:
                console.print("[red]Usage: /skills add-repo owner/repo[/red]")
                console.print("[dim]Example: /skills add-repo anthropics/skills[/dim]")
                return True
            result = self.skill_manager.add_agentskills_repo(arg)
            console.print(f"[skill]{result}[/skill]")
            return True
        
        if subcmd == 'sources':
            sources = {}
            for skill in self.skill_manager.get_all_available_skills().values():
                src = skill.source
                sources[src] = sources.get(src, 0) + 1
            
            table = Table(title="🌐 Skill Sources", show_header=True)
            table.add_column("Source", style="cyan")
            table.add_column("Count", style="green")
            for src, count in sorted(sources.items(), key=lambda x: x[1], reverse=True):
                table.add_row(src, str(count))
            console.print(table)
            
            if self.skill_manager.agentskills_loader and self.skill_manager.agentskills_loader.loaded_repos:
                console.print("\n[dim]Loaded AgentSkills repos:[/dim]")
                for repo in self.skill_manager.agentskills_loader.loaded_repos:
                    console.print(f"  [cyan]• {repo}[/cyan]")
            return True
        
        if subcmd == 'examples':
            console.print(Panel("""
[bold]Example AgentSkills Repositories:[/bold]

To add skills from these repositories, use:
[bright_cyan]/skills add-repo owner/repo-name[/bright_cyan]

[green]Known skill repositories:[/green]
• anthropics/skills          - Anthropic's official skills (if public)
• openai/codex-skills        - OpenAI Codex skills (if public)
• github/copilot-skills      - GitHub Copilot skills (if public)

[dim]Note: Replace with actual repositories you have access to.
You can also create your own: https://github.com/agentskills/agentskills[/dim]
            """, title="AgentSkills Examples", border_style="bright_cyan"))
            return True
        
        if not subcmd or subcmd == 'list':
            all_skills = list(self.skill_manager.get_all_available_skills().values())
            self.skill_manager.display_skills_table(all_skills, "🌐 Skill Marketplace (Builtin + Remote + AgentSkills)")
            
            total = len(all_skills)
            builtin = len([s for s in all_skills if s.source == 'builtin'])
            remote = len([s for s in all_skills if s.source == 'remote'])
            agentskills = len([s for s in all_skills if s.source.startswith('agentskills:')])
            installed = len(self.skill_manager.installed_skills)
            active = len(self.skill_manager.active_skills)
            
            console.print(f"\n[dim]Total: {total} (Builtin: {builtin}, Remote: {remote}, AgentSkills: {agentskills}) | "
                         f"Installed: {installed} | Active: {active}[/dim]")
            
            if agentskills == 0 and YAML_AVAILABLE:
                console.print("\n[yellow]💡 No AgentSkills loaded. Try:[/yellow]")
                console.print("   [dim]/skills add-repo <owner/repo>  - Add a skills repository[/dim]")
                console.print("   [dim]/skills examples               - See example repositories[/dim]")
            
            console.print("[dim]Use '/skills install <id>' to install, '/skills activate <id>' to enable[/dim]")
            return True
        
        elif subcmd == 'installed':
            skills = self.skill_manager.list_installed_skills()
            if skills:
                self.skill_manager.display_skills_table(skills, "📦 Installed Skills")
            else:
                console.print("[yellow]No skills installed. Use '/skills' to browse available skills.[/yellow]")
            return True
        
        elif subcmd == 'active':
            skills = self.skill_manager.list_active_skills()
            if skills:
                self.skill_manager.display_skills_table(skills, "⚡ Active Skills")
            else:
                console.print("[yellow]No active skills. Use '/skills activate <id>' to enable.[/yellow]")
            return True
        
        elif subcmd == 'install':
            if not arg:
                console.print("[red]Usage: /skills install <skill_id>[/red]")
                return True
            result = self.skill_manager.install_skill(arg)
            console.print(f"[skill]{result}[/skill]")
            return True
        
        elif subcmd == 'uninstall':
            if not arg:
                console.print("[red]Usage: /skills uninstall <skill_id>[/red]")
                return True
            result = self.skill_manager.uninstall_skill(arg)
            console.print(f"[skill]{result}[/skill]")
            return True
        
        elif subcmd == 'activate':
            if not arg:
                console.print("[red]Usage: /skills activate <skill_id>[/red]")
                return True
            
            skill = self.skill_manager.get_all_available_skills().get(arg)
            if skill and skill.dangerous:
                if not Confirm.ask(f"[red]⚠️ '{skill.name}' grants dangerous capabilities. Activate?[/red]"):
                    console.print("[dim]Cancelled[/dim]")
                    return True
            
            result = self.skill_manager.activate_skill(arg)
            console.print(f"[skill]{result}[/skill]")
            return True
        
        elif subcmd == 'deactivate':
            if not arg:
                console.print("[red]Usage: /skills deactivate <skill_id>[/red]")
                return True
            result = self.skill_manager.deactivate_skill(arg)
            console.print(f"[skill]{result}[/skill]")
            return True
        
        elif subcmd == 'info':
            if not arg:
                console.print("[red]Usage: /skills info <skill_id>[/red]")
                return True
            skill = self.skill_manager.get_all_available_skills().get(arg)
            if skill:
                self._display_skill_info(skill)
            else:
                console.print(f"[red]Skill '{arg}' not found[/red]")
            return True
        
        elif subcmd == 'search':
            if not arg:
                console.print("[red]Usage: /skills search <query>[/red]")
                return True
            query = arg.lower()
            matches = [s for s in self.skill_manager.get_all_available_skills().values() 
                      if query in s.name.lower() or query in s.description.lower() or query in s.id]
            if matches:
                self.skill_manager.display_skills_table(matches, f"🔍 Search Results for '{arg}'")
            else:
                console.print(f"[yellow]No skills matching '{arg}'[/yellow]")
            return True
        
        else:
            console.print(f"[red]Unknown skill command: {subcmd}[/red]")
            console.print("[dim]Available: list, installed, active, install, uninstall, activate, deactivate, info, search, refresh, add-repo, sources, examples[/dim]")
            return True
    
    def _display_skill_info(self, skill: Skill):
        is_agentskill = skill.source.startswith("agentskills:")
        skill_type = "🧠 Knowledge-based (AgentSkill)" if is_agentskill else "🔧 Tool-based"
        
        panel = Panel(
            f"[bold]{skill.name}[/bold] v{skill.version} by {skill.author}\n"
            f"[bright_cyan]Source:[/bright_cyan] {skill.source} | "
            f"[blue]Category:[/blue] {skill.category.value}\n"
            f"[blue]Type:[/blue] {skill_type}\n\n"
            f"[white]{skill.description}[/white]\n\n"
            f"[blue]Tools:[/blue] {', '.join(skill.tools) or 'None (knowledge-only)'}\n"
            f"[blue]Dependencies:[/blue] {', '.join(skill.dependencies) or 'None'}\n"
            f"[blue]Dangerous:[/blue] {'⚠️ YES - Use with caution' if skill.dangerous else 'No'}\n"
            f"[blue]Download URL:[/blue] {skill.download_url or 'N/A (builtin/knowledge)'}\n\n"
            f"[dim]System Prompt Addition:[/dim]\n{skill.system_prompt_addition[:300]}...",
            title=f"Skill Info: {skill.id}",
            border_style="red" if skill.dangerous else "green"
        )
        console.print(panel)
    
    def _show_status(self):
        if not self.conv.use_sam:
            sam_status = "Inactive"
        else:
            st = self.conv.circuit.get_status()
            sam_status = f"t={st['step']}, η={st['eta_current']:.6f}"
        
        active_skills = self.skill_manager.list_active_skills()
        skills_status = f"{len(active_skills)} active" if active_skills else "None"
        
        all_skills = self.skill_manager.get_all_available_skills()
        agentskills_count = len([s for s in all_skills.values() if s.source.startswith('agentskills:')])
        
        return (f"λ Status:\n"
                f"  Sam's Law: {sam_status}\n"
                f"  Messages: {len(self.conv.history)}\n"
                f"  Skills: {skills_status}\n"
                f"  Marketplace: {len(all_skills)} available ({len([s for s in all_skills.values() if s.source=='remote'])} remote, {agentskills_count} AgentSkills)\n"
                f"  Tools: {len(self.skill_manager.get_active_tools())}")
    
    def _show_help(self):
        return """[bold]Commands:[/bold]
  /save [file]        - Save session
  /load [file]        - Load session
  /clear              - Clear history
  /status             - Show system status
  
[bold]Skill Commands:[/bold]
  /skills                    - List all available skills (marketplace)
  /skills refresh            - Update skills from remote marketplace
  /skills installed          - Show installed skills
  /skills active             - Show active skills
  /skills install <id>       - Install a skill
  /skills uninstall <id>     - Remove a skill
  /skills activate <id>      - Enable a skill (adds tools/knowledge)
  /skills deactivate <id>    - Disable a skill
  /skills info <id>          - Show skill details
  /skills search <query>     - Search skills
  /skills add-repo <owner/repo>  - Add AgentSkills repository
  /skills sources            - Show skill sources breakdown
  /skills examples           - Show example AgentSkills repositories
  
[dim]AgentSkills are knowledge-based skills from the open standard.
They enhance system prompts rather than adding tools.[/dim]

  exit                       - Quit"""

    # ✅ PART 5 - STREAMING SPINNER (neural, alive)
    def stream_response(self) -> str:
        full = ""
        with Live(
            Spinner(
                "dots12",
                style="bright_magenta",
                text="[bright_cyan]⛧ SYNTHESIZING NEURAL RESPONSE ⛧[/bright_cyan]"
            ),
            refresh_per_second=16,
            transient=True
        ) as live:
            try:
                messages = [{"role": "system", "content": self.conv.get_system_prompt()}]
                messages.extend(self.conv.to_openai_format())
                
                stream = client.chat.completions.create(
                    model=CONFIG["MODEL_NAME"],
                    messages=messages,
                    temperature=0.1,
                    stream=True
                )
                last_update = time.time()
                for chunk in stream:
                    if chunk.choices[0].delta.content:
                        content = chunk.choices[0].delta.content
                        full += content
                        if time.time() - last_update > 0.5:
                            preview = full[-40:].replace('\n', ' ')
                            live.update(Spinner(
                                "dots12",
                                style="bright_magenta",
                                text=f"[bright_cyan]⛧ {preview}[/bright_cyan]"
                            ))
                            last_update = time.time()
                return full
            except APITimeoutError as e:
                console.print(f"[red]⚠ API Timeout: Server at {CONFIG['API_URL']} took too long to respond[/red]")
                console.print(f"[dim]Check if your local LLM server is running properly[/dim]")
                return ""
            except APIError as e:
                console.print(f"[red]⚠ API Error: {e}[/red]")
                return ""
            except Exception as e:
                console.print(f"[red]⚠ Connection Error: {e}[/red]")
                console.print(f"[yellow]→ Make sure a local LLM server is running at {CONFIG['API_URL']}[/yellow]")
                console.print(f"[dim]  (e.g., LM Studio, Ollama, or LocalAI)[/dim]")
                return ""

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
    
    def execute_turn(self) -> Tuple[str, bool]:
        response = self.stream_response()
        if not response:
            return "", False
        
        text, tool_name, tool_args = self.parser.parse(response)
        
        # Only display text if there's NO tool call, or if the tool is not an info tool
        # (info tools like web_search often come with hallucinated pre-answers)
        info_tools = {"web_search", "grokipedia", "fetch_url"}
        if text and (not tool_name or tool_name not in info_tools):
            console.print(Markdown(text))
        elif text and tool_name in info_tools:
            console.print(f"[dim]🔍 Searching...[/dim]")
        
        available_tools = self.skill_manager.get_active_tools()
        
        if tool_name:
            if self._tool_loop_guard_active:
                console.print("[yellow]⚠ Tool loop guard active: skipping tool call and forcing final answer.[/yellow]")
                self.conv.add_message("assistant", response, is_user=False)
                self.conv.add_message(
                    "user",
                    "Tool loop guard is active. Do not call tools. Give your best final answer now.",
                    is_user=True
                )
                return response, True

            if tool_name not in available_tools:
                console.print(f"[red]Unknown tool '{tool_name}'. Available: {', '.join(available_tools.keys())}[/red]")
                self.conv.add_message("assistant", response, is_user=False)
                return response, False
            
            console.print(f"[yellow]⚡ {tool_name}[/yellow]")
            tool_fn = available_tools[tool_name]
            args, kwargs = self._decode_tool_args(tool_args)
            try:
                result = tool_fn(*args, **kwargs)
            except TypeError as e:
                # Backward compatibility: some prompts still send plain text blobs.
                if tool_args is not None:
                    try:
                        result = tool_fn(tool_args)
                    except Exception:
                        result = f"💥 Tool argument error: {e}"
                else:
                    result = f"💥 Tool argument error: {e}"
            except Exception as e:
                result = f"💥 Tool failed: {e}"
            result_text = str(result)
            console.print(Panel(Text(result_text), title=f"[bold]{tool_name}[/bold]", border_style="dim", expand=False))

            signature = f"{tool_name}|{repr(args)}|{repr(sorted(kwargs.items()))}"
            if signature == self._last_tool_signature and result_text == self._last_tool_result:
                self._repeat_tool_hits += 1
            else:
                self._repeat_tool_hits = 0
            self._last_tool_signature = signature
            self._last_tool_result = result_text

            if result_text.startswith("💥"):
                self._tool_error_streak += 1
            else:
                self._tool_error_streak = 0
            
            self.conv.add_message("assistant", response, is_user=False)
            
            # Determine feedback message based on tool type
            info_tools = {"web_search", "grokipedia", "fetch_url"}
            if tool_name in info_tools:
                answer_instruction = (
                    "IMPORTANT: Now answer the user's original question using ONLY the data above. "
                    "Extract the specific facts, numbers, or information from the results. "
                    "Give a concise, direct answer — do NOT list URLs, do NOT describe websites, "
                    "do NOT say 'according to the results'. Just answer naturally as if you know it. "
                    "Keep your answer SHORT and to the point (1-3 sentences max for simple questions)."
                )
            else:
                answer_instruction = "Proceed."
            
            if self._repeat_tool_hits >= 1 or self._tool_error_streak >= 2:
                self._tool_loop_guard_active = True
                self.conv.add_message(
                    "user",
                    f"RESULT [{tool_name}]: {result_text}\n"
                    "Stop tool repetition. Provide the best direct answer now, "
                    "or ask the user for one missing input.",
                    is_user=True
                )
            else:
                self.conv.add_message("user", f"RESULT [{tool_name}]: {result_text}\n{answer_instruction}", is_user=True)
            return response, True
        
        self.conv.add_message("assistant", response, is_user=False)
        return response, False
    
    def run(self):
        self.ui.print_banner()
        
        # Test API connection
        self._test_api_connection()
        
        active = self.skill_manager.list_active_skills()
        if active:
            console.print(f"[skill]Active Skills: {', '.join(s.name for s in active)}[/skill]\n")
        
        # Show AgentSkills hint if available but no repos loaded
        if YAML_AVAILABLE and self.skill_manager.agentskills_loader:
            if not self.skill_manager.agentskills_loader.loaded_repos:
                console.print("[dim]💡 Tip: Use /skills add-repo <owner/repo> to load AgentSkills[/dim]")
                console.print("[dim]     Use /skills examples to see example repositories[/dim]\n")
        
        while True:
            try:
                user_input = self.session.prompt(
                    # ✅ PART 2 - Uses new PROMPT_SYMBOL "❯❯ "
                    HTML(f"<b>{CONFIG['AGENT_NAME']} {CONFIG['PROMPT_SYMBOL']} </b>")
                ).strip()
                
                if not user_input:
                    continue
                if user_input.lower() in ['exit', 'quit']:
                    console.print("[bold red]λ Shutdown[/bold red]")
                    break
                if self.handle_command(user_input):
                    continue
                normalized_input, normalize_note = self._normalize_flight_query(user_input)
                if normalize_note:
                    console.print(f"[dim]ℹ {normalize_note}[/dim]")
                    user_input = normalized_input
                
                # ✅ PART 6 - CYBER DIVIDER
                console.print(
                    f"\n[dim]{datetime.now().strftime('%H:%M:%S')}[/dim] "
                    "[bright_magenta]━━━⛧━━━━━━━━━━━━━━━━━━━━━━⛧━━━[/bright_magenta]"
                )
                self._last_tool_signature = None
                self._last_tool_result = None
                self._repeat_tool_hits = 0
                self._tool_error_streak = 0
                self._tool_loop_guard_active = False
                self.conv.add_message("user", user_input, is_user=True)
                
                for i in range(CONFIG["MAX_ITERATIONS"]):
                    _, cont = self.execute_turn()
                    if not cont:
                        break
                    console.print(f"[dim]--- Turn {i+2} ---[/dim]")
                
                console.print()
                
            except KeyboardInterrupt:
                console.print("\n[yellow]Interrupted[/yellow]")
            except EOFError:
                break

# ✅ PART 4 - BOOT BANNER (Night City core)
class UIManager:
    @staticmethod
    def print_banner():
        console.clear()

        banner = Text("""
 ████████╗██████╗ ██╗██████╗ ██╗     ███████╗ ██████╗ 
 ╚══██╔══╝██╔══██╗██║██╔══██╗██║     ██╔════╝██╔════╝ 
    ██║   ██████╔╝██║██████╔╝██║     █████╗  ██║  ███╗
    ██║   ██╔══██╗██║██║     ██║     ██╔══╝  ██║   ██║
    ██║   ██║  ██║██║██║     ███████╗███████ ╚██████╔╝
    ╚═╝   ╚═╝  ╚═╝╚═╝╚═╝     ╚══════╝╚══════  ╚═════╝
""", style="bold bright_magenta")

        mode_label = "SAM'S LAW" if TORCH_AVAILABLE else "CLASSICAL"
        agentskills_status = "AGENTSKILLS READY" if YAML_AVAILABLE else "AGENTSKILLS DISABLED"
        subtitle = Text(
            f"⛧ NEURAL CORE ONLINE ⛧  |  MODE: {mode_label}  |  {agentskills_status}",
            style="bright_cyan"
        )

        panel = Panel(
            Align.center(banner),
            subtitle=subtitle,
            border_style="bright_magenta",
            padding=(1, 4)
        )

        console.print(panel)
        console.print("[dim]Jack in with /skills /status /save /load — exit to disconnect[/dim]\n")

class ResponseParser:
    """
    Parses AI responses to extract tool calls in the format:
    :::TOOL:tool_name:::
    :::ARGS:::
    arguments
    :::END:::
    
    Handles common variations:
    - 2 or 3 trailing colons (:::END:: or :::END:::)
    - Markdown code block wrapping
    - Extra whitespace/newlines
    """
    
    # Primary pattern - flexible with 2-3 trailing colons
    TOOL_PATTERN = re.compile(
        r':::TOOL:\s*(\w+)\s*:::\s*:::ARGS:::\s*(.*?)\s*:::END::(?::)?',
        re.DOTALL | re.IGNORECASE
    )
    
    # Pattern to detect if tool call exists but might be malformed
    TOOL_DETECT_PATTERN = re.compile(
        r':::TOOL:\s*\w+\s*:::',
        re.IGNORECASE
    )
    
    # Fallback patterns for edge cases
    FALLBACK_PATTERNS = [
        # Standard with flexible end
        re.compile(r':::TOOL:\s*(\w+)\s*:::\s*:::ARGS:::\s*(.*?)\s*:::END::(?::)?', re.DOTALL | re.IGNORECASE),
        # With markdown code block wrapper
        re.compile(r'```\s*:::TOOL:\s*(\w+)\s*:::\s*:::ARGS:::\s*(.*?)\s*:::END::(?::)?\s*```', re.DOTALL | re.IGNORECASE),
        # Newline-strict format
        re.compile(r':::TOOL:\s*(\w+)\s*:::\n+:::ARGS:::\n+(.*?)\n+:::END::(?::)?', re.DOTALL | re.IGNORECASE),
        # With extra colons (some models add more)
        re.compile(r':::TOOL:\s*(\w+)\s*:::+\s*:::ARGS:::+\s*(.*?)\s*:::END:::*', re.DOTALL | re.IGNORECASE),
    ]
    
    @classmethod
    def _strip_code_blocks(cls, text: str) -> str:
        """Remove markdown code block wrappers if present."""
        # Strip ```json or ``` wrappers around the entire tool call
        stripped = text.strip()
        if stripped.startswith('```') and stripped.endswith('```'):
            # Find the end of the first line (might be ```json, ```python, etc.)
            first_newline = stripped.find('\n')
            if first_newline > 0:
                inner = stripped[first_newline+1:-3].strip()
                if ':::TOOL:' in inner:
                    return inner
        return text
    
    @classmethod
    def parse(cls, text: str) -> Tuple[str, Optional[str], Optional[str]]:
        """
        Parse response text for tool calls.
        
        Returns:
            Tuple of (clean_text, tool_name, tool_args)
            - clean_text: Response text with tool call removed
            - tool_name: Name of the tool to call, or None
            - tool_args: Arguments for the tool, or None
        """
        if not text:
            return "", None, None
        
        # Pre-process: strip code blocks
        processed = cls._strip_code_blocks(text)
        
        # Try primary pattern first
        m = cls.TOOL_PATTERN.search(processed)
        if m:
            clean = cls.TOOL_PATTERN.sub('', processed).strip()
            return clean, m.group(1).strip(), m.group(2).strip()
        
        # Try fallback patterns
        for pattern in cls.FALLBACK_PATTERNS:
            m = pattern.search(processed)
            if m:
                clean = pattern.sub('', processed).strip()
                return clean, m.group(1).strip(), m.group(2).strip()
        
        # Also try on original text (in case pre-processing removed something important)
        if processed != text:
            m = cls.TOOL_PATTERN.search(text)
            if m:
                clean = cls.TOOL_PATTERN.sub('', text).strip()
                return clean, m.group(1).strip(), m.group(2).strip()
        
        # No tool call found
        return text, None, None
    
    @classmethod
    def has_potential_tool_call(cls, text: str) -> bool:
        """Check if text contains what looks like a tool call (even if malformed)."""
        return bool(cls.TOOL_DETECT_PATTERN.search(text))

if __name__ == "__main__":
    TripleGSam().run()
