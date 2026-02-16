"""
TripleG Skills Marketplace
Fetches, installs, and manages Agent Skills from GitHub repositories.
Supports the standard SKILL.md format used by Claude, Codex, Copilot, etc.
"""

import json
import os
import re
import shutil
import tempfile
import time
import urllib.request
import urllib.error
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

# ==========================================
# SKILL REGISTRY - CURATED SOURCES
# ==========================================

# These are real GitHub repositories that contain Agent Skills
SKILL_SOURCES = {
    # === OFFICIAL REPOSITORIES ===
    "anthropics/skills": {
        "name": "Anthropic Official Skills",
        "description": "Official Claude skills for document processing (docx, xlsx, pptx, pdf)",
        "category": "official",
        "url": "https://github.com/anthropics/skills",
        "branch": "main",
        "verified": True,
    },
    "huggingface/skills": {
        "name": "HuggingFace Skills",
        "description": "HF skills for dataset creation, model evaluation, training, paper publishing",
        "category": "official",
        "url": "https://github.com/huggingface/skills",
        "branch": "main",
        "verified": True,
    },

    # === COMMUNITY COLLECTIONS ===
    "skillcreatorai/Ai-Agent-Skills": {
        "name": "SkillCreator.ai Collection",
        "description": "Community skills with CLI installer - various coding and automation skills",
        "category": "community",
        "url": "https://github.com/skillcreatorai/Ai-Agent-Skills",
        "branch": "main",
        "verified": False,
    },
    "karanb192/awesome-claude-skills": {
        "name": "Awesome Claude Skills",
        "description": "50+ verified skills for Claude Code and Claude.ai",
        "category": "community",
        "url": "https://github.com/karanb192/awesome-claude-skills",
        "branch": "main",
        "verified": False,
    },
    "mhattingpete/claude-skills-marketplace": {
        "name": "Claude Skills Marketplace",
        "description": "Git pushing, code review, test fixing, computer forensics skills",
        "category": "community",
        "url": "https://github.com/mhattingpete/claude-skills-marketplace",
        "branch": "main",
        "verified": False,
    },
    "hikanner/agent-skills": {
        "name": "Curated Agent Skills",
        "description": "Curated Claude Agent Skills collection",
        "category": "community",
        "url": "https://github.com/hikanner/agent-skills",
        "branch": "main",
        "verified": False,
    },
    "gotalab/skillport": {
        "name": "SkillPort",
        "description": "Skills distribution via CLI or MCP",
        "category": "community",
        "url": "https://github.com/gotalab/skillport",
        "branch": "main",
        "verified": False,
    },
    "GuDaStudio/skills": {
        "name": "GuDa Studio Skills",
        "description": "Multi-agent collaboration skills",
        "category": "community",
        "url": "https://github.com/GuDaStudio/skills",
        "branch": "main",
        "verified": False,
    },

    # === SPECIALIZED SKILLS ===
    "zxkane/aws-skills": {
        "name": "AWS Skills",
        "description": "AWS development with CDK best practices",
        "category": "specialized",
        "url": "https://github.com/zxkane/aws-skills",
        "branch": "main",
        "verified": False,
    },
    "lackeyjb/playwright-skill": {
        "name": "Playwright Automation",
        "description": "Browser automation for testing web apps",
        "category": "specialized",
        "url": "https://github.com/lackeyjb/playwright-skill",
        "branch": "main",
        "verified": False,
    },
    "fractalmind-ai/agent-manager-skill": {
        "name": "Agent Manager",
        "description": "Manage local CLI AI agents via tmux (start/stop/monitor/assign + cron scheduling)",
        "category": "specialized",
        "url": "https://github.com/fractalmind-ai/agent-manager-skill",
        "branch": "main",
        "verified": False,
    },
    "smerchek/claude-epub-skill": {
        "name": "Markdown to EPUB",
        "description": "Converts markdown documents into professional EPUB ebook files",
        "category": "specialized",
        "url": "https://github.com/smerchek/claude-epub-skill",
        "branch": "main",
        "verified": False,
    },
    "chrisvoncsefalvay/claude-d3js-skill": {
        "name": "D3.js Visualization",
        "description": "D3 charts and interactive data visualizations",
        "category": "specialized",
        "url": "https://github.com/chrisvoncsefalvay/claude-d3js-skill",
        "branch": "main",
        "verified": False,
    },
    "jthack/threat-hunting-with-sigma-rules-skill": {
        "name": "Threat Hunting",
        "description": "Hunt for threats using Sigma detection rules",
        "category": "specialized",
        "url": "https://github.com/jthack/threat-hunting-with-sigma-rules-skill",
        "branch": "main",
        "verified": False,
    },
    "jakedahn/pomodoro": {
        "name": "Pomodoro System Skill",
        "description": "System Skill Pattern - skills that remember and improve over time",
        "category": "specialized",
        "url": "https://github.com/jakedahn/pomodoro",
        "branch": "main",
        "verified": False,
    },
}


# ==========================================
# DATA CLASSES
# ==========================================

@dataclass
class MarketplaceSkill:
    """Represents a skill available in the marketplace."""
    id: str
    name: str
    description: str
    category: str  # official, community, specialized, custom, installed
    source_repo: str  # e.g. "anthropics/skills"
    source_url: str
    branch: str = "main"
    skill_path: str = ""  # relative path within repo
    version: str = "1.0.0"
    author: str = "Unknown"
    verified: bool = False
    installed: bool = False
    active: bool = False
    install_dir: Optional[str] = None
    content: str = ""  # Full SKILL.md content
    frontmatter: Dict[str, Any] = field(default_factory=dict)
    tools: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    dangerous: bool = False
    last_updated: str = ""

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "category": self.category,
            "source_repo": self.source_repo,
            "source_url": self.source_url,
            "branch": self.branch,
            "skill_path": self.skill_path,
            "version": self.version,
            "author": self.author,
            "verified": self.verified,
            "installed": self.installed,
            "active": self.active,
            "install_dir": self.install_dir,
            "content": self.content,
            "frontmatter": self.frontmatter,
            "tools": self.tools,
            "tags": self.tags,
            "dangerous": self.dangerous,
            "last_updated": self.last_updated,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "MarketplaceSkill":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


# ==========================================
# MARKETPLACE ENGINE
# ==========================================

class SkillsMarketplaceEngine:
    """
    Core marketplace engine that fetches, caches, installs, and manages skills
    from GitHub repositories using the standard SKILL.md format.
    """

    def __init__(self, base_dir: Optional[Path] = None):
        self.base_dir = base_dir or (Path.home() / ".tripleg")
        self.skills_dir = self.base_dir / "marketplace_skills"
        self.cache_dir = self.base_dir / "marketplace_cache"
        self.registry_file = self.base_dir / "marketplace_registry.json"
        self.custom_sources_file = self.base_dir / "custom_sources.json"

        # Create directories
        self.skills_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # State
        self.registry: Dict[str, MarketplaceSkill] = {}
        self.custom_sources: Dict[str, Dict] = {}
        self.fetch_log: List[str] = []

        # Load saved state
        self._load_registry()
        self._load_custom_sources()

    # ------------------------------------------
    # REGISTRY PERSISTENCE
    # ------------------------------------------

    def _load_registry(self):
        """Load the local skill registry from disk."""
        if self.registry_file.exists():
            try:
                with open(self.registry_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                for skill_data in data.get("skills", []):
                    skill = MarketplaceSkill.from_dict(skill_data)
                    self.registry[skill.id] = skill
            except Exception as e:
                self.fetch_log.append(f"Registry load error: {e}")

    def _save_registry(self):
        """Save the local skill registry to disk."""
        try:
            data = {
                "last_updated": datetime.now().isoformat(),
                "skills": [s.to_dict() for s in self.registry.values()],
            }
            with open(self.registry_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            self.fetch_log.append(f"Registry save error: {e}")

    def _load_custom_sources(self):
        """Load user-added custom sources."""
        if self.custom_sources_file.exists():
            try:
                with open(self.custom_sources_file, "r", encoding="utf-8") as f:
                    self.custom_sources = json.load(f)
            except Exception:
                self.custom_sources = {}

    def _save_custom_sources(self):
        """Save user-added custom sources."""
        try:
            with open(self.custom_sources_file, "w", encoding="utf-8") as f:
                json.dump(self.custom_sources, f, indent=2)
        except Exception:
            pass

    # ------------------------------------------
    # GITHUB FETCHING
    # ------------------------------------------

    def _download_repo_zip(self, owner: str, repo: str, branch: str = "main") -> Optional[Path]:
        """Download a GitHub repo as a zip and extract to cache."""
        cache_key = f"{owner}_{repo}_{branch}"
        extract_path = self.cache_dir / cache_key

        # Use cache if less than 24 hours old
        marker = extract_path / ".fetch_time"
        if extract_path.exists() and marker.exists():
            try:
                fetch_time = float(marker.read_text())
                if time.time() - fetch_time < 86400:  # 24 hours
                    return extract_path
            except Exception:
                pass

        zip_url = f"https://github.com/{owner}/{repo}/archive/refs/heads/{branch}.zip"
        self.fetch_log.append(f"Downloading {owner}/{repo}@{branch}...")

        try:
            req = urllib.request.Request(zip_url, headers={"User-Agent": "TripleG-Marketplace/2.0"})
            with urllib.request.urlopen(req, timeout=30) as response:
                zip_content = response.read()

            # Clean old cache
            if extract_path.exists():
                shutil.rmtree(extract_path, ignore_errors=True)

            with tempfile.TemporaryDirectory() as tmpdir:
                zip_path = Path(tmpdir) / "repo.zip"
                zip_path.write_bytes(zip_content)

                with zipfile.ZipFile(zip_path, "r") as zf:
                    zf.extractall(tmpdir)

                # Find extracted folder
                extracted = [p for p in Path(tmpdir).iterdir() if p.is_dir() and p.name != "__MACOSX"]
                if extracted:
                    shutil.move(str(extracted[0]), str(extract_path))
                else:
                    self.fetch_log.append(f"No folder found in zip for {owner}/{repo}")
                    return None

            # Write fetch timestamp
            marker.write_text(str(time.time()))
            self.fetch_log.append(f"Cached {owner}/{repo}@{branch}")
            return extract_path

        except urllib.error.HTTPError as e:
            self.fetch_log.append(f"HTTP {e.code} for {owner}/{repo} ({branch})")
            # Try 'master' branch as fallback
            if branch == "main":
                return self._download_repo_zip(owner, repo, "master")
            return None
        except Exception as e:
            self.fetch_log.append(f"Download failed for {owner}/{repo}: {e}")
            return None

    def _parse_skill_md(self, skill_path: Path) -> Optional[Dict[str, Any]]:
        """Parse a SKILL.md file with YAML frontmatter."""
        if not YAML_AVAILABLE:
            # Fallback: parse without YAML
            return self._parse_skill_md_no_yaml(skill_path)

        try:
            content = skill_path.read_text("utf-8", errors="replace")

            frontmatter = {}
            body = content

            if content.startswith("---"):
                parts = content.split("---", 2)
                if len(parts) >= 3:
                    try:
                        frontmatter = yaml.safe_load(parts[1]) or {}
                    except Exception:
                        frontmatter = {}
                    body = parts[2].strip()

            return {
                "frontmatter": frontmatter,
                "body": body,
                "full_content": content,
                "path": str(skill_path),
                "dir": str(skill_path.parent),
            }
        except Exception as e:
            self.fetch_log.append(f"Parse error for {skill_path}: {e}")
            return None

    def _parse_skill_md_no_yaml(self, skill_path: Path) -> Optional[Dict[str, Any]]:
        """Parse SKILL.md without YAML library - basic frontmatter extraction."""
        try:
            content = skill_path.read_text("utf-8", errors="replace")
            frontmatter = {}
            body = content

            if content.startswith("---"):
                parts = content.split("---", 2)
                if len(parts) >= 3:
                    # Basic key: value parsing
                    for line in parts[1].strip().splitlines():
                        line = line.strip()
                        if ":" in line:
                            key, _, value = line.partition(":")
                            frontmatter[key.strip()] = value.strip()
                    body = parts[2].strip()

            return {
                "frontmatter": frontmatter,
                "body": body,
                "full_content": content,
                "path": str(skill_path),
                "dir": str(skill_path.parent),
            }
        except Exception:
            return None

    def _discover_skills_in_repo(self, repo_path: Path) -> List[Dict[str, Any]]:
        """Find all SKILL.md files in a repository."""
        skills = []
        seen_paths = set()

        patterns = [
            "**/SKILL.md",
            "skills/**/SKILL.md",
            ".claude/skills/**/SKILL.md",
            ".codex/skills/**/SKILL.md",
            ".github/skills/**/SKILL.md",
        ]

        for pattern in patterns:
            for skill_file in repo_path.glob(pattern):
                if skill_file in seen_paths:
                    continue
                seen_paths.add(skill_file)

                parsed = self._parse_skill_md(skill_file)
                if parsed:
                    # Compute relative path within repo
                    try:
                        rel_path = skill_file.relative_to(repo_path)
                    except ValueError:
                        rel_path = skill_file.name
                    parsed["relative_path"] = str(rel_path)
                    skills.append(parsed)

        return skills

    def _make_skill_id(self, source_repo: str, skill_name: str, rel_path: str) -> str:
        """Generate a unique skill ID."""
        # Clean the name
        clean_name = re.sub(r"[^a-zA-Z0-9_-]", "_", skill_name.lower().replace(" ", "_"))
        clean_repo = source_repo.replace("/", "_")
        return f"{clean_repo}__{clean_name}"

    def _convert_to_marketplace_skill(
        self, parsed: Dict, source_repo: str, source_info: Dict
    ) -> Optional[MarketplaceSkill]:
        """Convert parsed SKILL.md data to a MarketplaceSkill."""
        fm = parsed.get("frontmatter", {})
        body = parsed.get("body", "")
        rel_path = parsed.get("relative_path", "")

        name = fm.get("name", "")
        if not name:
            # Try to extract name from directory or first heading
            dir_name = Path(parsed.get("dir", "")).name
            if dir_name and dir_name != ".":
                name = dir_name.replace("-", " ").replace("_", " ").title()
            else:
                heading_match = re.search(r"^#\s+(.+)$", body, re.MULTILINE)
                if heading_match:
                    name = heading_match.group(1).strip()
                else:
                    name = "Unnamed Skill"

        description = fm.get("description", "")
        if not description:
            # Extract first paragraph as description
            lines = [l.strip() for l in body.split("\n") if l.strip() and not l.strip().startswith("#")]
            description = lines[0][:200] if lines else f"Skill from {source_repo}"

        # Extract metadata
        metadata = fm.get("metadata", {})
        if isinstance(metadata, str):
            metadata = {}

        version = str(metadata.get("version", fm.get("version", "1.0.0")))
        author = str(metadata.get("author", fm.get("author", source_repo.split("/")[0])))

        # Extract tags from content
        tags = []
        if isinstance(fm.get("tags"), list):
            tags = fm["tags"]
        else:
            # Infer tags from content
            content_lower = (body + " " + name + " " + description).lower()
            tag_keywords = {
                "python": ["python", "pip", "pytest", "django", "flask"],
                "javascript": ["javascript", "typescript", "node", "npm", "react"],
                "web": ["html", "css", "web", "browser", "http"],
                "git": ["git", "commit", "branch", "merge", "rebase"],
                "docker": ["docker", "container", "kubernetes"],
                "aws": ["aws", "lambda", "s3", "ec2", "cloudformation"],
                "security": ["security", "vulnerability", "audit", "threat"],
                "testing": ["test", "testing", "unittest", "pytest", "jest"],
                "database": ["database", "sql", "postgres", "mysql", "mongodb"],
                "api": ["api", "rest", "graphql", "endpoint"],
                "docs": ["documentation", "docstring", "readme", "markdown"],
                "devops": ["ci/cd", "deploy", "pipeline", "github actions"],
                "data": ["data", "pandas", "numpy", "csv", "dataset"],
                "ai": ["model", "training", "inference", "llm", "embedding"],
            }
            for tag, keywords in tag_keywords.items():
                if any(kw in content_lower for kw in keywords):
                    tags.append(tag)

        skill_id = self._make_skill_id(source_repo, name, rel_path)

        return MarketplaceSkill(
            id=skill_id,
            name=name,
            description=description,
            category=source_info.get("category", "community"),
            source_repo=source_repo,
            source_url=source_info.get("url", f"https://github.com/{source_repo}"),
            branch=source_info.get("branch", "main"),
            skill_path=rel_path,
            version=version,
            author=author,
            verified=source_info.get("verified", False),
            content=body[:5000],  # Store truncated content
            frontmatter=fm,
            tools=[],  # Agent skills are knowledge-based
            tags=tags,
            dangerous=False,
            last_updated=datetime.now().isoformat(),
        )

    # ------------------------------------------
    # PUBLIC API
    # ------------------------------------------

    def get_all_sources(self) -> Dict[str, Dict]:
        """Get all skill sources (builtin + custom)."""
        sources = dict(SKILL_SOURCES)
        sources.update(self.custom_sources)
        return sources

    def add_source(self, repo_id: str, name: str = "", description: str = "", branch: str = "main") -> str:
        """Add a custom skill source repository."""
        if "/" not in repo_id:
            return f"Invalid repo format: {repo_id}. Use owner/repo format."

        # Parse if full URL
        if repo_id.startswith("https://github.com/"):
            repo_id = repo_id.replace("https://github.com/", "").rstrip("/").rstrip(".git")

        owner, repo = repo_id.split("/", 1)

        if not name:
            name = f"{owner}/{repo}"

        self.custom_sources[repo_id] = {
            "name": name,
            "description": description or f"Custom source: {repo_id}",
            "category": "custom",
            "url": f"https://github.com/{repo_id}",
            "branch": branch,
            "verified": False,
        }
        self._save_custom_sources()
        return f"Added source: {repo_id}"

    def remove_source(self, repo_id: str) -> str:
        """Remove a custom skill source."""
        if repo_id in self.custom_sources:
            del self.custom_sources[repo_id]
            self._save_custom_sources()
            return f"Removed source: {repo_id}"
        return f"Source not found: {repo_id}"

    def fetch_skills_from_source(self, repo_id: str) -> Tuple[int, List[str]]:
        """Fetch skills from a single source repository."""
        all_sources = self.get_all_sources()
        if repo_id not in all_sources:
            return 0, [f"Unknown source: {repo_id}"]

        source_info = all_sources[repo_id]
        owner, repo = repo_id.split("/", 1)
        branch = source_info.get("branch", "main")

        repo_path = self._download_repo_zip(owner, repo, branch)
        if not repo_path:
            return 0, [f"Failed to download {repo_id}"]

        raw_skills = self._discover_skills_in_repo(repo_path)
        if not raw_skills:
            return 0, [f"No SKILL.md files found in {repo_id}"]

        count = 0
        for raw in raw_skills:
            skill = self._convert_to_marketplace_skill(raw, repo_id, source_info)
            if skill:
                # Preserve installed/active state if already in registry
                if skill.id in self.registry:
                    existing = self.registry[skill.id]
                    skill.installed = existing.installed
                    skill.active = existing.active
                    skill.install_dir = existing.install_dir
                self.registry[skill.id] = skill
                count += 1

        self._save_registry()
        return count, [f"Fetched {count} skills from {repo_id}"]

    def fetch_all_sources(self, progress_callback=None) -> Tuple[int, List[str]]:
        """Fetch skills from all sources. Returns (total_count, log_messages)."""
        all_sources = self.get_all_sources()
        total = 0
        messages = []

        for i, (repo_id, source_info) in enumerate(all_sources.items()):
            if progress_callback:
                progress_callback(i, len(all_sources), repo_id)

            count, msgs = self.fetch_skills_from_source(repo_id)
            total += count
            messages.extend(msgs)

        messages.append(f"Total: {total} skills from {len(all_sources)} sources")
        return total, messages

    def install_skill(self, skill_id: str) -> str:
        """Install a skill from the registry to local storage."""
        if skill_id not in self.registry:
            return f"Skill not found: {skill_id}"

        skill = self.registry[skill_id]
        if skill.installed:
            return f"Already installed: {skill.name}"

        # Create install directory
        install_dir = self.skills_dir / skill_id
        install_dir.mkdir(parents=True, exist_ok=True)

        # Write SKILL.md
        skill_md_content = ""
        if skill.frontmatter:
            if YAML_AVAILABLE:
                skill_md_content = "---\n" + yaml.dump(skill.frontmatter, default_flow_style=False) + "---\n\n"
            else:
                fm_lines = [f"{k}: {v}" for k, v in skill.frontmatter.items() if isinstance(v, (str, int, float, bool))]
                skill_md_content = "---\n" + "\n".join(fm_lines) + "\n---\n\n"
        skill_md_content += skill.content

        (install_dir / "SKILL.md").write_text(skill_md_content, encoding="utf-8")

        # Write metadata
        meta = skill.to_dict()
        meta["installed"] = True
        meta["install_dir"] = str(install_dir)
        (install_dir / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

        # If the source repo was cached, copy any additional files (scripts, templates, etc.)
        if skill.skill_path:
            owner, repo = skill.source_repo.split("/", 1)
            cache_key = f"{owner}_{repo}_{skill.branch}"
            cached_repo = self.cache_dir / cache_key
            if cached_repo.exists():
                skill_src_dir = cached_repo / Path(skill.skill_path).parent
                if skill_src_dir.exists() and skill_src_dir.is_dir():
                    for item in skill_src_dir.iterdir():
                        if item.name != "SKILL.md" and item.name != "metadata.json":
                            dest = install_dir / item.name
                            if item.is_dir():
                                if dest.exists():
                                    shutil.rmtree(dest)
                                shutil.copytree(str(item), str(dest))
                            else:
                                shutil.copy2(str(item), str(dest))

        # Update registry
        skill.installed = True
        skill.install_dir = str(install_dir)
        self._save_registry()

        return f"Installed: {skill.name} ({skill.source_repo})"

    def uninstall_skill(self, skill_id: str) -> str:
        """Uninstall a skill."""
        if skill_id not in self.registry:
            return f"Skill not found: {skill_id}"

        skill = self.registry[skill_id]
        if not skill.installed:
            return f"Not installed: {skill.name}"

        # Deactivate first
        if skill.active:
            skill.active = False

        # Remove install directory
        if skill.install_dir and Path(skill.install_dir).exists():
            shutil.rmtree(skill.install_dir, ignore_errors=True)

        skill.installed = False
        skill.install_dir = None
        self._save_registry()

        return f"Uninstalled: {skill.name}"

    def activate_skill(self, skill_id: str) -> str:
        """Activate an installed skill."""
        if skill_id not in self.registry:
            return f"Skill not found: {skill_id}"

        skill = self.registry[skill_id]
        if not skill.installed:
            return f"Not installed: {skill.name}. Install first."

        skill.active = True
        self._save_registry()
        return f"Activated: {skill.name}"

    def deactivate_skill(self, skill_id: str) -> str:
        """Deactivate a skill."""
        if skill_id not in self.registry:
            return f"Skill not found: {skill_id}"

        skill = self.registry[skill_id]
        skill.active = False
        self._save_registry()
        return f"Deactivated: {skill.name}"

    def get_installed_skills(self) -> List[MarketplaceSkill]:
        """Get all installed skills."""
        return [s for s in self.registry.values() if s.installed]

    def get_active_skills(self) -> List[MarketplaceSkill]:
        """Get all active skills."""
        return [s for s in self.registry.values() if s.active]

    def get_skills_by_category(self, category: str) -> List[MarketplaceSkill]:
        """Get skills filtered by category."""
        return [s for s in self.registry.values() if s.category == category]

    def search_skills(self, query: str) -> List[MarketplaceSkill]:
        """Search skills by name, description, or tags."""
        query_lower = query.lower()
        results = []
        for skill in self.registry.values():
            score = 0
            if query_lower in skill.name.lower():
                score += 3
            if query_lower in skill.description.lower():
                score += 2
            if any(query_lower in t.lower() for t in skill.tags):
                score += 2
            if query_lower in skill.source_repo.lower():
                score += 1
            if query_lower in skill.content.lower():
                score += 1
            if score > 0:
                results.append((score, skill))
        results.sort(key=lambda x: x[0], reverse=True)
        return [s for _, s in results]

    def get_skill_content(self, skill_id: str) -> str:
        """Get the full SKILL.md content for a skill."""
        if skill_id not in self.registry:
            return "Skill not found."
        skill = self.registry[skill_id]

        # If installed, read from disk
        if skill.install_dir:
            skill_md = Path(skill.install_dir) / "SKILL.md"
            if skill_md.exists():
                return skill_md.read_text("utf-8", errors="replace")

        # Otherwise return cached content
        return skill.content or "No content available."

    def get_active_system_prompt_additions(self) -> str:
        """Get system prompt additions from all active skills."""
        additions = []
        for skill in self.get_active_skills():
            content = self.get_skill_content(skill.id)
            additions.append(
                f"\n=== Skill: {skill.name} (from {skill.source_repo}) ===\n"
                f"{skill.description}\n\n"
                f"{content[:3000]}\n"
                f"=== End Skill: {skill.name} ===\n"
            )
        return "\n".join(additions)

    def create_custom_skill(
        self,
        name: str,
        description: str,
        content: str,
        tags: Optional[List[str]] = None,
    ) -> str:
        """Create a custom skill from user input."""
        skill_id = "custom__" + re.sub(r"[^a-zA-Z0-9_-]", "_", name.lower().replace(" ", "_"))

        # Create install directory
        install_dir = self.skills_dir / skill_id
        install_dir.mkdir(parents=True, exist_ok=True)

        # Build SKILL.md
        frontmatter = {
            "name": name,
            "description": description,
        }
        if YAML_AVAILABLE:
            skill_md = "---\n" + yaml.dump(frontmatter, default_flow_style=False) + "---\n\n" + content
        else:
            skill_md = f"---\nname: {name}\ndescription: {description}\n---\n\n{content}"

        (install_dir / "SKILL.md").write_text(skill_md, encoding="utf-8")

        skill = MarketplaceSkill(
            id=skill_id,
            name=name,
            description=description,
            category="custom",
            source_repo="local/custom",
            source_url="",
            skill_path="SKILL.md",
            version="1.0.0",
            author="User",
            verified=False,
            installed=True,
            active=False,
            install_dir=str(install_dir),
            content=content,
            frontmatter=frontmatter,
            tags=tags or [],
            last_updated=datetime.now().isoformat(),
        )

        # Write metadata
        (install_dir / "metadata.json").write_text(
            json.dumps(skill.to_dict(), indent=2), encoding="utf-8"
        )

        self.registry[skill_id] = skill
        self._save_registry()

        return f"Created custom skill: {name} (id: {skill_id})"

    def install_from_github_url(self, url: str) -> str:
        """Install skills directly from a GitHub URL."""
        # Parse URL
        url = url.strip().rstrip("/").rstrip(".git")
        if url.startswith("https://github.com/"):
            parts = url.replace("https://github.com/", "").split("/")
            if len(parts) < 2:
                return f"Invalid GitHub URL: {url}"
            owner, repo = parts[0], parts[1]
        elif "/" in url and not url.startswith("http"):
            owner, repo = url.split("/", 1)
        else:
            return f"Invalid URL format: {url}. Use owner/repo or https://github.com/owner/repo"

        repo_id = f"{owner}/{repo}"

        # Add as custom source if not already known
        all_sources = self.get_all_sources()
        if repo_id not in all_sources:
            self.add_source(repo_id, description=f"Installed from URL: {url}")

        # Fetch skills
        count, messages = self.fetch_skills_from_source(repo_id)
        return "\n".join(messages)

    def get_stats(self) -> Dict[str, Any]:
        """Get marketplace statistics."""
        all_sources = self.get_all_sources()
        return {
            "total_skills": len(self.registry),
            "installed": len(self.get_installed_skills()),
            "active": len(self.get_active_skills()),
            "sources": len(all_sources),
            "builtin_sources": len(SKILL_SOURCES),
            "custom_sources": len(self.custom_sources),
            "categories": {
                cat: len(self.get_skills_by_category(cat))
                for cat in set(s.category for s in self.registry.values())
            },
        }

    def clear_cache(self) -> str:
        """Clear the download cache."""
        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir, ignore_errors=True)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        return "Cache cleared."
