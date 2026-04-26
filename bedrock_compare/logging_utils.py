"""Shared logging utilities for bedrock_compare scripts.

Provides ANSI color helpers via colorama, terminal-width truncation,
and a prompt-tracking set.
"""

from colorama import Fore, Style, init as _colorama_init
import hashlib
import shutil

_colorama_init(autoreset=True)  # auto-reset after each print

# ── ANSI color constants ────────────────────────────────────────────────────

DIM = Fore.LIGHTBLACK_EX          # grey / muted
BOLD = Style.BRIGHT               # bright bold
GREEN = Fore.GREEN                # success
BLUE = Fore.BLUE                  # announcements
YELLOW = Fore.YELLOW              # warning
RED = Fore.RED                    # error
CYAN = Fore.CYAN                  # filenames
RESET = Style.RESET_ALL           # reset to default

# ── Helpers ──────────────────────────────────────────────────────────────────

def truncate_line(
    text: str,
    width: int = shutil.get_terminal_size().columns,
) -> str:
    """Collapse *text* to a single line and truncate to *width* chars."""
    text = " ".join(text.replace("\n", " ").replace("\r", "").split())
    suffix = "..." if len(text) > width else ""
    return text[:max(0, width - len(suffix))] + suffix


# ── Prompt tracking (in-memory, per-run) ────────────────────────────────────

_SEEN_PROMPTS: set = set()


def _prompt_hash(prompt: str) -> int:
    """Deterministic hash for a prompt string."""
    return hash(prompt)


def seen_prompt(prompt: str) -> bool:
    """Return ``True`` if *prompt* has already been seen this run.

    First call always returns ``False`` (marks the prompt as seen).
    """
    h = _prompt_hash(prompt)
    if h in _SEEN_PROMPTS:
        return True
    _SEEN_PROMPTS.add(h)
    return False


# ── Logging helpers ─────────────────────────────────────────────────────────

def log_key(msg: str) -> None:
    """Bold — major milestones (file start, completion)."""
    print(f"{BOLD}{msg}{RESET}")


def log_info(msg: str) -> None:
    """Normal — routine info (source names, progress)."""
    print(msg)


def log_dim(msg: str) -> None:
    """Dimmed / greyed-out — background ops (health checks, model listing)."""
    print(f"{DIM}{msg}{RESET}")


def log_green(msg: str) -> None:
    """Green — success markers."""
    print(f"{GREEN}{msg}{RESET}")


def log_yellow(msg: str) -> None:
    """Yellow — warnings."""
    print(f"{YELLOW}{msg}{RESET}")


def log_blue(msg: str) -> None:
    """Blue — LLM judge announcements."""
    print(f"{BLUE}{msg}{RESET}")

def log_cyan(msg: str) -> None:
    """Cyan — filenames and identifiers."""
    print(f"{CYAN}{msg}{RESET}")


def log_red(msg: str) -> None:
    """Red — errors."""
    print(f"{RED}{msg}{RESET}")
