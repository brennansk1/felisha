"""Tab-completion helpers for the composer.

The composer's Tab key first autocompletes a slash command. After the
command head is filled in, subsequent Tabs walk the argument completer
defined here. For path-taking commands we glob the current directory
for the partial token under the cursor.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable


# Commands whose first positional argument is a filesystem path.
PATH_ARG_COMMANDS: frozenset[str] = frozenset(
    {
        "/discover",
        "/run",
        "/auto",
        "/init",
    }
)


def needs_path_completion(line: str) -> bool:
    """Return ``True`` when the trailing token on ``line`` is the path arg
    for a path-taking command.

    For ``/auto`` we treat the second word ("run") as a sub-verb and the
    third word as the path; for everything else the path is the first
    positional argument.
    """
    head, _, rest = line.partition(" ")
    head = head.lower()
    if head not in PATH_ARG_COMMANDS:
        return False
    tokens = rest.split()
    # Index of the token slot that is the path argument.
    if head == "/auto":
        # "/auto" - waiting for "run"
        if not tokens:
            return True
        # "/auto run" - the next token is the path
        if tokens[0].lower() == "run":
            path_slot = 1
        else:
            path_slot = 0
    else:
        path_slot = 0

    # If we're currently editing the path slot, we are if either:
    #   (a) cursor is on the (path_slot+1)-th token still being typed, OR
    #   (b) we've just finished the previous tokens with a trailing space.
    if line.endswith(" "):
        cursor_token = len(tokens)
    else:
        cursor_token = max(0, len(tokens) - 1)

    if cursor_token != path_slot:
        return False
    # Block completion on flag tokens (--treatment etc.).
    cur = tokens[cursor_token] if cursor_token < len(tokens) else ""
    if cur.startswith("--"):
        return False
    return True


def _last_token(line: str) -> str:
    if line.endswith(" "):
        return ""
    tokens = line.rsplit(" ", 1)
    return tokens[-1] if tokens else ""


def complete_path(
    line: str,
    cwd: Path,
    max_results: int = 10,
) -> tuple[str, ...]:
    """Return up to ``max_results`` filesystem completions for the last
    token in ``line`` relative to ``cwd``.

    Directories are returned with a trailing ``/`` so successive Tabs walk
    deeper into the tree.
    """
    token = _last_token(line)
    base = cwd
    prefix = token
    # If the user typed a partial like "data/coh", split into base + prefix.
    if "/" in token:
        head, _, prefix = token.rpartition("/")
        candidate = Path(head)
        base = candidate if candidate.is_absolute() else (cwd / candidate)
    try:
        if not base.is_dir():
            return ()
        children = sorted(base.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
    except OSError:
        return ()
    out: list[str] = []
    for child in children:
        name = child.name
        if not name.lower().startswith(prefix.lower()):
            continue
        if name.startswith(".") and not prefix.startswith("."):
            continue
        suffix = "/" if child.is_dir() else ""
        # Reconstruct the full token relative to the user's input
        if "/" in token:
            head = token.rsplit("/", 1)[0]
            out.append(f"{head}/{name}{suffix}")
        else:
            out.append(f"{name}{suffix}")
        if len(out) >= max_results:
            break
    return tuple(out)


def apply_completion(line: str, completion: str) -> str:
    """Return ``line`` with its trailing token replaced by ``completion``."""
    if line.endswith(" "):
        return line + completion
    head, _, _ = line.rpartition(" ")
    if head:
        return f"{head} {completion}"
    return completion


def common_prefix(strings: Iterable[str]) -> str:
    """Longest common prefix of an iterable of strings."""
    items = list(strings)
    if not items:
        return ""
    out = items[0]
    for s in items[1:]:
        i = 0
        while i < len(out) and i < len(s) and out[i] == s[i]:
            i += 1
        out = out[:i]
        if not out:
            break
    return out


__all__ = [
    "PATH_ARG_COMMANDS",
    "needs_path_completion",
    "complete_path",
    "apply_completion",
    "common_prefix",
]
