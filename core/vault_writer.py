"""Write notes to the Obsidian-style memory vault.

Writes markdown files with YAML frontmatter to the trading-memory vault.
Used by the trading system to auto-log incidents, trades, and decisions.
Pure Python, no dependencies beyond stdlib.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone, timedelta

from core.logger import setup_logger

log = setup_logger("trading.vault")

# SGT timezone
_SGT = timezone(timedelta(hours=8))

# Vault location — configurable via env, defaults to ~/trading-memory
VAULT_DIR = os.getenv("VAULT_DIR", os.path.expanduser("~/trading-memory"))


def _ensure_dir(folder: str) -> None:
    """Create vault subdirectory if it doesn't exist."""
    path = os.path.join(VAULT_DIR, folder)
    os.makedirs(path, exist_ok=True)


def _now_sgt() -> datetime:
    return datetime.now(_SGT)


def _timestamp() -> str:
    return _now_sgt().strftime("%Y-%m-%dT%H:%M:%S+08:00")


def _date_prefix() -> str:
    return _now_sgt().strftime("%Y-%m-%d")


def write_note(folder: str, title: str, frontmatter: dict, body: str) -> str | None:
    """Write a markdown note to the vault.

    Args:
        folder: Subfolder (e.g. "20-Incidents", "30-Trades")
        title: Note title (used in filename)
        frontmatter: YAML frontmatter dict (type, tags, status, etc.)
        body: Markdown body content

    Returns:
        File path if written, None on error.
    """
    try:
        _ensure_dir(folder)

        # Build frontmatter
        fm_lines = ["---"]
        for key, value in frontmatter.items():
            if isinstance(value, list):
                fm_lines.append(f"{key}:")
                for item in value:
                    fm_lines.append(f"  - {item}")
            else:
                fm_lines.append(f"{key}: {value}")
        fm_lines.append("---")

        content = "\n".join(fm_lines) + "\n\n" + body.strip() + "\n"

        # Sanitize filename
        safe_title = title.replace("/", "-").replace("\\", "-").replace(":", "-")
        filename = f"{_date_prefix()} {safe_title}.md"
        filepath = os.path.join(VAULT_DIR, folder, filename)

        with open(filepath, "w") as f:
            f.write(content)

        log.info("Vault note written: %s/%s", folder, filename)
        return filepath
    except Exception as e:
        log.warning("Failed to write vault note: %s", e)
        return None


def write_incident(title: str, what: str, root_cause: str, fix: str,
                   tags: list[str] | None = None, severity: str = "medium") -> str | None:
    """Write an incident note."""
    return write_note(
        folder="20-Incidents",
        title=title,
        frontmatter={
            "type": "incident",
            "created": _timestamp(),
            "tags": tags or [],
            "status": "active",
            "severity": severity,
        },
        body=f"# {title}\n\n## What happened\n{what}\n\n## Root cause\n{root_cause}\n\n## Fix\n{fix}",
    )


def write_trade(asset: str, direction: str, entry_price: float,
                thesis: str, verdict: str, outcome: str,
                tags: list[str] | None = None) -> str | None:
    """Write a trade journal note."""
    return write_note(
        folder="30-Trades",
        title=f"{asset} {direction} {outcome}",
        frontmatter={
            "type": "trade",
            "created": _timestamp(),
            "tags": [f"asset/{asset}"] + (tags or []),
            "status": "active",
        },
        body=(
            f"# {asset} {direction.upper()}\n\n"
            f"**Entry**: ${entry_price:.2f}\n"
            f"**Direction**: {direction}\n"
            f"**Outcome**: {outcome}\n\n"
            f"## Thesis\n{thesis}\n\n"
            f"## Verdict\n{verdict}"
        ),
    )


def write_decision(title: str, context: str, decision: str,
                   tags: list[str] | None = None) -> str | None:
    """Write a decision log note."""
    return write_note(
        folder="10-Decisions",
        title=title,
        frontmatter={
            "type": "decision",
            "created": _timestamp(),
            "tags": tags or [],
            "status": "active",
        },
        body=f"# {title}\n\n## Context\n{context}\n\n## Decision\n{decision}",
    )
