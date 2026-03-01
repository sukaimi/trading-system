"""Self-Optimizer — translates weekly strategy directives into config changes.

Takes WeeklyDirective from the Weekly Strategist and updates agent
parameter JSON files. Supports rollback to previous versions.
Phase 1 skeleton — full Opus integration in Phase 3.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any

from core.logger import setup_logger

log = setup_logger("trading.optimizer")

CONFIG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config")
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
OPT_LOG_FILE = os.path.join(DATA_DIR, "optimization_log.json")
VERSION_FILE = os.path.join(CONFIG_DIR, "strategy_version.json")


class SelfOptimizer:
    """Translates Opus strategy directives into actual config changes."""

    def apply_directives(self, weekly_directive: dict[str, Any]) -> list[dict[str, Any]]:
        """Apply parameter changes from a weekly directive.

        Args:
            weekly_directive: Dict with 'parameter_changes' list and 'week_reviewed'.

        Returns:
            List of change records applied.
        """
        changes_applied: list[dict[str, Any]] = []

        for change in weekly_directive.get("parameter_changes", []):
            target_agent = change["target"]
            param_path = change["parameter"]
            new_value = change["new_value"]
            old_value = change["old_value"]

            self._update_config(target_agent, param_path, new_value)
            version = self._increment_version()

            changes_applied.append(
                {
                    "timestamp": datetime.utcnow().isoformat(),
                    "agent": target_agent,
                    "parameter": param_path,
                    "old_value": old_value,
                    "new_value": new_value,
                    "reason": change.get("reason", ""),
                    "directive_source": weekly_directive.get("week_reviewed", ""),
                    "version": version,
                }
            )

        if changes_applied:
            self._append_to_log(changes_applied, weekly_directive)
            log.info("Applied %d parameter changes", len(changes_applied))

        return changes_applied

    def _update_config(self, agent: str, param_path: str, value: Any) -> None:
        """Update an agent's dynamic parameters JSON file."""
        config_file = os.path.join(CONFIG_DIR, f"{agent}_params.json")
        if not os.path.exists(config_file):
            log.warning("Config file not found: %s", config_file)
            return

        with open(config_file) as f:
            config = json.load(f)

        # Navigate nested path (e.g., "signal_weights.regulatory_crypto")
        keys = param_path.split(".")
        obj = config
        for key in keys[:-1]:
            obj = obj.setdefault(key, {})
        obj[keys[-1]] = value

        with open(config_file, "w") as f:
            json.dump(config, f, indent=2)

        log.info("Updated %s.%s = %s", agent, param_path, value)

    def _increment_version(self) -> int:
        """Increment and return the strategy version number."""
        if os.path.exists(VERSION_FILE):
            with open(VERSION_FILE) as f:
                v = json.load(f)
        else:
            v = {"version": 0}

        v["version"] += 1
        v["last_updated"] = datetime.utcnow().isoformat()

        with open(VERSION_FILE, "w") as f:
            json.dump(v, f, indent=2)

        return v["version"]

    def _append_to_log(
        self, changes: list[dict[str, Any]], directive: dict[str, Any]
    ) -> None:
        """Append optimization changes to the log file."""
        os.makedirs(DATA_DIR, exist_ok=True)

        log_entries: list[dict[str, Any]] = []
        if os.path.exists(OPT_LOG_FILE):
            with open(OPT_LOG_FILE) as f:
                log_entries = json.load(f)

        log_entries.append(
            {
                "version": changes[-1]["version"] if changes else 0,
                "timestamp": datetime.utcnow().isoformat(),
                "directive_week": directive.get("week_reviewed", ""),
                "changes": changes,
            }
        )

        with open(OPT_LOG_FILE, "w") as f:
            json.dump(log_entries, f, indent=2)

    def rollback(self, version_to_restore: int) -> bool:
        """Restore parameters from a previous version.

        Reads the optimization log, finds all changes after the target
        version, and reverses them in order.

        Returns True if rollback succeeded.
        """
        if not os.path.exists(OPT_LOG_FILE):
            log.warning("No optimization log found — nothing to rollback")
            return False

        with open(OPT_LOG_FILE) as f:
            log_entries = json.load(f)

        # Find entries to reverse (everything after target version)
        to_reverse = [
            entry
            for entry in log_entries
            if entry.get("version", 0) > version_to_restore
        ]

        if not to_reverse:
            log.info("No changes found after version %d", version_to_restore)
            return False

        # Reverse in newest-first order
        for entry in reversed(to_reverse):
            for change in reversed(entry.get("changes", [])):
                self._update_config(
                    change["agent"], change["parameter"], change["old_value"]
                )

        log.info("Rolled back to version %d (%d entries reversed)",
                 version_to_restore, len(to_reverse))
        return True
