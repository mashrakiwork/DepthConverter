"""Persistent app settings (folders, model choice, encoder options...).

Stored as JSON in the user's home directory so every field the user picks
survives restarts.
"""

import json
from pathlib import Path

CONFIG_DIR = Path.home() / ".depthconverter"
CONFIG_FILE = CONFIG_DIR / "config.json"


class Config:
    def __init__(self):
        self._data: dict = {}
        self._salvage_first_save = False
        try:
            # utf-8-sig: a file rewritten by another tool may carry a BOM.
            self._data = json.loads(CONFIG_FILE.read_text(encoding="utf-8-sig"))
        except OSError:
            pass
        except ValueError:
            # Damaged file. Starting from empty settings is fine, but the
            # first save would overwrite it - keep a copy so nothing the user
            # picked is lost for good.
            self._salvage_first_save = True

    def get(self, key: str, default=None):
        return self._data.get(key, default)

    def set(self, key: str, value) -> None:
        if self._data.get(key) == value:
            return
        self._data[key] = value
        self.save()

    def save(self) -> None:
        try:
            CONFIG_DIR.mkdir(parents=True, exist_ok=True)
            if self._salvage_first_save:
                self._salvage_first_save = False
                CONFIG_FILE.replace(CONFIG_FILE.with_suffix(".json.bak"))
            CONFIG_FILE.write_text(json.dumps(self._data, indent=2), encoding="utf-8")
        except OSError:
            pass
