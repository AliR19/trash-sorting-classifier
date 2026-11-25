"""Downstream step: turn the predicted material class into the bin the item goes in."""
import json
from pathlib import Path

DEFAULT_RULES = Path(__file__).resolve().parents[2] / "bins.json"


class BinRules:
    def __init__(self, path=None):
        rules = json.loads(Path(path or DEFAULT_RULES).read_text())
        self.rules = {k: v for k, v in rules.items() if not k.startswith("_")}

    def lookup(self, label: str) -> dict:
        return self.rules[label]
