from pathlib import Path
import yaml


class DotDict(dict):
    """支持点号访问的字典"""

    def __getattr__(self, name):
        value = self.get(name)
        if isinstance(value, dict):
            return DotDict(value)
        return value

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class Config:
    def __init__(self, path: str):
        self.path = Path(path)
        self.cfg = self._load()

    def _load(self):
        if not self.path.exists():
            raise FileNotFoundError(self.path)
        with open(self.path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return DotDict(data)

    def get(self, key, default=None):
        return self.cfg.get(key, default)
