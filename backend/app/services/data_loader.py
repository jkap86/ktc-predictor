"""Shared data loader for training data JSON (supports .json and .zip)."""

import json
import zipfile
from pathlib import Path
from typing import Optional

from app.config import TRAINING_DATA_PATH


class DataLoader:
    def __init__(self, data_path: Optional[Path] = None):
        self.data_path = data_path or TRAINING_DATA_PATH
        self._raw_data: Optional[dict] = None
        self._player_index: Optional[dict[str, dict]] = None

    def load(self) -> dict:
        if self._raw_data is None:
            path = self.data_path
            # Try zip first, fall back to json
            zip_path = path.with_suffix(".zip") if path.suffix != ".zip" else path
            json_path = path.with_suffix(".json") if path.suffix != ".json" else path

            if zip_path.exists():
                with zipfile.ZipFile(zip_path) as zf:
                    json_name = json_path.name
                    with zf.open(json_name) as f:
                        self._raw_data = json.load(f)
            elif json_path.exists():
                with open(json_path, "r") as f:
                    self._raw_data = json.load(f)
            else:
                raise FileNotFoundError(
                    f"Training data not found at {zip_path} or {json_path}"
                )
        return self._raw_data

    def get_players(self) -> list[dict]:
        data = self.load()
        return data.get("players", [])

    def get_player_by_id(self, player_id: str) -> Optional[dict]:
        if self._player_index is None:
            self._player_index = {p["player_id"]: p for p in self.get_players()}
        return self._player_index.get(player_id)


# Singleton
_data_loader: Optional[DataLoader] = None


def get_data_loader() -> DataLoader:
    global _data_loader
    if _data_loader is None:
        _data_loader = DataLoader()
    return _data_loader
