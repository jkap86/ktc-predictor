"""Transition model service for week-by-week KTC trajectory predictions.

Provides trajectory rollouts and what-if scenarios using the transition model.
"""

from pathlib import Path

from app.config import TRANSITION_MODELS_DIR, TRANSITION_MODEL_VERSION
from ktc_model.predict_transition import (
    generate_what_if_trajectory,
    predict_end_ktc_via_rollout,
    predict_next_ktc,
    rollout_season,
)
from ktc_model.train_transition import load_transition_bundle


class TransitionModelService:
    """Week-to-week KTC transition model service.

    Loads transition models and provides trajectory predictions.
    """

    def __init__(self):
        self._bundle: dict | None = None
        self._initialized = False

    def initialize(self) -> dict:
        """Load transition model bundle."""
        models_dir = Path(TRANSITION_MODELS_DIR)
        if not models_dir.exists():
            raise FileNotFoundError(
                f"Transition model directory not found: {models_dir}. "
                "Run `python -m ktc_model.train_transition` to train models."
            )

        required = ["QB.joblib", "RB.joblib", "WR.joblib", "TE.joblib"]
        missing = [f for f in required if not (models_dir / f).exists()]
        if missing:
            raise FileNotFoundError(
                f"Missing transition model files in {models_dir}: {missing}"
            )

        self._bundle = load_transition_bundle(str(models_dir))
        self._initialized = True

        return self._bundle.get("metrics", {})

    @property
    def bundle(self) -> dict:
        if not self._initialized:
            self.initialize()
        assert self._bundle is not None
        return self._bundle

    def predict_trajectory(
        self,
        player_id: str,
        data_loader,
    ) -> dict | None:
        """Predict KTC trajectory for a player using their weekly stats.

        Parameters
        ----------
        player_id : str
            Player ID to predict for.
        data_loader : DataLoaderService
            Data loader service to fetch player data.

        Returns
        -------
        dict or None
            {
                "player_id": str,
                "name": str,
                "position": str,
                "year": int,
                "start_ktc": float,
                "end_ktc": float,
                "delta_ktc": float,
                "trajectory": list[dict],  # [{week, ktc}, ...]
                "actual_weekly_ktc": list[dict],  # [{week, ktc}, ...] if available
            }
        """
        player = data_loader.get_player_by_id(player_id)
        if not player:
            return None

        position = player["position"]
        if position not in self.bundle["models"]:
            return None

        model = self.bundle["models"][position]
        clip_bounds = self.bundle["clip_bounds"].get(position, (-0.5, 0.5))

        seasons = player.get("seasons", [])
        if not seasons:
            return None

        # Use latest season
        latest = max(seasons, key=lambda s: s["year"])
        year = latest["year"]
        age = latest.get("age")

        # Get start KTC
        start_ktc = latest.get("start_ktc")
        if not start_ktc or start_ktc <= 0 or start_ktc >= 9999:
            return None

        # Get weekly stats
        weekly_stats = latest.get("weekly_stats", [])

        # Get actual weekly KTC for comparison
        actual_weekly_ktc = []
        for wk in latest.get("weekly_ktc", []):
            ktc = wk.get("ktc", 0)
            if ktc and ktc > 0 and ktc < 9999:
                actual_weekly_ktc.append({
                    "week": wk["week"],
                    "ktc": ktc,
                })

        # Run rollout
        result = predict_end_ktc_via_rollout(
            model=model,
            clip_bounds=clip_bounds,
            start_ktc=start_ktc,
            weekly_stats=weekly_stats,
            age=age,
            position=position,
        )

        return {
            "player_id": player_id,
            "name": player["name"],
            "position": position,
            "year": year,
            "start_ktc": result["start_ktc"],
            "end_ktc": result["end_ktc"],
            "delta_ktc": result["delta_ktc"],
            "delta_pct": result["delta_pct"],
            "trajectory": result["trajectory"],
            "actual_weekly_ktc": actual_weekly_ktc,
            "model_version": TRANSITION_MODEL_VERSION,
        }

    def predict_what_if(
        self,
        player_id: str,
        data_loader,
        target_ppg: float,
        target_games: int,
        current_week: int = 1,
    ) -> dict | None:
        """Generate what-if trajectory for custom PPG/games scenario.

        Parameters
        ----------
        player_id : str
            Player ID to predict for.
        data_loader : DataLoaderService
            Data loader service to fetch player data.
        target_ppg : float
            Target PPG for the scenario.
        target_games : int
            Target total games for the scenario.
        current_week : int
            Week to start the simulation from.

        Returns
        -------
        dict or None
            Same structure as predict_trajectory.
        """
        player = data_loader.get_player_by_id(player_id)
        if not player:
            return None

        position = player["position"]
        if position not in self.bundle["models"]:
            return None

        model = self.bundle["models"][position]
        clip_bounds = self.bundle["clip_bounds"].get(position, (-0.5, 0.5))

        seasons = player.get("seasons", [])
        if not seasons:
            return None

        latest = max(seasons, key=lambda s: s["year"])
        age = latest.get("age")

        # Get start KTC (could use current KTC if mid-season)
        start_ktc = latest.get("start_ktc")
        if not start_ktc or start_ktc <= 0 or start_ktc >= 9999:
            return None

        trajectory = generate_what_if_trajectory(
            model=model,
            clip_bounds=clip_bounds,
            start_ktc=start_ktc,
            target_ppg=target_ppg,
            target_games=target_games,
            age=age,
            position=position,
            current_week=current_week,
        )

        end_ktc = trajectory[-1]["ktc"] if trajectory else start_ktc
        delta_ktc = end_ktc - start_ktc

        return {
            "player_id": player_id,
            "name": player["name"],
            "position": position,
            "start_ktc": start_ktc,
            "end_ktc": end_ktc,
            "delta_ktc": delta_ktc,
            "delta_pct": (delta_ktc / start_ktc * 100) if start_ktc > 0 else 0,
            "trajectory": trajectory,
            "scenario": {
                "ppg": target_ppg,
                "games": target_games,
                "current_week": current_week,
            },
            "model_version": TRANSITION_MODEL_VERSION,
        }

    def predict_next_week(
        self,
        position: str,
        ktc_current: float,
        ppg_cumulative: float,
        games_played: int,
        week: int,
        weekly_fp: float = 0.0,
        games_this_week: int = 0,
        age: float | None = None,
    ) -> dict:
        """Predict KTC for next week from raw inputs.

        This is the low-level single-step prediction for API use.
        """
        if position not in self.bundle["models"]:
            raise ValueError(f"No model for position: {position}")

        model = self.bundle["models"][position]
        clip_bounds = self.bundle["clip_bounds"].get(position, (-0.5, 0.5))

        result = predict_next_ktc(
            model=model,
            clip_bounds=clip_bounds,
            ktc_current=ktc_current,
            ppg_cumulative=ppg_cumulative,
            games_played=games_played,
            week=week,
            weekly_fp=weekly_fp,
            games_this_week=games_this_week,
            age=age,
            position=position,
        )

        result["position"] = position
        result["week"] = week
        result["model_version"] = TRANSITION_MODEL_VERSION

        return result
