"""CorruptPoliceBench: a multi-agent deception benchmark based on the
Day-1 main match of The Devil's Plan: Death Room (Netflix, 2025)."""
from .config import GameConfig
from .board import Board, make_board
from .engine import Game, COP, CORRUPT, THIEF
from .env import CorruptPoliceEnv
from .runner import run_episode

__all__ = ["GameConfig", "Board", "make_board", "Game", "CorruptPoliceEnv", "run_episode",
           "COP", "CORRUPT", "THIEF"]
__version__ = "0.1.0"
