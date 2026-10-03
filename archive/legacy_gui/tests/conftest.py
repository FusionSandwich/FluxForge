"""Explicit archival regression setup; excluded from normal test discovery."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "archive/legacy_gui/src"))
