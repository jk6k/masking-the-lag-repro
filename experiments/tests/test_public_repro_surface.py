from pathlib import Path

from scripts.check_public_repro_repo import validate


def test_public_repro_surface_has_no_errors():
    report = validate(Path(__file__).resolve().parents[2])
    assert report.errors == [], report.render()
