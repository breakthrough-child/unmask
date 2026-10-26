"""Privacy rule: only sessions from data/<label>/divine_* folders may appear in public outputs."""
import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location("run_eval", Path(__file__).resolve().parents[1] / "eval" / "run_eval.py")
run_eval = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run_eval)


def clips(*paths):
    return {Path(p).stem: {"path": p} for p in paths}


def test_divine_sessions_are_public():
    assert run_eval.session_is_public(clips("real/divine_dim/passive.mp4", "real/divine_dim/head_turn.mp4"), "divine_")
    assert run_eval.session_is_public(clips("swapped/divine_daylight/hand_pass.mp4"), "divine_")


def test_other_people_are_aggregate_only():
    assert not run_eval.session_is_public(clips("real/ada_dim/passive.mp4"), "divine_")
    assert not run_eval.session_is_public(clips("real/divine_dim/passive.mp4", "real/ada_dim/head_turn.mp4"), "divine_")
    assert not run_eval.session_is_public(clips("real/p02/passive.mp4"), "divine_")


def test_malformed_or_flat_paths_are_not_public():
    assert not run_eval.session_is_public(clips("passive.mp4"), "divine_")
    assert not run_eval.session_is_public(clips("real/passive.mp4"), "divine_")


def test_no_prefix_means_everything_public_for_synthetic_runs():
    assert run_eval.session_is_public(clips("real/synth00/passive.mp4"), None)
