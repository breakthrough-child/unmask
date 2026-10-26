import csv
import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location("make_labels", Path(__file__).resolve().parents[1] / "scripts" / "make_labels.py")
ml = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ml)


def touch(root, label, session, clip):
    p = root / label / session / f"{clip}.mp4"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"x")


def test_owner_is_p01_and_others_are_pseudonymised(tmp_path):
    for label in ("real", "swapped"):
        for s in ("divine_dim", "ada_daylight", "bola_call_whatsapp"):
            touch(tmp_path, label, s, "passive")
    rows, mapping, warnings = ml.build_rows(tmp_path, "divine")
    by_session = {r["path"].split("/")[1]: r for r in rows}
    assert by_session["divine_dim"]["participant"] == "p01"
    assert {by_session["ada_daylight"]["participant"], by_session["bola_call_whatsapp"]["participant"]} == {"p02", "p03"}
    assert mapping["p01"] == "divine" and not warnings
    assert by_session["bola_call_whatsapp"]["condition"] == "call_whatsapp" and by_session["bola_call_whatsapp"]["via_call"] == "yes"
    assert by_session["divine_dim"]["via_call"] == "no"
    assert {r["label"] for r in rows} == {"real", "swapped"}
    assert not any("ada" in v or "bola" in v for r in rows for v in (r["participant"],))  # no real names in the column


def test_missing_counterpart_is_warned(tmp_path):
    touch(tmp_path, "real", "divine_dim", "passive")
    touch(tmp_path, "real", "divine_dim", "head_turn")
    touch(tmp_path, "swapped", "divine_dim", "passive")
    rows, _, warnings = ml.build_rows(tmp_path, "divine")
    assert len(rows) == 3
    assert any("swapped/divine_dim/head_turn" in w for w in warnings)   # real exists, swapped missing
    touch(tmp_path, "swapped", "divine_dim", "hand_pass")
    _, _, warnings = ml.build_rows(tmp_path, "divine")
    assert any("real/divine_dim/hand_pass" in w for w in warnings)      # swapped exists, real missing
