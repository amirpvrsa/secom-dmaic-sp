"""End-to-end smoke test: synthetic SECOM-format files -> full report.
Checks that the pipeline runs and that the honesty rules hold on the rendered page."""
import re
import subprocess
import sys

import numpy as np
import pandas as pd
import yaml

from src.config import REPO_ROOT, load_config


def _write_fake_secom(raw_dir, n=400, p=30, seed=0):
    rng = np.random.default_rng(seed)
    fail = rng.uniform(size=n) < 0.1
    X = rng.normal(size=(n, p))
    X[:, 0] += 1.5 * fail                     # one sensor associated with fails
    X[rng.uniform(size=(n, p)) < 0.03] = np.nan
    X[:, 5] = np.nan                          # all missing
    X[:, 6] = 3.0                             # constant
    lines = [" ".join("NaN" if np.isnan(v) else f"{v:.5f}" for v in row) for row in X]
    (raw_dir / "secom.data").write_text("\n".join(lines) + "\n")
    times = pd.date_range("2008-07-19", periods=n, freq="37min")
    (raw_dir / "secom_labels.data").write_text("".join(
        f'{1 if f else -1} "{t:%d/%m/%Y %H:%M:%S}"\n' for f, t in zip(fail, times)))


def test_full_pipeline_renders_honest_report(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    _write_fake_secom(raw)
    cfg = load_config()
    cfg["data"].update(raw_dir=str(raw), cache_dir=str(tmp_path / "cache"))
    cfg["outputs"]["dir"] = str(tmp_path / "out")
    cfg["analyze"]["bootstrap_n"] = 10
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))

    subprocess.run([sys.executable, str(REPO_ROOT / "run.py"), "--config", str(cfg_path)],
                   check=True, cwd=REPO_ROOT)

    html = (tmp_path / "out" / cfg["report"]["file"]).read_text()
    text = re.sub(r"<[^>]+>", " ", re.sub(r"<style>.*?</style>", "", html, flags=re.S))
    for phase in ("Define", "Measure", "Analyze", "Improve", "Control"):
        assert phase in text
    assert "Specification limits are assumed" in text      # assumption stated on the page
    assert "Observational data" in text
    lowered = text.lower()
    assert "doe" not in re.findall(r"[a-z]+", lowered)        # never called a DOE
    assert "design of experiments" not in lowered
    for phrase in ("causes failure", "caused by", "drives failure", "root cause is"):
        assert phrase not in lowered
