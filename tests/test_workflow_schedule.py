"""The daily cycle's schedule must be written in UTC and give evening attempts after the cutoff.

GitHub ran the `timezone:`-annotated crons as UTC, so an "18:23 ET" attempt fired before the close.
"""
import re
from pathlib import Path

WORKFLOW = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "daily.yml"


def _schedule():
    # Plain-text parse (no YAML dependency): every `- cron: '...'` line in the workflow.
    crons = re.findall(r"^\s*-\s*cron:\s*'([^']+)'", WORKFLOW.read_text(), re.M)
    assert crons, "no cron lines found"
    return [{"cron": c} for c in crons]


def test_daily_crons_are_utc_without_timezone_key():
    assert not re.search(r"^\s*timezone:", WORKFLOW.read_text(), re.M)


def test_evening_attempts_after_cutoff_in_both_est_and_edt():
    # Cutoff is 17:30 ET = 22:30 UTC (EST) / 21:30 UTC (EDT). Require two weekday attempts
    # between 22:30 and 23:59 UTC so the forecast can freeze early in the evening all year.
    evening = []
    for entry in _schedule():
        minute, hour, _, _, dow = entry["cron"].split()
        for h in hour.split(","):
            t = int(h) * 60 + int(minute)
            if 22 * 60 + 30 <= t < 24 * 60 and dow == "1-5":
                evening.append(t)
    assert len(evening) >= 2, evening


def test_a_morning_attempt_before_the_open_in_both_offsets():
    # 09:30 ET open = 14:30 UTC (EST) / 13:30 UTC (EDT); last retry must start well before 13:30 UTC.
    last = max(int(e["cron"].split()[1].split(",")[-1]) * 60 + int(e["cron"].split()[0])
               for e in _schedule() if int(e["cron"].split()[1].split(",")[-1]) < 14)
    assert last <= 13 * 60, last
