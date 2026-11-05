import pandas as pd

from unpriced.llm import reader


def test_guidance_rule_reads_common_phrasings():
    assert reader.rule_guidance("The company raised its full-year revenue guidance to $4.1 billion.")[0] == "raised"
    assert reader.rule_guidance("We are reaffirming our fiscal 2027 outlook.")[0] == "maintained"
    assert reader.rule_guidance("Given uncertainty, the company is withdrawing its full-year guidance.")[0] == "withdrawn"
    assert reader.rule_guidance("Management lowered its outlook for the second half.")[0] == "lowered"
    assert reader.rule_guidance("Revenue grew 12% on strong demand.")[0] == "none"
    assert reader.rule_guidance("We raised revenue guidance. We lowered margin guidance.")[0] == "unclear"


def test_signal_ignores_readings_finished_after_the_forecast(monkeypatch):
    rec = pd.DataFrame([
        {"ticker": "AAA", "guidance": "raised", "ruleGuidance": "raised", "reactionSession": "2026-11-04",
         "readAt": "2026-11-04T18:00:00-05:00"},
        {"ticker": "BBB", "guidance": "lowered", "ruleGuidance": "none", "reactionSession": "2026-11-04",
         "readAt": "2026-11-04T19:30:00-05:00"},
    ])
    monkeypatch.setattr(reader, "records", lambda since=None: rec)
    dates = pd.bdate_range("2026-10-20", "2026-11-04")
    s = reader.signal(pd.Timestamp("2026-11-04"), dates, source="llm", asof=pd.Timestamp("2026-11-04 19:00"))
    assert s.get("AAA", 0) > 0 and "BBB" not in s.index
    r = reader.signal(pd.Timestamp("2026-11-04"), dates, source="rule", asof=pd.Timestamp("2026-11-04 20:00"))
    assert r.get("AAA", 0) > 0 and r.get("BBB", 0) == 0
