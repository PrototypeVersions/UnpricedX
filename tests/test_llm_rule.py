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


def test_press_release_exhibits_are_found_by_document_type():
    from unpriced.data.sec import exhibits_from_index
    html = """<table class="tableFile"><tr><th>Seq</th><th>Description</th><th>Document</th><th>Type</th><th>Size</th></tr>
    <tr><td scope="row">1</td><td scope="row">8-K</td><td scope="row"><a href="/Archives/edgar/data/2488/000000248826000123/amd-20261104.htm">amd-20261104.htm</a></td><td scope="row">8-K</td><td>35123</td></tr>
    <tr><td scope="row">2</td><td scope="row">PRESS RELEASE</td><td scope="row"><a href="/Archives/edgar/data/2488/000000248826000123/q3-26pressrelease.htm">q3-26pressrelease.htm</a></td><td scope="row">EX-99.1</td><td>201234</td></tr>
    <tr><td scope="row">3</td><td scope="row">SLIDES</td><td scope="row"><a href="/ix?doc=/Archives/edgar/data/2488/000000248826000123/exhibit992.htm">exhibit992.htm</a></td><td scope="row">EX-99.2</td><td>9000</td></tr>
    <tr><td scope="row">4</td><td scope="row">GRAPHIC</td><td scope="row"><a href="/Archives/edgar/data/2488/000000248826000123/logo.jpg">logo.jpg</a></td><td scope="row">GRAPHIC</td><td>900</td></tr></table>"""
    assert exhibits_from_index(html) == ["q3-26pressrelease.htm", "exhibit992.htm"]
