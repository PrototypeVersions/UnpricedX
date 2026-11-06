"""AI filing reader: Claude reads new SEC filings and records a structured assessment.

This signal is *forward-only*. A language model has read the internet up to its training
cutoff, so asking it about filings from before that date can leak what happened next (see
He, Lv, Manela & Wu, "Chronologically Consistent Large Language Models", 2025). Backtests of
this signal are therefore not trusted; it is judged only on assessments frozen before the
outcome, exactly like the other live forecasts.

Evidence to temper expectations: Lopez-Lira & Tang (2023-2025) find LLM news signals predict
next-day returns, mostly for small stocks and bad news, decaying from 2021 to 2024 as adoption
grew, and unprofitable at ~20 bps round-trip costs. Filings are slower and longer than headlines,
which is where reading at scale might still add something.

One narrow, testable job (decided 2026-11-05, before any read): in earnings releases (8-K item
2.02 and its press-release exhibit), did management raise, maintain, lower or withdraw guidance,
and which sentence says so? The reader may answer "unclear". A simple keyword rule reads the same
text, so the live test is whether the AI adds anything beyond a cheap rule. Both feed separate
shadow challengers; neither changes the champion or the paper portfolio.

Timing: a reading can only affect a forecast frozen after the reading itself (readAt), and is
judged from the first open after it was read.

Off unless ANTHROPIC_API_KEY and SEC access are both configured.
"""
from __future__ import annotations

import json
import os
import re
import time

import pandas as pd
import requests

from .. import store
from ..config import paths, section
from ..util import append_jsonl, iso, log, now_et, read_jsonl

L = log("llm")
API = "https://api.anthropic.com/v1/messages"

TOOL = {
    "name": "record_assessment",
    "description": "Record a structured assessment of one SEC filing.",
    "input_schema": {
        "type": "object",
        "properties": {
            "event_type": {"type": "string", "enum": ["earnings", "guidance", "merger_acquisition", "management_change",
                                                      "financing", "legal_regulatory", "restatement_or_auditor",
                                                      "material_contract", "periodic_report", "other"]},
            "direction": {"type": "integer", "minimum": -2, "maximum": 2,
                          "description": "Likely effect on the stock relative to the market over the next week: -2 very negative ... +2 very positive."},
            "surprise": {"type": "integer", "minimum": -2, "maximum": 2,
                         "description": "How far the content is from what investors most likely expected beforehand (0 = in line or routine)."},
            "guidance": {"type": "string", "enum": ["raised", "maintained", "lowered", "withdrawn", "initiated", "none", "unclear"],
                         "description": "Change in management's forward guidance in this filing. 'none' if the filing gives no guidance; 'unclear' if you cannot tell."},
            "guidance_evidence": {"type": "string", "description": "The exact sentence from the text that supports the guidance answer, copied verbatim; empty if none or unclear."},
            "materiality": {"type": "number", "minimum": 0, "maximum": 1, "description": "How much this filing matters for the company's value."},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1, "description": "Confidence in the direction call."},
            "summary": {"type": "string", "description": "At most 40 words, plain English."},
        },
        "required": ["event_type", "guidance", "guidance_evidence", "direction", "surprise", "materiality", "confidence", "summary"],
    },
}

SYSTEM = (
    "You are an equity analyst reading a newly filed earnings release. Judge only from the text provided. "
    "You do not know how the stock reacted. Your main task: did management raise, maintain, lower, withdraw or "
    "first provide forward guidance (revenue, earnings, margins or similar) compared with what it said before, as "
    "stated in this text? Quote the one sentence that supports your answer. If the text does not let you tell, "
    "answer 'unclear'; if it gives no guidance, answer 'none'. Do not guess. Then give the other fields; be "
    "calibrated: most filings deserve direction 0."
)

# The cheap rule the AI must beat: keyword patterns on the same text.
_GUIDE = r"(?:guidance|outlook|forecast|full[- ]year (?:expectations|targets?))"
_RULES = [
    ("withdrawn", rf"\b(?:withdr[ae]w[sn]?|withdrawing|suspend(?:s|ed|ing)?|no longer (?:providing|expects?))\b[^.]{{0,80}}{_GUIDE}"),
    ("lowered", rf"\b(?:lower(?:s|ed|ing)?|reduc(?:e|es|ed|ing)|cut(?:s|ting)?|narrow(?:s|ed)? (?:the )?(?:low|bottom))\b[^.]{{0,80}}{_GUIDE}|{_GUIDE}[^.]{{0,40}}\b(?:lowered|reduced|cut)\b"),
    ("raised", rf"\b(?:rais(?:e|es|ed|ing)|increas(?:e|es|ed|ing)|boost(?:s|ed)?|lift(?:s|ed)?|improv(?:e|es|ed))\b[^.]{{0,80}}{_GUIDE}|{_GUIDE}[^.]{{0,40}}\b(?:raised|increased)\b"),
    ("maintained", rf"\b(?:reaffirm(?:s|ed|ing)?|reiterat(?:e|es|ed|ing)|maintain(?:s|ed|ing)?|confirm(?:s|ed|ing)?)\b[^.]{{0,80}}{_GUIDE}"),
]


def rule_guidance(text: str) -> tuple[str, str]:
    """Keyword baseline: the first sentence matching a guidance pattern, in priority order
    (withdrawn, lowered, raised, maintained). Raised and lowered both present counts as unclear."""
    sentences = re.split(r"(?<=[.!?])\s+", re.sub(r"\s+", " ", text))
    found = {}
    for label, pat in _RULES:
        rx = re.compile(pat, re.I)
        hit = next((s for s in sentences if rx.search(s)), None)
        if hit:
            found[label] = hit[:400]
    if "raised" in found and "lowered" in found:
        return "unclear", ""
    for label, _ in _RULES:
        if label in found:
            return label, found[label]
    return "none", ""


GUIDANCE_SCORE = {"raised": 1.0, "lowered": -1.0, "withdrawn": -1.0}


def api_key() -> str | None:
    return os.environ.get(section("llm")["apiKeyEnvVar"]) or None


def enabled() -> bool:
    cfg = section("llm")
    return bool(cfg.get("enabled")) and api_key() is not None and bool(os.environ.get(section("sec")["contactEnvVar"]))


def records(since: str | None = None) -> pd.DataFrame:
    rows = []
    folder = paths().live / "llm"
    for f in sorted(folder.glob("*.jsonl")) if folder.exists() else []:
        if since and f.stem < since:
            continue
        rows += read_jsonl(f)
    return pd.DataFrame(rows)


def _mdna(text: str) -> str:
    m = re.search(r"(?i)item\s*[27]\.?\s*management.s discussion and analysis", text)
    return text[m.start():] if m else text


def filing_text(sec, row) -> str:
    cfg = section("llm")
    limit = int(cfg["maxCharsPerFiling"])
    main = sec.document_text(row.cik, row.accession, row.primary_doc, max_chars=400_000) if row.primary_doc else ""
    if row.form in ("10-Q", "10-K"):
        return _mdna(main)[:limit]
    # the 8-K body is mostly a cover page; the results and guidance are in the EX-99 press release
    parts = [main[: limit // 8]]
    for name in sec.exhibits(row.cik, row.accession):
        if name != row.primary_doc:
            parts.append(sec.document_text(row.cik, row.accession, name, max_chars=limit))
    return "\n\n---\n\n".join(p for p in parts if p)[:limit]


def assess(text: str, row, model: str) -> dict:
    prompt = (f"Company: {row.ticker}\nForm: {row.form}\nItems: {row.items or 'n/a'}\nFiled: {row.filing_date}\n\n"
              f"Document text (may be truncated):\n<<<\n{text}\n>>>\n\nRecord your assessment with the tool.")
    body = {"model": model, "max_tokens": 700, "system": SYSTEM, "tools": [TOOL],
            "tool_choice": {"type": "tool", "name": TOOL["name"]}, "messages": [{"role": "user", "content": prompt}]}
    headers = {"x-api-key": api_key(), "anthropic-version": "2023-06-01", "content-type": "application/json"}
    for attempt in range(4):
        r = requests.post(API, headers=headers, data=json.dumps(body), timeout=120)
        if r.status_code == 200:
            js = r.json()
            for block in js.get("content", []):
                if block.get("type") == "tool_use":
                    return {**block["input"], "usage": js.get("usage", {})}
            raise RuntimeError("no tool_use block in response")
        if r.status_code in (429, 500, 502, 503, 529):
            time.sleep(5 * 2 ** attempt)
            continue
        raise RuntimeError(f"Claude API error {r.status_code}: {r.text[:200]}")
    raise RuntimeError("Claude API kept failing")


def run(session: pd.Timestamp, universe: list[str], liq_rank: pd.Series) -> dict:
    """Read filings that became public up to `session` and were not read before."""
    if not enabled():
        return {"enabled": False}
    from ..data.sec import SecClient, reaction_session
    from ..calendar import sessions
    cfg = section("llm")
    sec = SecClient()
    filings = store.read_or_empty("sec_filings")
    tick = store.read_or_empty("sec_tickers")
    if filings.empty or tick.empty:
        return {"enabled": True, "read": 0, "note": "no filings in store"}
    done = set(records()["accession"]) if len(records()) else set()
    cik_t = tick.drop_duplicates("cik").set_index("cik")["ticker"]
    f = filings.copy()
    f["ticker"] = f["cik"].map(cik_t)
    f = f[f["ticker"].isin(set(universe)) & f["form"].isin(cfg["forms"]) & ~f["accession"].isin(done)]
    f = f[pd.to_datetime(f["filing_date"]) >= pd.Timestamp(session) - pd.Timedelta(days=5)]
    is8k = f["form"] == "8-K"
    f = f[~is8k | f["items"].fillna("").apply(lambda s: any(i in s for i in cfg["eightKItems"]))]
    cal = sessions(pd.Timestamp(session) - pd.Timedelta(days=10), pd.Timestamp(session) + pd.Timedelta(days=10))
    f["session"] = reaction_session(f["filing_date"], f["acceptance"], cal)
    f = f[f["session"].notna() & (f["session"] <= pd.Timestamp(session) + pd.Timedelta(days=3))]
    f["liq"] = f["ticker"].map(liq_rank).fillna(1e6)
    f = f.sort_values(["liq"]).head(int(cfg["maxFilingsPerDay"]))
    out_path = paths().live / "llm" / f"{pd.Timestamp(session):%Y-%m-%d}.jsonl"
    n, tokens_in, tokens_out = 0, 0, 0
    started, budget = time.monotonic(), 60 * float(cfg.get("maxMinutes", 15))
    skipped = 0
    for row in f.itertuples():
        if time.monotonic() - started > budget:   # never let reading delay the forecast
            skipped += 1
            continue
        try:
            text = filing_text(sec, row)
            if len(text) < 200:
                continue
            a = assess(text, row, cfg["model"])
        except Exception as exc:
            L.warning("LLM read failed for %s %s: %s", row.ticker, row.accession, exc)
            continue
        usage = a.pop("usage", {})
        tokens_in += int(usage.get("input_tokens", 0))
        tokens_out += int(usage.get("output_tokens", 0))
        rule, rule_ev = rule_guidance(text)
        append_jsonl(out_path, {"accession": row.accession, "ticker": row.ticker, "form": row.form, "items": row.items,
                                "filingDate": str(row.filing_date), "reactionSession": pd.Timestamp(row.session).date().isoformat(),
                                "readAt": iso(now_et()), "model": cfg["model"], "promptVersion": cfg.get("promptVersion"),
                                "chars": len(text), "inputTokens": int(usage.get("input_tokens", 0)),
                                "outputTokens": int(usage.get("output_tokens", 0)),
                                "ruleGuidance": rule, "ruleEvidence": rule_ev, **a})
        n += 1
    L.info("LLM read %d filings (%d in / %d out tokens)", n, tokens_in, tokens_out)
    return {"enabled": True, "eligible": int(len(f)), "read": n, "deferredForTime": skipped,
            "inputTokens": tokens_in, "outputTokens": tokens_out, "model": cfg["model"], "promptVersion": cfg.get("promptVersion")}


def signal(session: pd.Timestamp, dates: pd.DatetimeIndex, source: str = "llm", asof=None) -> pd.Series:
    """Per-ticker guidance score (+1 raised, -1 lowered or withdrawn, else 0) from readings finished
    before `asof` (default: now, i.e. the moment the forecast is frozen). `source` is "llm" for the
    AI's reading or "rule" for the keyword baseline on the same documents. Older readings decay linearly.
    """
    cfg = section("llm")
    rec = records()
    if rec.empty:
        return pd.Series(dtype=float)
    asof = pd.Timestamp(asof or now_et())
    if asof.tzinfo is None:
        asof = asof.tz_localize("America/New_York")
    rec = rec[pd.to_datetime(rec["readAt"], utc=True) <= asof.tz_convert("UTC")].copy()
    col = "guidance" if source == "llm" else "ruleGuidance"
    if rec.empty or col not in rec:
        return pd.Series(dtype=float)
    rec["reactionSession"] = pd.to_datetime(rec["reactionSession"])
    decay = int(cfg["signalDecaySessions"])
    pos_now = dates.get_indexer([pd.Timestamp(session)])[0]
    if pos_now < 0:
        return pd.Series(dtype=float)
    first = dates[max(0, pos_now - decay + 1)]
    horizon_end = pd.Timestamp(session) + pd.Timedelta(days=5)
    r = rec[(rec["reactionSession"] >= first) & (rec["reactionSession"] <= horizon_end)].copy()
    if r.empty:
        return pd.Series(dtype=float)
    idx = dates.get_indexer(r["reactionSession"])
    age = pd.Series(pos_now - idx, index=r.index).where(idx >= 0, 0).clip(lower=0)
    r["w"] = 1 - age / decay
    r["s"] = r[col].map(GUIDANCE_SCORE).fillna(0.0) * r["w"]
    return r.groupby("ticker")["s"].sum()
