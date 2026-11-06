"""Exportable screening reports (Markdown and standalone printable HTML) from Screener.screen output."""
from __future__ import annotations

import datetime as dt
import html

DISCLAIMER = ("Educational research prototype. These are model predictions on the provided text only and "
              "must not be used to make clinical trial enrollment decisions.")
STATUS_TEXT = {"likely_eligible": "Likely eligible", "likely_ineligible": "Likely not eligible",
               "needs_review": "Needs review"}


def to_markdown(res: dict, title="Trial eligibility screening report", meta: dict | None = None) -> str:
    o = res["overall"]
    lines = [f"# {title}", "", f"*Generated {dt.datetime.now():%Y-%m-%d %H:%M}*  ", f"*{DISCLAIMER}*", ""]
    if meta:
        lines += [f"- **{k}:** {v}" for k, v in meta.items()] + [""]
    lines += [f"## Overall: {STATUS_TEXT.get(o['status'], o['status'])}", "", o["summary"], ""]
    if res.get("warnings"):
        lines += ["**Warnings:**"] + [f"- {w}" for w in res["warnings"]] + [""]
    lines += ["## Criteria", "", "| # | Type | Criterion | Decision | Evidence |", "|---|---|---|---|---|"]
    for r in res["results"]:
        ev = ", ".join(str(e) for e in r["evidence"]) or "none"
        crit = r["criterion_text"].replace("|", "\\|")
        lines.append(f"| {r['index'] + 1} | {r['criterion_type']} | {crit} | **{r['label']}** | {ev} |")
    lines += ["", "## Explanations", ""]
    for r in res["results"]:
        lines += [f"**{r['index'] + 1}. {r['criterion_text']}**  ", f"{r['explanation']}", ""]
    lines += ["## Patient note (as numbered for the model)", ""]
    lines += [f"{s['id']}. {s['text']}" for s in res["sentences"]]
    t = res["timing"]
    lines += ["", "## Run details", "",
              f"- Decoder: `{res['decoder']}`" + (f" (k={res['k']})" if res.get("k") else ""),
              f"- Total time: {t['total_s']:.2f} s for {len(res['results'])} criteria "
              f"({t['new_tokens']} generated tokens)"]
    return "\n".join(lines) + "\n"


def to_html(res: dict, title="Trial eligibility screening report", meta: dict | None = None) -> str:
    e = html.escape
    o = res["overall"]
    colors = {"likely_eligible": "#1f7a4d", "likely_ineligible": "#b3261e", "needs_review": "#8a5a00"}
    cited = {i for r in res["results"] for i in r["evidence"]}
    rows = "".join(
        f"<tr><td>{r['index'] + 1}</td><td>{e(r['criterion_type'])}</td><td>{e(r['criterion_text'])}</td>"
        f"<td><b>{e(r['label'])}</b></td><td>{e(', '.join(map(str, r['evidence'])) or 'none')}</td>"
        f"<td>{e(r['explanation'])}</td></tr>" for r in res["results"])
    note = "".join(f"<li value='{s['id']}'{' class=cited' if s['id'] in cited else ''}>{e(s['text'])}</li>"
                   for s in res["sentences"])
    meta_html = "".join(f"<div><b>{e(str(k))}:</b> {e(str(v))}</div>" for k, v in (meta or {}).items())
    warn = "".join(f"<li>{e(w)}</li>" for w in res.get("warnings", []))
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>{e(title)}</title>
<style>
body{{font-family:system-ui,Segoe UI,Arial,sans-serif;max-width:960px;margin:32px auto;padding:0 16px;color:#1b1f24}}
h1{{font-size:22px;margin:0 0 4px}} .muted{{color:#5b6470;font-size:13px}}
.banner{{border-left:6px solid {colors.get(o['status'], '#444')};background:#f6f7f9;padding:12px 16px;margin:16px 0}}
table{{border-collapse:collapse;width:100%;font-size:13px}} th,td{{border:1px solid #d6dae0;padding:6px 8px;vertical-align:top;text-align:left}}
th{{background:#eef1f4}} ol li{{margin:2px 0;font-size:13px}} li.cited{{background:#fff3c4}}
@media print{{body{{margin:0}} .banner{{-webkit-print-color-adjust:exact;print-color-adjust:exact}}}}
</style></head><body>
<h1>{e(title)}</h1><div class="muted">Generated {dt.datetime.now():%Y-%m-%d %H:%M} &middot; {e(DISCLAIMER)}</div>
{meta_html}
<div class="banner"><b>Overall: {e(STATUS_TEXT.get(o['status'], o['status']))}</b><br>{e(o['summary'])}</div>
{f'<ul>{warn}</ul>' if warn else ''}
<h2>Criteria</h2><table><tr><th>#</th><th>Type</th><th>Criterion</th><th>Decision</th><th>Evidence</th><th>Explanation</th></tr>{rows}</table>
<h2>Patient note</h2><p class="muted">Highlighted sentences were cited as evidence.</p><ol start="0">{note}</ol>
<p class="muted">Decoder: {e(res['decoder'])}{f" (k={res['k']})" if res.get('k') else ''} &middot;
{res['timing']['total_s']:.2f} s for {len(res['results'])} criteria</p>
</body></html>"""
