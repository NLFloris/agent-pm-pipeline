"""A small local website for the pipeline: drop in a trace file, watch it run, read every output.

    python -m agentpm.cli web            # then open http://127.0.0.1:8765

Runs on this machine only (bound to 127.0.0.1); nothing leaves it. Each run is
the `report` command (agentpm/report.py) on the uploaded file or on a path on
this machine. While it runs, the page shows the pipeline's own log as it is
written, the way the command line would. Reports are kept under --out
(default ~/agentpm_reports) with that log beside them.

Three demos fill the form in, one per corpus of the thesis, all bundled with
the repository: tau2-bench airline (examples/demos/tau2_airline), AgentHallu's
Magentic-One runs (examples/demos/agenthallu_magentic_one), and the harness
corpus with sub-agents (results/local_subagents/runs) with its reference model.

The format can be left on "detect": every ingester is tried in turn and the
first one that yields runs with events wins. Standard library only.
"""
from __future__ import annotations

import contextlib
import csv
import datetime
import email.parser
import email.policy
import html
import io
import json
import mimetypes
import os
import re
import shutil
import threading
import time
import traceback
import urllib.parse
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable, Optional

from .logbuild.objects import BUILTIN as BUILTIN_RULES, rules_from_json
from .reference import reference_from_dict
from .report import report

FILE_ORDER = ["otel", "tau2", "openai", "claude", "generic"]
DIR_ORDER = ["traceelephant", "agenthallu", "local_magentic"]
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEMOS = os.path.join(REPO, "examples", "demos")
_run_lock = threading.Lock()
_jobs: dict[str, dict[str, Any]] = {}

# noise that libraries print and nobody needs to read
_NOISE = re.compile(r"PM4Py|License|AGPL|Business use|Docs & Examples|processintelligence|^=+$|UserWarning|"
                    r"warnings\.warn|r4pm|rustxes|Community Version")


def _formats():
    from .cli import FORMATS
    return FORMATS


def load_any(path: str, fmt: str, say: Callable[[str], None] = lambda s: None) -> tuple[str, list]:
    """Load with the given format, or try every format that fits a file or a folder."""
    formats = _formats()
    if fmt != "detect":
        say(f"reading as {fmt}")
        return fmt, formats[fmt][0](path)
    order = DIR_ORDER if os.path.isdir(path) else FILE_ORDER
    errors = []
    for f in order:
        try:
            traces = formats[f][0](path)
        except Exception as exc:  # a wrong format raises; try the next
            say(f"  {f:14} no ({type(exc).__name__})")
            errors.append(f"{f}: {type(exc).__name__}")
            continue
        if traces and any(t.events for t in traces):
            say(f"  {f:14} yes")
            return f, traces
        say(f"  {f:14} no runs")
        errors.append(f"{f}: no runs")
    raise ValueError("no format recognised this input (" + "; ".join(errors) + "). Pick the format by hand.")


# ------------------------------------------------------------------ demos

def demos() -> list[dict[str, str]]:
    """The three corpora of the thesis; a demo whose files are missing is left out."""
    out = []
    tau2 = os.path.join(DEMOS, "tau2_airline", "claude-opus-4-5_high_airline_gpt-5.2_4trials.json")
    if os.path.exists(tau2):
        out.append({"key": "tau2", "label": "τ²-bench airline",
                    "note": "200 customer-service conversations by Claude Opus 4.5 (50 tasks, 4 trials each). "
                            "Every run whose task has gold actions is checked against them; users, reservations, "
                            "flights and payments become objects in the OCEL log.",
                    "path": tau2, "format": "tau2", "name": "Demo: τ²-bench airline",
                    "reference": "", "objects": "", "pictures": "1", "by_agent": "0"})
    ah = os.path.join(DEMOS, "agenthallu_magentic_one")
    if os.path.isdir(ah):
        out.append({"key": "agenthallu", "label": "AgentHallu, Magentic-One",
                    "note": "94 Magentic-One runs from AgentHallu: an orchestrator hands work to a web surfer, a "
                            "coder and a terminal, and each reports back. Every run has a human label "
                            "for whether and at which step a hallucination entered. No reference model, so no "
                            "conformance check.",
                    "path": ah, "format": "agenthallu", "name": "Demo: AgentHallu, Magentic-One",
                    "reference": "", "objects": "", "pictures": "1", "by_agent": "0"})
    runs = os.path.join(REPO, "results", "local_subagents", "runs")
    ref = os.path.join(REPO, "experiments", "local_agents", "claim_check_reference.json")
    if os.path.isdir(runs) and os.path.exists(ref):
        out.append({"key": "subagents", "label": "Own harness, sub-agents",
                    "note": "10 runs of one task type (claim_check) by a Magentic-One team on gpt-oss:120b, recorded "
                            "by the purpose-built harness. The Analyst starts two sub-agents, DataChecker and "
                            "Recalculator, which make their own tool calls. Checked against the claim_check "
                            "reference model, written before any run, with activities named by agent.",
                    "path": runs, "format": "local_magentic", "name": "Demo: own harness, sub-agents",
                    "reference": open(ref, encoding="utf-8").read().strip(), "objects": "", "pictures": "1",
                    "by_agent": "1"})
    return out


# ------------------------------------------------------------------ jobs

class _Tee(io.TextIOBase):
    """Collects what libraries print during a run into the job's log, one line at a time."""

    def __init__(self, add: Callable[[str], None]) -> None:
        self.add, self.buf = add, ""

    def write(self, s: str) -> int:
        self.buf += s.replace("\r", "\n")
        while "\n" in self.buf:
            line, self.buf = self.buf.split("\n", 1)
            line = line.rstrip()
            if line and not _NOISE.search(line):
                self.add(line, raw=True)
        return len(s)


def _start_job(form: dict[str, Any], out_root: str) -> str:
    jid = uuid.uuid4().hex[:12]
    job = {"lines": [], "progress": None, "status": "queued", "rid": None, "error": None, "t0": time.time()}
    _jobs[jid] = job

    def add(text: str, raw: bool = False) -> None:
        t = round(time.time() - job["t0"], 1)
        if raw and ("it/s" in text or "%|" in text or "s/it" in text):
            job["progress"] = {"t": t, "text": text, "raw": True}   # a bar redrawing itself: show live, keep last
            return
        if job.get("progress"):
            job["lines"].append(job["progress"])
            job["progress"] = None
        job["lines"].append({"t": t, "text": text, "raw": raw})

    threading.Thread(target=_run_job, args=(job, form, out_root, add), daemon=True).start()
    return jid


def _run_job(job: dict[str, Any], form: dict[str, Any], out_root: str, add) -> None:
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    upload = form.get("file")
    path = os.path.expanduser(form.get("path") or "")
    d = None
    try:
        if isinstance(upload, tuple) and upload[0] and upload[1]:
            base = os.path.basename(upload[0])
        elif path:
            if not os.path.exists(path):
                raise ValueError(f"no such file or folder on this machine: {path}")
            base = os.path.basename(path.rstrip("/"))
        else:
            raise ValueError("drop a file or give a path")
        rid = stamp + "_" + re.sub(r"[^A-Za-z0-9._-]+", "_", base)[:50]
        d = os.path.join(out_root, rid)
        os.makedirs(d, exist_ok=True)
        if isinstance(upload, tuple) and upload[0] and upload[1]:
            os.makedirs(os.path.join(d, "input"), exist_ok=True)
            path = os.path.join(d, "input", base)
            with open(path, "wb") as f:
                f.write(upload[1])
            add(f"received {base}, {len(upload[1]) / 1e6:.1f} MB")

        fmt = form.get("format") or "detect"
        cmd = f"$ agentpm report {base} --format {fmt}"
        ref = None
        if form.get("reference"):
            try:
                ref = reference_from_dict(json.loads(form["reference"]))
            except (ValueError, KeyError, TypeError) as exc:
                raise ValueError(f"the reference model is not valid JSON for a reference: {exc}")
            cmd += f" --reference {ref.name}.json"
        rules = None
        if form.get("objects"):
            try:
                rules = rules_from_json(json.loads(form["objects"]))
            except (ValueError, TypeError) as exc:
                raise ValueError(f"the object rules are not valid: {exc}")
            cmd += " --objects rules.json"
        noise = float(form.get("noise") or 0)
        pictures = form.get("pictures", "1") == "1"
        by_agent = ref is not None and form.get("by_agent") == "1"
        cmd += ("" if noise == 0 else f" --noise {noise}") + ("" if pictures else " --no-pictures")
        cmd += " --by-agent" if by_agent else ""
        add(cmd)

        if _run_lock.locked():
            add("waiting for the run before this one to finish")
        with _run_lock:
            job["status"] = "running"
            tee = _Tee(add)
            with contextlib.redirect_stdout(tee), contextlib.redirect_stderr(tee):
                add("reading " + path.replace(os.path.expanduser("~"), "~", 1))
                t = time.time()
                fmt, traces = load_any(path, fmt, add)
                add(f"{len(traces)} runs, {sum(len(x.events) for x in traces)} events "
                    f"({time.time() - t:.1f}s)")
                if rules is None:
                    rules = BUILTIN_RULES.get(fmt, [])
                    if rules:
                        add(f"using the {len(rules)} built-in object rules for {fmt}")
                if ref is not None:
                    add(f"reference model '{ref.name}': {len(ref.required)} required, "
                        f"{len(ref.precedence)} precedence, {len(ref.forbidden_after)} forbidden-after, "
                        f"{len(ref.directly_before)} directly-before")
                report(traces, d, source=f"{base} ({fmt})", reference=ref, object_rules=rules,
                       pictures=pictures, noise=noise, by_agent=by_agent, say=add)
        meta = {"name": form.get("name") or base, "source": path, "format": fmt, "runs": len(traces),
                "events": sum(len(x.events) for x in traces), "created": datetime.datetime.now().isoformat(),
                "seconds": round(time.time() - job["t0"], 1)}
        add(f"done in {meta['seconds']}s")
        with open(os.path.join(d, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=1)
        with open(os.path.join(d, "run.log"), "w", encoding="utf-8") as f:
            f.write("\n".join(f"{x['t']:6.1f}  {x['text']}" for x in job["lines"]) + "\n")
        job["rid"], job["status"] = rid, "done"
    except Exception as exc:
        msg = f"{type(exc).__name__}: {exc}"
        add("error: " + msg)
        if not isinstance(exc, ValueError):
            for line in traceback.format_exc(limit=4).splitlines():
                add(line, raw=True)
        if d:
            shutil.rmtree(d, ignore_errors=True)
        job["error"], job["status"] = msg, "error"


# ------------------------------------------------------------------ look

CSS = r"""
:root{--bg:#fff;--fg:#111;--mute:#888;--line:#e6e6e6;--soft:#f6f6f6;--red:#c4302b;
--sans:-apple-system,BlinkMacSystemFont,"Helvetica Neue",Helvetica,Arial,sans-serif;
--mono:ui-monospace,"SF Mono",Menlo,monospace}
@media (prefers-color-scheme:dark){:root{--bg:#111;--fg:#eee;--mute:#777;--line:#2a2a2a;--soft:#1a1a1a;--red:#ff6b61}}
*{box-sizing:border-box}
html,body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.6 var(--sans)}
.col{max-width:760px;margin:0 auto;padding:56px 20px 96px}
a{color:inherit}
h1{font-size:20px;font-weight:600;margin:0}
h2{font-size:15px;font-weight:600;margin:56px 0 14px}
p{margin:0}
.mute{color:var(--mute)}
.small{font-size:13px}
.top{display:flex;justify-content:space-between;align-items:baseline;margin-bottom:40px}
.top a{text-decoration:none}
.demos{margin:0 0 28px}
.demos button{background:none;border:0;padding:0;margin:0 18px 0 0;font:inherit;color:var(--fg);
text-decoration:underline;text-underline-offset:3px;cursor:pointer}
.drop{border:1px dashed var(--line);padding:28px;text-align:center;color:var(--mute);cursor:pointer}
.drop.on{border-color:var(--fg);color:var(--fg)}
.drop b{color:var(--fg);font-weight:500}
input[type=text],select,textarea{width:100%;padding:9px 0;border:0;border-bottom:1px solid var(--line);
background:transparent;color:var(--fg);font:14px var(--sans);border-radius:0}
textarea{font:13px/1.5 var(--mono);min-height:120px;resize:vertical;border:1px solid var(--line);padding:10px}
input:focus,select:focus,textarea:focus{outline:0;border-color:var(--fg)}
.field{margin-top:18px}.field label{display:block;font-size:13px;color:var(--mute);margin-bottom:2px}
details{margin-top:22px}summary{cursor:pointer;font-size:13px;color:var(--mute)}
.go{margin-top:28px;padding:10px 22px;border:1px solid var(--fg);background:var(--fg);color:var(--bg);
font:500 14px var(--sans);cursor:pointer}
.go:disabled{opacity:.35;cursor:default}
.log{display:none;margin-top:36px;font:12.5px/1.65 var(--mono);background:var(--soft);padding:14px 16px;
max-height:420px;overflow:auto;white-space:pre-wrap;word-break:break-word}
.log.on{display:block}
.log .t{color:var(--mute);display:inline-block;width:46px}
.log .raw{color:var(--mute)}.log .bad,.log .err{color:var(--red)}
.status{margin-top:10px;font-size:13px;color:var(--mute)}
table{border-collapse:collapse;width:100%;font-size:14px}
th{font-weight:400;color:var(--mute);text-align:left;padding:6px 12px 6px 0;border-bottom:1px solid var(--line);font-size:13px}
td{padding:7px 12px 7px 0;border-bottom:1px solid var(--line);vertical-align:top}
.clear{background:none;border:0;padding:0;font:13px var(--sans);color:var(--mute);cursor:pointer;text-decoration:underline}
.facts{display:flex;gap:36px;flex-wrap:wrap;margin-top:22px}
.facts b{display:block;font-size:22px;font-weight:500}.facts span{font-size:13px;color:var(--mute)}
.checks{list-style:none;padding:0;margin:0}
.checks li{display:flex;gap:14px;padding:6px 0;font-size:14px}
.checks .m{width:18px;flex:none;color:var(--mute);font-family:var(--mono)}
.checks .warn{color:var(--red)}.checks .warn .m{color:var(--red)}
.checks .info span:last-child{color:var(--mute)}
figure{margin:0 0 30px}
figure img{display:block;width:auto;height:auto;max-width:100%;max-height:520px;margin:0 auto;background:#fff;cursor:zoom-in}
figcaption{font-size:13px;color:var(--mute);margin-top:8px}
.scroll{overflow:auto;max-height:440px}
.fail{color:var(--red)}
pre{font:12.5px/1.6 var(--mono);background:var(--soft);padding:14px 16px;white-space:pre-wrap;max-height:380px;overflow:auto;margin:0}
.files a{margin-right:18px;font-size:14px}
select.inline{width:auto;max-width:100%;margin-bottom:14px}
.tr-task{background:var(--soft);padding:12px 14px;white-space:pre-wrap;font-size:14px;margin:6px 0 10px}
.tr-meta{font-size:13px;color:var(--mute);margin:0 0 8px}
.tr-ev{display:grid;grid-template-columns:44px 1fr;gap:4px 10px;padding:9px 0;border-bottom:1px solid var(--line);font-size:14px}
.tr-ev.bad{border-left:3px solid var(--red);padding-left:8px;margin-left:-11px}
.tr-ev .pos{font-family:var(--mono);color:var(--mute);font-size:12.5px;padding-top:2px}
.tr-ev .who b{font-weight:600}.tr-ev .who .act{font-family:var(--mono);font-size:13px}
.tr-ev .txt{white-space:pre-wrap;margin-top:3px}
.tr-ev pre{margin-top:5px;max-height:260px}
.tr-ev details{margin-top:4px}.tr-ev summary{font-size:12.5px}
.tr-ev .flag{color:var(--red);font-size:12.5px;margin-left:8px}
.tr-ev.book{opacity:.55}
#zoom{position:fixed;inset:0;background:rgba(0,0,0,.85);display:none;align-items:center;justify-content:center;padding:24px;z-index:9;cursor:zoom-out}
#zoom img{max-width:100%;max-height:100%;background:#fff}
"""

ZOOM_JS = """<div id=zoom onclick="this.style.display='none'"><img alt=""></div><script>
document.querySelectorAll('figure img').forEach(i=>i.onclick=()=>{const z=document.getElementById('zoom');
z.querySelector('img').src=i.src;z.style.display='flex'});</script>"""


TRANSCRIPT_JS = """<script>
(()=>{const $=id=>document.getElementById(id),rs=$('rs'),img=$('ri'),only=$('only'),box=$('tr');
const BOOK=new Set(['invoke_workflow','create_agent']);
function el(tag,cls,text){const e=document.createElement(tag);if(cls)e.className=cls;if(text!=null)e.textContent=text;return e}
function block(label,text,open){const d=el('details');if(open)d.open=true;d.appendChild(el('summary',null,label));d.appendChild(el('pre',null,text));return d}
async function transcript(run){box.innerHTML='';
  let doc;try{doc=await (await fetch(BASE+'transcripts/'+run+'.json')).json()}catch(e){box.appendChild(el('p','mute small','No transcript for this run.'));return}
  box.appendChild(el('h2',null,'Transcript'));
  if(doc.task){box.appendChild(el('p','tr-meta','What was asked'));box.appendChild(el('div','tr-task',doc.task))}
  const v=VERD[run],o=doc.outcome||{},bits=[];
  if(v)bits.push((v.ok?'compliant':'violation')+', fitness '+v.fitness.toFixed(2));
  for(const k of ['reward','termination_reason','mistake_agent','mistake_step','mistake_reason','is_hallucination','hallucination_step','hallucination_reason','harness_error'])
    if(o[k]!=null)bits.push(k.replace(/_/g,' ')+': '+o[k]);
  if(bits.length)box.appendChild(el('p','tr-meta',bits.join('  ·  ')));
  if(v&&v.violations.length)box.appendChild(el('p','tr-meta',''+v.violations.join('; '))).style.color='var(--red)';
  if(o.gold_actions)box.appendChild(block('Required actions (gold)',JSON.stringify(o.gold_actions.map(a=>({name:a.name,arguments:a.arguments})),null,1)));
  const tog=el('label','small');const cb=el('input');cb.type='checkbox';tog.appendChild(cb);tog.appendChild(document.createTextNode(' Show framework bookkeeping'));
  const tp=el('p',null);tp.style.margin='14px 0 4px';tp.appendChild(tog);box.appendChild(tp);
  const list=el('div');box.appendChild(list);const bad=new Set(VIOL[run]||[]);const expert=(o.decisive_event_index!=null)?Number(o.decisive_event_index):null;
  for(const e of doc.events){const r=el('div','tr-ev'+((bad.has(e.pos)||e.pos===expert)?' bad':'')+(BOOK.has(e.activity)?' book':''));
    if(BOOK.has(e.activity))r.dataset.book=1;
    r.appendChild(el('div','pos',e.label!=null?String(e.label):String(e.pos)));const body=el('div');r.appendChild(body);
    const who=el('div','who');who.appendChild(el('b',null,e.agent));who.appendChild(document.createTextNode('  '));
    who.appendChild(el('span','act',e.activity+(e.to?' \\u2192 '+e.to:'')));
    if(bad.has(e.pos))who.appendChild(el('span','flag','violation here'));if(e.pos===expert)who.appendChild(el('span','flag','expert: the decisive mistake'));if(e.error)who.appendChild(el('span','flag','error'));
    body.appendChild(who);
    for(const k of ['instruction','content'])if(e[k])body.appendChild(el('div','txt',e[k]));
    if(e.input)body.appendChild(el('pre',null,e.input));
    if(e.arguments)body.appendChild(el('pre',null,e.arguments));
    if(e.reasoning)body.appendChild(block('reasoning',e.reasoning,false));
    if(e.result)body.appendChild(block('result',e.result,false));
    list.appendChild(r)}
  const sync=()=>list.querySelectorAll('[data-book]').forEach(x=>x.style.display=cb.checked?'':'none');cb.onchange=sync;sync()}
function apply(){const run=rs.value,has=PICS.includes(run);
  $('rfig').style.display=has?'':'none';$('nopic').style.display=has?'none':'';if(has)img.src=BASE+'run_'+run+'.png';
  if(only){let shown=0;document.querySelectorAll('#ct tr[data-run]').forEach(tr=>{const hit=!only.checked||tr.dataset.run===run;
    tr.style.display=hit?'':'none';if(hit)shown++});$('none').style.display=(only.checked&&!shown)?'block':'none'}
  transcript(run)}
rs.onchange=apply;if(only)only.onchange=apply;
document.querySelectorAll('a.pick').forEach(a=>a.onclick=ev=>{ev.preventDefault();rs.value=a.dataset.run;apply();
  rs.scrollIntoView({behavior:'smooth',block:'start'})});
img.onclick=()=>{const z=$('zoom');z.querySelector('img').src=img.src;z.style.display='flex'};
apply()})();
</script>"""


def page(title: str, body: str) -> bytes:
    return (f"<!doctype html><html lang=en><head><meta charset=utf-8>"
            f"<meta name=viewport content='width=device-width,initial-scale=1'><title>{html.escape(title)}</title>"
            f"<style>{CSS}</style></head><body><div class=col>{body}</div></body></html>").encode()


def _past(out_root: str) -> str:
    rows = []
    for d in sorted(os.listdir(out_root), reverse=True)[:40] if os.path.isdir(out_root) else []:
        meta = os.path.join(out_root, d, "meta.json")
        if os.path.exists(meta):
            m = json.load(open(meta))
            rows.append(f"<tr><td><a href='/r/{urllib.parse.quote(d)}/'>{html.escape(m['name'])}</a></td>"
                        f"<td class=mute>{m['format']}</td><td class=mute>{m['runs']} runs</td>"
                        f"<td class=mute>{m['created'][:16].replace('T', ' ')}</td></tr>")
    if not rows:
        return ""
    return ("<h2>History</h2><table>" + "".join(rows) + "</table>"
            "<p style='margin-top:12px'><button class=clear id=clear>Clear history</button></p>")


def index_page(out_root: str) -> bytes:
    formats = _formats()
    opts = "<option value=detect>Detect automatically</option>" + "".join(
        f"<option value={k}>{k}</option>" for k in formats)
    ds = demos()
    demo_btns = "".join(f"<button type=button data-k='{x['key']}' title='{html.escape(x['note'])}'>"
                        f"{html.escape(x['label'])}</button>" for x in ds)
    body = f"""<div class=top><h1>agentpm</h1><span class="mute small">runs locally</span></div>
<p class=mute style="margin-bottom:28px">Turn agent traces into event logs and process models.</p>
{('<div class=demos><span class="mute small" style="margin-right:14px">Demo:</span>' + demo_btns + '</div>') if ds else ''}
<form id=f>
  <div class=drop id=drop><b>Drop a trace file</b> or click to choose<br>
  <span class=small id=fname>OpenTelemetry, tau2, OpenAI Agents SDK, Claude Code</span>
  <input type=file name=file id=file hidden></div>
  <div class=field><label>Or a path on this machine</label>
  <input type=text name=path id=path placeholder="a file, or the folder of a per-run format such as agenthallu"></div>
  <div class=field><label>Format</label><select name=format id=format>{opts}</select></div>
  <details><summary>Reference model and options</summary>
    <div class=field><label>Name</label><input type=text name=name id=name placeholder="optional"></div>
    <div class=field><label>Reference model (JSON). Empty: tau2 uses its gold actions, other sources are not checked.</label>
    <textarea name=reference id=reference placeholder='{{"name": "my-policy", "required": ["read_file"], "precedence": [["read_file", "run_python"]]}}'></textarea></div>
    <div class=field><label>Activity names in the reference</label><select name=by_agent id=by_agent>
    <option value=0>Tool names (read_file)</option><option value=1>Agent and tool (Analyst.read_file, Analyst.spawn:DataChecker)</option></select></div>
    <div class=field><label>Object rules (JSON)</label>
    <textarea name=objects id=objects style="min-height:70px" placeholder='[{{"type": "ticket", "path": "arguments.ticket_id"}}]'></textarea></div>
    <div class=field><label>Inductive miner noise threshold</label><input type=text name=noise id=noise value="0"></div>
    <div class=field><label>Pictures</label><select name=pictures id=pictures><option value=1>Yes</option><option value=0>No, logs and checks only</option></select></div>
  </details>
  <button class=go id=go>Run</button>
</form>
<div class=log id=log></div><div class=status id=status></div>
{_past(out_root)}
<script>
const DEMOS={json.dumps({x['key']: x for x in ds})};
const $=id=>document.getElementById(id);
const drop=$('drop'),file=$('file');
drop.onclick=()=>file.click();
file.onchange=()=>{{$('fname').textContent=file.files[0]?file.files[0].name:'';if(file.files[0])$('path').value=''}};
['dragenter','dragover'].forEach(e=>drop.addEventListener(e,ev=>{{ev.preventDefault();drop.classList.add('on')}}));
['dragleave','drop'].forEach(e=>drop.addEventListener(e,ev=>{{ev.preventDefault();drop.classList.remove('on')}}));
drop.addEventListener('drop',ev=>{{file.files=ev.dataTransfer.files;file.onchange()}});
document.querySelectorAll('.demos button').forEach(b=>b.onclick=()=>{{const d=DEMOS[b.dataset.k];
  file.value='';$('fname').textContent=d.label+' (demo)';
  for(const k of ['path','format','name','reference','objects','pictures','by_agent'])$(k).value=d[k];
  if(d.reference)document.querySelector('details').open=true}});
const c=$('clear');if(c)c.onclick=async()=>{{if(!confirm('Delete every report in the history? The files are removed from disk.'))return;
  await fetch('/clear',{{method:'POST'}});location.reload()}};
const esc=s=>s.replace(/[&<>]/g,c=>({{'&':'&amp;','<':'&lt;','>':'&gt;'}})[c]);
function cls(l){{if(l.raw)return'raw';const t=l.text;if(t.startsWith('error'))return'err';if(t.startsWith('[!!]'))return'bad';return''}}
$('f').onsubmit=async ev=>{{ev.preventDefault();
  const go=$('go'),log=$('log'),st=$('status');go.disabled=true;log.innerHTML='';log.classList.add('on');st.textContent='Starting';
  let j;try{{j=await (await fetch('/run',{{method:'POST',body:new FormData(ev.target)}})).json()}}
  catch(e){{st.textContent=String(e);go.disabled=false;return}}
  let n=0;const t0=Date.now();
  while(true){{await new Promise(r=>setTimeout(r,350));
    const s=await (await fetch('/job/'+j.job+'?since='+n)).json();
    const live=$('live');if(live)live.remove();
    for(const l of s.lines){{const div=document.createElement('div');div.className=cls(l);
      div.innerHTML='<span class=t>'+l.t.toFixed(1)+'</span>'+esc(l.text);log.appendChild(div)}}
    if(s.progress){{const div=document.createElement('div');div.id='live';div.className='raw';
      div.innerHTML='<span class=t>'+s.progress.t.toFixed(1)+'</span>'+esc(s.progress.text);log.appendChild(div)}}
    n=s.n;log.scrollTop=log.scrollHeight;
    st.textContent=(s.status==='running'?'Running':s.status==='done'?'Done':s.status==='error'?'Failed':'Waiting')+', '+((Date.now()-t0)/1000).toFixed(1)+'s';
    if(s.status==='done'){{setTimeout(()=>location.href='/r/'+encodeURIComponent(s.rid)+'/',800);break}}
    if(s.status==='error'){{go.disabled=false;break}}}}}};
const qs=new URLSearchParams(location.search);
if(qs.get('demo')){{const b=document.querySelector('.demos button[data-k="'+qs.get('demo')+'"]');if(b){{b.click();
  if(qs.get('run'))$('f').requestSubmit()}}}}
</script>"""
    return page("agentpm", body)


def _read(p: str, default: Any = None) -> Any:
    if not os.path.exists(p):
        return default
    with open(p, encoding="utf-8") as f:
        return json.load(f) if p.endswith(".json") else f.read()


def report_page(rid: str, d: str) -> bytes:
    meta = _read(os.path.join(d, "meta.json"), {})
    checks = _read(os.path.join(d, "checks.json"), [])
    verdicts = _read(os.path.join(d, "verdicts.json"))
    files = sorted(os.listdir(d))
    q = urllib.parse.quote(rid)
    n = [0]

    def sec(title: str, inner: str, note: str = "") -> str:
        return (f"<h2>{title}</h2>" + (f"<p class='mute small' style='margin:-8px 0 14px'>{note}</p>" if note else "")
                + inner)

    def fig(name: str, title: str, cap: str, big: bool = False) -> str:
        return (f"<figure><img loading=lazy alt='{html.escape(title)}' src='/r/{q}/{name}'>"
                f"<figcaption>{title} {cap}</figcaption></figure>") if name in files else ""

    facts = [(meta.get("runs", "?"), "runs"), (meta.get("events", "?"), "events"), (meta.get("format", "?"), "format")]
    if verdicts:
        facts.append((f"{sum(1 for v in verdicts if v['compliant'])}/{len(verdicts)}", "compliant"))
    if meta.get("seconds") is not None:
        facts.append((f"{meta['seconds']}s", "pipeline time"))
    warns = sum(1 for c in checks if c["status"] == "warn")
    out = [f"<div class=top><h1>{html.escape(meta.get('name', rid))}</h1><a class='mute small' href='/'>New run</a></div>"
           f"<p class='mute small' style='margin-top:-30px'>{html.escape(meta.get('source', '').replace(os.path.expanduser('~'), '~', 1))}</p>",
           "<div class=facts>" + "".join(f"<div><b>{html.escape(str(a))}</b><span>{b}</span></div>" for a, b in facts) + "</div>"]

    mark = {"ok": "&#10003;", "warn": "!", "info": "&middot;"}
    out.append(sec("What the source exposes",
                   "<ul class=checks>" + "".join(f"<li class={c['status']}><span class=m>{mark[c['status']]}</span>"
                                                 f"<span>{html.escape(c['text'])}</span></li>" for c in checks) + "</ul>",
                   f"{warns} limitation{'s' if warns != 1 else ''} to keep in mind" if warns else ""))

    own = [fig("lane_model.png", "Lanes.", "One lane per agent; grey edges stay inside an agent, coloured edges hand work over.", True),
           fig("handover.png", "Handover.", "Who passes work to whom, with counts."),
           fig("dfg_by_agent.png", "Directly-follows.", "Each activity coloured by the agent that performs it.")]
    if any(own):
        out.append(sec("Who did what", "".join(own)))
    pm = [fig("pm4py_dfg.png", "DFG.", "pm4py.discover_dfg"),
          fig("pm4py_petri_net.png", "Petri net.", "inductive miner"),
          fig("pm4py_bpmn.png", "BPMN.", "the same model"),
          fig("pm4py_process_tree.png", "Process tree.", "the same model as operators")]
    if any(pm):
        summary = _read(os.path.join(d, "pm4py_summary.txt"), "")
        out.append(sec("Process models", "".join(pm)
                       + (f"<details><summary>variants, start and end activities</summary><pre>{html.escape(summary)}</pre></details>" if summary else ""),
                       "pm4py, inductive miner. Messages and framework bookkeeping are left out."))
    safe = lambda c: "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in c)[:60]
    pics = sorted(f[4:-4] for f in files if f.startswith("run_") and f.endswith(".png"))
    tdir = os.path.join(d, "transcripts")
    runs = sorted(f[:-5] for f in os.listdir(tdir) if f.endswith(".json")) if os.path.isdir(tdir) else list(pics)
    vmap = {safe(v["case_id"]): v for v in (verdicts or [])}
    failed = {k for k, v in vmap.items() if not v["compliant"]}
    runs.sort(key=lambda r: (r not in failed, r not in pics, r))     # failing runs first, then drawn ones
    if runs:
        opts = "".join(f"<option value='{html.escape(r)}'>{html.escape(r)}{'  (fail)' if r in failed else ''}</option>"
                       for r in runs)
        out.append(sec("One run",
                       f"<select id=rs class=inline>{opts}</select>"
                       f"<figure id=rfig><img id=ri alt=''></figure><p id=nopic class='mute small' style='display:none'>"
                       f"No picture for this run. Pictures are drawn for {len(pics)} runs, failing runs first.</p>"
                       "<div id=tr></div>",
                       "The picture and the transcript below it show the run picked here. "
                       "Box numbers are steps; two boxes with the same number and a letter were issued together. The verdicts and the transcript use the same numbers."))
    if verdicts:
        rows = []
        for v in sorted(verdicts, key=lambda v: (v["compliant"], v["case_id"])):
            viol = "<br>".join(html.escape(x["constraint"]) + (f" <span class=mute>at step {x.get('at_label', x['at'])}</span>" if x["at"] is not None else "")
                               for x in v["violations"])
            sid = safe(v["case_id"])
            link = (f"<a href='#rs' class=pick data-run='{html.escape(sid)}'>{html.escape(v['case_id'])}</a>"
                    if sid in runs else html.escape(v["case_id"]))
            rows.append(f"<tr data-run='{html.escape(sid)}'><td>{link}</td><td class={'pass' if v['compliant'] else 'fail'}>"
                        f"{'pass' if v['compliant'] else 'fail'}</td><td>{v['fitness']:.2f}</td><td>{viol}</td></tr>")
        controls = ("<p class=small style='margin:0 0 12px'>"
                    "<label><input type=checkbox id=only checked> Only the run shown above</label>"
                    "<span class=mute style='margin-left:18px'>Click a run name to show it above.</span></p>") if runs else ""
        out.append(sec("Conformance", controls + "<div class=scroll><table id=ct><tr><th>run</th><th>verdict</th><th>fitness</th><th>violated constraint</th></tr>"
                       + "".join(rows) + "</table></div>"
                       + "<p class='mute small' id=none style='display:none;margin-top:10px'>This run was not checked against the reference.</p>",
                       f"Reference: {html.escape(verdicts[0].get('reference', ''))}. Fitness is the share of constraints that hold."))
    if runs:
        viol_at = {k: sorted({x["at"] for x in v["violations"] if x["at"] is not None}) for k, v in vmap.items()}
        verdict_js = {k: {"ok": v["compliant"], "fitness": v["fitness"],
                          "violations": [x["constraint"] + (f" at step {x.get('at_label', x['at'])}" if x["at"] is not None else "") for x in v["violations"]]}
                      for k, v in vmap.items()}
        out.append("<script>const PICS=" + json.dumps(pics) + ",VIOL=" + json.dumps(viol_at) + ",VERD=" + json.dumps(verdict_js)
                   + ",BASE='/r/" + q + "/';</script>" + TRANSCRIPT_JS)
    log = os.path.join(d, "event_log.csv")
    if os.path.exists(log):
        with open(log, newline="", encoding="utf-8") as f:
            rd = list(csv.reader(f))
        head, body = rd[0], rd[1:201]
        keep = [i for i, h in enumerate(head) if any(r[i] for r in body)]
        cut = lambda s: html.escape(s if len(s) <= 110 else s[:110] + "…")
        out.append(sec("Event log", "<div class=scroll><table><tr>" + "".join(f"<th>{html.escape(head[i])}</th>" for i in keep) + "</tr>"
                       + "".join("<tr>" + "".join(f"<td>{cut(r[i])}</td>" for i in keep) + "</tr>" for r in body) + "</table></div>",
                       f"First {len(body)} of {len(rd) - 1} events."))
    downloads = [("event_log.csv", "event_log.csv"), ("event_log.xes", "event_log.xes"), ("runs.ocel.json", "runs.ocel.json"),
                 ("verdicts.json", "verdicts.json"), ("CHECKS.txt", "CHECKS.txt"), ("inventory.txt", "inventory.txt"), ("lenses.txt", "lenses.txt"),
                 ("run.log", "run.log"), ("README.md", "README.md")]
    out.append(sec("Files", "<div class=files>" + "".join(f"<a href='/r/{q}/{a}' download>{b}</a>" for a, b in downloads if a in files)
                   + "</div>"))
    runlog = _read(os.path.join(d, "run.log"))
    if runlog:
        out.append(sec("Run log", f"<pre>{html.escape(runlog)}</pre>", ""))
    return page(meta.get("name", rid), "".join(out) + ZOOM_JS)


# ------------------------------------------------------------------ server

class Handler(BaseHTTPRequestHandler):
    out_root = os.path.expanduser("~/agentpm_reports")

    def log_message(self, *a):
        pass

    def _send(self, code: int, body: bytes, ctype: str = "text/html; charset=utf-8") -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj: Any) -> None:
        self._send(200, json.dumps(obj).encode(), "application/json")

    def do_GET(self) -> None:  # noqa: N802
        url = urllib.parse.urlparse(self.path)
        path = urllib.parse.unquote(url.path)
        if path == "/":
            return self._send(200, index_page(self.out_root))
        m = re.match(r"^/job/([0-9a-f]+)$", path)
        if m:
            job = _jobs.get(m.group(1))
            if not job:
                return self._json({"status": "error", "lines": [], "n": 0, "error": "unknown job"})
            since = int(urllib.parse.parse_qs(url.query).get("since", ["0"])[0])
            lines = job["lines"]
            return self._json({"status": job["status"], "lines": lines[since:], "n": len(lines),
                               "progress": job["progress"], "rid": job["rid"], "error": job["error"]})
        m = re.match(r"^/r/([^/]+)/(.*)$", path)
        if not m:
            return self._send(404, page("Not found", "<section class=main><p>Not found. <a href='/'>Back</a></p></section>"))
        rid, name = m.group(1), m.group(2)
        d = os.path.realpath(os.path.join(self.out_root, rid))
        if not d.startswith(os.path.realpath(self.out_root) + os.sep) or not os.path.isdir(d):
            return self._send(404, page("Not found", "<section class=main><p>No such report. <a href='/'>Back</a></p></section>"))
        if name == "":
            return self._send(200, report_page(rid, d))
        f = os.path.realpath(os.path.join(d, name))
        if not f.startswith(d + os.sep) or not os.path.isfile(f):
            return self._send(404, b"not found", "text/plain")
        ctype = mimetypes.guess_type(f)[0] or "application/octet-stream"
        if ctype.startswith("text/") or f.endswith((".json", ".md", ".csv", ".xes", ".log")):
            ctype = (ctype if ctype.startswith("text/") else "text/plain") + "; charset=utf-8"
        with open(f, "rb") as fh:
            self._send(200, fh.read(), ctype)

    def do_POST(self) -> None:  # noqa: N802
        if self.path == "/clear":
            root = os.path.realpath(self.out_root)
            removed = 0
            for name in os.listdir(root):
                d = os.path.join(root, name)
                # only folders this site made: a report folder carries meta.json
                if os.path.isdir(d) and os.path.exists(os.path.join(d, "meta.json")):
                    shutil.rmtree(d, ignore_errors=True)
                    removed += 1
            return self._json({"removed": removed})
        if self.path != "/run":
            return self._send(404, b"{}", "application/json")
        self._json({"job": _start_job(self._form(), self.out_root)})

    def _form(self) -> dict[str, Any]:
        n = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(n)
        msg = email.parser.BytesParser(policy=email.policy.HTTP).parsebytes(
            b"Content-Type: " + self.headers.get("Content-Type", "").encode() + b"\r\n\r\n" + raw)
        fields: dict[str, Any] = {}
        for part in msg.iter_parts():
            name = part.get_param("name", header="content-disposition")
            filename = part.get_filename()
            data = part.get_payload(decode=True) or b""
            fields[name] = (filename, data) if filename is not None else data.decode("utf-8", "replace").strip()
        return fields


def serve(port: int = 8765, out_root: Optional[str] = None, open_browser: bool = True) -> None:
    Handler.out_root = os.path.expanduser(out_root or "~/agentpm_reports")
    os.makedirs(Handler.out_root, exist_ok=True)
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    url = f"http://127.0.0.1:{port}/"
    print(f"agentpm web on {url}  (reports in {Handler.out_root}; ctrl-c to stop)")
    if open_browser:
        import webbrowser
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")


if __name__ == "__main__":
    serve()
