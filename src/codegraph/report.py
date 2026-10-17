"""Human-readable views of the whole graph: GRAPH_REPORT.md and a self-contained graph.html.

Both are written next to the index (`.codegraph/`) and need nothing but a browser or a markdown viewer.
"""
from __future__ import annotations

import json
import posixpath
from collections import Counter, defaultdict
from pathlib import Path

HTML_NODES = 400  # a force layout stays readable and smooth up to roughly this many nodes
GOD_NODES = 10
COMMUNITIES = 12
SURPRISES = 8
NON_TARGET = ("module", "impl", "namespace")


def _area(file: str) -> str:
    parts = file.split("/")
    return "/".join(parts[:2]) if len(parts) > 2 else parts[0]


def _gather(db):
    syms = {r["id"]: r for r in db.execute(
        f"SELECT id, file, qualname, kind, rank, community FROM symbols WHERE kind NOT IN {NON_TARGET!r}")}
    edges = [(r["src"], r["dst"], r["kind"], r["conf"]) for r in db.execute("SELECT src, dst, kind, conf FROM edges")
             if r["src"] in syms and r["dst"] in syms]
    deg: Counter = Counter()
    for s, d, _, _ in edges:
        deg[s] += 1
        deg[d] += 1
    return syms, edges, deg


def _communities(syms, edges, deg):
    members: dict[int, list[int]] = defaultdict(list)
    for i, r in syms.items():
        if r["community"] >= 0:
            members[r["community"]].append(i)
    inside: Counter = Counter()
    touching: Counter = Counter()
    for s, d, _, _ in edges:
        cs, cd = syms[s]["community"], syms[d]["community"]
        touching[cs] += 1
        if cs != cd:
            touching[cd] += 1
        else:
            inside[cs] += 1
    out = []
    for c, ids in sorted(members.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        ids.sort(key=lambda i: -syms[i]["rank"])
        areas = Counter(_area(syms[i]["file"]) for i in ids)
        out.append({"id": c, "size": len(ids), "area": areas.most_common(1)[0][0],
                    "files": len({syms[i]["file"] for i in ids}),
                    "top": [syms[i]["qualname"] for i in ids[:4]],
                    "cohesion": inside[c] / touching[c] if touching[c] else 1.0})
    return out


def _surprises(syms, edges, deg):
    """Certain edges that join different communities *and* different areas of the tree, between
    symbols that are not hubs: the links a reader would not have guessed from the file layout."""
    scored = []
    for s, d, kind, conf in edges:
        a, b = syms[s], syms[d]
        if conf != "extracted" or a["community"] == b["community"] or _area(a["file"]) == _area(b["file"]):
            continue
        if a["community"] < 0 or b["community"] < 0:
            continue
        scored.append((deg[s] + deg[d], a["qualname"], b["qualname"], kind, a["file"], b["file"]))
    scored.sort()
    return scored[:SURPRISES]


def build_report(db) -> tuple[str, dict]:
    syms, edges, deg = _gather(db)
    conf = Counter(c for *_, c in edges)
    comms = _communities(syms, edges, deg)
    files = db.execute("SELECT COUNT(*) FROM files").fetchone()[0]
    kinds = Counter(r["kind"] for r in syms.values())
    out = ["# Codegraph report", "",
           f"{files} files, {len(syms)} symbols, {len(edges)} edges "
           f"({conf['extracted']} extracted, {conf['inferred']} inferred, {conf['ambiguous']} ambiguous), "
           f"{len(comms)} communities.", ""]
    docs = [k for k in ("section", "table", "view", "procedure") if kinds.get(k)]
    if docs:
        out += ["Beyond code: " + ", ".join(f"{kinds[k]} {k}s" for k in docs) + ".", ""]

    out += ["## Key symbols", "", "Highest PageRank: the code everything else leans on.", ""]
    top = sorted(syms, key=lambda i: -syms[i]["rank"])[:GOD_NODES]
    for i in top:
        r = syms[i]
        out.append(f"- `{r['qualname']}` ({r['kind']}, {r['file']}) rank {r['rank']:.4f}, {deg[i]} links")

    out += ["", "## Communities", "", "Groups of symbols that call each other more than they call out. "
            "Cohesion is the share of their edges that stay inside.", ""]
    for c in comms[:COMMUNITIES]:
        out.append(f"- **#{c['id']} {c['area']}** — {c['size']} symbols in {c['files']} files, "
                   f"cohesion {c['cohesion']:.0%}: " + ", ".join(f"`{q}`" for q in c["top"]))
    if len(comms) > COMMUNITIES:
        out.append(f"- ... and {len(comms) - COMMUNITIES} smaller")

    sur = _surprises(syms, edges, deg)
    if sur:
        out += ["", "## Surprising connections", "",
                "Certain links between different communities *and* different directories, between "
                "non-hub symbols.", ""]
        out += [f"- `{a}` ({fa}) -{k}-> `{b}` ({fb})" for _, a, b, k, fa, fb in sur]

    out += ["", "## Questions worth asking", ""]
    if top:
        q0 = syms[top[0]]["qualname"]
        out += [f"- What breaks if `{q0}` changes? -> `codegraph impact {q0}`",
                f"- What is `{q0}` closest to? -> `codegraph related {q0}`"]
    if conf["ambiguous"]:
        out.append(f"- {conf['ambiguous']} edges are ambiguous (several plausible targets). "
                   "Treat their callers/impact as upper bounds.")
    if comms:
        out.append(f"- Why is the biggest area ({comms[0]['area']}, {comms[0]['size']} symbols) one community?")
    meta = {"symbols": len(syms), "edges": len(edges), "communities": len(comms),
            "confidence": dict(conf)}
    return "\n".join(out) + "\n", meta


def build_html(db) -> str:
    syms, edges, deg = _gather(db)
    keep = sorted(syms, key=lambda i: -syms[i]["rank"])[:HTML_NODES]
    ix = {sid: n for n, sid in enumerate(keep)}
    nodes = [{"n": syms[i]["qualname"], "f": syms[i]["file"], "c": syms[i]["community"],
              "r": syms[i]["rank"], "k": syms[i]["kind"]} for i in keep]
    links = [{"s": ix[s], "t": ix[d], "q": c} for s, d, _, c in edges if s in ix and d in ix]
    data = json.dumps({"nodes": nodes, "links": links}).replace("</", "<\\/")
    return _HTML.replace("__DATA__", data).replace("__TOTAL__", str(len(syms))).replace("__SHOWN__", str(len(keep)))


def write_all(db, root: Path) -> tuple[str, list[Path]]:
    out_dir = root / ".codegraph"
    out_dir.mkdir(exist_ok=True)
    text, _ = build_report(db)
    md, html = out_dir / "GRAPH_REPORT.md", out_dir / "graph.html"
    md.write_text(text)
    html.write_text(build_html(db))
    return text, [md, html]


def short(path: Path, root: Path) -> str:
    return posixpath.normpath(path.relative_to(root).as_posix())


_HTML = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Codegraph</title>
<style>
:root{--bg:#fafafa;--fg:#1a1a1a;--mute:#6b7280;--edge:#9ca3af}
@media (prefers-color-scheme:dark){:root{--bg:#111318;--fg:#e5e7eb;--mute:#9ca3af;--edge:#4b5563}}
html,body{margin:0;height:100%;background:var(--bg);color:var(--fg);font:14px system-ui,sans-serif}
canvas{display:block;width:100%;height:100%}
#tip{position:fixed;pointer-events:none;background:var(--bg);border:1px solid var(--edge);padding:4px 8px;
border-radius:4px;display:none;max-width:60ch;word-break:break-all}
#hud{position:fixed;left:12px;top:10px;color:var(--mute)}
</style></head><body>
<canvas id="c"></canvas><div id="tip"></div>
<div id="hud">Top __SHOWN__ of __TOTAL__ symbols by PageRank. Colour = community. Dashed = inferred/ambiguous link. Hover to read, click to pin a neighbourhood.</div>
<script>
const D=__DATA__, N=D.nodes, L=D.links, cv=document.getElementById('c'), cx=cv.getContext('2d'), tip=document.getElementById('tip');
const dpr=devicePixelRatio||1; let W,H;
function size(){W=innerWidth;H=innerHeight;cv.width=W*dpr;cv.height=H*dpr;cx.setTransform(dpr,0,0,dpr,0,0)}
addEventListener('resize',size);size();
const col=c=>c<0?'#9ca3af':`hsl(${(c*137.5)%360},65%,52%)`;
N.forEach((n,i)=>{n.x=W/2+Math.cos(i)*W/4*Math.random();n.y=H/2+Math.sin(i)*H/4*Math.random();n.vx=n.vy=0;n.d=0;n.rad=3+Math.sqrt(n.r*4000)});
L.forEach(l=>{N[l.s].d++;N[l.t].d++});
let alpha=1, pin=null, hov=null;
function step(){
 for(let i=0;i<N.length;i++){const a=N[i];
  for(let j=i+1;j<N.length;j++){const b=N[j];let dx=a.x-b.x,dy=a.y-b.y,d2=dx*dx+dy*dy+.01;
   if(d2<40000){const f=900/d2;dx*=f;dy*=f;a.vx+=dx;a.vy+=dy;b.vx-=dx;b.vy-=dy}}
  a.vx+=(W/2-a.x)*.002;a.vy+=(H/2-a.y)*.002}
 for(const l of L){const a=N[l.s],b=N[l.t];let dx=b.x-a.x,dy=b.y-a.y,d=Math.hypot(dx,dy)||1;
  const same=a.c===b.c,f=(d-(same?40:90))*.012*(l.q==='extracted'?1:.5);dx*=f/d;dy*=f/d;a.vx+=dx;a.vy+=dy;b.vx-=dx;b.vy-=dy}
 for(const n of N){n.vx*=.82;n.vy*=.82;n.x+=n.vx*alpha;n.y+=n.vy*alpha}
 alpha=Math.max(.02,alpha*.992)}
const near=i=>{const s=new Set([i]);L.forEach(l=>{if(l.s===i)s.add(l.t);if(l.t===i)s.add(l.s)});return s};
function draw(){
 cx.clearRect(0,0,W,H);const focus=pin!==null?near(pin):null;
 cx.lineWidth=1;
 for(const l of L){const a=N[l.s],b=N[l.t],on=!focus||(focus.has(l.s)&&focus.has(l.t));
  cx.globalAlpha=on?.55:.06;cx.strokeStyle=getComputedStyle(document.documentElement).getPropertyValue('--edge');
  cx.setLineDash(l.q==='extracted'?[]:[3,3]);cx.beginPath();cx.moveTo(a.x,a.y);cx.lineTo(b.x,b.y);cx.stroke()}
 cx.setLineDash([]);
 N.forEach((n,i)=>{cx.globalAlpha=!focus||focus.has(i)?1:.12;cx.fillStyle=col(n.c);cx.beginPath();cx.arc(n.x,n.y,n.rad,0,6.3);cx.fill()});
 cx.globalAlpha=1}
(function loop(){step();draw();requestAnimationFrame(loop)})();
function pick(e){let best=null,bd=1e9;N.forEach((n,i)=>{const d=Math.hypot(n.x-e.clientX,n.y-e.clientY);if(d<n.rad+4&&d<bd){bd=d;best=i}});return best}
cv.addEventListener('mousemove',e=>{hov=pick(e);if(hov===null){tip.style.display='none';return}
 const n=N[hov];tip.textContent=`${n.k} ${n.n}  ${n.f}  (${n.d} links)`;tip.style.display='block';tip.style.left=e.clientX+12+'px';tip.style.top=e.clientY+12+'px'});
cv.addEventListener('click',e=>{const p=pick(e);pin=p===pin?null:p;alpha=.3});
</script></body></html>
"""
