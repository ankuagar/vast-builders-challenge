"""Aisle Conflict Board: near misses and blocked aisles from warehouse cameras.

Reads re-ingested VSS segments whose captions end with
`EVENTS: <list> | RISK: <level>`, groups the same moment across cameras into one
incident, and serves a queue for a shift lead. Stdlib only.
"""

import hashlib
import json
import os
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

VSS_URL = os.environ["VSS_URL"].rstrip("/")
VSS_USERNAME = os.environ["VSS_USERNAME"]
VSS_PASSWORD = os.environ["VSS_PASSWORD"]
PORT = int(os.environ.get("PORT", "8080"))
LOCATIONS = [l.strip() for l in os.environ.get("BOARD_LOCATIONS", "warehouse3").split(",") if l.strip()]
REFRESH_SEC = int(os.environ.get("REFRESH_SEC", "120"))
WANDB_API_KEY = os.environ.get("WANDB_API_KEY", "")
WANDB_PROJECT = os.environ.get("WANDB_PROJECT_PATH", "")
LLM_MODEL = os.environ.get("LLM_MODEL", "Qwen/Qwen3-30B-A3B-Instruct-2507")
LLM_URL = "https://api.inference.wandb.ai/v1/chat/completions"
HERE = os.path.dirname(os.path.abspath(__file__))

RISK_RANK = {"high": 3, "medium": 2, "low": 1, "": 0}
EVENT_INFO = {
    "NEAR_MISS": ("Near miss", "near_miss"),
    "PERSON_IN_PATH": ("Person in vehicle path", "near_miss"),
    "VEHICLE_CONFLICT": ("Vehicles blocking each other", "blocked"),
    "BLOCKED_AISLE": ("Blocked aisle", "blocked"),
    "PALLET_IN_WALKWAY": ("Pallet in walkway", "blocked"),
}
EVENTS_RE = re.compile(r"EVENTS:\s*(?P<events>[^|\n]*)(?:\|\s*RISK:\s*(?P<risk>\w+))?", re.I)
RUN_RE = re.compile(r"run_(?P<run>\d+)_seed_\d+\.(?P<cam>[a-z]+_\d+)")
GAP_RE = re.compile(r"(under 1 ?m|1\s*-\s*3 ?m|over 3 ?m|less than (?:one|1) met)", re.I)


class VSS:
    def __init__(self):
        self.token = None
        self.lock = threading.Lock()

    def login(self):
        body = json.dumps({"username": VSS_USERNAME, "password": VSS_PASSWORD}).encode()
        req = urllib.request.Request(f"{VSS_URL}/api/v1/auth/login", data=body,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=30) as r:
            self.token = json.load(r)["access_token"]

    def get_token(self):
        with self.lock:
            if not self.token:
                self.login()
            return self.token

    def get(self, path, retry=True):
        req = urllib.request.Request(VSS_URL + path, headers={"Authorization": "Bearer " + self.get_token()})
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code == 401 and retry:
                with self.lock:
                    self.token = None
                return self.get(path, retry=False)
            raise


vss = VSS()


def parse_caption(text):
    text = text or ""
    m = None
    for m in EVENTS_RE.finditer(text):
        pass
    if not m:
        return None
    events = [e.strip().upper().replace(" ", "_") for e in m.group("events").split(",")]
    events = [e for e in events if e in EVENT_INFO]
    risk = (m.group("risk") or "").lower()
    risk = risk if risk in RISK_RANK else ""
    narrative = text[:m.start()].strip()
    gap = GAP_RE.search(narrative)
    return {"events": events, "risk": risk, "narrative": narrative,
            "gap": gap.group(1) if gap else None}


def first_sentences(text, n=2):
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return " ".join(parts[:n])


def fetch_chunks():
    chunks = []
    for loc in LOCATIONS:
        off = 0
        while True:
            q = urllib.parse.urlencode({"scope": "all", "limit": 100, "offset": off, "location": loc})
            page = vss.get(f"/api/v1/videos/explore?{q}").get("chunks") or []
            chunks += page
            if len(page) < 100:
                break
            off += 100
    return chunks


def build_board(chunks):
    incidents, views_total, parsed_total = {}, 0, 0
    for c in chunks:
        fn = c.get("filename", "")
        rm = RUN_RE.search(fn)
        run = f"run {rm.group('run')}" if rm else fn.split("_chunk")[0]
        cam = rm.group("cam").replace("_", " ") if rm else (c.get("camera_id") or "camera")
        for seg in c.get("timeline") or []:
            views_total += 1
            p = parse_caption(seg.get("reasoning_content"))
            if not p:
                continue
            parsed_total += 1
            key = f"{run}|{seg.get('segment_number')}"
            inc = incidents.setdefault(key, {
                "id": hashlib.sha1(key.encode()).hexdigest()[:10],
                "scene": run,
                "start_sec": seg.get("segment_start_sec"),
                "end_sec": seg.get("segment_end_sec"),
                "time": c.get("upload_timestamp"),
                "location": c.get("location"),
                "camera_id": c.get("camera_id"),
                "views": [],
            })
            inc["views"].append({
                "camera": cam,
                "source": seg.get("source"),
                "events": p["events"],
                "risk": p["risk"],
                "gap": p["gap"],
                "narrative": p["narrative"],
            })

    flagged = []
    for inc in incidents.values():
        inc["views"].sort(key=lambda v: (-RISK_RANK[v["risk"]], -len(v["events"]), v["camera"]))
        events = sorted({e for v in inc["views"] for e in v["events"]})
        if not events:
            continue
        inc["events"] = events
        inc["categories"] = sorted({EVENT_INFO[e][1] for e in events})
        inc["risk"] = max((v["risk"] for v in inc["views"] if v["events"]), key=lambda r: RISK_RANK[r], default="low") or "low"
        inc["flagging_cameras"] = sum(1 for v in inc["views"] if v["events"])
        top = next(v for v in inc["views"] if v["events"])
        inc["summary"] = first_sentences(top["narrative"])
        inc["action"] = None
        flagged.append(inc)

    flagged.sort(key=lambda i: (-RISK_RANK[i["risk"]], -i["flagging_cameras"], i["scene"], i["start_sec"] or 0))
    stats = {
        "segments_indexed": views_total,
        "segments_analyzed": parsed_total,
        "moments": len(incidents),
        "incidents": len(flagged),
        "high": sum(i["risk"] == "high" for i in flagged),
        "medium": sum(i["risk"] == "medium" for i in flagged),
        "near_miss": sum("near_miss" in i["categories"] for i in flagged),
        "blocked": sum("blocked" in i["categories"] for i in flagged),
    }
    return flagged, stats


_llm_cache = {}


def llm_summarize(inc):
    if not (WANDB_API_KEY and WANDB_PROJECT):
        return None
    evidence = "\n".join(
        f"- {v['camera']} (risk {v['risk'] or 'n/a'}, events {','.join(v['events']) or 'NONE'}): {v['narrative']}"
        for v in inc["views"][:6])
    key = hashlib.sha1(evidence.encode()).hexdigest()
    if key in _llm_cache:
        return _llm_cache[key]
    prompt = (
        "Several warehouse cameras saw the same 5-second moment. Camera notes:\n"
        f"{evidence}\n\n"
        "Reply with JSON only: {\"what_happened\": one plain sentence for a shift lead, "
        "describing who was near or blocking whom, \"action\": one short imperative fix "
        "(e.g. add floor marking, slow zone, spotter)}. Only use facts in the notes.")
    body = json.dumps({"model": LLM_MODEL, "temperature": 0.2, "max_tokens": 200,
                       "messages": [{"role": "user", "content": prompt}]}).encode()
    req = urllib.request.Request(LLM_URL, data=body, headers={
        "Authorization": "Bearer " + WANDB_API_KEY,
        "OpenAI-Project": WANDB_PROJECT,
        "Content-Type": "application/json",
        "User-Agent": "aisle-board/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            text = json.load(r)["choices"][0]["message"]["content"]
        out = json.loads(text[text.index("{"):text.rindex("}") + 1])
        _llm_cache[key] = out
        return out
    except Exception as e:
        print("llm error:", e, flush=True)
        return None


class Board:
    def __init__(self):
        self.incidents, self.stats = [], {}
        self.updated, self.error = None, None
        self.sources = set()
        self.lock = threading.Lock()

    def refresh(self):
        try:
            incidents, stats = build_board(fetch_chunks())
            with self.lock:
                old = {i["id"]: i for i in self.incidents}
                for inc in incidents:
                    prev = old.get(inc["id"])
                    if prev and prev.get("llm_key") == self._key(inc):
                        inc.update({k: prev[k] for k in ("summary", "action", "llm_key")})
                self.incidents, self.stats = incidents, stats
                self.sources = {v["source"] for i in incidents for v in i["views"]}
                self.updated, self.error = time.time(), None
            print(f"refreshed: {stats}", flush=True)
            for inc in incidents:
                if inc.get("llm_key") == self._key(inc):
                    continue
                out = llm_summarize(inc)
                if out:
                    with self.lock:
                        inc["summary"] = out.get("what_happened") or inc["summary"]
                        inc["action"] = out.get("action")
                        inc["llm_key"] = self._key(inc)
        except Exception as e:
            self.error = str(e)
            print("refresh error:", e, flush=True)

    @staticmethod
    def _key(inc):
        return hashlib.sha1("".join(v["narrative"] for v in inc["views"]).encode()).hexdigest()

    def loop(self):
        while True:
            self.refresh()
            time.sleep(REFRESH_SEC)

    def snapshot(self):
        with self.lock:
            return {"incidents": self.incidents, "stats": self.stats, "updated": self.updated,
                    "error": self.error, "locations": LOCATIONS}


board = Board()


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        pass

    def _send(self, code, body, ctype="application/json"):
        data = body if isinstance(body, bytes) else body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        url = urllib.parse.urlparse(self.path)
        if url.path in ("/", "/index.html"):
            with open(os.path.join(HERE, "index.html"), "rb") as f:
                return self._send(200, f.read(), "text/html; charset=utf-8")
        if url.path == "/health":
            return self._send(200, json.dumps({"ok": True, "updated": board.updated}))
        if url.path == "/api/board":
            return self._send(200, json.dumps(board.snapshot()))
        if url.path == "/clip":
            return self._clip(urllib.parse.parse_qs(url.query).get("source", [""])[0])
        self._send(404, json.dumps({"error": "not found"}))

    def do_POST(self):
        if urllib.parse.urlparse(self.path).path == "/api/refresh":
            threading.Thread(target=board.refresh, daemon=True).start()
            return self._send(202, json.dumps({"ok": True}))
        self._send(404, json.dumps({"error": "not found"}))

    def _clip(self, source):
        if source not in board.sources:
            return self._send(403, json.dumps({"error": "unknown clip"}))
        q = urllib.parse.urlencode({"source": source, "token": vss.get_token()})
        headers = {}
        if self.headers.get("Range"):
            headers["Range"] = self.headers["Range"]
        req = urllib.request.Request(f"{VSS_URL}/api/v1/videos/stream?{q}", headers=headers)
        try:
            upstream = urllib.request.urlopen(req, timeout=60)
        except urllib.error.HTTPError as e:
            if e.code == 401:
                vss.token = None
            return self._send(e.code, json.dumps({"error": f"upstream {e.code}"}))
        with upstream:
            self.send_response(upstream.status)
            ctype = upstream.headers.get("Content-Type", "")
            self.send_header("Content-Type", ctype if ctype.startswith("video/") else "video/mp4")
            for h in ("Content-Length", "Content-Range", "Accept-Ranges"):
                if upstream.headers.get(h):
                    self.send_header(h, upstream.headers[h])
            self.end_headers()
            try:
                while True:
                    buf = upstream.read(64 * 1024)
                    if not buf:
                        break
                    self.wfile.write(buf)
            except (BrokenPipeError, ConnectionResetError):
                pass


if __name__ == "__main__":
    threading.Thread(target=board.loop, daemon=True).start()
    print(f"aisle board on :{PORT}, locations={LOCATIONS}", flush=True)
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
