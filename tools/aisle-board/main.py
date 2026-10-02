"""Warehouse Safety Monitor: near misses, blocked aisles and crowding from warehouse cameras.

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
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

VSS_URL = os.environ["VSS_URL"].rstrip("/")
VSS_USERNAME = os.environ["VSS_USERNAME"]
VSS_PASSWORD = os.environ["VSS_PASSWORD"]
PORT = int(os.environ.get("PORT", "8080"))
LOCATIONS = [l.strip() for l in os.environ.get("BOARD_LOCATIONS", "warehouse3,warehouse017").split(",") if l.strip()]
ASK_FILTERS = json.loads(os.environ.get("ASK_FILTERS", '{"capture_type": "warehouse"}'))
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
    "CROWDING": ("Unusual crowding", "crowding"),
}
CROWD_GAP_M = 0.5
CROWD_MIN = int(os.environ.get("CROWD_MIN", "5"))
CROWD_ABOVE_TYPICAL = int(os.environ.get("CROWD_ABOVE_TYPICAL", "2"))
CROWD_PERCENTILE = float(os.environ.get("CROWD_PERCENTILE", "0.95"))
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
        return self._call(path, None, retry)

    def post(self, path, body, retry=True):
        return self._call(path, body, retry)

    def _call(self, path, body, retry):
        headers = {"Authorization": "Bearer " + self.get_token()}
        data = None
        if body is not None:
            data = json.dumps(body).encode()
            headers["Content-Type"] = "application/json"
        req = urllib.request.Request(VSS_URL + path, data=data, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=90) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code == 401 and retry:
                with self.lock:
                    self.token = None
                return self._call(path, body, retry=False)
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


SITE_CAM_RE = re.compile(r"Warehouse_(?P<site>\d+)_Camera(?:_(?P<cam>\d+))?_chunk_(?P<chunk>\d+)")
CAM_KIND = {"ceiling": "Ceiling cam", "eye": "Eye-level cam"}


def describe_chunk(c):
    """Returns (scene, camera, group, offset_sec). Views sharing a group show the same moment."""
    fn = c.get("filename") or ""
    rm = RUN_RE.search(fn)
    if rm:
        scene = f"Warehouse 3 · scenario {rm.group('run')}"
        kind, num = rm.group("cam").split("_")
        return scene, f"{CAM_KIND.get(kind, kind.title() + ' cam')} {int(num)}", scene, 0.0
    sm = SITE_CAM_RE.search(fn)
    if sm:
        cam = f"Camera {int(sm.group('cam') or 0)}"
        offset = int(sm.group("chunk")) * float(c.get("chunk_duration_sec") or 30)
        return f"Warehouse {sm.group('site')} · {cam}", cam, fn, offset
    return fn.split("_chunk")[0], c.get("camera_id") or "Camera", fn, 0.0


def gap_status(view):
    """'measured' when the detector distance is usable, 'inconclusive' when it contradicts the caption."""
    y = view.get("yolo_gap_m")
    if y is None:
        return "none"
    said = (view.get("gap") or "").lower()
    close = "under" in said or "less than" in said
    far = "over" in said
    if (close and y >= 2) or (far and y < 1):
        return "inconclusive"
    return "measured"


def merge_consecutive(flagged):
    """Back-to-back 5 s slices of the same scene and kind of trouble become one incident."""
    out, last = [], {}
    for inc in sorted(flagged, key=lambda i: (i["scene"], i["start_sec"] or 0)):
        prev = last.get(inc["scene"])
        if prev and (inc["start_sec"] or 0) - (prev["end_sec"] or 0) <= 0.5 and set(prev["categories"]) & set(inc["categories"]):
            if (RISK_RANK[inc["risk"]], inc["flagging_cameras"]) > (RISK_RANK[prev["risk"]], prev["flagging_cameras"]):
                prev["summary"] = inc["summary"]
            prev["end_sec"] = inc["end_sec"]
            prev["views"] += inc["views"]
            prev["events"] = sorted(set(prev["events"]) | set(inc["events"]))
            prev["categories"] = sorted(set(prev["categories"]) | set(inc["categories"]))
            prev["risk"] = max(prev["risk"], inc["risk"], key=RISK_RANK.get)
            prev["moments"] += 1
        else:
            last[inc["scene"]] = inc
            out.append(inc)
    for inc in out:
        inc["views"].sort(key=lambda v: (-RISK_RANK[v["risk"]], -len(v["events"]), v["start_sec"], v["camera"]))
        inc["flagging_cameras"] = len({v["camera"] for v in inc["views"] if v["events"]})
        inc["cameras"] = len({v["camera"] for v in inc["views"]})
    return out


def _percentile(values, q):
    values = sorted(values)
    return values[min(len(values) - 1, int(q * len(values)))] if values else 0


def mark_crowding(incidents):
    """Flags views whose tightest group is unusual for that camera, not merely large."""
    by_cam = {}
    for inc in incidents.values():
        for v in inc["views"]:
            if v.get("crowd") is not None:
                by_cam.setdefault((inc["scene"].split(" · ")[0], v["camera"]), []).append(v)
    for views in by_cam.values():
        sizes = [v["crowd"] for v in views]
        typical = _percentile(sizes, 0.5)
        threshold = max(CROWD_MIN, _percentile(sizes, CROWD_PERCENTILE), typical + CROWD_ABOVE_TYPICAL)
        for v in views:
            v["crowd_typical"] = typical
            if v["crowd"] >= threshold:
                near_vehicle = v.get("yolo_gap_m") is not None and v["yolo_gap_m"] < 1
                crowd_risk = "medium" if near_vehicle or v["crowd"] >= threshold + 2 else "low"
                v["risk"] = max(v["risk"], crowd_risk, key=RISK_RANK.get) if v["events"] else crowd_risk
                v["events"] = v["events"] + ["CROWDING"]


def build_board(chunks, dets=None):
    dets = dets or {}
    incidents, views_total, parsed_total, captions = {}, 0, 0, {}
    for c in chunks:
        run, cam, group, offset = describe_chunk(c)
        for seg in c.get("timeline") or []:
            # Mid-relabel chunks are listed under both their old and new location.
            if seg.get("source") in captions:
                continue
            views_total += 1
            p = parse_caption(seg.get("reasoning_content"))
            start = offset + float(seg.get("segment_start_sec") or 0)
            end = offset + float(seg.get("segment_end_sec") or 0)
            captions[seg.get("source")] = {
                "scene": run, "camera": cam, "start_sec": start, "end_sec": end,
                "narrative": p["narrative"] if p else (seg.get("reasoning_content") or ""),
                "events": p["events"] if p else [], "risk": p["risk"] if p else "",
            }
            if p:
                parsed_total += 1
            else:
                p = {"events": [], "risk": "", "gap": None, "narrative": seg.get("reasoning_content") or ""}
            key = f"{group}|{seg.get('segment_number')}"
            inc = incidents.setdefault(key, {
                "id": hashlib.sha1(key.encode()).hexdigest()[:10],
                "scene": run,
                "start_sec": start,
                "end_sec": end,
                "time": c.get("upload_timestamp"),
                "location": c.get("location"),
                "camera_id": c.get("camera_id"),
                "views": [],
            })
            view = {
                "camera": cam,
                "start_sec": start,
                "end_sec": end,
                "source": seg.get("source"),
                "events": p["events"],
                "risk": p["risk"],
                "gap": p["gap"],
                "narrative": p["narrative"],
            }
            d = dets.get(seg.get("source"))
            if d:
                if d["closest"]:
                    view["yolo_gap_m"], view["yolo_t"] = d["closest"]["gap_m"], d["closest"]["t"]
                view["crowd"], view["crowd_t"], view["people"] = d["crowd"], d["crowd_t"], d["people"]
            view["gap_status"] = gap_status(view)
            inc["views"].append(view)

    mark_crowding(incidents)
    for src_inc in incidents.values():
        for v in src_inc["views"]:
            if "CROWDING" in v["events"]:
                captions[v["source"]]["events"] = v["events"]
                captions[v["source"]]["risk"] = v["risk"]

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
        if top["events"] == ["CROWDING"]:
            inc["summary"] = (f"{top['crowd']} people bunched within arm's reach on {top['camera']} "
                              f"(it usually sees groups of {top['crowd_typical']}). " + inc["summary"])
        inc["moments"] = 1
        inc["action"] = None
        flagged.append(inc)

    flagged = merge_consecutive(flagged)
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
        "crowding": sum("crowding" in i["categories"] for i in flagged),
    }
    return flagged, stats, captions


# COCO has no forklift class; on these cameras YOLO lands on forklifts under these labels.
EQUIPMENT_LABELS = {"truck", "car", "bus", "train", "motorcycle", "boat", "airplane", "suitcase",
                    "oven", "refrigerator", "bench", "couch", "dining table", "microwave", "tv"}
PERSON_HEIGHT_M = 1.7
HOLD_FRAMES = 45
MIN_CONF = 0.35
MIN_AREA_FRAC = 0.003


def _overlap_or_near(a, b, pad=10):
    return not (a[2] + pad < b[0] or b[2] + pad < a[0] or a[3] + pad < b[1] or b[3] + pad < a[1])


def _merge_boxes(boxes):
    boxes = [list(b) for b in boxes]
    merged = True
    while merged:
        merged = False
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                if _overlap_or_near(boxes[i], boxes[j]):
                    a, b = boxes[i], boxes.pop(j)
                    boxes[i] = [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])]
                    merged = True
                    break
            if merged:
                break
    return boxes


def _gap(a, b):
    dx = max(0, b[0] - a[2], a[0] - b[2])
    dy = max(0, b[1] - a[3], a[1] - b[3])
    bcx, bcy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
    pa = (min(max(bcx, a[0]), a[2]), min(max(bcy, a[1]), a[3]))
    pb = (min(max(pa[0], b[0]), b[2]), min(max(pa[1], b[1]), b[3]))
    return (dx * dx + dy * dy) ** 0.5, [round(pa[0]), round(pa[1]), round(pb[0]), round(pb[1])]


def analyze_detections(det):
    h, w = (det.get("video_shape") or [1080, 1920])[:2]
    min_area = MIN_AREA_FRAC * w * h
    fps = det.get("fps") or 30.0
    frames, labels, heights = [], {}, []
    last_eq, last_eq_idx = [], -10 ** 9
    for f in det.get("frames") or []:
        persons, eq = [], []
        for d in f.get("detections") or []:
            x1, y1, x2, y2 = d["bbox"]
            if d["confidence"] < MIN_CONF:
                continue
            if d["label"] == "person":
                persons.append([x1, y1, x2, y2])
                heights.append(y2 - y1)
            elif d["label"] in EQUIPMENT_LABELS and (x2 - x1) * (y2 - y1) >= min_area:
                eq.append([x1, y1, x2, y2])
                labels[d["label"]] = labels.get(d["label"], 0) + 1
        idx = f.get("frame_index", len(frames))
        held = False
        if eq:
            eq = _merge_boxes(eq)
            last_eq, last_eq_idx = eq, idx
        elif last_eq and idx - last_eq_idx <= HOLD_FRAMES:
            eq, held = last_eq, True
        frames.append({"t": f.get("time_sec", idx / fps), "p": persons, "e": eq, "held": held})

    heights.sort()
    person_h = heights[len(heights) // 2] if heights else None
    closest = None
    for fr in frames:
        best = None
        for p in fr["p"]:
            for e in fr["e"]:
                g, line = _gap(p, e)
                g_m = g * PERSON_HEIGHT_M / max(p[3] - p[1], 1)
                if best is None or g_m < best[0]:
                    best = (g_m, line, g)
        if best:
            fr["gap_px"] = round(best[2])
            fr["line"] = best[1]
            fr["gap_m"] = round(best[0], 2)
            if closest is None or fr["gap_m"] < closest["gap_m"]:
                closest = {"gap_m": fr["gap_m"], "t": round(fr["t"], 2), "held": fr["held"]}
        group = _largest_group(fr["p"])
        fr["n"] = len(fr["p"])
        if len(group) >= 3:
            fr["c"] = group

    hold = max(1, int(fps))
    sizes = sorted((len(fr.get("c", [])) for fr in frames), reverse=True)
    counts = sorted((fr["n"] for fr in frames), reverse=True)
    crowd = sizes[hold - 1] if len(sizes) >= hold else 0
    peak = max(frames, key=lambda fr: len(fr.get("c", [])), default=None)
    return {"shape": [h, w], "fps": fps, "frames": frames, "closest": closest,
            "equipment_labels": labels, "person_height_px": person_h,
            "equipment_frames": sum(1 for fr in frames if fr["e"] and not fr["held"]),
            "person_frames": sum(1 for fr in frames if fr["p"]),
            "crowd": crowd, "people": counts[hold - 1] if len(counts) >= hold else 0,
            "crowd_t": round(peak["t"], 2) if peak and crowd else None}


def _largest_group(persons):
    """Indices of the biggest set of people chained within CROWD_GAP_M of each other."""
    n = len(persons)
    if n < 2:
        return list(range(n))
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(n):
        for j in range(i + 1, n):
            a, b = persons[i], persons[j]
            height = ((a[3] - a[1]) + (b[3] - b[1])) / 2
            if _gap(a, b)[0] * PERSON_HEIGHT_M / max(height, 1) <= CROWD_GAP_M:
                parent[find(i)] = find(j)
    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return max(groups.values(), key=len)


_det_cache = {}


def get_detections(source):
    if source not in _det_cache:
        det = vss.get("/api/v1/videos/detections?" + urllib.parse.urlencode({"source": source}))
        _det_cache[source] = analyze_detections(det)
    return _det_cache[source]


def get_detections_or_none(source):
    try:
        return get_detections(source)
    except Exception as e:
        print("detections error:", source[-60:], e, flush=True)
        return None


def yolo_facts(v):
    facts = []
    if v.get("gap_status") == "measured":
        facts.append(f"closest person-vehicle gap ~{v['yolo_gap_m']} m")
    elif v.get("gap_status") == "inconclusive":
        facts.append("measured distance contradicts the caption, so treat distance as unknown")
    if "CROWDING" in v["events"]:
        facts.append(f"group of {v['crowd']} people within {CROWD_GAP_M} m (camera typical {v.get('crowd_typical')})")
    return f" [YOLO: {'; '.join(facts)}]" if facts else ""


_llm_cache = {}


def llm_summarize(inc):
    if not (WANDB_API_KEY and WANDB_PROJECT):
        return None
    evidence = "\n".join(
        f"- {v['camera']} at {int(v['start_sec']) // 60}:{int(v['start_sec']) % 60:02d} "
        f"(risk {v['risk'] or 'n/a'}, events {','.join(v['events']) or 'NONE'}): {v['narrative']}{yolo_facts(v)}"
        for v in inc["views"][:6])
    key = hashlib.sha1(evidence.encode()).hexdigest()
    if key in _llm_cache:
        return _llm_cache[key]
    prompt = (
        "Warehouse cameras recorded one incident (one or more back-to-back 5-second moments). "
        "Robots and AGVs count as vehicles. Camera notes:\n"
        f"{evidence}\n\n"
        "Reply with JSON only: {\"what_happened\": one plain sentence for a shift lead, "
        "describing who was near, blocking, or crowding whom, \"action\": one short imperative fix "
        "(e.g. add floor marking, slow zone, spotter, stagger breaks)}. Only use facts in the notes; "
        "YOLO facts are measurements, prefer them over caption guesses. Never mention YOLO, captions, "
        "detectors or camera names, and only state a distance if the notes agree on it.")
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
        self.captions = {}
        self.ask_sources = set()
        self.lock = threading.Lock()

    def refresh(self):
        try:
            chunks = fetch_chunks()
            sources = [s.get("source") for c in chunks for s in c.get("timeline") or [] if s.get("source")]
            with ThreadPoolExecutor(8) as pool:
                dets = dict(zip(sources, pool.map(get_detections_or_none, sources)))
            missing = [s for s, d in dets.items() if d is None]
            if missing:
                time.sleep(5)
                with ThreadPoolExecutor(2) as pool:
                    dets.update(zip(missing, pool.map(get_detections_or_none, missing)))
            failed = sum(d is None for d in dets.values())
            if self.incidents and failed > 0.2 * len(dets):
                self.error = f"Video backend busy ({failed} of {len(dets)} clips unreadable); showing the last good board."
                print("refresh skipped:", self.error, flush=True)
                return
            incidents, stats, captions = build_board(chunks, dets)
            with self.lock:
                old = {i["id"]: i for i in self.incidents}
                for inc in incidents:
                    prev = old.get(inc["id"])
                    if prev and prev.get("llm_key") == self._key(inc):
                        inc.update({k: prev[k] for k in ("summary", "action", "llm_key")})
                self.incidents, self.stats, self.captions = incidents, stats, captions
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
        return hashlib.sha1("".join(f"{v['start_sec']}{v['narrative']}{v['events']}{v.get('crowd')}{v.get('gap_status')}"
                                    for v in inc["views"]).encode()).hexdigest()

    def loop(self):
        while True:
            self.refresh()
            time.sleep(REFRESH_SEC)

    def snapshot(self):
        with self.lock:
            return {"incidents": self.incidents, "stats": self.stats, "updated": self.updated,
                    "error": self.error, "locations": LOCATIONS}

    def can_stream(self, source):
        with self.lock:
            return source in self.captions or source in self.ask_sources

    def ask(self, question):
        body = {"query": question, "top_k": 8, "llm_top_n": 3, "min_similarity": 0.25,
                "metadata_filters": ASK_FILTERS}
        res = vss.post("/api/v1/agent/search-and-answer", body)
        clips = []
        for ch in (res.get("evidence") or {}).get("chunks") or []:
            src = ch.get("preview_source")
            if not src:
                continue
            with self.lock:
                cap = self.captions.get(src)
            scene, cam, _, offset = describe_chunk(ch)
            clips.append({
                "source": src,
                "scene": cap["scene"] if cap else scene,
                "camera": cap["camera"] if cap else cam,
                "start_sec": cap["start_sec"] if cap else offset + float(ch.get("best_match_start_sec") or 0),
                "end_sec": cap["end_sec"] if cap else offset + float(ch.get("best_match_end_sec") or 0),
                "similarity": ch.get("similarity_score"),
                "narrative": cap["narrative"] if cap else "",
                "events": cap["events"] if cap else [],
                "risk": cap["risk"] if cap else "",
            })
        with self.lock:
            self.ask_sources.update(c["source"] for c in clips)
        return {"question": question, "answer": res.get("answer") or "", "clips": clips}


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
        if url.path == "/api/detections":
            source = urllib.parse.parse_qs(url.query).get("source", [""])[0]
            if not board.can_stream(source):
                return self._send(403, json.dumps({"error": "unknown clip"}))
            try:
                return self._send(200, json.dumps(get_detections(source)))
            except urllib.error.HTTPError as e:
                return self._send(404 if e.code == 404 else 502, json.dumps({"error": f"detections {e.code}"}))
        self._send(404, json.dumps({"error": "not found"}))

    def do_POST(self):
        path = urllib.parse.urlparse(self.path).path
        if path == "/api/refresh":
            threading.Thread(target=board.refresh, daemon=True).start()
            return self._send(202, json.dumps({"ok": True}))
        if path == "/api/ask":
            try:
                length = min(int(self.headers.get("Content-Length") or 0), 10_000)
                question = (json.loads(self.rfile.read(length) or b"{}").get("question") or "").strip()
            except (ValueError, AttributeError):
                return self._send(400, json.dumps({"error": "invalid JSON"}))
            if not question or len(question) > 300:
                return self._send(400, json.dumps({"error": "question must be 1-300 characters"}))
            try:
                return self._send(200, json.dumps(board.ask(question)))
            except Exception as e:
                print("ask error:", e, flush=True)
                return self._send(502, json.dumps({"error": f"search failed: {e}"}))
        self._send(404, json.dumps({"error": "not found"}))

    def _clip(self, source):
        if not board.can_stream(source):
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
