from email.policy import default
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, PlainTextResponse
from openai import OpenAI, BadRequestError
import os, json, traceback, re, threading, asyncio, time
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime, timezone, timedelta

app = FastAPI()

CODE_VERSION = "v1.01"
print(f"🔁 New GPT-agent — code version: {CODE_VERSION}")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
s2_KEY = os.getenv("s2_assist", "")
m50_KEY = os.getenv("m50_assist", "")
if not OPENAI_API_KEY:
    print("⚠️ OPENAI_API_KEY is not set. The /evaluate route will fail until it is provided.")
client = OpenAI(api_key=OPENAI_API_KEY)

STRICT_FAIL_ON_UNPARSABLE = os.getenv("STRICT_FAIL_ON_UNPARSABLE", "0").strip() == "1"

# ===================== ASSISTANTS (pid -> assistant_id) =====================
ASSISTANTS_MAP = {
    "s2": s2_KEY,
    "m50": m50_KEY, 
    "default": s2_KEY,
}

# ===================== CACHE (GPTarr) =====================
# Structure: {sym, tf, pid, time (iso), time_dt (UTC), answer, explain, key, ts_added}
GPTarr: List[Dict[str, Any]] = []
_GPTARR_LOCK = threading.Lock()
_MAX_AGE = timedelta(hours=1)

def _now_utc() -> datetime:
    return datetime.now(timezone.utc)

# ----------------- Generic helpers -----------------
def coerce_bool(val: Any) -> bool:
    if isinstance(val, bool):
        return val
    if isinstance(val, (int, float)):
        return val != 0
    if isinstance(val, str):
        return val.strip().lower() in ("1", "true", "yes", "y", "on")
    return False

def assistant_id_for_pid(pid: str) -> str:
    pid_key = (pid or "").strip().lower()
    return ASSISTANTS_MAP.get(pid_key) or ASSISTANTS_MAP.get("DEFAULT") or ASSISTANTS_MAP.get("default", "")

def parse_json_strict_but_safe(body_bytes: bytes) -> dict:
    tail = body_bytes[-16:] if len(body_bytes) >= 16 else body_bytes
    # print(f"📦 Incoming bytes: len={len(body_bytes)} tail={repr(tail)}")
    cleaned = body_bytes.replace(b"\x00", b"")
    start = cleaned.find(b"{")
    end = cleaned.rfind(b"}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("No valid JSON object delimiters found in body")
    s = cleaned[start:end + 1].decode("utf-8", errors="ignore").strip()
    return json.loads(s)

def extract_probability(text: str) -> Optional[float]:
    if not text:
        return None
    text = text.strip()
    try:
        val = float(text)
        if 0.0 <= val <= 1.0:
            return val
    except Exception:
        pass
    m = re.search(r"(?<!\d)(?:0(?:\.\d+)?|1(?:\.0+)?)", text)
    if m:
        try:
            val = float(m.group(0))
            if 0.0 <= val <= 1.0:
                return val
        except Exception:
            return None
    return None


def extract_prob_and_explain(text: str) -> Tuple[Optional[float], Optional[str]]:
    if not text:
        return None, None
    s = text.strip()
    if s.startswith("{") and s.endswith("}"):
        try:
            obj = json.loads(s)
            prob = obj.get("prob") if "prob" in obj else obj.get("probability")
            explain = obj.get("explain") or obj.get("reason")
            if prob is not None:
                try:
                    prob = float(prob)
                except Exception:
                    prob = None
            return prob, explain if isinstance(explain, str) else None
        except Exception:
            pass
    prob = None
    m = re.search(r"prob\s*[:=]\s*([01](?:\.0+)?)", s, re.IGNORECASE)
    if m:
        try:
            prob = float(m.group(1))
        except Exception:
            prob = None
    if prob is None:
        prob = extract_probability(s)
    explain = None
    m2 = re.search(r"explain\s*[:=]\s*(.+)", s, re.IGNORECASE)
    if m2:
        explain = m2.group(1).strip()
        explain = explain.rstrip("}").strip()
    return prob, explain

def fallback_explain_from_text(text: str) -> Optional[str]:
    s = (text or "").strip()
    if not s:
        return None
    s = s.strip("{}").strip()
    s = re.sub(r"(?i)\bprob(?:ability)?\s*[:=]\s*[01](?:\.0+)?\s*[;,\s]*", "", s)
    s = re.sub(r"(?i)\bexplain\s*[:=]\s*", "", s)
    s = s.strip(" ;,")
    return s or None

_TF_RE = re.compile(r"^\s*([mMhH])\s*([0-9]+)\s*$")

def timeframe_to_seconds(tf: Optional[str]) -> int:
    if not tf or not isinstance(tf, str):
        return 60
    m = _TF_RE.match(tf)
    if not m:
        m2 = re.search(r"(M|H)(\d+)", tf, re.IGNORECASE)
        if not m2:
            return 60
        unit, num = m2.group(1).upper(), int(m2.group(2))
    else:
        unit, num = m.group(1).upper(), int(m.group(2))
    return (num * 60) if unit == "M" else (num * 3600)

def _parse_bar_time_string(t: str) -> Optional[datetime]:
    if not t or not isinstance(t, str):
        return None
    t = t.strip()
    if not t:
        return None
    try:
        iso = t[:-1] + "+00:00" if t.endswith("Z") else t
        dt = datetime.fromisoformat(iso)
        return dt.replace(tzinfo=dt.tzinfo or timezone.utc).astimezone(timezone.utc)
    except Exception:
        pass
    for fmt in ("%Y.%m.%d %H:%M:%S", "%Y.%m.%d %H:%M"):
        try:
            return datetime.strptime(t, fmt).replace(tzinfo=timezone.utc)
        except Exception:
            continue
    return None

def parse_first_bar_time(payload: Dict[str, Any]) -> Optional[datetime]:
    try:
        bars = payload.get("bars") or []
        if bars and isinstance(bars[0], dict):
            t = bars[0].get("t")
            dt = _parse_bar_time_string(t)
            if dt:
                return dt
        meta = payload.get("meta") or {}
        t_fallback = payload.get("bar1_close_time", meta.get("bar1_close_time"))
        return _parse_bar_time_string(t_fallback)
    except Exception as e:
        print("⚠️ parse_first_bar_time error:", e)
        return None

def floor_to_bar(dt: datetime, bar_sec: int) -> datetime:
    dt = dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    epoch = int(dt.timestamp())
    floored = epoch - (epoch % max(1, bar_sec))
    return datetime.fromtimestamp(floored, tz=timezone.utc)

def canonical_key(sym: str, tf: str, pid: str, first_bar_dt: datetime, bar_sec: int) -> str:
    base_dt = floor_to_bar(first_bar_dt, bar_sec)
    return f"{sym}|{tf}|{pid}|{base_dt.isoformat()}"

def compute_tolerance_seconds(bar_sec: int) -> int:
    return max(30, min(180, bar_sec // 2 if bar_sec > 0 else 60))

def _extract_sym_tf(payload: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    meta = payload.get("meta") or {}
    sym = payload.get("sym") or payload.get("symbol") or meta.get("sym") or meta.get("symbol")
    tf  = payload.get("tf")  or payload.get("timeframe") or payload.get("TF") \
          or meta.get("tf")  or meta.get("timeframe")    or meta.get("TF")
    if isinstance(sym, str): sym = sym.strip().upper()
    if isinstance(tf, str):  tf  = tf.strip().upper()
    return sym, tf

def _extract_pid(payload: Dict[str, Any]) -> str:
    meta = payload.get("meta") or {}
    pid = payload.get("pid") or meta.get("pid") or ""
    if isinstance(pid, str):
        pid = pid.strip().lower()
    return pid or "default"

# ----------------- Cache lookup/store/clean -----------------
def find_cached_answer(payload: Dict[str, Any]) -> Optional[Tuple[float, str]]:
    sym, tf = _extract_sym_tf(payload)
    pid = _extract_pid(payload)
    if not sym or not tf:
        return None
    first_bar_dt = parse_first_bar_time(payload)
    if not first_bar_dt:
        return None

    bar_sec = timeframe_to_seconds(tf)
    tol = compute_tolerance_seconds(bar_sec)
    key_exact = canonical_key(sym, tf, pid, first_bar_dt, bar_sec)

    with _GPTARR_LOCK:
        # 1) Exact key match
        for row in reversed(GPTarr):
            if row.get("key") == key_exact and isinstance(row.get("answer"), (int, float)):
                prob = float(row["answer"])
                print(f"💾 Cache HIT (exact) — key={key_exact} prob={prob:.4f}")
                return prob, str(row.get("explain") or "")

        # 2) Proximity by sym/tf and |dt diff| ≤ tol
        fb_epoch = int(first_bar_dt.timestamp())
        for row in reversed(GPTarr):
            if row.get("sym") != sym or row.get("tf") != tf:
                continue
            if row.get("pid") != pid:
                continue
            row_dt = row.get("time_dt")
            if not isinstance(row_dt, datetime):
                continue
            if abs(int(row_dt.timestamp()) - fb_epoch) <= tol:
                ans = row.get("answer")
                if isinstance(ans, (int, float)):
                    prob = float(ans)
                    print(f"💾 Cache HIT (prox ±{tol}s) — sym={sym} tf={tf} "
                          f"req={first_bar_dt.isoformat()} cached={row_dt.isoformat()} prob={prob:.4f}")
                    return prob, str(row.get("explain") or "")
    return None

def add_cache_record(payload: Dict[str, Any], answer: float, explain: str) -> str:
    sym, tf = _extract_sym_tf(payload)
    pid = _extract_pid(payload)
    first_bar_dt = parse_first_bar_time(payload)
    if not sym or not tf or not first_bar_dt:
        return "N/A"
    bar_sec = timeframe_to_seconds(tf)
    key = canonical_key(sym, tf, pid, first_bar_dt, bar_sec)
    row = {
        "sym": sym,
        "tf": tf,
        "pid": pid,
        "time": first_bar_dt.isoformat(),
        "time_dt": first_bar_dt,
        "answer": float(answer),
        "explain": explain or "",
        "key": key,
        "ts_added": _now_utc().isoformat(),
    }
    with _GPTARR_LOCK:
        GPTarr.append(row)
    print(f"🆕 Unique result stored — key={key} prob={answer:.4f}")
    return key

def clean_cache() -> None:
    cutoff = _now_utc() - _MAX_AGE
    with _GPTARR_LOCK:
        before = len(GPTarr)
        GPTarr[:] = [r for r in GPTarr if isinstance(r.get("time_dt"), datetime) and r["time_dt"] >= cutoff]
        after = len(GPTarr)
    if before != after:
        print(f"🧹 Cache cleaned: {before} -> {after} (drop if first_bar_dt < {cutoff.isoformat()})")

# ===================== IN-FLIGHT DEDUPE =====================
# Followers wait briefly for the leader’s result to avoid duplicate GPT calls.
_INFLIGHT: Dict[str, asyncio.Future] = {}
_INFLIGHT_LOCK = asyncio.Lock()

def inflight_bucket_key(sym: str, tf: str, pid: str, first_bar_dt: datetime, bar_sec: int) -> str:
    """
    Proximity bucket (coarser than cache key) so tiny time differences coalesce:
    bucket by round(epoch / tol), where tol = half a bar (clamped 30..180).
    """
    tol = compute_tolerance_seconds(bar_sec)
    buck = int(round(first_bar_dt.timestamp() / tol))
    return f"{sym}|{tf}|{pid}|prox|{tol}|{buck}"

async def inflight_acquire(key: str) -> Tuple[bool, asyncio.Future]:
    async with _INFLIGHT_LOCK:
        fut = _INFLIGHT.get(key)
        if fut and not fut.done():
            return False, fut  # follower
        fut = asyncio.get_event_loop().create_future()
        _INFLIGHT[key] = fut
        return True, fut       # leader

async def inflight_finish(key: str, result: Tuple[float, str] = None, err: Exception = None):
    async with _INFLIGHT_LOCK:
        fut = _INFLIGHT.pop(key, None)
    if fut and not fut.done():
        if err is not None:
            fut.set_exception(err)
        else:
            fut.set_result(result)

# ===================== ASSISTANT CALLERS =====================
def run_assistant(assistant_id: str, description_text: str) -> str:
    run = client.beta.threads.create_and_run(
        assistant_id=assistant_id,
        thread={
            "messages": [
                {"role": "user", "content": description_text or ""}
            ]
        }
    )

    thread_id = run.thread_id
    run_id = run.id
    status = run.status
    for _ in range(60):
        if status in ("completed", "failed", "cancelled", "expired"):
            break
        time.sleep(0.5)
        run = client.beta.threads.runs.retrieve(thread_id=thread_id, run_id=run_id)
        status = run.status

    if status != "completed":
        raise RuntimeError(f"Assistant run not completed: {status}")

    messages = client.beta.threads.messages.list(thread_id=thread_id, order="desc", limit=1)
    if not messages.data:
        return ""
    msg = messages.data[0]
    parts = []
    for part in msg.content or []:
        if getattr(part, "type", "") == "text" and getattr(part, "text", None):
            parts.append(part.text.value)
    return "".join(parts).strip()

def auto_heal_and_call(args):
    try:
        return client.chat.completions.create(**args)
    except BadRequestError as e:
        msg = str(e)
        print("⚠️ BadRequestError, attempting auto-fix:", msg)
        for p in re.findall(r"Unsupported parameter: '([^']+)'", msg):
            args.pop(p, None)
        if "max_tokens" in msg and "Unsupported" in msg:
            args.pop("max_tokens", None)
            args["max_completion_tokens"] = args.get("max_completion_tokens", 3)
        if "max_completion_tokens" in msg and "Unsupported" in msg:
            args.pop("max_completion_tokens", None)
            args["max_tokens"] = args.get("max_tokens", 3)
        if "temperature" in msg and "Unsupported" in msg:
            args.pop("temperature", None)
        if "stop" in msg and "Unsupported" in msg:
            args.pop("stop", None)
        return client.chat.completions.create(**args)

# ===================== ROUTES =====================
@app.get("/health")
async def health():
    return {"status": "ok", "version": CODE_VERSION, "cache_size": len(GPTarr)}

@app.post("/evaluate")
async def evaluate(request: Request):
    print(f"\n📥 New request — version: {CODE_VERSION}")
    try:
        body_bytes = await request.body()
        # preview = body_bytes[:400]
        # print("Raw request (first 400 bytes):", preview.decode("utf-8", errors="ignore"))

        try:
            payload = parse_json_strict_but_safe(body_bytes)

            meta = payload.get("meta") or {}
            print("Payload keys:", list(payload.keys()))
            print("Payload meta:", meta)
            gpt_exp = coerce_bool(payload.get("gpt_exp", meta.get("gpt_exp")))
            news_within_90m = payload.get("news_within_90m", meta.get("news_within_90m"))
            if coerce_bool(news_within_90m):
                return {
                    "probability": 0.0,
                    "version": CODE_VERSION,
                    "cache": "skip_news",
                    "explain": "Skipped: High impact news within 90m window.",
                }
        except Exception as pe:
            print("❌ JSON parse error:", str(pe))
            traceback.print_exc()
            return JSONResponse(status_code=400, content={"error": "Invalid JSON", "details": str(pe), "version": CODE_VERSION})

        # Extract + log key parts
        sym, tf = _extract_sym_tf(payload)
        pid = _extract_pid(payload)
        first_bar_dt = parse_first_bar_time(payload)
        if not sym or not tf or not first_bar_dt:
            return JSONResponse(status_code=400, content={"error": "Missing sym/tf/bars[0].t", "version": CODE_VERSION})
        bar_sec = timeframe_to_seconds(tf)
        tol = compute_tolerance_seconds(bar_sec)
        key = canonical_key(sym, tf, pid, first_bar_dt, bar_sec)
        now = _now_utc()
        print(f"🔑 Key={key} | tol_sec={tol} | first_bar_dt={first_bar_dt.isoformat()} | now_utc={now.isoformat()} | Δ={(now-first_bar_dt).total_seconds():.0f}s")

        # Housekeeping
        clean_cache()

        # Try cache
        cached = find_cached_answer(payload)
        if isinstance(cached, tuple):
            cached_prob, cached_explain = cached
            if gpt_exp and not cached_explain:
                cached = None
            else:
                cached = cached_prob
                print(f"🔁 Already have result for response — key={key} prob={cached:.4f}")
                resp_out = {"probability": float(cached), "version": CODE_VERSION, "cache": "hit"}
                if gpt_exp:
                    resp_out["explain"] = cached_explain or ""
                return resp_out

        # In-flight dedupe (proximity bucket)
        prox_key = inflight_bucket_key(sym, tf, pid, first_bar_dt, bar_sec)
        leader, fut = await inflight_acquire(prox_key)
        if not leader:
            print(f"⏳ In-flight pending — waiting for leader (prox_key={prox_key})")
            try:
                # Wait up to 5s; afterwards, re-check cache and possibly compute
                prob, cached_explain = await asyncio.wait_for(fut, timeout=5.0)
                print(f"🔁 Already have result for response (in-flight reuse) — key={key} prob={prob:.4f}")
                resp_out = {"probability": float(prob), "version": CODE_VERSION, "cache": "hit_inflight"}
                if gpt_exp:
                    resp_out["explain"] = cached_explain or ""
                return resp_out
            except asyncio.TimeoutError:
                print(f"⏱️ In-flight wait timed out (prox_key={prox_key}); rechecking cache...")
                cached2 = find_cached_answer(payload)
                if isinstance(cached2, tuple):
                    cached_prob, cached_explain = cached2
                    if gpt_exp and not cached_explain:
                        cached2 = None
                    else:
                        cached2 = cached_prob
                        print(f"🔁 Already have result for response (post-timeout cache) — key={key} prob={cached2:.4f}")
                        resp_out = {"probability": float(cached2), "version": CODE_VERSION, "cache": "hit"}
                        if gpt_exp:
                            resp_out["explain"] = cached_explain or ""
                        return resp_out
                # proceed as ad-hoc leader

        # Leader path — compute with GPT (offload blocking call to a thread)
        meta_for_model = dict(meta)
        meta_for_model.pop("bar1_close_time", None)
        meta_for_model.pop("news_within_90m", None)
        meta_for_model.pop("sma_slope", None)
        meta_for_model.pop("rsi_mom", None)
        meta_for_model.pop("vol_ratio", None)

        description_raw = payload.get("description", meta.get("description"))
        compact_json = coerce_bool(payload.get("description", meta.get("description")))
        description_text = "" if description_raw is None else str(description_raw)

        assistant_id = assistant_id_for_pid(pid)
        if not assistant_id:
            msg = f"Assistant ID not configured for pid={pid!r}"
            return JSONResponse(status_code=500, content={"error": msg, "version": CODE_VERSION})
        if not description_text:
            msg = "Missing description for assistant call"
            return JSONResponse(status_code=400, content={"error": msg, "version": CODE_VERSION})
        try:
            reply = await asyncio.to_thread(run_assistant, assistant_id, description_text)
            reply = (reply or "").strip()
            explain = ""
            if gpt_exp:
                prob, explain = extract_prob_and_explain(reply)
            else:
                prob = extract_probability(reply)

            if prob is None:
                msg = "Assistant did not return a numeric probability"
                print(f"⚠️ {msg}. Using fallback.")
                if STRICT_FAIL_ON_UNPARSABLE:
                    await inflight_finish(prox_key, err=RuntimeError(msg))
                    return JSONResponse(status_code=502, content={"error": msg, "version": CODE_VERSION, "cache": "miss"})
                prob = 0.5

            prob = min(1.0, max(0.0, float(prob)))
            if gpt_exp and not explain:
                explain = fallback_explain_from_text(reply) or ""
            if gpt_exp:
                print(f"Assistant reply raw: {reply}")
                # print(f"GPT explain parsed: {explain!r}")
            stored_key = add_cache_record(payload, prob, explain or "")
            await inflight_finish(prox_key, result=(prob, explain or ""))

            print(f"✅ Final probability (NEW) — key={stored_key} prob={prob:.4f}")
            resp_out = {"probability": prob, "version": CODE_VERSION, "cache": "miss"}
            if gpt_exp:
                resp_out["explain"] = explain or ""
            return resp_out

        except Exception as e:
            await inflight_finish(prox_key, err=e)
            print("❌ Unhandled ERROR during assistant call:", str(e))
            traceback.print_exc()
            return JSONResponse(status_code=500, content={"error": str(e), "version": CODE_VERSION})

    except BadRequestError as e:
        print("❌ OpenAI BadRequestError:", str(e))
        return JSONResponse(status_code=400, content={"error": str(e), "version": CODE_VERSION})
    except Exception as e:
        print("❌ Unhandled ERROR:", str(e))
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e), "version": CODE_VERSION})

@app.get("/", response_class=PlainTextResponse)
async def root():
    return f"OK: {CODE_VERSION}\n"

