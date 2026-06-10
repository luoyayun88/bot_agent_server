from email.policy import default
from fastapi import FastAPI, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse, RedirectResponse
from openai import OpenAI, BadRequestError
import os, json, traceback, re, threading, asyncio, time, tempfile, base64, hashlib, hmac, secrets, html as html_lib
from urllib.parse import parse_qs
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime, timezone, timedelta, date
from decimal import Decimal, InvalidOperation
from pathlib import Path

try:
    import psycopg2
    from psycopg2 import pool as psycopg2_pool
    from psycopg2.extras import DictCursor, execute_values
except Exception as _pg_exc:  # Allow app to start without psycopg2
    psycopg2 = None
    psycopg2_pool = None
    DictCursor = None
    _PSYCOPG2_IMPORT_ERROR = _pg_exc
from pydantic import BaseModel

app = FastAPI()

CODE_VERSION = "v1.09"
print(f"🔁 New GPT-agent — code version: {CODE_VERSION}")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
s2_KEY = os.getenv("s2_assist", "")
m50_KEY = os.getenv("m50_assist", "")
if not OPENAI_API_KEY:
    print("⚠️ OPENAI_API_KEY is not set. The /evaluate route will fail until it is provided.")
client = OpenAI(api_key=OPENAI_API_KEY)

DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
PROMPT_S2_MA50 = os.getenv("PROMT_S2_MA50", "").strip()
MAXTOKENS = int(os.getenv("MAXTOKENS", "80") or "80")
MINTOKENS = int(os.getenv("MINTOKENS", "0") or "0")
USE_DEFAULT_PROMPT_JSON = os.getenv("USE_DEFAULT_PROMPT_JSON", "1").strip().lower() not in ("0", "false", "no")
VS_ID = os.getenv("VS_ID", "").strip()
DB_API_KEY = os.getenv("DB_API_KEY", "").strip()
USE_VECTOR_DB = os.getenv("USE_VECTOR_DB", "0").strip().lower() in ("1", "true", "yes")
print(f"USE_VECTOR_DB={'ON' if USE_VECTOR_DB else 'OFF'}")

DEFAULT_PROMPT = ("OUTPUT REQUIREMENTS:\n"
"- Return ONLY a valid JSON object.\n"
"- \"prob\": A float between 0.0 and 1.0.\n"
"- \"explain\": One short sentence (max 18 words).\n"
"OUTPUT FORMAT:\n"
"{\"prob\": <float 0.0 to 1.0>, \"explain\": \"<string>\"}\n"
"If you cannot comply, return: {\"prob\": 0.5, \"explain\": \"format_error\"}" )

MINIMAL_PROMPT = "Provide a probability estimate for the input."





STRICT_FAIL_ON_UNPARSABLE = os.getenv("STRICT_FAIL_ON_UNPARSABLE", "0").strip() == "1"

# ===================== ASSISTANTS (pid -> assistant_id) =====================
ASSISTANTS_MAP = {
    "s2": s2_KEY,
    "m50": s2_KEY, 
    "default": s2_KEY,
}

# ===================== NEURO REFRESH =====================
NEURO_DB_CONF = Path(__file__).resolve().parent / "analytics" / "db.conf"
NEURO_TABLE_DEFAULT = "neuro.gpt_base"
NEURO_SOURCE_DEFAULT = "s2"


class DbReadRequest(BaseModel):
    schema: str
    table: str
    where: Optional[Dict[str, Any]] = None
    limit: Optional[int] = None
    order_by: Optional[str] = None
    order_dir: Optional[str] = None
    db_mode: Optional[str] = None


class DbWriteRequest(BaseModel):
    schema: str
    table: str
    rows: List[Dict[str, Any]]
    db_mode: Optional[str] = None

class DbDeleteRequest(BaseModel):
    schema: str
    table: str
    where: Dict[str, Any]
    db_mode: Optional[str] = None


class NeuroRefreshRequest(BaseModel):
    pid: str = "s2"
    source_id: Optional[str] = None
    table: str = NEURO_TABLE_DEFAULT
    limit: Optional[int] = None
    db_mode: Optional[str] = None


class ConfigUiChange(BaseModel):
    row_id: int
    value: str
    input_param: Optional[str] = None
    reason: Optional[str] = None


class ConfigUiSaveRequest(BaseModel):
    account_login: int
    bot: str
    copy_to_account_login: Optional[int] = None
    copy_to_account_logins: Optional[List[int]] = None
    changes: List[ConfigUiChange]


class ParamCatalogChange(BaseModel):
    bot_kind: Optional[str] = None
    param_key: str
    section_name: Optional[str] = None
    input_param_name: Optional[str] = None
    display_name: Optional[str] = None
    param_desc: Optional[str] = None
    param_path: Optional[str] = None
    value_type: Optional[str] = None
    min_numeric: Optional[str] = None
    max_numeric: Optional[str] = None
    allowed_values: Optional[List[str]] = None
    sort_order: Optional[int] = None
    user_editable: Optional[bool] = None


class ParamCatalogSaveRequest(BaseModel):
    account_login: int
    bot: str
    changes: List[ParamCatalogChange]


class RmControlCommandRequest(BaseModel):
    account_logins: List[int]
    action: str
    bot_ids: Optional[List[str]] = None
    duration: Optional[str] = None
    until: Optional[str] = None
    reason: Optional[str] = None


class RecommendationDecisionRequest(BaseModel):
    reason: Optional[str] = None


class BotRuntimeBaseRequest(BaseModel):
    env: str = "prod"
    account_login: int
    bot_kind: str
    bot_id: str
    source_id: Optional[str] = None
    instance_id: str
    applied_version_no: Optional[int] = None
    command_id: Optional[int] = None


class BotRuntimeParamValue(BaseModel):
    input_param: str
    value: Optional[str] = None


class BotRuntimeResolveOnInitRequest(BotRuntimeBaseRequest):
    program_defaults: List[BotRuntimeParamValue] = []
    input_values: List[BotRuntimeParamValue] = []


class BotRuntimeFinishRequest(BaseModel):
    command_id: int
    instance_id: str
    status: str
    result_json: Optional[Dict[str, Any]] = None
    error_text: Optional[str] = None


class BotRuntimeStatusRequest(BotRuntimeBaseRequest):
    status: str = "running"
    allow_new_entries: bool = True
    applied_version_no: Optional[int] = None
    applied_config_hash: Optional[str] = None
    last_error: Optional[str] = None
    runtime_json: Optional[Dict[str, Any]] = None


class BotRuntimeRmStateUpdateRequest(BotRuntimeBaseRequest):
    scope: str
    target_bot_kind: Optional[str] = None
    active: bool
    action: str = "flatten_and_halt"
    reset_mode: str = "manual"
    reset_at_epoch: Optional[int] = None
    source_command_id: Optional[int] = None
    reason: Optional[str] = None


class BotRuntimeRmStateAckRequest(BotRuntimeBaseRequest):
    scope: str
    target_bot_kind: Optional[str] = None
    observed_state_version: int
    decision: str
    last_error: Optional[str] = None


def resolve_db_url(path=NEURO_DB_CONF, db_mode: Optional[str] = None):
    mode = (db_mode or "").strip().lower()
    if mode == "test":
        env_url = os.getenv("TSTDATABASE_URL", "").strip()
        if env_url:
            return env_url, "test_env"
    env_url = os.getenv("DATABASE_URL", "").strip()
    if env_url:
        return env_url, "prod_env"
    cfg_path = Path(path)
    if not cfg_path.is_file():
        alt = Path(__file__).resolve().parent / "db.conf"
        cfg_path = alt if alt.is_file() else cfg_path
    with open(cfg_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("DATABASE_URL="):
                return line.strip().split("=", 1)[1], "prod_file"
    raise RuntimeError("DATABASE_URL not found")


def load_db_url(path=NEURO_DB_CONF, db_mode: Optional[str] = None):
    url, _ = resolve_db_url(path=path, db_mode=db_mode)
    return url


def split_table_name(value):
    parts = (value or "").split(".")
    if len(parts) == 1:
        return "public", parts[0].lower()
    if len(parts) == 2:
        return parts[0].lower(), parts[1].lower()
    raise ValueError(f"Invalid table name: {value}")


def is_safe_ident(value):
    return bool(re.match(r"^[a-z_][a-z0-9_]*$", value or ""))


def require_api_key(request: Request):
    if not DB_API_KEY:
        return None
    key = request.headers.get("X-API-Key", "")
    if key != DB_API_KEY:
        return JSONResponse(status_code=401, content={"error": "unauthorized", "version": CODE_VERSION})
    return None


CONFIG_UI_COOKIE_NAME = "config_ui_session"
CONFIG_UI_COOKIE_MAX_AGE = 8 * 60 * 60
CONFIG_UI_DB_POOL_LOCK = threading.Lock()
CONFIG_UI_DB_POOL = None
CONFIG_UI_DB_POOL_DSN = None
CONFIG_UI_DB_POOL_LIMITS = None


class ConfigUiPooledConnection:
    def __init__(self, pool, conn):
        self._pool = pool
        self._conn = conn
        self._closed = False

    def __getattr__(self, name):
        return getattr(self._conn, name)

    def close(self):
        if self._closed:
            return
        close_conn = False
        try:
            if not self._conn.closed:
                try:
                    self._conn.rollback()
                except Exception:
                    close_conn = True
        finally:
            self._pool.putconn(self._conn, close=close_conn)
            self._closed = True


def config_ui_enabled():
    return os.getenv("CONFIG_UI_ENABLED", "0").strip().lower() in ("1", "true", "yes", "on")


def get_config_ui_actor():
    return (os.getenv("CONFIG_UI_ACTOR", "finexpert") or "").strip()


def get_config_ui_username():
    return (os.getenv("CONFIG_UI_USERNAME", "runtime_config_admin") or "").strip()


def get_config_ui_password_hash():
    return (os.getenv("CONFIG_UI_PASSWORD_HASH", "") or "").strip()


def get_config_ui_session_secret():
    return (os.getenv("CONFIG_UI_SESSION_SECRET", "") or "").strip()


def config_ui_cookie_secure():
    return os.getenv("CONFIG_UI_COOKIE_SECURE", "1").strip().lower() not in ("0", "false", "no", "off")


def config_ui_json_error(status_code, error):
    return JSONResponse(status_code=status_code, content={"ok": False, "error": error, "version": CODE_VERSION})


def make_config_ui_password_hash(password, iterations=260000):
    salt = secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), bytes.fromhex(salt), iterations).hex()
    return f"pbkdf2_sha256${iterations}${salt}${digest}"


def verify_config_ui_password(password, password_hash):
    if not password or not password_hash:
        return False
    parts = password_hash.split("$")
    if len(parts) != 4 or parts[0] != "pbkdf2_sha256":
        return False
    try:
        iterations = int(parts[1])
        salt = bytes.fromhex(parts[2])
        expected = bytes.fromhex(parts[3])
    except Exception:
        return False
    actual = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
    return hmac.compare_digest(actual, expected)


def _b64url_encode(raw):
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _b64url_decode(value):
    padding = "=" * (-len(value) % 4)
    return base64.urlsafe_b64decode((value + padding).encode("ascii"))


def sign_config_ui_session(username):
    secret = get_config_ui_session_secret()
    if not secret:
        raise RuntimeError("CONFIG_UI_SESSION_SECRET is not set")
    payload = {
        "u": username,
        "exp": int(time.time()) + CONFIG_UI_COOKIE_MAX_AGE,
    }
    body = _b64url_encode(json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8"))
    sig = hmac.new(secret.encode("utf-8"), body.encode("ascii"), hashlib.sha256).hexdigest()
    return f"{body}.{sig}"


def read_config_ui_session(cookie_value):
    if not cookie_value or "." not in cookie_value:
        return None
    secret = get_config_ui_session_secret()
    if not secret:
        return None
    body, sig = cookie_value.rsplit(".", 1)
    expected = hmac.new(secret.encode("utf-8"), body.encode("ascii"), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(sig, expected):
        return None
    try:
        payload = json.loads(_b64url_decode(body).decode("utf-8"))
    except Exception:
        return None
    if int(payload.get("exp") or 0) < int(time.time()):
        return None
    username = payload.get("u")
    return username if isinstance(username, str) and username else None


def get_config_ui_session_user(request: Request):
    return read_config_ui_session(request.cookies.get(CONFIG_UI_COOKIE_NAME))


def config_ui_requirements_error():
    if not config_ui_enabled():
        return "CONFIG_UI_ENABLED is not 1"
    if psycopg2 is None:
        return f"psycopg2 not available: {_PSYCOPG2_IMPORT_ERROR}"
    if not get_config_ui_session_secret():
        return "CONFIG_UI_SESSION_SECRET is not set"
    if not get_config_ui_username():
        return "CONFIG_UI_USERNAME is not set"
    if not get_config_ui_password_hash():
        return "CONFIG_UI_PASSWORD_HASH is not set"
    if not get_config_ui_actor():
        return "CONFIG_UI_ACTOR is not set"
    return None


def require_config_ui_api(request: Request):
    error = config_ui_requirements_error()
    if error:
        return None, None, config_ui_json_error(503, error)
    username = get_config_ui_session_user(request)
    if not username:
        return None, None, config_ui_json_error(401, "login required")
    return username, get_config_ui_actor(), None


def config_ui_pool_limits():
    try:
        minconn = int(os.getenv("CONFIG_UI_DB_POOL_MIN", "1") or "1")
    except ValueError:
        minconn = 1
    try:
        maxconn = int(os.getenv("CONFIG_UI_DB_POOL_MAX", "8") or "8")
    except ValueError:
        maxconn = 8
    minconn = max(1, minconn)
    maxconn = max(minconn, maxconn)
    return minconn, maxconn


def config_ui_conn():
    db_url, _ = resolve_db_url()
    if psycopg2_pool is None:
        return psycopg2.connect(db_url, sslmode="require")
    limits = config_ui_pool_limits()
    global CONFIG_UI_DB_POOL, CONFIG_UI_DB_POOL_DSN, CONFIG_UI_DB_POOL_LIMITS
    with CONFIG_UI_DB_POOL_LOCK:
        if (
            CONFIG_UI_DB_POOL is None
            or CONFIG_UI_DB_POOL_DSN != db_url
            or CONFIG_UI_DB_POOL_LIMITS != limits
        ):
            if CONFIG_UI_DB_POOL is not None:
                CONFIG_UI_DB_POOL.closeall()
            CONFIG_UI_DB_POOL = psycopg2_pool.ThreadedConnectionPool(
                limits[0],
                limits[1],
                db_url,
                sslmode="require",
            )
            CONFIG_UI_DB_POOL_DSN = db_url
            CONFIG_UI_DB_POOL_LIMITS = limits
        pool = CONFIG_UI_DB_POOL
    return ConfigUiPooledConnection(pool, pool.getconn())


@app.on_event("shutdown")
def close_config_ui_db_pool():
    global CONFIG_UI_DB_POOL, CONFIG_UI_DB_POOL_DSN, CONFIG_UI_DB_POOL_LIMITS
    with CONFIG_UI_DB_POOL_LOCK:
        if CONFIG_UI_DB_POOL is not None:
            CONFIG_UI_DB_POOL.closeall()
            CONFIG_UI_DB_POOL = None
            CONFIG_UI_DB_POOL_DSN = None
            CONFIG_UI_DB_POOL_LIMITS = None


def config_ui_db_text(value):
    if value is None:
        return None
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


def config_ui_db_decimal(value):
    if value is None:
        return Decimal("0")
    if isinstance(value, Decimal):
        return value
    return Decimal(str(value))


def parse_config_ui_date(value, field_name="filedate"):
    text = (value or "").strip()
    if not text:
        return None
    for fmt in ("%Y-%m-%d", "%Y%m%d"):
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            pass
    raise ValueError(f"{field_name} must be YYYY-MM-DD")


def config_ui_analytics_filedate_date_sql(alias="cp"):
    prefix = f"{alias}." if alias else ""
    raw = f"NULLIF(btrim({prefix}filedate::text), '')"
    return (
        "CASE "
        f"WHEN left({raw}, 10) ~ '^\\d{{4}}-\\d{{2}}-\\d{{2}}$' THEN left({raw}, 10)::date "
        f"WHEN {raw} ~ '^\\d{{8}}$' THEN to_date({raw}, 'YYYYMMDD') "
        "ELSE NULL END"
    )


def config_ui_analytics_source_fallback(source_id):
    sid = (source_id or "").strip().lower()
    return {
        "m50": "n4",
        "nm50": "n4",
        "s2": "n5",
        "m10": "n6",
        "dj": "n7",
        "inbr": "n8",
        "smbrk": "n9",
        "pivot": "n10",
        "kcbb": "n11",
        "fib": "n12",
        "bot123": "bot123",
    }.get(sid)


def set_actor(cur, actor):
    cur.execute("SELECT set_config('bot_param.actor', %s, true)", (actor,))


def ensure_actor_account_access(cur, actor, account_login, can_apply=False):
    right_column = "can_apply" if can_apply else "can_edit"
    cur.execute(
        f"""
        SELECT 1
          FROM bot_param.operator_account
         WHERE db_user = %s
           AND env = 'prod'
           AND account_login = %s
           AND enabled = true
           AND {right_column} = true
        """,
        (actor, account_login),
    )
    return cur.fetchone() is not None


def resolve_new_value_columns(cur, bot, input_param, value):
    cur.execute(
        """
        SELECT EXISTS (
            SELECT 1
              FROM bot_param.bot_config_allowed_value
             WHERE bot = %s
               AND input_param = %s
        )
        """,
        (bot, input_param),
    )
    has_dictionary = bool(cur.fetchone()[0])
    if has_dictionary:
        return value, None
    return None, value


def clean_optional_text(value):
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def clean_required_text(value, field_name):
    text = clean_optional_text(value)
    if not text:
        raise ValueError(f"{field_name} is required")
    return text


def normalize_catalog_numeric(value, field_name):
    text = clean_optional_text(value)
    if text is None:
        return None
    try:
        return Decimal(text)
    except (InvalidOperation, ValueError):
        raise ValueError(f"{field_name} must be numeric")


def normalize_catalog_allowed_values(values):
    if not values:
        return None
    cleaned = []
    seen = set()
    for value in values:
        text = clean_optional_text(value)
        if not text or text in seen:
            continue
        cleaned.append(text)
        seen.add(text)
    return cleaned or None


def sync_catalog_allowed_value_rows(cur, bot, input_param, allowed_values):
    bot = (bot or "").strip().lower()
    input_param = clean_required_text(input_param, "input_param")
    if allowed_values:
        cur.execute(
            """
            DELETE FROM bot_param.bot_config_allowed_value
             WHERE bot = %s
               AND input_param = %s
               AND NOT (allowed_value = ANY(%s))
            """,
            (bot, input_param, allowed_values),
        )
        for sort_order, allowed_value in enumerate(allowed_values, start=1):
            cur.execute(
                """
                INSERT INTO bot_param.bot_config_allowed_value (
                    bot, input_param, allowed_value, value_desc, sort_order
                )
                VALUES (%s, %s, %s, NULL, %s)
                ON CONFLICT (bot, input_param, allowed_value)
                DO UPDATE SET sort_order = EXCLUDED.sort_order
                """,
                (bot, input_param, allowed_value, sort_order),
            )
    else:
        cur.execute(
            """
            DELETE FROM bot_param.bot_config_allowed_value
             WHERE bot = %s
               AND input_param = %s
            """,
            (bot, input_param),
        )


def bot_config_param_catalog_columns(cur):
    cur.execute(
        """
        SELECT column_name
          FROM information_schema.columns
         WHERE table_schema = 'bot_param'
           AND table_name = 'bot_config_param_catalog'
        """
    )
    return {row[0] for row in cur.fetchall()}


def table_columns(cur, schema, table):
    cur.execute(
        """
        SELECT column_name
          FROM information_schema.columns
         WHERE table_schema = %s
           AND table_name = %s
        """,
        (schema, table),
    )
    return {row[0] for row in cur.fetchall()}


def table_exists(cur, schema, table):
    cur.execute(
        """
        SELECT EXISTS (
            SELECT 1
              FROM information_schema.tables
             WHERE table_schema = %s
               AND table_name = %s
        )
        """,
        (schema, table),
    )
    return bool(cur.fetchone()[0])


def refresh_bot_config_editors(cur, account_login, bot):
    bot = (bot or "").strip().lower()
    for fn_name in ("refresh_bot_config_editor", "refresh_bot_config_user_editor"):
        savepoint = f"cfg_{fn_name}"
        cur.execute(f"SAVEPOINT {savepoint}")
        try:
            cur.execute(f"SELECT bot_param.{fn_name}(%s, %s, %s)", ("prod", int(account_login), bot))
            cur.execute(f"RELEASE SAVEPOINT {savepoint}")
        except Exception:
            cur.execute(f"ROLLBACK TO SAVEPOINT {savepoint}")
            cur.execute(f"RELEASE SAVEPOINT {savepoint}")
            raise


def catalog_refresh_accounts(cur, actor, source_account_login, bot):
    bot = (bot or "").strip().lower()
    cur.execute(
        """
        SELECT DISTINCT oa.account_login
          FROM bot_param.operator_account oa
         WHERE oa.env = 'prod'
           AND oa.db_user = %s
           AND oa.enabled = true
           AND oa.can_apply = true
           AND (
                oa.account_login = %s
                OR EXISTS (
                    SELECT 1
                      FROM bot_param.bot_config_current c
                     WHERE c.env = oa.env
                       AND c.account_login = oa.account_login
                       AND c.bot_kind = %s
                )
           )
         ORDER BY oa.account_login
        """,
        (actor, int(source_account_login), bot),
    )
    accounts = [int(row[0]) for row in cur.fetchall()]
    refreshed = []
    for account_login in accounts:
        refresh_bot_config_editors(cur, account_login, bot)
        refreshed.append(account_login)
    return refreshed


def first_error_line(exc):
    text = str(exc).strip()
    return text.splitlines()[0] if text else exc.__class__.__name__


def require_bot_runtime_api(request: Request):
    auth = require_api_key(request)
    if auth:
        return auth
    if psycopg2 is None:
        return JSONResponse(
            status_code=500,
            content={"ok": False, "error": f"psycopg2 not available: {_PSYCOPG2_IMPORT_ERROR}", "version": CODE_VERSION},
        )
    return None


def bot_runtime_json_error(status_code, error):
    return JSONResponse(status_code=status_code, content={"ok": False, "error": error, "version": CODE_VERSION})


def normalize_runtime_identity(req: BotRuntimeBaseRequest):
    env = (req.env or "prod").strip().lower()
    bot_kind = (req.bot_kind or "").strip().lower()
    bot_id = (req.bot_id or "").strip()
    source_id = (req.source_id or bot_id).strip()
    instance_id = (req.instance_id or "").strip()
    if not env or not bot_kind or not bot_id or not source_id or not instance_id:
        raise ValueError("env, account_login, bot_kind, bot_id, source_id and instance_id are required")
    return env, int(req.account_login), bot_kind, bot_id, source_id, instance_id


def bot_runtime_touch(cur, env, account_login, instance_id, column_name):
    if column_name not in ("last_config_check_at", "last_command_check_at", "last_reinit_at"):
        return
    cur.execute(
        f"""
        UPDATE bot_param.bot_runtime_status
           SET {column_name} = now(),
               last_seen_at = now()
         WHERE env = %s
           AND account_login = %s
           AND instance_id = %s
        """,
        (env, account_login, instance_id),
    )


def runtime_param_map(items):
    out = {}
    for item in items or []:
        input_param = (item.input_param or "").strip()
        if not input_param:
            continue
        value = "" if item.value is None else str(item.value).strip()
        out[input_param] = value
    return out


def json_path_get(config, path):
    cur = config or {}
    for part in (path or "").split("."):
        if not part:
            continue
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def json_path_set(config, path, value):
    cur = config
    parts = [p for p in (path or "").split(".") if p]
    if not parts:
        return
    for part in parts[:-1]:
        nxt = cur.get(part)
        if not isinstance(nxt, dict):
            nxt = {}
            cur[part] = nxt
        cur = nxt
    cur[parts[-1]] = value


def config_json_hash(config):
    payload = json.dumps(config or {}, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def coerce_config_json_value(value, value_type):
    text = "" if value is None else str(value).strip()
    vt = (value_type or "text").strip().lower()
    if vt == "bool":
        return text.lower() in ("true", "t", "yes", "y", "on", "1")
    if vt == "int":
        return int(Decimal(text))
    if vt in ("numeric", "double", "float", "number"):
        dec = Decimal(text)
        if dec == dec.to_integral_value():
            return int(dec)
        return float(dec)
    if vt == "json":
        try:
            return json.loads(text)
        except Exception:
            return text
    return text


def normalize_runtime_param_value(value, value_type):
    if value is None:
        return ""
    vt = (value_type or "text").strip().lower()
    text = str(value).strip()
    if vt == "bool":
        return "true" if text.lower() in ("true", "t", "yes", "y", "on", "1") else "false"
    if vt == "int":
        try:
            return str(int(Decimal(text)))
        except Exception:
            return text
    if vt in ("numeric", "double", "float", "number"):
        try:
            dec = Decimal(text).normalize()
            return format(dec, "f").rstrip("0").rstrip(".") if "." in format(dec, "f") else format(dec, "f")
        except Exception:
            return text
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return text


def load_runtime_param_catalog(cur, bot_kind):
    cur.execute(
        """
        SELECT pc.input_param_name AS input_param,
               pc.param_key,
               pc.param_path,
               pc.value_type,
               pc.sort_order
          FROM bot_param.bot_config_param_catalog pc
         WHERE pc.bot_kind = %s
           AND pc.input_param_name IS NOT NULL
           AND pc.param_path IS NOT NULL
           AND COALESCE(pc.user_editable, true) = true
         ORDER BY pc.sort_order, pc.input_param_name
        """,
        (bot_kind,),
    )
    return [dict(row) for row in cur.fetchall()]


def load_runtime_current_config(cur, env, account_login, bot_kind, bot_id):
    cur.execute(
        """
        SELECT c.active_version_no,
               c.active_config_id,
               c.config_hash,
               v.config_json
          FROM bot_param.bot_config_current c
          JOIN bot_param.bot_config_version v
            ON v.config_version_id = c.active_config_id
         WHERE c.env = %s
           AND c.account_login = %s
           AND c.bot_kind = %s
           AND c.bot_id = %s
        """,
        (env, account_login, bot_kind, bot_id),
    )
    row = cur.fetchone()
    return dict(row) if row else None


def insert_runtime_config_version(cur, env, account_login, bot_kind, bot_id, config, config_hash, reason, source):
    cur.execute(
        """
        SELECT COALESCE(MAX(version_no), 0) + 1
          FROM bot_param.bot_config_version
         WHERE env = %s
           AND account_login = %s
           AND bot_kind = %s
           AND bot_id = %s
        """,
        (env, account_login, bot_kind, bot_id),
    )
    version_no = int(cur.fetchone()[0] or 1)
    config_text = json.dumps(config, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    cur.execute(
        """
        INSERT INTO bot_param.bot_config_version (
            env, account_login, bot_kind, bot_id, config_scope, version_no,
            status, config_json, config_hash, validation_status, validation_errors,
            created_by, created_source, created_reason,
            approved_by, approved_at, activated_by, activated_at
        )
        VALUES (
            %s, %s, %s, %s, 'bot', %s,
            'active', %s::jsonb, %s, 'ok', '[]'::jsonb,
            current_user::text, %s, %s,
            current_user::text, now(), current_user::text, now()
        )
        RETURNING config_version_id
        """,
        (env, account_login, bot_kind, bot_id, version_no, config_text, config_hash, source, reason),
    )
    config_id = int(cur.fetchone()[0])
    cur.execute(
        """
        INSERT INTO bot_param.bot_config_current (
            env, account_login, bot_kind, bot_id, active_version_no,
            active_config_id, config_hash, updated_by, updated_source
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, current_user::text, %s)
        ON CONFLICT (env, account_login, bot_kind, bot_id)
        DO UPDATE SET
            active_version_no = EXCLUDED.active_version_no,
            active_config_id = EXCLUDED.active_config_id,
            config_hash = EXCLUDED.config_hash,
            updated_by = EXCLUDED.updated_by,
            updated_source = EXCLUDED.updated_source,
            updated_at = now()
        """,
        (env, account_login, bot_kind, bot_id, version_no, config_id, config_hash, source),
    )
    return version_no, config_id


def runtime_params_from_config(catalog, config):
    params = []
    for item in catalog:
        value = json_path_get(config, item["param_path"])
        if value is None:
            continue
        params.append(
            {
                "input_param": item["input_param"],
                "param_key": item.get("param_key"),
                "param_path": item["param_path"],
                "value_type": item.get("value_type"),
                "value": normalize_runtime_param_value(value, item.get("value_type")),
            }
        )
    return params


RM_STATE_SCOPES = {"account", "bot"}
RM_STATE_ACTIONS = {"halt_only", "flatten_and_halt"}
RM_STATE_RESET_MODES = {"manual", "next_day", "next_week", "defined_date"}
RM_STATE_DECISIONS = {"continue", "halt_only", "flatten_and_halt"}


def ensure_rm_state_schema(cur):
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS bot_param.rm_state (
            env               TEXT NOT NULL DEFAULT 'prod',
            account_login     BIGINT NOT NULL,
            scope             TEXT NOT NULL,
            bot_kind          TEXT NOT NULL DEFAULT '',
            active            BOOLEAN NOT NULL DEFAULT false,
            action            TEXT NOT NULL DEFAULT 'halt_only',
            reset_mode        TEXT NOT NULL DEFAULT 'manual',
            reset_at          TIMESTAMPTZ,
            state_version     BIGINT NOT NULL DEFAULT 0,
            source_command_id BIGINT,
            reason            TEXT,
            updated_by        TEXT,
            updated_source    TEXT,
            updated_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
            PRIMARY KEY (env, account_login, scope, bot_kind),
            CHECK (scope IN ('account', 'bot')),
            CHECK (action IN ('halt_only', 'flatten_and_halt')),
            CHECK (reset_mode IN ('manual', 'next_day', 'next_week', 'defined_date')),
            CHECK ((scope = 'account' AND bot_kind = '') OR (scope = 'bot' AND bot_kind <> ''))
        )
        """
    )
    cur.execute(
        """
        CREATE INDEX IF NOT EXISTS ix_rm_state_active
            ON bot_param.rm_state (env, account_login, active, scope, bot_kind)
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS bot_param.rm_state_ack (
            env                    TEXT NOT NULL DEFAULT 'prod',
            account_login          BIGINT NOT NULL,
            bot_kind               TEXT NOT NULL,
            bot_id                 TEXT NOT NULL,
            instance_id            TEXT NOT NULL,
            scope                  TEXT NOT NULL,
            target_bot_kind        TEXT NOT NULL DEFAULT '',
            observed_state_version BIGINT NOT NULL,
            decision               TEXT NOT NULL,
            last_error             TEXT,
            applied_at             TIMESTAMPTZ NOT NULL DEFAULT now(),
            PRIMARY KEY (env, account_login, bot_kind, bot_id, instance_id, scope, target_bot_kind),
            CHECK (scope IN ('account', 'bot')),
            CHECK (decision IN ('continue', 'halt_only', 'flatten_and_halt'))
        )
        """
    )
    cur.execute(
        """
        CREATE INDEX IF NOT EXISTS ix_rm_state_ack_account
            ON bot_param.rm_state_ack (env, account_login, scope, target_bot_kind, applied_at DESC)
        """
    )


def normalize_rm_state_scope(scope):
    value = (scope or "").strip().lower()
    if value not in RM_STATE_SCOPES:
        raise ValueError("scope must be account or bot")
    return value


def normalize_rm_state_bot_kind(scope, target_bot_kind):
    if scope == "account":
        return ""
    value = (target_bot_kind or "").strip().lower()
    if not re.match(r"^(n\d+|bot123)$", value):
        raise ValueError("target_bot_kind must be n* or bot123 for bot scope")
    return value


def normalize_rm_state_action(action):
    value = (action or "").strip().lower()
    if value not in RM_STATE_ACTIONS:
        raise ValueError("action must be halt_only or flatten_and_halt")
    return value


def normalize_rm_state_reset_mode(reset_mode):
    value = (reset_mode or "").strip().lower()
    if value not in RM_STATE_RESET_MODES:
        raise ValueError("reset_mode must be manual, next_day, next_week, or defined_date")
    return value


def rm_state_reset_at(reset_at_epoch):
    if reset_at_epoch is None:
        return None
    value = int(reset_at_epoch)
    if value <= 0:
        return None
    return datetime.fromtimestamp(value, tz=timezone.utc)


def rm_state_decision(active, action):
    if not active:
        return "continue"
    return "flatten_and_halt" if action == "flatten_and_halt" else "halt_only"


def rm_state_row_dict(row):
    if not row:
        return None
    result = dict(row)
    result["decision"] = rm_state_decision(bool(result.get("active")), result.get("action"))
    return result


def expire_rm_states(cur, env, account_login):
    ensure_rm_state_schema(cur)
    cur.execute(
        """
        UPDATE bot_param.rm_state
           SET active = false,
               state_version = state_version + 1,
               updated_source = 'runtime-expire',
               updated_at = now(),
               reason = COALESCE(reason, '') || CASE WHEN reason IS NULL OR reason = '' THEN '' ELSE ' | ' END || 'auto expired'
         WHERE env = %s
           AND account_login = %s
           AND active = true
           AND reset_mode IN ('next_day', 'next_week', 'defined_date')
           AND reset_at IS NOT NULL
           AND reset_at <= now()
        """,
        (env, int(account_login)),
    )


def upsert_rm_state(cur, env, account_login, scope, bot_kind, active, action, reset_mode, reset_at, source_command_id, reason, updated_by, updated_source):
    ensure_rm_state_schema(cur)
    cur.execute(
        """
        INSERT INTO bot_param.rm_state (
            env, account_login, scope, bot_kind, active, action, reset_mode,
            reset_at, state_version, source_command_id, reason, updated_by, updated_source
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, 1, %s, %s, %s, %s)
        ON CONFLICT (env, account_login, scope, bot_kind)
        DO UPDATE SET
            active = EXCLUDED.active,
            action = EXCLUDED.action,
            reset_mode = EXCLUDED.reset_mode,
            reset_at = EXCLUDED.reset_at,
            state_version = bot_param.rm_state.state_version + 1,
            source_command_id = EXCLUDED.source_command_id,
            reason = EXCLUDED.reason,
            updated_by = EXCLUDED.updated_by,
            updated_source = EXCLUDED.updated_source,
            updated_at = now()
        RETURNING env,
                  account_login,
                  scope,
                  NULLIF(bot_kind, '') AS bot_kind,
                  active,
                  action,
                  reset_mode,
                  reset_at,
                  state_version,
                  source_command_id,
                  reason,
                  updated_by,
                  updated_source,
                  updated_at
        """,
        (
            env,
            int(account_login),
            scope,
            bot_kind,
            bool(active),
            action,
            reset_mode,
            reset_at,
            source_command_id,
            reason,
            updated_by,
            updated_source,
        ),
    )
    return rm_state_row_dict(cur.fetchone())


def load_effective_rm_state(cur, env, account_login, bot_kind):
    expire_rm_states(cur, env, account_login)
    cur.execute(
        """
        SELECT env,
               account_login,
               scope,
               NULLIF(bot_kind, '') AS bot_kind,
               active,
               action,
               reset_mode,
               reset_at,
               state_version,
               source_command_id,
               reason,
               updated_by,
               updated_source,
               updated_at
          FROM bot_param.rm_state
         WHERE env = %s
           AND account_login = %s
           AND (
                (scope = 'account' AND bot_kind = '')
                OR
                (scope = 'bot' AND bot_kind = %s)
           )
         ORDER BY scope
        """,
        (env, int(account_login), bot_kind),
    )
    states = [rm_state_row_dict(row) for row in cur.fetchall()]
    account_state = next((row for row in states if row["scope"] == "account"), None)
    bot_state = next((row for row in states if row["scope"] == "bot"), None)
    active_states = [row for row in (account_state, bot_state) if row and row["active"]]
    decision = "continue"
    if any(row["decision"] == "flatten_and_halt" for row in active_states):
        decision = "flatten_and_halt"
    elif any(row["decision"] == "halt_only" for row in active_states):
        decision = "halt_only"
    effective = {
        "decision": decision,
        "state_version": max([int(row["state_version"]) for row in active_states] or [0]),
        "active_scopes": [row["scope"] for row in active_states],
        "reason": " | ".join([row.get("reason") or "" for row in active_states if row.get("reason")]),
    }
    return account_state, bot_state, effective


def bot_runtime_load_changed_params(cur, env, account_login, bot_kind, bot_id, old_version_no, new_version_no, active_config_id):
    if not old_version_no or int(old_version_no) <= 0 or int(old_version_no) == int(new_version_no or 0):
        return []

    cur.execute(
        """
        SELECT pc.input_param_name AS input_param,
               a.old_value #>> '{}' AS old_value,
               a.new_value #>> '{}' AS new_value,
               a.changed_by,
               a.changed_reason AS reason
          FROM bot_param.bot_param_audit a
          JOIN bot_param.bot_config_param_catalog pc
            ON pc.bot_kind = a.bot_kind
           AND pc.param_path = a.param_path
         WHERE a.env = %s
           AND a.account_login = %s
           AND a.bot_kind = %s
           AND a.bot_id = %s
           AND a.old_version_no = %s
           AND a.new_version_no = %s
           AND pc.input_param_name IS NOT NULL
         ORDER BY pc.sort_order, pc.input_param_name, a.audit_id
        """,
        (env, account_login, bot_kind, bot_id, old_version_no, new_version_no),
    )
    rows = [dict(row) for row in cur.fetchall()]
    if rows:
        return rows

    cur.execute(
        """
        SELECT pc.input_param_name AS input_param,
               old_v.config_json #>> string_to_array(pc.param_path, '.') AS old_value,
               new_v.config_json #>> string_to_array(pc.param_path, '.') AS new_value,
               NULL::text AS changed_by,
               NULL::text AS reason
          FROM bot_param.bot_config_version new_v
          JOIN bot_param.bot_config_version old_v
            ON old_v.env = new_v.env
           AND old_v.account_login = new_v.account_login
           AND old_v.bot_kind = new_v.bot_kind
           AND old_v.bot_id = new_v.bot_id
           AND old_v.version_no = %s
          JOIN bot_param.bot_config_param_catalog pc
            ON pc.bot_kind = new_v.bot_kind
         WHERE new_v.config_version_id = %s
           AND pc.input_param_name IS NOT NULL
           AND pc.param_path IS NOT NULL
           AND COALESCE(pc.user_editable, true) = true
           AND (old_v.config_json #>> string_to_array(pc.param_path, '.'))
               IS DISTINCT FROM
               (new_v.config_json #>> string_to_array(pc.param_path, '.'))
         ORDER BY pc.sort_order, pc.input_param_name
        """,
        (old_version_no, active_config_id),
    )
    return [dict(row) for row in cur.fetchall()]


RM_CONTROLLER_BOT = "rm_controller"
RM_CONTROLLER_BOT_LABEL = "RM* Controller"
RM_CONTROLLER_BOT_ID = "rm_controller"
RM_CONTROLLER_SOURCE_ID = "rm"
RM_CONTROL_ACTION_COMMANDS = {
    "status": "status",
    "config": "config",
    "resume": "resume",
    "rm_daystart": "rm_daystart",
    "rm_reset": "rm_reset",
}
RM_CONTROL_COMMAND_DESCRIPTIONS = {
    "status": "Show account RM state, active account stop, per-bot stops and runtime heartbeat.",
    "config": "Show current RM limits and owner configuration.",
    "stop_account": "Stop trading on the selected account until the selected period expires or resume is sent.",
    "pause_account": "Pause new trading on the selected account without flattening open positions.",
    "resume": "Clear current account stop and keep today's baseline.",
    "rm_daystart": "Clear current account stop and restore today's day-start baseline.",
    "rm_reset": "Rearm RM from current balance/equity.",
    "stop_bots": "Hard stop for selected bot families: FLATTEN_AND_HALT. Blocks new entries and tells bots to flatten/close open positions.",
    "pause_bots": "Soft pause for selected bot families: HALT_ONLY. Blocks new entries, but keeps open positions running. No flatten.",
    "resume_bots": "Clear stops for selected bot families.",
}
RM_CONTROL_INPUT_PARAMS = {
    "RM_UseProfitTargetPct": "bool",
    "RM_ProfitTargetPct": "double",
    "RM_UseProfitTargetAbs": "bool",
    "RM_ProfitTargetAbs": "double",
    "RM_UseLossLimitPct": "bool",
    "RM_LossLimitPct": "double",
    "RM_UseLossLimitAbs": "bool",
    "RM_LossLimitAbs": "double",
    "RM_MetricMode": "metric",
    "RM_ActionOnProfit": "action",
    "RM_ActionOnLoss": "action",
    "RM_ResetMode": "reset",
    "RM_OwnerHeartbeatSec": "int",
    "RM_ManualResetNow": "bool",
}
RM_CONTROLLER_PARAM_DEFS = [
    ("Account RM Owner", "RM_UseProfitTargetPct", "Use account profit target as percent from baseline.", "bool"),
    ("Account RM Owner", "RM_ProfitTargetPct", "Account profit target percentage from baseline.", "double"),
    ("Account RM Owner", "RM_UseProfitTargetAbs", "Use account profit target as absolute account currency amount.", "bool"),
    ("Account RM Owner", "RM_ProfitTargetAbs", "Account profit target absolute amount.", "double"),
    ("Account RM Owner", "RM_UseLossLimitPct", "Use account loss limit as percent from baseline.", "bool"),
    ("Account RM Owner", "RM_LossLimitPct", "Account loss limit percentage from baseline.", "double"),
    ("Account RM Owner", "RM_UseLossLimitAbs", "Use account loss limit as absolute account currency amount.", "bool"),
    ("Account RM Owner", "RM_LossLimitAbs", "Account loss limit absolute amount.", "double"),
    ("Account RM Owner", "RM_MetricMode", "Account metric used for RM thresholds.", "metric"),
    ("Account RM Owner", "RM_ActionOnProfit", "Action taken when a profit target is reached.", "action"),
    ("Account RM Owner", "RM_ActionOnLoss", "Action taken when a loss limit is reached.", "action"),
    ("Account RM Owner", "RM_ResetMode", "Automatic reset mode for normal RM stops.", "reset"),
    ("Account RM Owner", "RM_OwnerHeartbeatSec", "Maximum owner heartbeat age in seconds.", "int"),
    ("Account RM Owner", "RM_ManualResetNow", "One-shot manual reset request for triggered RM state.", "bool"),
]
RM_CONTROLLER_DEFAULT_VALUES = {
    "RM_UseProfitTargetPct": "false",
    "RM_ProfitTargetPct": "2.0",
    "RM_UseProfitTargetAbs": "true",
    "RM_ProfitTargetAbs": "60.0",
    "RM_UseLossLimitPct": "true",
    "RM_LossLimitPct": "2.0",
    "RM_UseLossLimitAbs": "false",
    "RM_LossLimitAbs": "30.0",
    "RM_MetricMode": "RM_METRIC_BALANCE",
    "RM_ActionOnProfit": "RM_ACTION_FLATTEN_AND_HALT",
    "RM_ActionOnLoss": "RM_ACTION_FLATTEN_AND_HALT",
    "RM_ResetMode": "RM_RESET_NEXT_DAY",
    "RM_OwnerHeartbeatSec": "30",
    "RM_ManualResetNow": "false",
}


def rm_controller_config_json(account_login):
    config = {
        "schema_version": 1,
        "env": "prod",
        "account_login": int(account_login),
        "bot_kind": RM_CONTROLLER_BOT,
        "bot_id": RM_CONTROLLER_BOT_ID,
        "source_id": RM_CONTROLLER_SOURCE_ID,
        "version_no": 1,
        "enabled": True,
        "allow_new_entries": True,
    }
    for key, value in RM_CONTROLLER_DEFAULT_VALUES.items():
        value_type = RM_CONTROL_INPUT_PARAMS.get(key)
        if value_type == "bool":
            config[key] = normalize_rm_bool_value(value, key) == "true"
        elif value_type == "double":
            config[key] = float(value)
        elif value_type == "int":
            config[key] = int(value)
        else:
            config[key] = value
    return config


def rm_controller_config_hash(config):
    payload = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def rm_controller_catalog_value_type(value_type):
    if value_type == "double":
        return "numeric"
    if value_type in ("metric", "action", "reset"):
        return "text"
    return value_type


def ensure_rm_controller_catalog(cur):
    cur.execute("SELECT 1 FROM bot_param.bot_catalog WHERE bot_kind = %s", (RM_CONTROLLER_BOT,))
    if cur.fetchone():
        cur.execute(
            """
            UPDATE bot_param.bot_catalog
               SET bot_id = %s,
                   source_id = %s,
                   display_name = %s,
                   ea_file = %s,
                   sort_order = %s,
                   enabled = true
             WHERE bot_kind = %s
            """,
            (RM_CONTROLLER_BOT_ID, RM_CONTROLLER_SOURCE_ID, RM_CONTROLLER_BOT_LABEL, "RM_Controller.mq5", 50, RM_CONTROLLER_BOT),
        )
    else:
        cur.execute(
            """
            INSERT INTO bot_param.bot_catalog (
                bot_kind, bot_id, source_id, display_name, ea_file, sort_order, enabled
            )
            VALUES (%s, %s, %s, %s, %s, %s, true)
            """,
            (RM_CONTROLLER_BOT, RM_CONTROLLER_BOT_ID, RM_CONTROLLER_SOURCE_ID, RM_CONTROLLER_BOT_LABEL, "RM_Controller.mq5", 50),
        )

    for sort_order, (section_name, input_param, desc, value_type) in enumerate(RM_CONTROLLER_PARAM_DEFS, start=1):
        allowed_values = [choice["allowed_value"] for choice in rm_controller_choices(input_param)] or None
        cur.execute(
            """
            SELECT 1
              FROM bot_param.bot_config_param_catalog
             WHERE bot_kind = %s
               AND param_key = %s
            """,
            (RM_CONTROLLER_BOT, input_param),
        )
        catalog_value_type = rm_controller_catalog_value_type(value_type)
        if cur.fetchone():
            cur.execute(
                """
                UPDATE bot_param.bot_config_param_catalog
                   SET section_name = %s,
                       input_param_name = %s,
                       display_name = %s,
                       param_desc = %s,
                       param_path = %s,
                       value_type = %s,
                       allowed_values = %s,
                       sort_order = %s,
                       user_editable = COALESCE(user_editable, true)
                 WHERE bot_kind = %s
                   AND param_key = %s
                """,
                (
                    section_name,
                    input_param,
                    input_param,
                    desc,
                    input_param,
                    catalog_value_type,
                    allowed_values,
                    sort_order,
                    RM_CONTROLLER_BOT,
                    input_param,
                ),
            )
        else:
            cur.execute(
                """
                INSERT INTO bot_param.bot_config_param_catalog (
                    bot_kind, section_name, param_key, input_param_name, display_name,
                    param_desc, param_path, value_type, allowed_values, sort_order, user_editable
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, true)
                """,
                (RM_CONTROLLER_BOT, section_name, input_param, input_param, input_param, desc, input_param, catalog_value_type, allowed_values, sort_order),
            )

    for input_param, value_type in RM_CONTROL_INPUT_PARAMS.items():
        for sort_order, choice in enumerate(rm_controller_choices(input_param), start=1):
            cur.execute(
                """
                INSERT INTO bot_param.bot_config_allowed_value (
                    bot, input_param, allowed_value, value_desc, sort_order
                )
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (bot, input_param, allowed_value)
                DO UPDATE SET
                    value_desc = EXCLUDED.value_desc,
                    sort_order = EXCLUDED.sort_order
                """,
                (
                    RM_CONTROLLER_BOT,
                    input_param,
                    choice["allowed_value"],
                    choice.get("value_desc"),
                    sort_order,
                ),
            )


def refresh_rm_controller_editor(cur, account_login):
    for fn_name in ("refresh_bot_config_editor", "refresh_bot_config_user_editor"):
        savepoint = f"rm_{fn_name}"
        cur.execute(f"SAVEPOINT {savepoint}")
        try:
            cur.execute(f"SELECT bot_param.{fn_name}(%s, %s, %s)", ("prod", int(account_login), RM_CONTROLLER_BOT))
            cur.execute(f"RELEASE SAVEPOINT {savepoint}")
        except Exception:
            cur.execute(f"ROLLBACK TO SAVEPOINT {savepoint}")
            cur.execute(f"RELEASE SAVEPOINT {savepoint}")
            raise


def ensure_rm_controller_db_config(cur, account_login):
    ensure_rm_controller_catalog(cur)
    cur.execute(
        """
        SELECT 1
          FROM bot_param.bot_config_current
         WHERE env = 'prod'
           AND account_login = %s
           AND bot_kind = %s
           AND bot_id = %s
        """,
        (int(account_login), RM_CONTROLLER_BOT, RM_CONTROLLER_BOT_ID),
    )
    if not cur.fetchone():
        config = rm_controller_config_json(account_login)
        config_hash = rm_controller_config_hash(config)
        cur.execute(
            """
            SELECT config_version_id, config_hash
              FROM bot_param.bot_config_version
             WHERE env = 'prod'
               AND account_login = %s
               AND bot_kind = %s
               AND bot_id = %s
               AND version_no = 1
            """,
            (int(account_login), RM_CONTROLLER_BOT, RM_CONTROLLER_BOT_ID),
        )
        version_row = cur.fetchone()
        if version_row:
            config_id = version_row["config_version_id"] if hasattr(version_row, "keys") else version_row[0]
            config_hash = version_row["config_hash"] if hasattr(version_row, "keys") else version_row[1]
        else:
            cur.execute(
                """
                INSERT INTO bot_param.bot_config_version (
                    env, account_login, bot_kind, bot_id, config_scope, version_no,
                    status, config_json, config_hash, validation_status, validation_errors,
                    created_by, created_source, created_reason,
                    approved_by, approved_at, activated_by, activated_at
                )
                VALUES (
                    'prod', %s, %s, %s, 'bot', 1,
                    'active', %s::jsonb, %s, 'ok', '[]'::jsonb,
                    current_user::text, 'config-ui-bootstrap', 'RM controller initial config',
                    current_user::text, now(), current_user::text, now()
                )
                RETURNING config_version_id
                """,
                (int(account_login), RM_CONTROLLER_BOT, RM_CONTROLLER_BOT_ID, json.dumps(config, separators=(",", ":")), config_hash),
            )
            config_id = cur.fetchone()[0]
        cur.execute(
            """
            INSERT INTO bot_param.bot_config_current (
                env, account_login, bot_kind, bot_id, active_version_no,
                active_config_id, config_hash, updated_by, updated_source
            )
            VALUES ('prod', %s, %s, %s, 1, %s, %s, current_user::text, 'config-ui-bootstrap')
            """,
            (int(account_login), RM_CONTROLLER_BOT, RM_CONTROLLER_BOT_ID, config_id, config_hash),
        )
    refresh_rm_controller_editor(cur, account_login)


def normalize_rm_bot_ids(bot_ids):
    normalized = []
    for bot_id in bot_ids or []:
        value = (bot_id or "").strip().lower()
        if not re.match(r"^(n\d+|bot123)$", value):
            raise ValueError(f"unsupported bot id: {bot_id}")
        if value not in normalized:
            normalized.append(value)
    if not normalized:
        raise ValueError("choose at least one bot")
    return normalized


def is_rm_controller_bot(bot):
    return (bot or "").strip().lower() in (RM_CONTROLLER_BOT, "rm*", "rm")


def normalize_rm_until(value):
    text = (value or "").strip()
    if not re.match(r"^\d{4}\.\d{2}\.\d{2} \d{2}:\d{2}$", text):
        raise ValueError("until must use YYYY.MM.DD HH:MM")
    try:
        datetime.strptime(text, "%Y.%m.%d %H:%M")
    except ValueError:
        raise ValueError("until must be a valid YYYY.MM.DD HH:MM datetime")
    return text


def normalize_rm_bool_value(value, param_name):
    text = (value or "").strip().lower()
    if text in ("true", "1", "on", "enable", "enabled", "yes"):
        return "true"
    if text in ("false", "0", "off", "disable", "disabled", "no"):
        return "false"
    raise ValueError(f"{param_name} expects true/false")


def build_rm_control_command(account_login, req: RmControlCommandRequest):
    action = (req.action or "").strip().lower()
    if action in RM_CONTROL_ACTION_COMMANDS:
        return f"/{int(account_login)} {RM_CONTROL_ACTION_COMMANDS[action]}"

    if action == "stop_account":
        duration = (req.duration or "").strip().lower()
        if duration == "manual":
            return f"/{int(account_login)} halt"
        if duration == "day":
            return f"/{int(account_login)} day_stop"
        if duration == "week":
            return f"/{int(account_login)} week_stop"
        if duration == "until":
            return f"/{int(account_login)} stop {normalize_rm_until(req.until)}"
        raise ValueError("duration must be manual, day, week, or until")

    if action == "pause_account":
        duration = (req.duration or "").strip().lower()
        if duration == "manual":
            return f"/{int(account_login)} pause"
        if duration == "day":
            return f"/{int(account_login)} day_pause"
        if duration == "week":
            return f"/{int(account_login)} week_pause"
        if duration == "until":
            return f"/{int(account_login)} pause {normalize_rm_until(req.until)}"
        raise ValueError("duration must be manual, day, week, or until")

    if action == "stop_bots":
        bots = ",".join(normalize_rm_bot_ids(req.bot_ids))
        duration = (req.duration or "").strip().lower()
        if duration in ("manual", "day", "week"):
            stop_until = duration
        elif duration == "until":
            stop_until = normalize_rm_until(req.until)
        else:
            raise ValueError("duration must be manual, day, week, or until")
        return f"/{int(account_login)} stop bots {bots} {stop_until}"

    if action == "pause_bots":
        bots = ",".join(normalize_rm_bot_ids(req.bot_ids))
        duration = (req.duration or "").strip().lower()
        if duration in ("manual", "day", "week"):
            pause_until = duration
        elif duration == "until":
            pause_until = normalize_rm_until(req.until)
        else:
            raise ValueError("duration must be manual, day, week, or until")
        return f"/{int(account_login)} pause bots {bots} {pause_until}"

    if action == "resume_bots":
        bots = ",".join(normalize_rm_bot_ids(req.bot_ids))
        return f"/{int(account_login)} resume bots {bots}"

    raise ValueError("unsupported RM action")


def rm_controller_choices(input_param):
    value_type = RM_CONTROL_INPUT_PARAMS.get(input_param)
    if value_type == "bool":
        return [
            {"allowed_value": "true", "value_desc": "enabled"},
            {"allowed_value": "false", "value_desc": "disabled"},
        ]
    if value_type == "metric":
        return [
            {"allowed_value": "RM_METRIC_BALANCE", "value_desc": "balance"},
            {"allowed_value": "RM_METRIC_EQUITY", "value_desc": "equity"},
        ]
    if value_type == "action":
        return [
            {"allowed_value": "RM_ACTION_HALT_ONLY", "value_desc": "halt only"},
            {"allowed_value": "RM_ACTION_FLATTEN_AND_HALT", "value_desc": "flatten and halt"},
        ]
    if value_type == "reset":
        return [
            {"allowed_value": "RM_RESET_NEXT_DAY", "value_desc": "next day"},
            {"allowed_value": "RM_RESET_MANUAL", "value_desc": "manual"},
        ]
    return []


def insert_rm_control_command(cur, account, actor, command_text, reason, action):
    payload = {
        "command": command_text,
        "actor": actor,
        "source": "config-ui",
        "action": action,
    }
    cur.execute(
        """
        INSERT INTO bot_param.bot_command (
            env,
            account_login,
            target_bot_kind,
            target_bot_id,
            command_type,
            command_payload,
            priority,
            created_by,
            created_source,
            created_reason
        )
        VALUES (
            'prod',
            %s,
            'rm_controller',
            'rm_controller',
            'RM_CONTROL',
            %s::jsonb,
            250,
            %s,
            'config-ui',
            %s
        )
        RETURNING command_id,
                  account_login,
                  target_bot_kind,
                  target_bot_id,
                  command_type,
                  command_payload,
                  status,
                  created_at
        """,
        (
            account,
            json.dumps(payload, separators=(",", ":")),
            actor,
            reason,
        ),
    )
    return dict(cur.fetchone())


RECOMMENDATION_VALUE_COLUMNS = ("recommended_value", "new_value", "target_value")
RECOMMENDATION_INPUT_COLUMNS = ("input_param", "param_key", "input_param_name")
RECOMMENDATION_OLD_VALUE_COLUMNS = ("old_value", "current_value")


def recommendation_json_expr(columns):
    json_cols = [name for name in ("evidence", "evidence_json") if name in columns]
    if not json_cols:
        return "'{}'::jsonb"
    return "COALESCE(" + ", ".join(f"r.{name}" for name in json_cols) + ", '{}'::jsonb)"


def recommendation_text_expr(columns, column_names, evidence_keys, default_sql="NULL::text"):
    evidence = recommendation_json_expr(columns)
    parts = []
    for name in column_names:
        if name in columns:
            parts.append(f"NULLIF(to_jsonb(r.{name}) #>> '{{}}', '')")
    for key in evidence_keys:
        parts.append(f"NULLIF({evidence}->>'{key}', '')")
    if not parts:
        return default_sql
    return "COALESCE(" + ", ".join(parts) + ")"


def recommendation_numeric_expr(columns, column_name):
    if column_name not in columns:
        return "NULL::numeric"
    return f"r.{column_name}::numeric"


def recommendation_bool_expr(columns, column_name):
    if column_name not in columns:
        return "NULL::boolean"
    return f"r.{column_name}::boolean"


def recommendation_timestamp_expr(columns, column_name):
    if column_name not in columns:
        return "NULL::timestamptz"
    return f"r.{column_name}"


def recommendation_select_sql(columns, for_update=False, by_id=False):
    evidence = recommendation_json_expr(columns)
    bot_kind_expr = "r.bot_kind" if "bot_kind" in columns else "NULL::text"
    bot_id_expr = "r.bot_id" if "bot_id" in columns else bot_kind_expr
    symbol_expr = "r.symbol" if "symbol" in columns else "NULL::text"
    tf_expr = "r.tf" if "tf" in columns else ("r.timeframe" if "timeframe" in columns else "NULL::text")
    status_expr = "r.status" if "status" in columns else "'new'::text"
    decision_type_expr = "r.decision_type" if "decision_type" in columns else "NULL::text"
    severity_expr = "r.severity" if "severity" in columns else "NULL::text"
    confidence_expr = recommendation_numeric_expr(columns, "confidence")
    trend_expr = recommendation_numeric_expr(columns, "trend_strength")
    min_sample_expr = recommendation_bool_expr(columns, "min_sample_reached")
    created_at_expr = recommendation_timestamp_expr(columns, "created_at")
    expires_at_expr = recommendation_timestamp_expr(columns, "expires_at")
    cooldown_expr = recommendation_timestamp_expr(columns, "cooldown_until")
    input_expr = recommendation_text_expr(columns, RECOMMENDATION_INPUT_COLUMNS, ("input_param", "param_key", "input_param_name"))
    recommended_expr = recommendation_text_expr(columns, RECOMMENDATION_VALUE_COLUMNS, ("recommended_value", "new_value", "target_value"))
    old_expr = recommendation_text_expr(columns, RECOMMENDATION_OLD_VALUE_COLUMNS, ("old_value", "current_value"))
    reason_expr = "r.reason" if "reason" in columns else "NULL::text"
    order_expr = "r.created_at DESC, r.recommendation_id DESC" if "created_at" in columns else "r.recommendation_id DESC"
    lock_expr = " FOR UPDATE OF r" if for_update else ""

    id_filter = "AND r.recommendation_id = %s" if by_id else ""
    status_filter = "" if by_id or "status" not in columns else "AND r.status IN ('new', 'approved')"
    return f"""
        SELECT r.recommendation_id,
               r.account_login,
               {bot_kind_expr} AS bot_kind,
               {bot_id_expr} AS bot_id,
               {symbol_expr} AS symbol,
               {tf_expr} AS tf,
               {decision_type_expr} AS decision_type,
               {severity_expr} AS severity,
               {status_expr} AS status,
               {confidence_expr} AS confidence,
               {trend_expr} AS trend_strength,
               {min_sample_expr} AS min_sample_reached,
               {created_at_expr} AS created_at,
               {expires_at_expr} AS expires_at,
               {cooldown_expr} AS cooldown_until,
               {input_expr} AS input_param,
               {old_expr} AS old_value,
               {recommended_expr} AS recommended_value,
               {reason_expr} AS reason,
               {evidence} AS evidence
          FROM bot_online.bot_control_recommendation r
          JOIN bot_param.operator_account oa
            ON oa.env = 'prod'
           AND oa.account_login = r.account_login
           AND oa.db_user = %s
           AND oa.enabled = true
         WHERE r.account_login = %s
           {"" if "env" not in columns else "AND r.env = 'prod'"}
           {id_filter}
           {status_filter}
         ORDER BY {order_expr}
         LIMIT 100
         {lock_expr}
    """


def recommendation_status_update_sql(columns, status, actor, reason):
    set_parts = ["status = %s"]
    values = [status]
    if status == "approved":
        if "approved_by" in columns:
            set_parts.append("approved_by = %s")
            values.append(actor)
        if "approved_at" in columns:
            set_parts.append("approved_at = now()")
    elif status == "rejected":
        if "rejected_by" in columns:
            set_parts.append("rejected_by = %s")
            values.append(actor)
        if "rejected_at" in columns:
            set_parts.append("rejected_at = now()")
    if reason and "operator_reason" in columns:
        set_parts.append("operator_reason = %s")
        values.append(reason)
    sql = f"""
        UPDATE bot_online.bot_control_recommendation
           SET {", ".join(set_parts)}
         WHERE recommendation_id = %s
    """
    return sql, values


def config_ui_login_html(error=None):
    error_block = ""
    if error:
        error_block = f'<div class="error">{html_lib.escape(error)}</div>'
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>BOT parameters Login</title>
  <style>
    :root {{ color-scheme: light; font-family: Inter, Segoe UI, Arial, sans-serif; }}
    body {{ margin: 0; min-height: 100vh; display: grid; place-items: center; background: #f4f6f8; color: #17202a; }}
    main {{ width: min(420px, calc(100vw - 32px)); background: #fff; border: 1px solid #d8dee6; border-radius: 8px; padding: 24px; box-shadow: 0 10px 30px rgba(20, 30, 45, .08); }}
    h1 {{ font-size: 22px; margin: 0 0 18px; letter-spacing: 0; }}
    label {{ display: block; font-size: 13px; color: #465466; margin: 14px 0 6px; }}
    input {{ width: 100%; box-sizing: border-box; border: 1px solid #b8c2cc; border-radius: 6px; padding: 12px; font-size: 16px; }}
    button {{ width: 100%; margin-top: 18px; border: 0; border-radius: 6px; padding: 12px 14px; font-size: 16px; font-weight: 600; background: #1769aa; color: #fff; cursor: pointer; }}
    .error {{ border: 1px solid #e09a9a; background: #fff1f1; color: #8d1f1f; border-radius: 6px; padding: 10px 12px; margin-bottom: 12px; font-size: 14px; }}
  </style>
</head>
<body>
  <main>
    <h1>BOT parameters</h1>
    {error_block}
    <form method="post" action="/config-ui/login">
      <label for="username">Login</label>
      <input id="username" name="username" autocomplete="username" required>
      <label for="password">Password</label>
      <input id="password" name="password" type="password" autocomplete="current-password" required>
      <button type="submit">Sign in</button>
    </form>
  </main>
</body>
</html>"""


CONFIG_UI_APP_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>BOT parameters</title>
  <style>
    :root {
      color-scheme: light;
      font-family: Inter, Segoe UI, Arial, sans-serif;
      --border: #d7dde5;
      --text: #152033;
      --muted: #667085;
      --accent: #1769aa;
      --bg: #f5f7fa;
      --panel: #ffffff;
      --danger: #9f1d1d;
      --rm-bg: #edf8f4;
      --rm-border: #8bc9ba;
      --rm-accent: #0f766e;
      --rm-soft: #d9f0e9;
    }
    * { box-sizing: border-box; }
    body { margin: 0; background: var(--bg); color: var(--text); }
    header { position: sticky; top: 0; z-index: 20; background: var(--panel); border-bottom: 1px solid var(--border); }
    .header-inner { max-width: 1280px; margin: 0 auto; padding: 12px 16px; display: flex; gap: 12px; align-items: center; justify-content: space-between; }
    .title { font-size: 18px; font-weight: 700; letter-spacing: 0; }
    .meta { display: flex; gap: 10px; align-items: center; color: var(--muted); font-size: 13px; flex-wrap: wrap; }
    .logout-form { margin: 0; }
    button, select, input { font: inherit; }
    button { border: 0; border-radius: 6px; padding: 10px 12px; font-weight: 600; cursor: pointer; background: var(--accent); color: #fff; }
    button.secondary { background: #eef2f6; color: #1f2a3d; border: 1px solid var(--border); }
    button:disabled { opacity: .55; cursor: not-allowed; }
    main { max-width: 1280px; margin: 0 auto; padding: 16px; }
    .toolbar { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 10px; align-items: end; margin-bottom: 12px; }
    .field { display: flex; flex-direction: column; gap: 5px; min-width: 0; }
    label { color: var(--muted); font-size: 12px; font-weight: 600; }
    select, input[type="text"], input[type="search"], input[type="date"], textarea { border: 1px solid #b8c2cc; border-radius: 6px; background: #fff; color: var(--text); min-height: 38px; padding: 8px 10px; width: 100%; }
    textarea { resize: vertical; min-height: 58px; line-height: 1.3; }
    .copy-line { display: flex; gap: 8px; align-items: center; min-height: 38px; border: 1px solid var(--border); border-radius: 6px; background: #fff; padding: 8px 10px; }
    .copy-line input { width: auto; }
    .target-list { min-height: 86px; max-height: 112px; overflow-y: auto; border: 1px solid #b8c2cc; border-radius: 6px; background: #fff; padding: 4px; }
    .target-list.is-disabled { opacity: .65; background: #f8fafc; }
    .target-option { display: flex; align-items: center; gap: 8px; min-height: 26px; padding: 3px 5px; border-radius: 4px; color: var(--text); font-size: 13px; font-weight: 500; }
    .target-option:hover { background: #eef4fb; }
    .target-option.is-unavailable { color: var(--muted); }
    .target-option input { width: auto; flex: 0 0 auto; }
    .target-option span { min-width: 0; overflow-wrap: anywhere; }
    .tabs { display: flex; gap: 6px; align-items: center; margin: 10px 0 12px; border-bottom: 1px solid var(--border); overflow-x: auto; }
    .tab-btn { flex: 0 0 auto; border: 1px solid transparent; border-bottom: 0; border-radius: 8px 8px 0 0; padding: 10px 14px; background: transparent; color: #314052; font-weight: 700; }
    .tab-btn:hover { background: #eef4fb; }
    .tab-btn.is-active { background: var(--panel); border-color: var(--border); color: var(--accent); }
    #rmTab:hover { background: var(--rm-soft); }
    #rmTab.is-active { background: var(--rm-bg); border-color: var(--rm-border); color: var(--rm-accent); }
    .tab-panel[hidden] { display: none; }
    .rm-control { margin-bottom: 12px; padding: 12px; background: var(--rm-bg); border: 1px solid var(--rm-border); border-radius: 8px; box-shadow: inset 4px 0 0 var(--rm-accent); }
    .rm-head { display: flex; align-items: center; justify-content: space-between; gap: 10px; margin-bottom: 10px; }
    .rm-title { font-size: 15px; font-weight: 700; color: #0b534d; }
    .rm-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 10px; align-items: start; }
    .rm-check-list { min-height: 38px; max-height: 132px; overflow-y: auto; border: 1px solid var(--rm-border); border-radius: 6px; background: #fff; padding: 4px; }
    .rm-option { display: flex; align-items: center; gap: 8px; min-height: 28px; padding: 3px 5px; border-radius: 4px; color: var(--text); font-size: 13px; font-weight: 500; }
    .rm-option:hover { background: var(--rm-soft); }
    .rm-option input { width: auto; flex: 0 0 auto; }
    .rm-desc { min-height: 34px; padding: 7px 0 0; color: var(--muted); font-size: 12px; line-height: 1.35; white-space: pre-line; }
    .rm-preview { min-height: 38px; max-height: 96px; overflow: auto; margin: 0; padding: 8px 10px; border: 1px solid var(--rm-border); border-radius: 6px; background: #f7fcfa; color: #26413d; font: 12px Consolas, Menlo, monospace; white-space: pre-wrap; overflow-wrap: anywhere; }
    .rm-footer { display: grid; grid-template-columns: minmax(0, 1fr) auto; gap: 10px; align-items: start; margin-top: 10px; }
    .runtime-list { margin-top: 10px; overflow-x: hidden; }
    .runtime-list table { min-width: 0; table-layout: fixed; }
    .runtime-list th:nth-child(1), .runtime-list td:nth-child(1) { width: 18%; }
    .runtime-list th:nth-child(2), .runtime-list td:nth-child(2) { width: 18%; }
    .runtime-list th:nth-child(3), .runtime-list td:nth-child(3) { width: 18%; }
    .runtime-list th:nth-child(4), .runtime-list td:nth-child(4) { width: 18%; }
    .runtime-list th:nth-child(5), .runtime-list td:nth-child(5) { width: 28%; }
    .runtime-list td { overflow-wrap: anywhere; }
    .status { min-height: 28px; font-size: 14px; color: var(--muted); margin: 8px 0; }
    .status.error { color: var(--danger); }
    .table-wrap { overflow-x: hidden; background: var(--panel); border: 1px solid var(--border); border-radius: 8px; }
    table { width: 100%; border-collapse: collapse; table-layout: fixed; }
    th, td { border-bottom: 1px solid #e8edf3; padding: 9px 10px; text-align: left; vertical-align: top; }
    th { background: #f8fafc; color: #3a4656; font-size: 12px; }
    td { font-size: 13px; }
    tr.hidden { display: none; }
    tr.group-break td { border-top: 10px solid var(--bg); }
    tr.group-break td:first-child { box-shadow: inset 4px 0 0 #3d7cae; }
    tr.group-break .group { background: #eef4fb; color: #1f5f8c; }
    .params-table th:nth-child(1), .params-table td.group { width: 9%; }
    .params-table th:nth-child(2), .params-table td.param { width: 16%; }
    .params-table th:nth-child(3), .params-table td.desc { width: 21%; }
    .params-table th:nth-child(4), .params-table td.current { width: 14%; }
    .params-table th:nth-child(5), .params-table td.prev-value { width: 15%; }
    .params-table th:nth-child(6), .params-table td.new-value { width: 14%; }
    .params-table th:nth-child(7), .params-table td.reason { width: 11%; }
    .group { color: #49566a; font-weight: 600; white-space: normal; overflow-wrap: anywhere; }
    .param { font-family: Consolas, Menlo, monospace; font-size: 12px; white-space: normal; overflow-wrap: anywhere; }
    .desc { overflow-wrap: anywhere; }
    .current { font-family: Consolas, Menlo, monospace; color: #147a3d; white-space: pre-wrap; overflow-wrap: anywhere; }
    .prev-value, .new-value, .reason { min-width: 0; }
    .prev-value select, .new-value input, .new-value select, .reason input { width: 100%; min-width: 0; }
    .prev-value select:disabled { color: var(--muted); background: #f8fafc; }
    .catalog-wrap { overflow-x: auto; }
    .catalog-table { min-width: 1040px; }
    .catalog-table th:nth-child(1), .catalog-table td.catalog-editable { width: 7%; }
    .catalog-table th:nth-child(2), .catalog-table td.catalog-section { width: 11%; }
    .catalog-table th:nth-child(3), .catalog-table td.catalog-input { width: 15%; }
    .catalog-table th:nth-child(4), .catalog-table td.catalog-display { width: 15%; }
    .catalog-table th:nth-child(5), .catalog-table td.catalog-desc { width: 27%; }
    .catalog-table th:nth-child(6), .catalog-table td.catalog-values { width: 18%; }
    .catalog-table th:nth-child(7), .catalog-table td.catalog-sort { width: 7%; }
    .catalog-table input, .catalog-table select, .catalog-table textarea { min-width: 0; font-size: 12px; }
    .catalog-table .catalog-number { text-align: right; }
    .catalog-range-grid { display: grid; grid-template-columns: minmax(0, 1fr); gap: 5px; }
    .catalog-toggle { min-height: 38px; display: flex; align-items: center; justify-content: center; }
    .catalog-toggle input { width: auto; }
    .consul-table th:nth-child(1), .consul-table td.rec-id { width: 8%; }
    .consul-table th:nth-child(2), .consul-table td.rec-bot { width: 11%; }
    .consul-table th:nth-child(3), .consul-table td.rec-param { width: 16%; }
    .consul-table th:nth-child(4), .consul-table td.rec-values { width: 18%; }
    .consul-table th:nth-child(5), .consul-table td.rec-signal { width: 15%; }
    .consul-table th:nth-child(6), .consul-table td.rec-reason { width: 22%; }
    .consul-table th:nth-child(7), .consul-table td.rec-actions { width: 10%; }
    .rec-values, .rec-param { font-family: Consolas, Menlo, monospace; font-size: 12px; overflow-wrap: anywhere; }
    .rec-reason { overflow-wrap: anywhere; }
    .rec-signal { color: var(--muted); overflow-wrap: anywhere; }
    .rec-actions { display: flex; gap: 6px; flex-wrap: wrap; }
    .rec-actions button { padding: 7px 9px; font-size: 12px; }
    .rec-actions button.reject { background: #eef2f6; color: #8d1f1f; border: 1px solid #e3b0b0; }
    .analytics-head { display: flex; align-items: center; justify-content: space-between; gap: 10px; margin-bottom: 12px; }
    .analytics-title { font-size: 15px; font-weight: 700; color: #263445; }
    .analytics-filters { margin-bottom: 12px; }
    .analytics-filter-actions { display: flex; gap: 8px; align-items: flex-end; }
    .analytics-filter-actions button { flex: 1 1 0; min-width: 0; }
    .analytics-summary { display: grid; grid-template-columns: repeat(3, minmax(120px, 1fr)); gap: 12px; margin-bottom: 12px; }
    .analytics-stat { min-width: 0; padding: 8px 0; border-bottom: 2px solid #dfe7ef; }
    .analytics-label { color: var(--muted); font-size: 12px; font-weight: 700; text-transform: uppercase; }
    .analytics-value { margin-top: 4px; color: var(--text); font-size: 20px; font-weight: 800; overflow-wrap: anywhere; }
    .analytics-table th:nth-child(1), .analytics-table td.analytics-source { width: 46%; }
    .analytics-table th:nth-child(2), .analytics-table td.analytics-deals { width: 18%; }
    .analytics-table th:nth-child(3), .analytics-table td.analytics-profit { width: 36%; }
    .analytics-source { overflow-wrap: anywhere; }
    .analytics-source-main { color: var(--text); font-size: 16px; font-weight: 800; }
    .analytics-source-bot { color: var(--accent); font-size: 17px; font-weight: 900; }
    .analytics-deals, .analytics-profit { text-align: right; font-family: Consolas, Menlo, monospace; }
    .analytics-table th:nth-child(2), .analytics-deals { border-left: 2px solid #d0d8e2; border-right: 2px solid #d0d8e2; }
    .analytics-deals { background: #f8fbfd; font-weight: 800; }
    .analytics-profit { font-weight: 800; }
    .profit-positive { background: #1f8f4d; color: #fff; }
    .profit-negative { background: #c73535; color: #fff; }
    .footer { position: sticky; bottom: 0; z-index: 15; margin-top: 12px; padding: 10px; display: flex; justify-content: space-between; align-items: center; gap: 12px; background: rgba(245, 247, 250, .96); border: 1px solid var(--border); border-radius: 8px; }
    .changed-count { color: var(--muted); font-size: 14px; }
    @media (max-width: 1100px) {
      .params-table th:first-child, .params-table td.group { display: none; }
      .params-table tr.group-break td:nth-child(2) { box-shadow: inset 4px 0 0 #3d7cae; }
      .params-table th:nth-child(2), .params-table td.param { width: 20%; }
      .params-table th:nth-child(3), .params-table td.desc { width: 25%; }
      .params-table th:nth-child(4), .params-table td.current { width: 17%; }
      .params-table th:nth-child(5), .params-table td.prev-value { width: 16%; }
      .params-table th:nth-child(6), .params-table td.new-value { width: 13%; }
      .params-table th:nth-child(7), .params-table td.reason { width: 9%; }
    }
    @media (max-width: 760px) {
      .header-inner { align-items: flex-start; display: grid; grid-template-columns: minmax(0, 1fr) auto; }
      .logout-form { grid-column: 2; grid-row: 1; }
      .logout-form button { margin-top: 0; min-height: 36px; padding: 8px 10px; }
      .meta { grid-column: 1 / -1; }
      .toolbar { grid-template-columns: 1fr; }
      main { padding: 10px; }
      .tabs { margin-top: 8px; }
      .tab-btn { flex: 1 0 auto; min-height: 42px; }
      .table-wrap { border: 0; border-radius: 0; overflow-x: visible; background: transparent; }
      .params-table, .params-table thead, .params-table tbody, .params-table tr, .params-table td,
      .catalog-table, .catalog-table thead, .catalog-table tbody, .catalog-table tr, .catalog-table td,
      .consul-table, .consul-table thead, .consul-table tbody, .consul-table tr, .consul-table td,
      .analytics-table, .analytics-table thead, .analytics-table tbody, .analytics-table tr, .analytics-table td { display: block; width: 100%; }
      .params-table, .catalog-table, .consul-table, .analytics-table { min-width: 0; border-collapse: separate; }
      .params-table thead, .catalog-table thead, .consul-table thead, .analytics-table thead { display: none; }
      .params-table tr, .catalog-table tr, .consul-table tr, .analytics-table tr { display: grid; grid-template-columns: minmax(0, 1fr); gap: 12px; padding: 14px 12px; border: 1px solid var(--border); border-radius: 8px; margin-bottom: 10px; background: var(--panel); }
      .params-table tr.hidden, .catalog-table tr.hidden, .consul-table tr.hidden, .analytics-table tr.hidden { display: none; }
      .params-table tr.group-break { margin-top: 18px; border-top: 3px solid #3d7cae; }
      .params-table tr.group-break td { border-top: 0; }
      .params-table tr.group-break td:nth-child(2) { box-shadow: none; }
      .params-table td, .catalog-table td, .consul-table td, .analytics-table td { border-bottom: 0; padding: 0; min-width: 0; width: 100% !important; }
      .params-table td.group, .params-table td.reason { display: none; }
      .catalog-table td::before, .consul-table td::before, .analytics-table td::before { content: attr(data-label); display: block; margin-bottom: 5px; font-size: 12px; font-weight: 700; color: var(--muted); }
      .analytics-summary { grid-template-columns: 1fr; }
      .analytics-deals, .analytics-profit { text-align: left; }
      .analytics-profit { padding: 8px 10px !important; }
      .analytics-profit::before { color: rgba(255, 255, 255, .82) !important; }
      .catalog-range-grid { grid-template-columns: 1fr 1fr; }
      .param { grid-column: 1 / -1; font-family: Consolas, Menlo, monospace; font-size: 13px; font-weight: 700; white-space: normal; overflow-wrap: anywhere; }
      .param::after { content: attr(data-desc); display: block; margin-top: 4px; font-family: Inter, Segoe UI, Arial, sans-serif; font-weight: 500; color: var(--muted); }
      .params-table td.desc { display: none; }
      .current, .prev-value, .new-value { grid-column: 1 / -1; min-width: 0; max-width: none; }
      .current::before, .prev-value::before, .new-value::before { display: block; margin-bottom: 5px; font-family: Inter, Segoe UI, Arial, sans-serif; font-size: 12px; font-weight: 700; color: var(--muted); white-space: nowrap; }
      .current::before { content: "current value"; color: #147a3d; }
      .prev-value::before { content: "prev value"; }
      .new-value::before { content: "new"; }
      .current { font-family: Consolas, Menlo, monospace; white-space: pre-wrap; overflow-wrap: anywhere; overflow: visible; }
      .footer { align-items: stretch; flex-direction: column; }
      .footer button { width: 100%; min-height: 44px; }
      select, input[type="text"], input[type="search"], input[type="date"], textarea { min-height: 44px; font-size: 16px; }
      .target-list { max-height: 144px; }
      .target-option { min-height: 34px; font-size: 14px; }
      .rm-control { padding: 10px; }
      .rm-head, .rm-footer { display: grid; grid-template-columns: 1fr; }
      .rm-grid { grid-template-columns: 1fr; }
      .rm-option { min-height: 34px; font-size: 14px; }
      .runtime-list table, .runtime-list tbody, .runtime-list tr, .runtime-list td { display: block; width: 100%; }
      .runtime-list thead { display: none; }
      .runtime-list tr { display: grid; grid-template-columns: 1fr; gap: 6px; padding: 10px; border: 1px solid var(--border); border-radius: 8px; margin-bottom: 8px; }
      .runtime-list td { display: block; width: 100% !important; }
    }
  </style>
</head>
<body>
  <header>
    <div class="header-inner">
      <div>
        <div class="title">BOT parameters</div>
        <div class="meta">
          <span>user: <strong id="sessionUser"></strong></span>
          <span>actor: <strong id="actorName"></strong></span>
          <span>version: <strong id="codeVersion"></strong></span>
        </div>
      </div>
      <form class="logout-form" method="post" action="/config-ui/logout">
        <button class="secondary" type="submit">Logout</button>
      </form>
    </div>
  </header>
  <main>
    <div id="status" class="status"></div>

    <section class="toolbar" aria-label="context selectors">
      <div class="field">
        <label for="accountSelect">Account</label>
        <select id="accountSelect"></select>
      </div>
      <div class="field">
        <label for="botSelect">Bot</label>
        <select id="botSelect"></select>
      </div>
      <div class="field">
        <label>Mode</label>
        <label class="copy-line"><input id="adminModeToggle" type="checkbox"> Admin mode</label>
      </div>
    </section>

    <nav class="tabs" role="tablist" aria-label="form tabs">
      <button id="paramsTab" class="tab-btn is-active" type="button" role="tab" aria-selected="true" aria-controls="paramsPanel">Params</button>
      <button id="paramsConfigTab" class="tab-btn" type="button" role="tab" aria-selected="false" aria-controls="paramsConfigPanel" hidden>Params_config</button>
      <button id="consulTab" class="tab-btn" type="button" role="tab" aria-selected="false" aria-controls="consulPanel" hidden>Consul</button>
      <button id="rmTab" class="tab-btn" type="button" role="tab" aria-selected="false" aria-controls="rmPanel">RM command</button>
      <button id="analyticsTab" class="tab-btn" type="button" role="tab" aria-selected="false" aria-controls="analyticsPanel">Analytics</button>
    </nav>

    <section id="paramsPanel" class="tab-panel" role="tabpanel" aria-labelledby="paramsTab">
      <section class="toolbar" aria-label="filters">
        <div class="field">
          <label for="groupFilter">Group</label>
          <select id="groupFilter"><option value="">All groups</option></select>
        </div>
        <div class="field">
          <label for="searchInput">Search</label>
          <input id="searchInput" type="search" placeholder="input_param or description">
        </div>
        <div class="field">
          <label>Copy</label>
          <label class="copy-line"><input id="copyToggle" type="checkbox"> Apply same values</label>
        </div>
        <div class="field">
          <label id="targetAccountLabel">Target accounts</label>
          <div id="targetAccountList" class="target-list is-disabled" role="group" aria-labelledby="targetAccountLabel"></div>
        </div>
      </section>

      <section class="table-wrap">
        <table class="params-table">
          <thead>
            <tr>
              <th>group</th>
              <th>input_param</th>
              <th>param_desc</th>
              <th>current_value</th>
              <th>prev_value</th>
              <th>new value</th>
              <th>reason</th>
            </tr>
          </thead>
          <tbody id="paramsBody"></tbody>
        </table>
      </section>

      <section class="footer">
        <div id="changedCount" class="changed-count">0 changed rows</div>
        <button id="saveBtn" type="button" disabled>Save changes</button>
      </section>
    </section>

    <section id="paramsConfigPanel" class="tab-panel" role="tabpanel" aria-labelledby="paramsConfigTab" hidden>
      <section class="toolbar" aria-label="catalog filters">
        <div class="field">
          <label for="catalogSearchInput">Search</label>
          <input id="catalogSearchInput" type="search" placeholder="input_param, display, description">
        </div>
      </section>

      <section class="table-wrap catalog-wrap">
        <table class="catalog-table">
          <thead>
            <tr>
              <th>editable</th>
              <th>section</th>
              <th>input_param</th>
              <th>display</th>
              <th>description</th>
              <th>allowed_values</th>
              <th>sort</th>
            </tr>
          </thead>
          <tbody id="paramsConfigBody"></tbody>
        </table>
      </section>

      <section class="footer">
        <div id="paramsConfigChangedCount" class="changed-count">0 catalog rows changed</div>
        <button id="paramsConfigSaveBtn" type="button" disabled>Save catalog</button>
      </section>
    </section>

    <section id="consulPanel" class="tab-panel" role="tabpanel" aria-labelledby="consulTab" hidden>
      <section class="rm-control" aria-label="consul-recommendations">
        <div class="rm-head">
          <div class="rm-title">Parameter recommendations</div>
          <button id="consulRefreshBtn" class="secondary" type="button">Refresh</button>
        </div>
        <section class="table-wrap">
          <table class="consul-table">
            <thead>
              <tr>
                <th>id</th>
                <th>bot</th>
                <th>input_param</th>
                <th>values</th>
                <th>signal</th>
                <th>reason</th>
                <th>action</th>
              </tr>
            </thead>
            <tbody id="consulBody"></tbody>
          </table>
        </section>
      </section>
    </section>

    <section id="rmPanel" class="tab-panel" role="tabpanel" aria-labelledby="rmTab" hidden>
      <section class="rm-control" aria-label="rm-control">
        <div class="rm-head">
          <div class="rm-title">RM control</div>
          <button id="rmRefreshStatusBtn" class="secondary" type="button">Refresh status</button>
        </div>
        <div class="rm-grid">
          <div class="field">
            <label>Accounts</label>
            <div id="rmAccountList" class="rm-check-list"></div>
          </div>
          <div class="field">
            <label for="rmActionSelect">Command</label>
            <select id="rmActionSelect">
              <option value="status">Status</option>
              <option value="config">Config</option>
              <option value="stop_account">Stop account</option>
              <option value="pause_account">Pause account</option>
              <option value="resume">Resume account</option>
              <option value="rm_daystart">RM daystart</option>
              <option value="rm_reset">RM reset</option>
              <option value="stop_bots">Stop selected bots - close positions</option>
              <option value="pause_bots">Pause selected bots - keep positions</option>
              <option value="resume_bots">Resume selected bots</option>
            </select>
            <div id="rmCommandDescription" class="rm-desc"></div>
          </div>
          <div class="field" id="rmBotField">
            <label>Bots</label>
            <div id="rmBotList" class="rm-check-list"></div>
          </div>
          <div class="field" id="rmPeriodField">
            <label>Period</label>
            <div id="rmPeriodList" class="rm-check-list">
              <label class="rm-option"><input type="checkbox" value="manual" checked> Manual resume</label>
              <label class="rm-option"><input type="checkbox" value="day"> Day</label>
              <label class="rm-option"><input type="checkbox" value="week"> Week</label>
              <label class="rm-option"><input type="checkbox" value="until"> Until</label>
            </div>
          </div>
          <div class="field" id="rmUntilField">
            <label for="rmUntilInput">Until</label>
            <input id="rmUntilInput" type="text" placeholder="YYYY.MM.DD HH:MM">
          </div>
        </div>
        <div class="rm-footer">
          <pre id="rmCommandPreview" class="rm-preview"></pre>
          <button id="rmSendBtn" type="button">Send RM command</button>
        </div>
        <div id="rmRuntimeList" class="runtime-list"></div>
      </section>
    </section>

    <section id="analyticsPanel" class="tab-panel" role="tabpanel" aria-labelledby="analyticsTab" hidden>
      <div class="analytics-head">
        <div class="analytics-title">Profit by source_id</div>
        <button id="analyticsRefreshBtn" class="secondary" type="button">Refresh</button>
      </div>
      <section class="toolbar analytics-filters" aria-label="analytics filters">
        <div class="field">
          <label for="analyticsYearInput">Year</label>
          <input id="analyticsYearInput" type="text" inputmode="numeric" placeholder="2026">
        </div>
        <div class="field">
          <label for="analyticsMonthSelect">Month</label>
          <select id="analyticsMonthSelect">
            <option value="">All / latest</option>
            <option value="1">January</option>
            <option value="2">February</option>
            <option value="3">March</option>
            <option value="4">April</option>
            <option value="5">May</option>
            <option value="6">June</option>
            <option value="7">July</option>
            <option value="8">August</option>
            <option value="9">September</option>
            <option value="10">October</option>
            <option value="11">November</option>
            <option value="12">December</option>
          </select>
        </div>
        <div class="field">
          <label for="analyticsFiledateInput">Date</label>
          <input id="analyticsFiledateInput" type="date">
        </div>
        <div class="field">
          <label>Filter</label>
          <div class="analytics-filter-actions">
            <button id="analyticsApplyBtn" type="button">Apply</button>
            <button id="analyticsClearBtn" class="secondary" type="button">Clear</button>
          </div>
        </div>
      </section>
      <section class="analytics-summary" aria-label="analytics totals">
        <div class="analytics-stat">
          <div class="analytics-label">Period</div>
          <div id="analyticsFiledate" class="analytics-value">-</div>
        </div>
        <div class="analytics-stat">
          <div class="analytics-label">Total deals</div>
          <div id="analyticsTotalDeals" class="analytics-value">0</div>
        </div>
        <div class="analytics-stat">
          <div class="analytics-label">Total profit</div>
          <div id="analyticsTotalProfit" class="analytics-value">0.00</div>
        </div>
      </section>
      <section class="table-wrap">
        <table class="analytics-table">
          <thead>
            <tr>
              <th>source_id (bot)</th>
              <th>deals</th>
              <th>profit</th>
            </tr>
          </thead>
          <tbody id="analyticsBody"></tbody>
        </table>
      </section>
    </section>
  </main>
  <script>
    const SESSION_USER = __SESSION_USER__;
    const CONFIG_ACTOR = __CONFIG_ACTOR__;
    const CODE_VERSION = __CODE_VERSION__;
    const choiceCache = new Map();
    const choiceBulkCache = new Set();
    const RM_COMMAND_DESCRIPTIONS = {
      status: 'Show account RM state, active account stop, selected bot stops and runtime heartbeat.',
      config: 'Show current RM limits and owner configuration.',
      stop_account: 'Stop trading on the selected account for the selected period.',
      pause_account: 'Pause new trading on the selected account without flattening open positions.',
      resume: "Clear current account stop and keep today's baseline.",
      rm_daystart: "Clear current account stop and restore today's day-start baseline.",
      rm_reset: 'Rearm RM from current balance/equity.',
      stop_bots: 'Hard stop for selected bot families: FLATTEN_AND_HALT.\nBlocks new entries and tells bots to flatten/close open positions for the selected period.',
      pause_bots: 'Soft pause for selected bot families: HALT_ONLY.\nBlocks new entries for the selected period, but keeps open positions running. No flatten.',
      resume_bots: 'Clear stops for selected bot families.'
    };
    let currentBotRows = [];
    let currentCatalogRows = [];
    let paramsLoadSeq = 0;
    let analyticsLoaded = false;
    const CATALOG_VALUE_TYPES = ['bool', 'int', 'numeric', 'text', 'json'];

    const els = {
      sessionUser: document.getElementById('sessionUser'),
      actorName: document.getElementById('actorName'),
      codeVersion: document.getElementById('codeVersion'),
      accountSelect: document.getElementById('accountSelect'),
      botSelect: document.getElementById('botSelect'),
      adminModeToggle: document.getElementById('adminModeToggle'),
      groupFilter: document.getElementById('groupFilter'),
      searchInput: document.getElementById('searchInput'),
      copyToggle: document.getElementById('copyToggle'),
      targetAccountList: document.getElementById('targetAccountList'),
      rmAccountList: document.getElementById('rmAccountList'),
      rmActionSelect: document.getElementById('rmActionSelect'),
      rmCommandDescription: document.getElementById('rmCommandDescription'),
      rmBotField: document.getElementById('rmBotField'),
      rmBotList: document.getElementById('rmBotList'),
      rmPeriodField: document.getElementById('rmPeriodField'),
      rmPeriodList: document.getElementById('rmPeriodList'),
      rmUntilField: document.getElementById('rmUntilField'),
      rmUntilInput: document.getElementById('rmUntilInput'),
      rmCommandPreview: document.getElementById('rmCommandPreview'),
      rmSendBtn: document.getElementById('rmSendBtn'),
      rmRefreshStatusBtn: document.getElementById('rmRefreshStatusBtn'),
      rmRuntimeList: document.getElementById('rmRuntimeList'),
      paramsTab: document.getElementById('paramsTab'),
      paramsConfigTab: document.getElementById('paramsConfigTab'),
      consulTab: document.getElementById('consulTab'),
      rmTab: document.getElementById('rmTab'),
      analyticsTab: document.getElementById('analyticsTab'),
      paramsPanel: document.getElementById('paramsPanel'),
      paramsConfigPanel: document.getElementById('paramsConfigPanel'),
      consulPanel: document.getElementById('consulPanel'),
      rmPanel: document.getElementById('rmPanel'),
      analyticsPanel: document.getElementById('analyticsPanel'),
      analyticsRefreshBtn: document.getElementById('analyticsRefreshBtn'),
      analyticsApplyBtn: document.getElementById('analyticsApplyBtn'),
      analyticsClearBtn: document.getElementById('analyticsClearBtn'),
      analyticsYearInput: document.getElementById('analyticsYearInput'),
      analyticsMonthSelect: document.getElementById('analyticsMonthSelect'),
      analyticsFiledateInput: document.getElementById('analyticsFiledateInput'),
      analyticsFiledate: document.getElementById('analyticsFiledate'),
      analyticsTotalDeals: document.getElementById('analyticsTotalDeals'),
      analyticsTotalProfit: document.getElementById('analyticsTotalProfit'),
      analyticsBody: document.getElementById('analyticsBody'),
      consulBody: document.getElementById('consulBody'),
      consulRefreshBtn: document.getElementById('consulRefreshBtn'),
      status: document.getElementById('status'),
      paramsBody: document.getElementById('paramsBody'),
      changedCount: document.getElementById('changedCount'),
      saveBtn: document.getElementById('saveBtn'),
      catalogSearchInput: document.getElementById('catalogSearchInput'),
      paramsConfigBody: document.getElementById('paramsConfigBody'),
      paramsConfigChangedCount: document.getElementById('paramsConfigChangedCount'),
      paramsConfigSaveBtn: document.getElementById('paramsConfigSaveBtn')
    };

    els.sessionUser.textContent = SESSION_USER;
    els.actorName.textContent = CONFIG_ACTOR;
    els.codeVersion.textContent = CODE_VERSION;

    function setStatus(text, isError = false) {
      els.status.textContent = text || '';
      els.status.className = isError ? 'status error' : 'status';
    }

    function updateAdminModeUi() {
      const showAdmin = !!els.adminModeToggle.checked;
      els.paramsConfigTab.hidden = !showAdmin;
      if (!showAdmin && !els.paramsConfigPanel.hidden) switchFormTab('params');
    }

    function setConsulTabVisible(visible) {
      els.consulTab.hidden = !visible;
      if (!visible && !els.consulPanel.hidden) switchFormTab('params');
    }

    function switchFormTab(tab) {
      if (tab === 'catalog' && !els.adminModeToggle.checked) tab = 'params';
      if (tab === 'consul' && els.consulTab.hidden) tab = 'params';
      const showRm = tab === 'rm';
      const showCatalog = tab === 'catalog';
      const showConsul = tab === 'consul';
      const showAnalytics = tab === 'analytics';
      els.paramsPanel.hidden = showRm || showCatalog || showConsul || showAnalytics;
      els.paramsConfigPanel.hidden = !showCatalog;
      els.consulPanel.hidden = !showConsul;
      els.rmPanel.hidden = !showRm;
      els.analyticsPanel.hidden = !showAnalytics;
      els.paramsTab.classList.toggle('is-active', !showRm && !showCatalog && !showConsul && !showAnalytics);
      els.paramsConfigTab.classList.toggle('is-active', showCatalog);
      els.consulTab.classList.toggle('is-active', showConsul);
      els.rmTab.classList.toggle('is-active', showRm);
      els.analyticsTab.classList.toggle('is-active', showAnalytics);
      els.paramsTab.setAttribute('aria-selected', showRm || showCatalog || showConsul || showAnalytics ? 'false' : 'true');
      els.paramsConfigTab.setAttribute('aria-selected', showCatalog ? 'true' : 'false');
      els.consulTab.setAttribute('aria-selected', showConsul ? 'true' : 'false');
      els.rmTab.setAttribute('aria-selected', showRm ? 'true' : 'false');
      els.analyticsTab.setAttribute('aria-selected', showAnalytics ? 'true' : 'false');
      if (showCatalog) loadParamCatalog().catch(exc => setStatus(exc.message, true));
      if (showConsul) loadConsulRecommendations().catch(exc => setStatus(exc.message, true));
      if (showRm) loadRuntimeStatus().catch(exc => setStatus(exc.message, true));
      if (showAnalytics && !analyticsLoaded) loadAnalyticsProfit().catch(exc => setStatus(exc.message, true));
    }

    async function api(path, options = {}) {
      const res = await fetch(path, {
        credentials: 'same-origin',
        headers: Object.assign({'Accept': 'application/json'}, options.headers || {}),
        ...options
      });
      if (res.status === 401) {
        window.location.href = '/config-ui/login';
        throw new Error('login required');
      }
      const data = await res.json().catch(() => ({}));
      if (!res.ok || data.ok === false) {
        throw new Error(data.error || res.statusText || 'Request failed');
      }
      return data;
    }

    function fillSelect(select, rows, valueKey, labelKey, emptyLabel) {
      select.innerHTML = '';
      if (emptyLabel) {
        const opt = document.createElement('option');
        opt.value = '';
        opt.textContent = emptyLabel;
        select.appendChild(opt);
      }
      for (const row of rows) {
        const opt = document.createElement('option');
        opt.value = String(row[valueKey]);
        opt.textContent = row[labelKey] || String(row[valueKey]);
        select.appendChild(opt);
      }
    }

    function setTargetAccountListEnabled(enabled) {
      els.targetAccountList.classList.toggle('is-disabled', !enabled);
      for (const input of els.targetAccountList.querySelectorAll('input[type="checkbox"]')) {
        input.disabled = !enabled || input.dataset.hasConfig !== '1';
      }
    }

    function fillTargetAccountList(rows) {
      els.targetAccountList.innerHTML = '';
      for (const row of rows) {
        const option = document.createElement('label');
        option.className = 'target-option' + (row.has_bot_config ? '' : ' is-unavailable');

        const input = document.createElement('input');
        input.type = 'checkbox';
        input.className = 'target-account-input';
        input.value = String(row.account_login);
        input.dataset.hasConfig = row.has_bot_config ? '1' : '0';
        input.disabled = !row.has_bot_config;

        const text = document.createElement('span');
        text.textContent = row.has_bot_config ? row.account_label : row.account_label + ' (no config for this bot)';

        option.appendChild(input);
        option.appendChild(text);
        els.targetAccountList.appendChild(option);
      }
    }

    function checkedValues(container) {
      return Array.from(container.querySelectorAll('input[type="checkbox"]:checked')).map(input => input.value);
    }

    function singleCheckedValue(container, fallback) {
      const checked = container.querySelector('input[type="checkbox"]:checked');
      return checked ? checked.value : fallback;
    }

    function fillRmAccountList(rows) {
      els.rmAccountList.innerHTML = '';
      const currentAccount = els.accountSelect.value;
      for (const row of rows) {
        const option = document.createElement('label');
        option.className = 'rm-option';

        const input = document.createElement('input');
        input.type = 'checkbox';
        input.value = String(row.account_login);
        input.checked = String(row.account_login) === currentAccount;
        input.addEventListener('change', updateRmCommandUi);

        const text = document.createElement('span');
        text.textContent = row.account_label || String(row.account_login);

        option.appendChild(input);
        option.appendChild(text);
        els.rmAccountList.appendChild(option);
      }
      if (!checkedValues(els.rmAccountList).length) {
        const first = els.rmAccountList.querySelector('input[type="checkbox"]');
        if (first) first.checked = true;
      }
      updateRmCommandUi();
    }

    function syncRmCurrentAccount() {
      const currentAccount = els.accountSelect.value;
      if (!currentAccount) return;
      for (const input of els.rmAccountList.querySelectorAll('input[type="checkbox"]')) {
        input.checked = input.value === currentAccount;
      }
      updateRmCommandUi();
    }

    function isRmStopBot(bot) {
      return /^(n\d+|bot123)$/i.test(String(bot || '').trim());
    }

    function fillRmBotList(rows = currentBotRows) {
      els.rmBotList.innerHTML = '';
      const rmBots = (rows || [])
        .filter(row => isRmStopBot(row.bot))
        .sort((a, b) => String(a.bot).localeCompare(String(b.bot), undefined, {numeric: true, sensitivity: 'base'}));

      if (!rmBots.length) {
        const empty = document.createElement('div');
        empty.className = 'rm-option';
        empty.textContent = 'No RM-stop bots for selected account';
        els.rmBotList.appendChild(empty);
        updateRmCommandUi();
        return;
      }

      for (const row of rmBots) {
        const botId = String(row.bot).toLowerCase();
        const displayName = row.display_name || row.bot;
        const option = document.createElement('label');
        option.className = 'rm-option';

        const input = document.createElement('input');
        input.type = 'checkbox';
        input.value = botId;
        input.checked = true;
        input.addEventListener('change', updateRmCommandUi);

        const text = document.createElement('span');
        text.textContent = displayName && String(displayName).toLowerCase() !== botId
          ? botId + ' - ' + displayName
          : botId;

        option.appendChild(input);
        option.appendChild(text);
        els.rmBotList.appendChild(option);
      }
    }

    function rmUntilValue() {
      return els.rmUntilInput.value.trim();
    }

    function formatRmLocalNow() {
      const now = new Date();
      const pad = value => String(value).padStart(2, '0');
      return now.getFullYear() + '.' + pad(now.getMonth() + 1) + '.' + pad(now.getDate()) +
        ' ' + pad(now.getHours()) + ':' + pad(now.getMinutes());
    }

    function ensureRmUntilDefault() {
      if (!els.rmUntilInput.value.trim()) {
        els.rmUntilInput.value = formatRmLocalNow();
      }
    }

    function selectedRmPeriod() {
      return singleCheckedValue(els.rmPeriodList, 'manual');
    }

    function buildRmCommandCore() {
      const action = els.rmActionSelect.value;
      const period = selectedRmPeriod();
      if (action === 'stop_account' || action === 'pause_account') {
        const isPause = action === 'pause_account';
        if (period === 'manual') return isPause ? 'pause' : 'halt';
        if (period === 'day') return isPause ? 'day_pause' : 'day_stop';
        if (period === 'week') return isPause ? 'week_pause' : 'week_stop';
        return (isPause ? 'pause ' : 'stop ') + (rmUntilValue() || 'YYYY.MM.DD HH:MM');
      }
      if (action === 'stop_bots' || action === 'pause_bots') {
        const bots = checkedValues(els.rmBotList).join(',');
        const duration = period === 'until' ? (rmUntilValue() || 'YYYY.MM.DD HH:MM') : period;
        return bots ? (action === 'pause_bots' ? 'pause bots ' : 'stop bots ') + bots + ' ' + duration : 'choose bots';
      }
      if (action === 'resume_bots') {
        const bots = checkedValues(els.rmBotList).join(',');
        return bots ? 'resume bots ' + bots : 'choose bots';
      }
      return {
        status: 'status',
        config: 'config',
        resume: 'resume',
        rm_daystart: 'rm_daystart',
        rm_reset: 'rm_reset'
      }[action] || action;
    }

    function updateRmCommandUi() {
      const action = els.rmActionSelect.value;
      const needsBots = action === 'stop_bots' || action === 'pause_bots' || action === 'resume_bots';
      const needsPeriod = action === 'stop_account' || action === 'pause_account' || action === 'stop_bots' || action === 'pause_bots';
      const needsUntil = needsPeriod && selectedRmPeriod() === 'until';

      if (needsPeriod) ensureRmUntilDefault();
      els.rmBotField.style.display = needsBots ? '' : 'none';
      els.rmPeriodField.style.display = needsPeriod ? '' : 'none';
      els.rmUntilField.style.display = needsUntil ? '' : 'none';
      els.rmCommandDescription.textContent = RM_COMMAND_DESCRIPTIONS[action] || '';

      const accounts = checkedValues(els.rmAccountList);
      if (!accounts.length) {
        els.rmCommandPreview.textContent = 'No account selected';
        return;
      }
      const core = buildRmCommandCore();
      els.rmCommandPreview.textContent = accounts.map(account => '/' + account + ' ' + core).join('\n');
    }

    function rmCommandPayload() {
      return {
        account_logins: checkedValues(els.rmAccountList).map(value => Number(value)),
        action: els.rmActionSelect.value,
        bot_ids: checkedValues(els.rmBotList),
        duration: selectedRmPeriod(),
        until: rmUntilValue(),
        reason: 'config-ui ' + CONFIG_ACTOR
      };
    }

    function renderRuntimeStatus(rows) {
      if (!rows.length) {
        els.rmRuntimeList.innerHTML = '';
        return;
      }
      const table = document.createElement('table');
      table.innerHTML = '<thead><tr><th>account</th><th>bot</th><th>status</th><th>version</th><th>last_seen</th></tr></thead>';
      const body = document.createElement('tbody');
      for (const row of rows) {
        const tr = document.createElement('tr');
        tr.appendChild(textCell('', row.account_login));
        tr.appendChild(textCell('', (row.bot_kind || '') + '/' + (row.bot_id || '')));
        tr.appendChild(textCell('', row.status || ''));
        tr.appendChild(textCell('', row.applied_version_no == null ? '' : row.applied_version_no));
        tr.appendChild(textCell('', row.last_seen_at || ''));
        body.appendChild(tr);
      }
      table.appendChild(body);
      els.rmRuntimeList.innerHTML = '';
      els.rmRuntimeList.appendChild(table);
    }

    async function loadRuntimeStatus() {
      const accounts = checkedValues(els.rmAccountList);
      const query = accounts.map(account => 'account_login=' + encodeURIComponent(account)).join('&');
      const data = await api('/config-ui/api/runtime-status' + (query ? '?' + query : ''));
      renderRuntimeStatus(data.statuses || []);
    }

    async function sendRmCommand() {
      const payload = rmCommandPayload();
      if (!payload.account_logins.length) {
        setStatus('Choose at least one RM account', true);
        return;
      }
      els.rmSendBtn.disabled = true;
      setStatus('Queueing RM command...');
      try {
        const data = await api('/config-ui/api/rm-command', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify(payload)
        });
        setStatus('Queued ' + (data.commands || []).length + ' RM command(s)');
        await loadRuntimeStatus();
      } catch (exc) {
        setStatus(exc.message, true);
      } finally {
        els.rmSendBtn.disabled = false;
      }
    }

    async function loadAccounts() {
      setStatus('Loading accounts...');
      const data = await api('/config-ui/api/accounts');
      const accounts = data.accounts || [];
      fillSelect(els.accountSelect, accounts, 'account_login', 'account_label', '');
      fillRmAccountList(accounts);
      if (!els.accountSelect.value) {
        setStatus('No accounts available for actor ' + CONFIG_ACTOR, true);
        return;
      }
      await loadBots();
      loadRuntimeStatus().catch(exc => setStatus(exc.message, true));
    }

    async function loadBots() {
      const account = els.accountSelect.value;
      fillSelect(els.botSelect, [], 'bot', 'display_name', '');
      setConsulTabVisible(false);
      els.consulBody.innerHTML = '';
      if (!account) return;
      setStatus('Loading bots...');
      const data = await api('/config-ui/api/bots?account_login=' + encodeURIComponent(account));
      currentBotRows = data.bots || [];
      fillSelect(els.botSelect, currentBotRows, 'bot', 'display_name', '');
      fillRmBotList(currentBotRows);
      const n9 = Array.from(els.botSelect.options).find(o => o.value === 'n9');
      if (n9) els.botSelect.value = 'n9';
      await loadCopyTargets();
      await loadParams();
      if (!els.paramsConfigPanel.hidden) await loadParamCatalog();
      const wasConsulOpen = !els.consulPanel.hidden;
      try {
        await loadConsulRecommendations({silent: !wasConsulOpen});
      } catch (exc) {
        setConsulTabVisible(false);
        if (wasConsulOpen) setStatus(exc.message, true);
      }
    }

    async function loadCopyTargets() {
      const account = els.accountSelect.value;
      const bot = els.botSelect.value;
      els.targetAccountList.innerHTML = '';
      setTargetAccountListEnabled(false);
      if (!account || !bot) return;
      const data = await api('/config-ui/api/copy-target-accounts?source_account_login=' + encodeURIComponent(account) + '&bot=' + encodeURIComponent(bot));
      fillTargetAccountList(data.accounts || []);
      const enabledTargets = Array.from(els.targetAccountList.querySelectorAll('input[type="checkbox"]')).some(input => input.dataset.hasConfig === '1');
      setTargetAccountListEnabled(els.copyToggle.checked && enabledTargets);
    }

    async function getChoices(bot, inputParam) {
      const key = bot + '|' + inputParam;
      if (choiceCache.has(key)) return choiceCache.get(key);
      const data = await api('/config-ui/api/choices?bot=' + encodeURIComponent(bot) + '&input_param=' + encodeURIComponent(inputParam));
      choiceCache.set(key, data.choices || []);
      return data.choices || [];
    }

    async function preloadChoices(bot) {
      if (!bot || choiceBulkCache.has(bot)) return;
      const data = await api('/config-ui/api/choices-bulk?bot=' + encodeURIComponent(bot));
      const choicesByParam = data.choices_by_param || {};
      for (const inputParam of Object.keys(choicesByParam)) {
        choiceCache.set(bot + '|' + inputParam, choicesByParam[inputParam] || []);
      }
      choiceBulkCache.add(bot);
    }

    function clearChoiceCaches() {
      choiceCache.clear();
      choiceBulkCache.clear();
    }

    function textCell(className, text) {
      const td = document.createElement('td');
      td.className = className || '';
      td.textContent = text == null ? '' : String(text);
      return td;
    }

    function labeledTextCell(className, label, text) {
      const td = textCell(className, text);
      td.dataset.label = label;
      return td;
    }

    function formatAnalyticsNumber(value, fractionDigits = 2) {
      const number = Number(value || 0);
      if (!Number.isFinite(number)) return '';
      return number.toLocaleString(undefined, {
        minimumFractionDigits: fractionDigits,
        maximumFractionDigits: fractionDigits
      });
    }

    function formatAnalyticsInt(value) {
      const number = Number(value || 0);
      if (!Number.isFinite(number)) return '0';
      return Math.trunc(number).toLocaleString();
    }

    function analyticsDefaultYear() {
      return String(new Date().getFullYear());
    }

    function analyticsQueryString() {
      const params = new URLSearchParams();
      const filedate = els.analyticsFiledateInput.value.trim();
      let year = els.analyticsYearInput.value.trim();
      const month = els.analyticsMonthSelect.value.trim();
      if (filedate) {
        params.set('filedate', filedate);
      } else {
        if (month && !year) {
          year = analyticsDefaultYear();
          els.analyticsYearInput.value = year;
        }
        if (year) params.set('year', year);
        if (month) params.set('month', month);
      }
      return params.toString();
    }

    function analyticsSourceCell(row) {
      const td = document.createElement('td');
      td.className = 'analytics-source';
      td.dataset.label = 'source_id';
      const source = document.createElement('span');
      source.className = 'analytics-source-main';
      source.textContent = row.source_id == null ? '' : String(row.source_id);
      td.appendChild(source);
      if (row.bot_kind) {
        const bot = document.createElement('span');
        bot.className = 'analytics-source-bot';
        bot.textContent = ' (' + row.bot_kind + ')';
        td.appendChild(bot);
      }
      return td;
    }

    function renderAnalyticsProfit(data) {
      const rows = data.rows || [];
      els.analyticsFiledate.textContent = data.period_label || data.filedate || '-';
      els.analyticsTotalDeals.textContent = formatAnalyticsInt(data.total_deals);
      els.analyticsTotalProfit.textContent = formatAnalyticsNumber(data.total_profit, 2);
      els.analyticsTotalProfit.classList.toggle('profit-negative', Number(data.total_profit || 0) < 0);
      els.analyticsTotalProfit.classList.toggle('profit-positive', Number(data.total_profit || 0) >= 0);
      els.analyticsBody.innerHTML = '';

      if (!rows.length) {
        const tr = document.createElement('tr');
        const td = labeledTextCell('', 'analytics', 'No data');
        td.colSpan = 3;
        tr.appendChild(td);
        els.analyticsBody.appendChild(tr);
        return;
      }

      for (const row of rows) {
        const profit = Number(row.profit || 0);
        const tr = document.createElement('tr');
        tr.appendChild(analyticsSourceCell(row));
        tr.appendChild(labeledTextCell('analytics-deals', 'deals', formatAnalyticsInt(row.deals)));
        tr.appendChild(labeledTextCell('analytics-profit ' + (profit < 0 ? 'profit-negative' : 'profit-positive'), 'profit', formatAnalyticsNumber(profit, 2)));
        els.analyticsBody.appendChild(tr);
      }
    }

    async function loadAnalyticsProfit() {
      setStatus('Loading analytics...');
      const tr = document.createElement('tr');
      const td = labeledTextCell('', 'analytics', 'Loading...');
      td.colSpan = 3;
      tr.appendChild(td);
      els.analyticsBody.innerHTML = '';
      els.analyticsBody.appendChild(tr);
      const query = analyticsQueryString();
      const data = await api('/config-ui/api/analytics/profit-by-source' + (query ? '?' + query : ''));
      renderAnalyticsProfit(data);
      analyticsLoaded = true;
      setStatus('Analytics loaded for ' + (data.period_label || data.filedate || 'no data'));
    }

    function recommendationSignalText(row) {
      const parts = [];
      if (row.status) parts.push('status=' + row.status);
      if (row.trend_strength != null) parts.push('trend=' + row.trend_strength);
      if (row.confidence != null) parts.push('confidence=' + row.confidence);
      if (row.min_sample_reached != null) parts.push('sample=' + (row.min_sample_reached ? 'ok' : 'low'));
      if (row.cooldown_until) parts.push('cooldown until ' + row.cooldown_until);
      return parts.join('\n');
    }

    function recommendationValuesText(row) {
      const oldValue = row.old_value == null ? '' : String(row.old_value);
      const newValue = row.recommended_value == null ? '' : String(row.recommended_value);
      return oldValue ? oldValue + ' -> ' + newValue : newValue;
    }

    function renderConsulRecommendations(rows) {
      els.consulBody.innerHTML = '';
      if (!rows.length) {
        const tr = document.createElement('tr');
        const td = labeledTextCell('', 'recommendations', 'No active parameter recommendations');
        td.colSpan = 7;
        tr.appendChild(td);
        els.consulBody.appendChild(tr);
        return;
      }
      for (const row of rows) {
        const tr = document.createElement('tr');
        tr.appendChild(labeledTextCell('rec-id', 'id', row.recommendation_id));
        tr.appendChild(labeledTextCell('rec-bot', 'bot', [row.bot_kind, row.symbol, row.tf].filter(Boolean).join(' / ')));
        tr.appendChild(labeledTextCell('rec-param', 'input_param', row.input_param || ''));
        tr.appendChild(labeledTextCell('rec-values', 'values', recommendationValuesText(row)));
        tr.appendChild(labeledTextCell('rec-signal', 'signal', recommendationSignalText(row)));
        tr.appendChild(labeledTextCell('rec-reason', 'reason', row.reason || ''));

        const actions = document.createElement('td');
        actions.className = 'rec-actions';
        actions.dataset.label = 'action';
        const canAct = String(row.status || '').toLowerCase() === 'new';
        const approve = document.createElement('button');
        approve.type = 'button';
        approve.textContent = 'Approve';
        approve.disabled = !canAct;
        approve.addEventListener('click', () => decideRecommendation(row.recommendation_id, 'approve'));
        const reject = document.createElement('button');
        reject.type = 'button';
        reject.className = 'reject';
        reject.textContent = 'Reject';
        reject.disabled = !canAct;
        reject.addEventListener('click', () => decideRecommendation(row.recommendation_id, 'reject'));
        actions.appendChild(approve);
        actions.appendChild(reject);
        tr.appendChild(actions);
        els.consulBody.appendChild(tr);
      }
    }

    async function loadConsulRecommendations(options = {}) {
      const silent = !!options.silent;
      const account = els.accountSelect.value;
      els.consulBody.innerHTML = '';
      if (!account) {
        setConsulTabVisible(false);
        return [];
      }
      if (!silent) setStatus('Loading Consul recommendations...');
      const data = await api('/config-ui/api/recommendations?account_login=' + encodeURIComponent(account));
      const rows = data.recommendations || [];
      setConsulTabVisible(rows.length > 0);
      renderConsulRecommendations(rows);
      if (!silent) setStatus(rows.length + ' active recommendation(s) loaded');
      return rows;
    }

    async function decideRecommendation(recommendationId, action) {
      if (!recommendationId) return;
      setStatus((action === 'approve' ? 'Approving' : 'Rejecting') + ' recommendation #' + recommendationId + '...');
      try {
        await api('/config-ui/api/recommendations/' + encodeURIComponent(recommendationId) + '/' + action, {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({reason: 'Consul ' + CONFIG_ACTOR})
        });
        await loadConsulRecommendations();
        await loadParams();
        setStatus('Recommendation #' + recommendationId + ' ' + (action === 'approve' ? 'approved' : 'rejected'));
      } catch (exc) {
        setStatus(exc.message, true);
      }
    }

    function paramCell(row) {
      const td = textCell('param', row.input_param);
      td.dataset.desc = row.param_desc || '';
      td.title = row.param_desc || '';
      return td;
    }

    async function buildValueControl(row) {
      if (row.has_choices) {
        const select = document.createElement('select');
        select.className = 'new-value-input';
        select.dataset.rowId = row.row_id;
        select.dataset.inputParam = row.input_param;
        const empty = document.createElement('option');
        empty.value = '';
        empty.textContent = 'No change';
        select.appendChild(empty);
        const choices = await getChoices(row.bot, row.input_param);
        for (const choice of choices) {
          const opt = document.createElement('option');
          opt.value = choice.allowed_value;
          opt.textContent = choice.value_desc ? choice.allowed_value + ' - ' + choice.value_desc : choice.allowed_value;
          select.appendChild(opt);
        }
        if (row.new_value_ui) select.value = row.new_value_ui;
        select.addEventListener('change', updateChangedCount);
        return select;
      }
      const input = document.createElement('input');
      input.type = 'text';
      input.className = 'new-value-input';
      input.dataset.rowId = row.row_id;
      input.dataset.inputParam = row.input_param;
      input.placeholder = 'New value';
      input.value = row.new_value_ui || '';
      input.addEventListener('input', updateChangedCount);
      return input;
    }

    function auditOptionLabel(item) {
      const when = item.changed_at || '';
      const oldValue = item.old_value == null ? '' : String(item.old_value);
      const newValue = item.new_value == null ? '' : String(item.new_value);
      return (when ? when + ' | ' : '') + oldValue + ' -> ' + newValue;
    }

    function applyPrevValue(select) {
      const selected = select.selectedOptions && select.selectedOptions.length ? select.selectedOptions[0] : null;
      const value = selected ? selected.dataset.oldValue || '' : '';
      if (!value) return;
      const rowId = select.dataset.rowId;
      const target = els.paramsBody.querySelector('.new-value-input[data-row-id="' + rowId + '"]');
      if (!target) return;
      if (target.tagName === 'SELECT') {
        const exists = Array.from(target.options).some(option => option.value === value);
        if (!exists) {
          setStatus('Previous value is not allowed for this dictionary parameter', true);
          select.value = '';
          return;
        }
      }
      target.value = value;
      const reason = els.paramsBody.querySelector('.reason-input[data-row-id="' + rowId + '"]');
      if (reason && !reason.value.trim()) {
        reason.value = 'rollback from audit ' + (selected.dataset.changedAt || '');
      }
      updateChangedCount();
    }

    async function buildPrevValueControl(row) {
      const select = document.createElement('select');
      select.className = 'prev-value-input';
      select.dataset.rowId = row.row_id;
      const empty = document.createElement('option');
      empty.value = '';
      empty.textContent = 'No history';
      select.appendChild(empty);

      const currentValue = row.current_value == null ? '' : String(row.current_value);
      const history = Array.isArray(row.prev_values) ? row.prev_values : [];
      let allowed = null;
      if (row.has_choices) {
        const choices = await getChoices(row.bot, row.input_param);
        allowed = new Set(choices.map(choice => String(choice.allowed_value)));
      }

      let count = 0;
      const seen = new Set();
      for (const item of history) {
        const oldValue = item && item.old_value != null ? String(item.old_value) : '';
        if (!oldValue || oldValue === currentValue || seen.has(oldValue)) continue;
        if (allowed && !allowed.has(oldValue)) continue;
        seen.add(oldValue);
        const opt = document.createElement('option');
        opt.value = String(count + 1);
        opt.textContent = auditOptionLabel(item);
        opt.dataset.oldValue = oldValue;
        opt.dataset.changedAt = item.changed_at || '';
        select.appendChild(opt);
        count++;
      }

      if (!count) {
        select.disabled = true;
      } else {
        select.firstChild.textContent = 'Choose previous';
        select.addEventListener('change', () => applyPrevValue(select));
      }
      return select;
    }

    async function loadParams() {
      const account = els.accountSelect.value;
      const bot = els.botSelect.value;
      const loadSeq = ++paramsLoadSeq;
      els.paramsBody.innerHTML = '';
      if (!account || !bot) return;
      setStatus('Loading parameters...');
      const data = await api('/config-ui/api/params?account_login=' + encodeURIComponent(account) + '&bot=' + encodeURIComponent(bot));
      if (loadSeq !== paramsLoadSeq || account !== els.accountSelect.value || bot !== els.botSelect.value) return;
      const rows = data.params || [];
      if (rows.some(row => row.has_choices)) {
        await preloadChoices(bot);
        if (loadSeq !== paramsLoadSeq || account !== els.accountSelect.value || bot !== els.botSelect.value) return;
      }
      const groups = Array.from(new Set(rows.map(r => r.param_group || '').filter(Boolean))).sort();
      const currentGroup = els.groupFilter.value;
      els.groupFilter.innerHTML = '<option value="">All groups</option>';
      for (const group of groups) {
        const opt = document.createElement('option');
        opt.value = group;
        opt.textContent = group;
        els.groupFilter.appendChild(opt);
      }
      if (groups.includes(currentGroup)) els.groupFilter.value = currentGroup;

      let previousGroup = null;
      const fragment = document.createDocumentFragment();
      for (const row of rows) {
        if (loadSeq !== paramsLoadSeq || account !== els.accountSelect.value || bot !== els.botSelect.value) return;
        const rowGroup = row.param_group || '';
        const tr = document.createElement('tr');
        if (previousGroup !== null && rowGroup !== previousGroup) tr.classList.add('group-break');
        previousGroup = rowGroup;
        tr.dataset.group = rowGroup;
        tr.dataset.search = ((row.input_param || '') + ' ' + (row.param_desc || '')).toLowerCase();
        tr.dataset.rowId = row.row_id;
        tr.appendChild(textCell('group', row.param_group));
        tr.appendChild(paramCell(row));
        tr.appendChild(textCell('desc', row.param_desc));
        tr.appendChild(textCell('current', row.current_value));

        const prevTd = document.createElement('td');
        prevTd.className = 'prev-value';
        prevTd.appendChild(await buildPrevValueControl(row));
        if (loadSeq !== paramsLoadSeq || account !== els.accountSelect.value || bot !== els.botSelect.value) return;
        tr.appendChild(prevTd);

        const newTd = document.createElement('td');
        newTd.className = 'new-value';
        newTd.appendChild(await buildValueControl(row));
        if (loadSeq !== paramsLoadSeq || account !== els.accountSelect.value || bot !== els.botSelect.value) return;
        tr.appendChild(newTd);

        const reasonTd = document.createElement('td');
        reasonTd.className = 'reason';
        const reason = document.createElement('input');
        reason.type = 'text';
        reason.className = 'reason-input';
        reason.dataset.rowId = row.row_id;
        reason.placeholder = 'Optional';
        reason.value = row.reason || '';
        reason.addEventListener('input', updateChangedCount);
        reasonTd.appendChild(reason);
        tr.appendChild(reasonTd);
        fragment.appendChild(tr);
      }
      els.paramsBody.innerHTML = '';
      els.paramsBody.appendChild(fragment);
      applyFilters();
      updateChangedCount();
      setStatus(rows.length + ' parameters loaded');
    }

    function catalogLabelCell(className, label, content) {
      const td = document.createElement('td');
      td.className = className || '';
      td.dataset.label = label;
      if (content instanceof Node) {
        td.appendChild(content);
      } else {
        td.textContent = content == null ? '' : String(content);
      }
      return td;
    }

    function normalizedControlValue(control) {
      if (!control) return '';
      if (control.type === 'checkbox') return control.checked ? '1' : '0';
      return String(control.value || '').trim();
    }

    function catalogControlChanged(control) {
      return normalizedControlValue(control) !== String(control.dataset.original || '');
    }

    function catalogTextInput(row, field, placeholder) {
      const input = document.createElement('input');
      input.type = 'text';
      input.className = 'catalog-input-control';
      input.dataset.field = field;
      input.placeholder = placeholder || '';
      input.value = row[field] == null ? '' : String(row[field]);
      input.dataset.original = input.value.trim();
      input.addEventListener('input', updateCatalogChangedCount);
      return input;
    }

    function catalogTextarea(row, field, placeholder) {
      const textarea = document.createElement('textarea');
      textarea.className = 'catalog-input-control';
      textarea.dataset.field = field;
      textarea.placeholder = placeholder || '';
      textarea.value = Array.isArray(row[field]) ? row[field].join('\n') : (row[field] == null ? '' : String(row[field]));
      textarea.dataset.original = field === 'allowed_values' ? normalizeAllowedValuesText(textarea.value) : textarea.value.trim();
      textarea.addEventListener('input', updateCatalogChangedCount);
      return textarea;
    }

    function catalogTypeSelect(row) {
      const select = document.createElement('select');
      select.className = 'catalog-input-control';
      select.dataset.field = 'value_type';
      const current = row.value_type == null ? '' : String(row.value_type);
      const values = current && !CATALOG_VALUE_TYPES.includes(current) ? [current, ...CATALOG_VALUE_TYPES] : CATALOG_VALUE_TYPES;
      for (const value of values) {
        const opt = document.createElement('option');
        opt.value = value;
        opt.textContent = value;
        select.appendChild(opt);
      }
      select.value = current || 'text';
      select.dataset.original = select.value.trim();
      select.addEventListener('change', updateCatalogChangedCount);
      return select;
    }

    function catalogCheckbox(row, field) {
      const label = document.createElement('label');
      label.className = 'catalog-toggle';
      const input = document.createElement('input');
      input.type = 'checkbox';
      input.className = 'catalog-input-control';
      input.dataset.field = field;
      input.checked = row[field] !== false;
      input.dataset.original = input.checked ? '1' : '0';
      input.addEventListener('change', updateCatalogChangedCount);
      label.appendChild(input);
      return label;
    }

    function catalogRangeControls(row) {
      const wrap = document.createElement('div');
      wrap.className = 'catalog-range-grid';
      const minInput = catalogTextInput(row, 'min_numeric', 'min');
      const maxInput = catalogTextInput(row, 'max_numeric', 'max');
      minInput.classList.add('catalog-number');
      maxInput.classList.add('catalog-number');
      wrap.appendChild(minInput);
      wrap.appendChild(maxInput);
      return wrap;
    }

    function normalizeAllowedValuesText(text) {
      return parseAllowedValues(text).join('\n');
    }

    function parseAllowedValues(text) {
      const parts = String(text || '').split(/[\n,]+/);
      const values = [];
      const seen = new Set();
      for (const part of parts) {
        const value = part.trim();
        if (!value || seen.has(value)) continue;
        values.push(value);
        seen.add(value);
      }
      return values;
    }

    function catalogControl(tr, field) {
      return tr.querySelector('.catalog-input-control[data-field="' + field + '"]');
    }

    function catalogTextValue(tr, field) {
      const control = catalogControl(tr, field);
      const value = control ? String(control.value || '').trim() : '';
      return value || null;
    }

    function catalogDataValue(tr, field) {
      const value = tr.dataset[field] == null ? '' : String(tr.dataset[field]).trim();
      return value || null;
    }

    function catalogIntegerValue(tr, field, validate) {
      const value = catalogTextValue(tr, field);
      if (value == null) return null;
      if (!/^-?\d+$/.test(value)) {
        if (validate) throw new Error(field + ' must be integer');
        return null;
      }
      return Number(value);
    }

    function catalogRowPayload(tr, validate) {
      const userEditable = catalogControl(tr, 'user_editable');
      return {
        bot_kind: tr.dataset.botKind,
        param_key: tr.dataset.paramKey,
        section_name: catalogTextValue(tr, 'section_name'),
        input_param_name: catalogTextValue(tr, 'input_param_name'),
        display_name: catalogTextValue(tr, 'display_name'),
        param_desc: catalogTextValue(tr, 'param_desc'),
        param_path: catalogTextValue(tr, 'param_path') || catalogDataValue(tr, 'paramPath'),
        value_type: catalogTextValue(tr, 'value_type') || catalogDataValue(tr, 'valueType'),
        min_numeric: catalogTextValue(tr, 'min_numeric') || catalogDataValue(tr, 'minNumeric'),
        max_numeric: catalogTextValue(tr, 'max_numeric') || catalogDataValue(tr, 'maxNumeric'),
        allowed_values: parseAllowedValues(catalogControl(tr, 'allowed_values')?.value || ''),
        sort_order: catalogIntegerValue(tr, 'sort_order', validate),
        user_editable: userEditable ? userEditable.checked : true
      };
    }

    function renderCatalogRow(row) {
      const tr = document.createElement('tr');
      tr.dataset.botKind = row.bot_kind || '';
      tr.dataset.paramKey = row.param_key || '';
      tr.dataset.paramPath = row.param_path || '';
      tr.dataset.valueType = row.value_type || '';
      tr.dataset.minNumeric = row.min_numeric == null ? '' : String(row.min_numeric);
      tr.dataset.maxNumeric = row.max_numeric == null ? '' : String(row.max_numeric);
      tr.dataset.search = [
        row.param_key,
        row.section_name,
        row.input_param_name,
        row.display_name,
        row.param_desc,
        row.param_path,
        row.value_type,
        Array.isArray(row.allowed_values) ? row.allowed_values.join(' ') : ''
      ].filter(Boolean).join(' ').toLowerCase();
      tr.appendChild(catalogLabelCell('catalog-editable', 'editable', catalogCheckbox(row, 'user_editable')));
      tr.appendChild(catalogLabelCell('catalog-section', 'section', catalogTextInput(row, 'section_name', 'section')));
      tr.appendChild(catalogLabelCell('catalog-input', 'input_param', catalogTextInput(row, 'input_param_name', 'input_param')));
      tr.appendChild(catalogLabelCell('catalog-display', 'display', catalogTextInput(row, 'display_name', 'display name')));
      tr.appendChild(catalogLabelCell('catalog-desc', 'description', catalogTextarea(row, 'param_desc', 'description')));
      tr.appendChild(catalogLabelCell('catalog-values', 'allowed_values', catalogTextarea(row, 'allowed_values', 'one value per line')));
      tr.appendChild(catalogLabelCell('catalog-sort', 'sort', catalogTextInput(row, 'sort_order', 'sort')));
      return tr;
    }

    async function loadParamCatalog() {
      const account = els.accountSelect.value;
      const bot = els.botSelect.value;
      els.paramsConfigBody.innerHTML = '';
      currentCatalogRows = [];
      updateCatalogChangedCount();
      if (!account || !bot) return;
      setStatus('Loading parameter catalog...');
      const data = await api('/config-ui/api/param-catalog?account_login=' + encodeURIComponent(account) + '&bot=' + encodeURIComponent(bot));
      currentCatalogRows = data.params || [];
      for (const row of currentCatalogRows) {
        els.paramsConfigBody.appendChild(renderCatalogRow(row));
      }
      applyCatalogFilter();
      updateCatalogChangedCount();
      setStatus(currentCatalogRows.length + ' catalog rows loaded');
    }

    function applyCatalogFilter() {
      const query = els.catalogSearchInput.value.trim().toLowerCase();
      for (const tr of els.paramsConfigBody.querySelectorAll('tr')) {
        tr.classList.toggle('hidden', !!query && !tr.dataset.search.includes(query));
      }
    }

    function getCatalogChangedRows(validate) {
      const changes = [];
      for (const tr of els.paramsConfigBody.querySelectorAll('tr')) {
        let changed = false;
        for (const control of tr.querySelectorAll('.catalog-input-control')) {
          if (control.dataset.field === 'allowed_values') {
            if (normalizeAllowedValuesText(control.value) !== String(control.dataset.original || '')) changed = true;
          } else if (catalogControlChanged(control)) {
            changed = true;
          }
          if (changed) break;
        }
        if (changed) changes.push(catalogRowPayload(tr, !!validate));
      }
      return changes;
    }

    function updateCatalogChangedCount() {
      const count = getCatalogChangedRows(false).length;
      els.paramsConfigChangedCount.textContent = count + (count === 1 ? ' catalog row changed' : ' catalog rows changed');
      els.paramsConfigSaveBtn.disabled = count === 0;
    }

    async function saveParamCatalogChanges() {
      let changes;
      try {
        changes = getCatalogChangedRows(true);
      } catch (exc) {
        setStatus(exc.message, true);
        return;
      }
      if (!changes.length) return;
      for (const change of changes) {
        if (!change.param_path) {
          setStatus(change.param_key + ': param_path is required', true);
          return;
        }
        if (!change.value_type) {
          setStatus(change.param_key + ': value_type is required', true);
          return;
        }
      }
      els.paramsConfigSaveBtn.disabled = true;
      setStatus('Saving catalog...');
      const payload = {
        account_login: Number(els.accountSelect.value),
        bot: els.botSelect.value,
        changes
      };
      try {
        const data = await api('/config-ui/api/param-catalog/save', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify(payload)
        });
        const refreshed = (data.refreshed_accounts || []).length;
        const savedMessage = 'Saved ' + (data.updated_count || 0) + ' catalog row(s); refreshed ' + refreshed + ' account(s)';
        setStatus(savedMessage);
        clearChoiceCaches();
        await loadParamCatalog();
        await loadParams();
        await loadCopyTargets();
        setStatus(savedMessage);
      } catch (exc) {
        setStatus(exc.message, true);
      } finally {
        updateCatalogChangedCount();
      }
    }

    function applyFilters() {
      const group = els.groupFilter.value;
      const query = els.searchInput.value.trim().toLowerCase();
      for (const tr of els.paramsBody.querySelectorAll('tr')) {
        const groupOk = !group || tr.dataset.group === group;
        const queryOk = !query || tr.dataset.search.includes(query);
        tr.classList.toggle('hidden', !(groupOk && queryOk));
      }
    }

    function getChangedRows() {
      const changes = [];
      for (const control of els.paramsBody.querySelectorAll('.new-value-input')) {
        const value = (control.value || '').trim();
        if (!value) continue;
        const rowId = Number(control.dataset.rowId);
        const reason = els.paramsBody.querySelector('.reason-input[data-row-id="' + rowId + '"]');
        changes.push({row_id: rowId, input_param: control.dataset.inputParam || null, value, reason: reason ? reason.value.trim() || null : null});
      }
      return changes;
    }

    function getSelectedTargetAccounts() {
      if (!els.copyToggle.checked) return [];
      return Array.from(els.targetAccountList.querySelectorAll('input[type="checkbox"]:checked'))
        .filter(input => !input.disabled && input.value)
        .map(input => Number(input.value));
    }

    function updateChangedCount() {
      const count = getChangedRows().length;
      els.changedCount.textContent = count + (count === 1 ? ' changed row' : ' changed rows');
      els.saveBtn.disabled = count === 0;
    }

    async function saveChanges() {
      const changes = getChangedRows();
      if (!changes.length) return;
      const targetAccounts = getSelectedTargetAccounts();
      if (els.copyToggle.checked && !targetAccounts.length) {
        setStatus('Choose at least one available target account or turn off Apply same values', true);
        return;
      }
      els.saveBtn.disabled = true;
      setStatus('Saving...');
      const payload = {
        account_login: Number(els.accountSelect.value),
        bot: els.botSelect.value,
        copy_to_account_logins: targetAccounts,
        changes
      };
      try {
        const data = await api('/config-ui/api/save', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify(payload)
        });
        setStatus('Saved ' + (data.applied_count || 0) + ' row updates');
        await loadParams();
      } catch (exc) {
        setStatus(exc.message, true);
      } finally {
        updateChangedCount();
      }
    }

    els.accountSelect.addEventListener('change', async () => { syncRmCurrentAccount(); await loadBots(); });
    els.botSelect.addEventListener('change', async () => {
      await loadCopyTargets();
      await loadParams();
      if (!els.paramsConfigPanel.hidden) await loadParamCatalog();
      const wasConsulOpen = !els.consulPanel.hidden;
      try {
        await loadConsulRecommendations({silent: !wasConsulOpen});
      } catch (exc) {
        setConsulTabVisible(false);
        if (wasConsulOpen) setStatus(exc.message, true);
      }
    });
    els.groupFilter.addEventListener('change', applyFilters);
    els.searchInput.addEventListener('input', applyFilters);
    els.catalogSearchInput.addEventListener('input', applyCatalogFilter);
    els.copyToggle.addEventListener('change', loadCopyTargets);
    els.adminModeToggle.addEventListener('change', updateAdminModeUi);
    els.saveBtn.addEventListener('click', saveChanges);
    els.paramsTab.addEventListener('click', () => switchFormTab('params'));
    els.paramsConfigTab.addEventListener('click', () => switchFormTab('catalog'));
    els.consulTab.addEventListener('click', () => switchFormTab('consul'));
    els.rmTab.addEventListener('click', () => switchFormTab('rm'));
    els.analyticsTab.addEventListener('click', () => switchFormTab('analytics'));
    els.paramsConfigSaveBtn.addEventListener('click', saveParamCatalogChanges);
    els.rmActionSelect.addEventListener('change', updateRmCommandUi);
    for (const input of els.rmPeriodList.querySelectorAll('input[type="checkbox"]')) {
      input.addEventListener('change', () => {
        if (input.checked) {
          for (const other of els.rmPeriodList.querySelectorAll('input[type="checkbox"]')) {
            if (other !== input) other.checked = false;
          }
        } else if (!checkedValues(els.rmPeriodList).length) {
          input.checked = true;
        }
        updateRmCommandUi();
      });
    }
    els.rmUntilInput.addEventListener('input', updateRmCommandUi);
    els.rmSendBtn.addEventListener('click', sendRmCommand);
    els.rmRefreshStatusBtn.addEventListener('click', () => loadRuntimeStatus().catch(exc => setStatus(exc.message, true)));
    els.consulRefreshBtn.addEventListener('click', () => loadConsulRecommendations().catch(exc => setStatus(exc.message, true)));
    els.analyticsYearInput.addEventListener('input', () => { analyticsLoaded = false; });
    els.analyticsMonthSelect.addEventListener('change', () => {
      if (els.analyticsMonthSelect.value && !els.analyticsYearInput.value.trim()) {
        els.analyticsYearInput.value = analyticsDefaultYear();
      }
      analyticsLoaded = false;
    });
    els.analyticsFiledateInput.addEventListener('input', () => { analyticsLoaded = false; });
    els.analyticsApplyBtn.addEventListener('click', () => {
      analyticsLoaded = false;
      loadAnalyticsProfit().catch(exc => setStatus(exc.message, true));
    });
    els.analyticsClearBtn.addEventListener('click', () => {
      els.analyticsYearInput.value = '';
      els.analyticsMonthSelect.value = '';
      els.analyticsFiledateInput.value = '';
      analyticsLoaded = false;
      loadAnalyticsProfit().catch(exc => setStatus(exc.message, true));
    });
    els.analyticsRefreshBtn.addEventListener('click', () => {
      analyticsLoaded = false;
      loadAnalyticsProfit().catch(exc => setStatus(exc.message, true));
    });

    updateAdminModeUi();
    setConsulTabVisible(false);
    fillRmBotList([]);
    updateRmCommandUi();
    loadAccounts().catch(exc => setStatus(exc.message, true));
  </script>
</body>
</html>"""


def config_ui_app_html(username, actor):
    return (
        CONFIG_UI_APP_HTML
        .replace("__SESSION_USER__", json.dumps(username))
        .replace("__CONFIG_ACTOR__", json.dumps(actor))
        .replace("__CODE_VERSION__", json.dumps(CODE_VERSION))
    )


def infer_pg_type(value):
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int) and not isinstance(value, bool):
        return "integer"
    if isinstance(value, float):
        return "double precision"
    return "text"


def ensure_table_for_rows(conn, schema, table, columns, sample_row):
    if not is_safe_ident(schema) or not is_safe_ident(table):
        raise ValueError("Unsafe table name")
    col_defs = []
    for col in columns:
        if not is_safe_ident(col):
            raise ValueError("Unsafe column name")
        col_type = infer_pg_type(sample_row.get(col))
        col_defs.append(f"{col} {col_type}")
    create_sql = f"""
    CREATE SCHEMA IF NOT EXISTS {schema};
    CREATE TABLE IF NOT EXISTS {schema}.{table} (
        {', '.join(col_defs)}
    );
    """
    with conn.cursor() as cur:
        cur.execute(create_sql)
        for col in columns:
            col_type = infer_pg_type(sample_row.get(col))
            cur.execute(f"ALTER TABLE {schema}.{table} ADD COLUMN IF NOT EXISTS {col} {col_type}")


def ensure_column_exists(cur, schema, table, column):
    cur.execute(
        """
        SELECT 1
        FROM information_schema.columns
        WHERE table_schema = %s
          AND table_name = %s
          AND column_name = %s
        """,
        (schema, table, column),
    )
    return cur.fetchone() is not None


def format_neuro_text(row):
    pid = row.get("source_id") or ""
    sym = row["sym"] or ""
    tf = row["tf"] or ""
    target = row["target"]
    lcb_hour = row["lcb_hour"]
    lcb_dow = row["lcb_dow"]
    dist_sma = row["dist_sma"]
    rsi_mom = row["rsi_mom"]
    vol_ratio = row["vol_ratio"]
    sma_slope = row["sma_slope"]
    htf_rsi = row["htf_rsi"]
    desc = (row["description"] or "").strip()
    header = (
        f"[{pid}] [{sym} {tf}] target={target} hour={lcb_hour} dow={lcb_dow} "
        f"dist_sma={dist_sma} rsi_mom={rsi_mom} vol_ratio={vol_ratio} "
        f"sma_slope={sma_slope} htf_rsi={htf_rsi}."
    )
    return f"{header} {desc}".strip()


def get_source_row_count(db_url, table_name, source_id):
    schema, table = split_table_name(table_name)
    if not is_safe_ident(schema) or not is_safe_ident(table):
        raise ValueError(f"Unsafe table name: {table_name}")
    conn = psycopg2.connect(db_url, sslmode="require")
    try:
        with conn.cursor() as cur:
            cur.execute(
                f"SELECT COUNT(*) FROM {schema}.{table} WHERE source_id = %s",
                (source_id,),
            )
            return cur.fetchone()[0]
    finally:
        conn.close()


def export_neuro_jsonl(db_url, table_name, source_id, limit, out_path):
    schema, table = split_table_name(table_name)
    if not is_safe_ident(schema) or not is_safe_ident(table):
        raise ValueError(f"Unsafe table name: {table_name}")

    conn = psycopg2.connect(db_url, sslmode="require")
    try:
        with conn.cursor() as cur:
            if not ensure_column_exists(cur, schema, table, "source_id"):
                raise RuntimeError(f"{schema}.{table} missing source_id column")

        query = (
            f"SELECT source_id, sym, tf, target, lcb_hour, lcb_dow, dist_sma, rsi_mom, vol_ratio, "
            f"sma_slope, htf_rsi, description "
            f"FROM {schema}.{table} "
            # f"WHERE source_id = %s"
        )
        params = []
        if limit:
            query += " LIMIT %s"
            params.append(limit)

        count = 0
        with conn.cursor(name="neuro_export", cursor_factory=DictCursor) as cur:
            cur.itersize = 2000
            cur.execute(query, params)
            with open(out_path, "w", encoding="utf-8", newline="") as f:
                for row in cur:
                    text = format_neuro_text(row)
                    doc = {
                        "text": text,
                        "metadata": {
                            "pid": row["source_id"],
                            "sym": row["sym"],
                            "tf": row["tf"],
                            "target": row["target"],
                            "lcb_hour": row["lcb_hour"],
                            "lcb_dow": row["lcb_dow"],
                            "dist_sma": row["dist_sma"],
                            "rsi_mom": row["rsi_mom"],
                            "vol_ratio": row["vol_ratio"],
                            "sma_slope": row["sma_slope"],
                            "htf_rsi": row["htf_rsi"],
                        },
                    }
                    f.write(json.dumps(doc, ensure_ascii=True))
                    f.write("\n")
                    count += 1
        return count
    finally:
        conn.close()


def vector_store_api(client_obj):
    if hasattr(client_obj, "beta") and hasattr(client_obj.beta, "vector_stores"):
        return client_obj.beta.vector_stores
    if hasattr(client_obj, "vector_stores"):
        return client_obj.vector_stores
    raise RuntimeError("Vector store API not available in client")


def assistant_api(client_obj):
    if hasattr(client_obj, "beta") and hasattr(client_obj.beta, "assistants"):
        return client_obj.beta.assistants
    if hasattr(client_obj, "assistants"):
        return client_obj.assistants
    raise RuntimeError("Assistant API not available in client")


def attach_file_to_vector_store(client_obj, vector_store_id, file_id):
    vs_api = vector_store_api(client_obj)
    if hasattr(vs_api, "file_batches"):
        fb_api = vs_api.file_batches
        if hasattr(fb_api, "create_and_poll"):
            return fb_api.create_and_poll(vector_store_id=vector_store_id, file_ids=[file_id])
        batch = fb_api.create(vector_store_id=vector_store_id, file_ids=[file_id])
        return wait_for_batch(fb_api, vector_store_id, batch.id)
    if hasattr(vs_api, "files"):
        return vs_api.files.create(vector_store_id=vector_store_id, file_id=file_id)
    raise RuntimeError("Vector store file attach API not available")


def list_vector_store_files(client_obj, vector_store_id):
    vs_api = vector_store_api(client_obj)
    if hasattr(vs_api, "files") and hasattr(vs_api.files, "list"):
        return vs_api.files.list(vector_store_id=vector_store_id, limit=100)
    return None


def delete_vector_store_file(client_obj, vector_store_id, file_id):
    vs_api = vector_store_api(client_obj)
    if hasattr(vs_api, "files") and hasattr(vs_api.files, "delete"):
        return vs_api.files.delete(vector_store_id=vector_store_id, file_id=file_id)
    return None


def wait_for_batch(fb_api, vector_store_id, batch_id, timeout_s=900, interval_s=2):
    start = time.time()
    while True:
        batch = fb_api.retrieve(vector_store_id=vector_store_id, batch_id=batch_id)
        status = getattr(batch, "status", None) or batch.get("status", None)
        if status in ("completed", "failed", "cancelled"):
            return batch
        if time.time() - start > timeout_s:
            raise TimeoutError("Vector store indexing timed out")
        time.sleep(interval_s)


def update_assistant_vector_store(client_obj, assistant_id, vector_store_id):
    as_api = assistant_api(client_obj)
    assistant = as_api.retrieve(assistant_id=assistant_id)
    tool_resources = getattr(assistant, "tool_resources", None) or {}
    old_ids = []
    if isinstance(tool_resources, dict):
        file_search = tool_resources.get("file_search") or {}
        old_ids = file_search.get("vector_store_ids") or []
    as_api.update(
        assistant_id=assistant_id,
        tool_resources={"file_search": {"vector_store_ids": [vector_store_id]}},
    )
    return old_ids


def get_vector_store_file_count(client_obj, vector_store_id):
    vs_api = vector_store_api(client_obj)
    store = vs_api.retrieve(vector_store_id=vector_store_id)
    file_counts = getattr(store, "file_counts", None) or {}
    if isinstance(file_counts, dict):
        total = file_counts.get("total")
        if total is not None:
            return total
    if hasattr(vs_api, "files"):
        files_api = vs_api.files
        total = 0
        cursor = None
        while True:
            if cursor:
                page = files_api.list(vector_store_id=vector_store_id, after=cursor, limit=100)
            else:
                page = files_api.list(vector_store_id=vector_store_id, limit=100)
            data = getattr(page, "data", None) or page.get("data", [])
            total += len(data)
            has_more = getattr(page, "has_more", None)
            if has_more is None:
                has_more = page.get("has_more", False)
            if not has_more:
                break
            cursor = data[-1].id if data else None
            if not cursor:
                break
        return total
    return None


def run_neuro_refresh(pid, source_id, table_name, limit, db_mode: Optional[str] = None):
    if psycopg2 is None:
        raise RuntimeError(f"psycopg2 not available: {_PSYCOPG2_IMPORT_ERROR}")
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is not set")
    if not VS_ID:
        raise RuntimeError("VS_ID is not set")

    db_url = load_db_url(db_mode=db_mode)
    db_row_count = get_source_row_count(db_url, table_name, source_id)
    print(f"[NEURO_WS] db rows for {table_name} source_id={source_id}: {db_row_count}")

    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = Path(tmpdir) / f"neuro_{source_id}.txt"
        row_count = export_neuro_jsonl(db_url, table_name, source_id, limit, out_path)

        local_client = OpenAI(api_key=OPENAI_API_KEY)

        vs_api = vector_store_api(local_client)
        files_page = list_vector_store_files(local_client, VS_ID)
        file_items = []
        if files_page is not None:
            data_attr = getattr(files_page, "data", None)
            if data_attr is not None:
                file_items = data_attr
            elif isinstance(files_page, dict):
                file_items = files_page.get("data", [])
            else:
                try:
                    file_items = list(files_page)
                except Exception:
                    file_items = []
        for item in file_items:
            file_id = getattr(item, "id", None)
            if not file_id and isinstance(item, dict):
                file_id = item.get("id")
            if file_id:
                delete_vector_store_file(local_client, VS_ID, file_id)

        with open(out_path, "rb") as f:
            file_obj = local_client.files.create(file=f, purpose="assistants")
        attach_file_to_vector_store(local_client, VS_ID, file_obj.id)
        vs_file_count = get_vector_store_file_count(local_client, VS_ID)
        print(f"[NEURO_WS] vector store files for VS_ID={VS_ID}: {vs_file_count}")

    return {
        "db_row_count": db_row_count,
        "rows": row_count,
        "assistant_id": None,
        "vector_store_id": VS_ID,
        "vector_store_file_count": vs_file_count,
        "old_vector_store_ids": [],
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

def strip_code_fences(text: str) -> str:
    s = (text or "").replace("\r", "").strip()
    if s.startswith("```"):
        first_nl = s.find("\n")
        if first_nl != -1:
            s = s[first_nl + 1:]
        s = s.strip()
        if s.endswith("```"):
            s = s[:-3]
    return s.strip()

def sanitize_explain(explain: Optional[str]) -> Optional[str]:
    if explain is None:
        return None
    s = strip_code_fences(str(explain).strip())
    if len(s) >= 2 and ((s[0] == '"' and s[-1] == '"') or (s[0] == "'" and s[-1] == "'")):
        s = s[1:-1].strip()
    s = s.replace("\r", " ")
    if re.fullmatch(r"[+-]?\d+(?:\.\d+)?", s or ""):
        return None
    s = s.replace("\\n", " ").replace("\n", " ").strip()
    return s or None


def extract_prob_and_explain(text: str) -> Tuple[Optional[float], Optional[str]]:
    if not text:
        return None, None
    s = strip_code_fences(text.strip())
    obj_text = s
    if not (s.startswith("{") and s.endswith("}")):
        start = s.find("{")
        end = s.rfind("}")
        if start != -1 and end != -1 and end > start:
            obj_text = s[start:end + 1]
    if obj_text.startswith("{") and obj_text.endswith("}"):
        try:
            obj = json.loads(obj_text)
            prob = obj.get("prob") if "prob" in obj else obj.get("probability")
            explain = sanitize_explain(obj.get("explain") or obj.get("reason"))
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
        explain = sanitize_explain(m2.group(1))
        if explain:
            explain = explain.rstrip("}").strip()
    return prob, sanitize_explain(explain)

def fallback_explain_from_text(text: str) -> Optional[str]:
    s = strip_code_fences((text or "").strip())
    if not s:
        return None
    s = s.strip("{}").strip()
    s = re.sub(r"(?i)\bprob(?:ability)?\s*[:=]\s*[01](?:\.0+)?\s*[;,\s]*", "", s)
    s = re.sub(r"(?i)\bexplain\s*[:=]\s*", "", s)
    s = s.strip(" ;,")
    return sanitize_explain(s or None)

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
def adjust_prompt_for_explain(prompt: str, gpt_exp: bool) -> str:
    if not prompt or not gpt_exp:
        return prompt
    replacement = 'Output JSON only: {"prob": 0.00, "explain": "<25 words max>"}'
    pat1 = r'Output JSON only:\s*\{[^\n]*\}'
    if re.search(pat1, prompt):
        return re.sub(pat1, replacement, prompt)
    pat2 = r'OUTPUT FORMAT:\s*\{[^\n]*\}'
    if re.search(pat2, prompt):
        return re.sub(pat2, replacement, prompt)
    pat3 = r'Output JSON only:\s*\{[^\n]*\}'
    if re.search(pat3, prompt, flags=re.IGNORECASE):
        return re.sub(pat3, replacement, prompt, flags=re.IGNORECASE)
    if '{"prob": 0.00}' in prompt:
        return prompt.replace('{"prob": 0.00}', '{"prob": 0.00, "explain": "<25 words max>"}')
    if '{"prob": number}' in prompt:
        return prompt.replace('{"prob": number}', '{"prob": 0.00, "explain": "<25 words max>"}')
    return prompt + "\n" + replacement

def run_response(model: str, description_text: str, gpt_exp: bool, pid: Optional[str] = None) -> dict:
    if PROMPT_S2_MA50:
        prompt = PROMPT_S2_MA50
    else:
        prompt = DEFAULT_PROMPT if USE_DEFAULT_PROMPT_JSON else MINIMAL_PROMPT
    prompt = adjust_prompt_for_explain(prompt, gpt_exp)
    if prompt and ("json" not in prompt.lower()):
        prompt = prompt + "\nRespond in JSON only."
    user_text = description_text or ""
    if "json" not in user_text.lower():
        user_text = "JSON ONLY. " + user_text
    input_payload = [{"role": "user", "content": user_text}]
    kwargs = {
        "model": model,
        "instructions": prompt,
        "input": input_payload,
        "text": {"format": {"type": "json_object"}},
        "temperature": 0.1,
        "top_p": 0.1,
        "max_output_tokens": MAXTOKENS,
    }
    if MINTOKENS > 0:
        kwargs["min_output_tokens"] = MINTOKENS
    if USE_VECTOR_DB and VS_ID:
        tool = {
            "type": "file_search",
            "vector_store_ids": [VS_ID],
            "max_num_results": 3,
        }
        if pid:
            tool["filters"] = {"type": "in", "key": "pid", "value": [pid]}
        kwargs["tools"] = [tool]
    response = client.responses.create(**kwargs)
    text = (getattr(response, "output_text", "") or "").strip()
    try:
        return json.loads(text) if text else {}
    except Exception:
        return {}

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



def extract_first_json_object(text: str) -> str:
    if not text:
        return ""
    start = text.find("{")
    if start < 0:
        return ""
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0:
                    return text[start:i + 1]
    return ""

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

        model = DEFAULT_MODEL
        if not description_text:
            msg = "Missing description for assistant call"
            return JSONResponse(status_code=400, content={"error": msg, "version": CODE_VERSION})
        try:
            reply_obj = await asyncio.to_thread(run_response, model, description_text, gpt_exp, pid)
            explain = ""
            prob = None
            if isinstance(reply_obj, dict):
                prob = reply_obj.get("prob")
                if prob is None:
                    prob = reply_obj.get("probability")
                if gpt_exp:
                    explain = sanitize_explain(reply_obj.get("explain")) or ""
            else:
                reply_obj = {}

            if prob is None:
                msg = "Assistant did not return a numeric probability"
                print(f"⚠️ {msg}. Using fallback.")
                if STRICT_FAIL_ON_UNPARSABLE:
                    await inflight_finish(prox_key, err=RuntimeError(msg))
                    return JSONResponse(status_code=502, content={"error": msg, "version": CODE_VERSION, "cache": "miss"})
                prob = 0.5

            prob = min(1.0, max(0.0, float(prob)))
            if gpt_exp and not explain:
                explain = ""
            if gpt_exp:
                print(f"Assistant reply raw: {json.dumps(reply_obj, ensure_ascii=True)}")
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


@app.post("/analyzer")
async def analyzer(request: Request):
    try:
        payload = await request.json()
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": f"Invalid JSON: {e}", "version": CODE_VERSION})

    description_text = payload.get("description") or ""
    prompt = payload.get("prompt") or DEFAULT_PROMPT
    model = DEFAULT_MODEL
    kwargs = {
        "model": model,
        "instructions": prompt,
        "input": [{"role": "user", "content": description_text}],
        "temperature": 0.1,
        "top_p": 0.1,
        "max_output_tokens": MAXTOKENS,
    }
    response = client.responses.create(**kwargs)
    text = (getattr(response, "output_text", "") or "").strip()
    return {"text": text, "version": CODE_VERSION}


@app.get("/config-ui/login", response_class=HTMLResponse)
async def config_ui_login_page(request: Request):
    if not config_ui_enabled():
        return HTMLResponse("Config UI is disabled", status_code=404)
    if get_config_ui_session_user(request):
        return RedirectResponse(url="/config-ui", status_code=303)
    error = config_ui_requirements_error()
    if error:
        return HTMLResponse(config_ui_login_html(error), status_code=503)
    return HTMLResponse(config_ui_login_html())


@app.post("/config-ui/login")
async def config_ui_login_submit(request: Request):
    if not config_ui_enabled():
        return HTMLResponse("Config UI is disabled", status_code=404)
    body = (await request.body()).decode("utf-8", errors="replace")
    form = parse_qs(body, keep_blank_values=True)
    username = (form.get("username", [""])[0] or "").strip()
    password = form.get("password", [""])[0] or ""

    error = config_ui_requirements_error()
    if error:
        return HTMLResponse(config_ui_login_html(error), status_code=503)
    if username != get_config_ui_username() or not verify_config_ui_password(password, get_config_ui_password_hash()):
        return HTMLResponse(config_ui_login_html("Invalid login or password"), status_code=401)

    response = RedirectResponse(url="/config-ui", status_code=303)
    response.set_cookie(
        CONFIG_UI_COOKIE_NAME,
        sign_config_ui_session(username),
        max_age=CONFIG_UI_COOKIE_MAX_AGE,
        httponly=True,
        secure=config_ui_cookie_secure(),
        samesite="lax",
    )
    return response


@app.post("/config-ui/logout")
async def config_ui_logout():
    response = RedirectResponse(url="/config-ui/login", status_code=303)
    response.delete_cookie(CONFIG_UI_COOKIE_NAME)
    return response


@app.get("/config-ui", response_class=HTMLResponse)
async def config_ui_page(request: Request):
    if not config_ui_enabled():
        return HTMLResponse("Config UI is disabled", status_code=404)
    error = config_ui_requirements_error()
    if error:
        return HTMLResponse(config_ui_login_html(error), status_code=503)
    username = get_config_ui_session_user(request)
    if not username:
        return RedirectResponse(url="/config-ui/login", status_code=303)
    return HTMLResponse(config_ui_app_html(username, get_config_ui_actor()))


@app.get("/config-ui/api/accounts")
async def config_ui_accounts(request: Request):
    _, actor, auth = require_config_ui_api(request)
    if auth:
        return auth
    conn = config_ui_conn()
    try:
        with conn.cursor(cursor_factory=DictCursor) as cur:
            cur.execute(
                """
                SELECT account_login, account_label
                  FROM bot_param.operator_account
                 WHERE db_user = %s
                   AND env = 'prod'
                   AND enabled = true
                   AND can_edit = true
                 ORDER BY account_label, account_login
                """,
                (actor,),
            )
            rows = [dict(row) for row in cur.fetchall()]
        return {"ok": True, "actor": actor, "accounts": rows, "version": CODE_VERSION}
    except Exception as exc:
        return config_ui_json_error(500, first_error_line(exc))
    finally:
        conn.close()


@app.get("/config-ui/api/bots")
async def config_ui_bots(request: Request, account_login: int):
    _, actor, auth = require_config_ui_api(request)
    if auth:
        return auth
    conn = config_ui_conn()
    try:
        with conn.cursor(cursor_factory=DictCursor) as cur:
            if not ensure_actor_account_access(cur, actor, account_login, can_apply=False):
                return config_ui_json_error(403, "account is not available for actor")
            set_actor(cur, actor)
            ensure_rm_controller_db_config(cur, account_login)
            cur.execute(
                """
                SELECT DISTINCT
                       e.bot,
                       COALESCE(c.display_name, e.bot) AS display_name,
                       COALESCE(c.sort_order, 9999) AS sort_order
                  FROM bot_param.bot_config_user_editor e
                  JOIN bot_param.bot_config_param_catalog pc
                    ON pc.bot_kind = e.bot
                   AND COALESCE(pc.input_param_name, pc.param_key) = e.input_param
                   AND COALESCE(pc.user_editable, true) = true
                  LEFT JOIN bot_param.bot_catalog c
                    ON c.bot_kind = e.bot
                  JOIN bot_param.operator_account oa
                    ON oa.env = 'prod'
                   AND oa.account_login = e.account_login
                   AND oa.db_user = %s
                   AND oa.enabled = true
                   AND oa.can_edit = true
                 WHERE e.account_login = %s
                 ORDER BY sort_order, e.bot
                """,
                (actor, account_login),
            )
            rows = [dict(row) for row in cur.fetchall()]
            rows.sort(key=lambda row: (int(row.get("sort_order") or 9999), row.get("display_name") or row.get("bot") or ""))
        conn.commit()
        return {"ok": True, "bots": rows, "version": CODE_VERSION}
    except Exception as exc:
        return config_ui_json_error(500, first_error_line(exc))
    finally:
        conn.close()


@app.get("/config-ui/api/params")
async def config_ui_params(request: Request, account_login: int, bot: str):
    _, actor, auth = require_config_ui_api(request)
    if auth:
        return auth
    bot = (bot or "").strip().lower()
    if not bot:
        return config_ui_json_error(400, "bot is required")
    conn = config_ui_conn()
    try:
        with conn.cursor(cursor_factory=DictCursor) as cur:
            if not ensure_actor_account_access(cur, actor, account_login, can_apply=False):
                return config_ui_json_error(403, "account is not available for actor")
            cur.execute(
                """
                SELECT
                    e.row_id,
                    e.account_login,
                    e.bot,
                    COALESCE(pc.section_name, e.param_group) AS param_group,
                    e.input_param,
                    e.param_desc,
                    e.current_value,
                    CASE
                        WHEN EXISTS (
                            SELECT 1
                              FROM bot_param.bot_config_allowed_value av
                             WHERE av.bot = e.bot
                               AND av.input_param = e.input_param
                        )
                        THEN e.new_choice
                        ELSE e.new_value
                    END AS new_value_ui,
                    e.reason,
                    EXISTS (
                        SELECT 1
                          FROM bot_param.bot_config_allowed_value av
                        WHERE av.bot = e.bot
                           AND av.input_param = e.input_param
                    ) AS has_choices,
                    COALESCE(audit.prev_values, '[]'::jsonb) AS prev_values
                  FROM bot_param.bot_config_user_editor e
                  JOIN bot_param.bot_config_param_catalog pc
                    ON pc.bot_kind = e.bot
                   AND COALESCE(pc.input_param_name, pc.param_key) = e.input_param
                   AND COALESCE(pc.user_editable, true) = true
                  LEFT JOIN LATERAL (
                    SELECT jsonb_agg(
                               jsonb_build_object(
                                 'changed_at', to_char(hist.changed_at AT TIME ZONE 'UTC', 'YYYY-MM-DD HH24:MI') || ' UTC',
                                 'old_value', hist.old_value,
                                 'new_value', hist.new_value,
                                 'changed_by', hist.changed_by
                               )
                               ORDER BY hist.changed_at DESC
                           ) AS prev_values
                      FROM (
                        SELECT a.changed_at,
                               a.old_value #>> '{}' AS old_value,
                               a.new_value #>> '{}' AS new_value,
                               a.changed_by
                          FROM bot_param.bot_param_audit a
                         WHERE a.env = 'prod'
                           AND a.account_login = e.account_login
                           AND a.bot_kind = e.bot
                           AND a.param_path = pc.param_path
                           AND a.old_value IS NOT NULL
                           AND NULLIF(a.old_value #>> '{}', '') IS NOT NULL
                         ORDER BY a.changed_at DESC
                         LIMIT 10
                      ) hist
                  ) audit ON true
                  JOIN bot_param.operator_account oa
                    ON oa.env = 'prod'
                   AND oa.account_login = e.account_login
                   AND oa.db_user = %s
                   AND oa.enabled = true
                   AND oa.can_edit = true
                 WHERE e.account_login = %s
                   AND e.bot = %s
                 ORDER BY COALESCE(pc.section_name, e.param_group, ''),
                          COALESCE(pc.sort_order, 999999),
                          e.input_param
                """,
                (actor, account_login, bot),
            )
            rows = [dict(row) for row in cur.fetchall()]
        return {"ok": True, "params": rows, "version": CODE_VERSION}
    except Exception as exc:
        return config_ui_json_error(500, first_error_line(exc))
    finally:
        conn.close()


@app.get("/config-ui/api/param-catalog")
async def config_ui_param_catalog(request: Request, account_login: int, bot: str):
    _, actor, auth = require_config_ui_api(request)
    if auth:
        return auth
    bot = (bot or "").strip().lower()
    if not bot:
        return config_ui_json_error(400, "bot is required")
    conn = config_ui_conn()
    try:
        with conn.cursor(cursor_factory=DictCursor) as cur:
            if not ensure_actor_account_access(cur, actor, account_login, can_apply=False):
                return config_ui_json_error(403, "account is not available for actor")
            set_actor(cur, actor)
            if is_rm_controller_bot(bot):
                ensure_rm_controller_db_config(cur, account_login)
                bot = RM_CONTROLLER_BOT
            columns = bot_config_param_catalog_columns(cur)
            min_numeric_expr = "pc.min_numeric" if "min_numeric" in columns else "NULL::numeric AS min_numeric"
            max_numeric_expr = "pc.max_numeric" if "max_numeric" in columns else "NULL::numeric AS max_numeric"
            cur.execute(
                f"""
                SELECT pc.bot_kind,
                       pc.section_name,
                       pc.param_key,
                       pc.input_param_name,
                       pc.display_name,
                       pc.param_desc,
                       pc.param_path,
                       pc.value_type,
                       {min_numeric_expr},
                       {max_numeric_expr},
                       COALESCE(pc.allowed_values, av.allowed_values) AS allowed_values,
                       pc.sort_order,
                       COALESCE(pc.user_editable, true) AS user_editable
                  FROM bot_param.bot_config_param_catalog pc
                  LEFT JOIN LATERAL (
                    SELECT array_agg(v.allowed_value ORDER BY v.sort_order, v.allowed_value) AS allowed_values
                      FROM bot_param.bot_config_allowed_value v
                     WHERE v.bot = pc.bot_kind
                       AND v.input_param = COALESCE(pc.input_param_name, pc.param_key)
                  ) av ON true
                 WHERE pc.bot_kind = %s
                 ORDER BY COALESCE(pc.sort_order, 999999),
                          COALESCE(pc.section_name, ''),
                          pc.param_key
                """,
                (bot,),
            )
            rows = [dict(row) for row in cur.fetchall()]
            for row in rows:
                for numeric_field in ("min_numeric", "max_numeric"):
                    if row.get(numeric_field) is not None:
                        row[numeric_field] = str(row[numeric_field])
                if row.get("allowed_values") is None:
                    row["allowed_values"] = []
        conn.commit()
        return {"ok": True, "params": rows, "version": CODE_VERSION}
    except Exception as exc:
        conn.rollback()
        return config_ui_json_error(500, first_error_line(exc))
    finally:
        conn.close()


@app.post("/config-ui/api/param-catalog/save")
async def config_ui_param_catalog_save(req: ParamCatalogSaveRequest, request: Request):
    _, actor, auth = require_config_ui_api(request)
    if auth:
        return auth
    bot = (req.bot or "").strip().lower()
    if not bot:
        return config_ui_json_error(400, "bot is required")
    if not req.changes:
        return config_ui_json_error(400, "changes is empty")
    if len(req.changes) > 200:
        return config_ui_json_error(400, "too many changes")

    conn = config_ui_conn()
    conn.autocommit = False
    try:
        updated = []
        refreshed_accounts = []
        with conn.cursor(cursor_factory=DictCursor) as cur:
            set_actor(cur, actor)
            if not ensure_actor_account_access(cur, actor, req.account_login, can_apply=True):
                return config_ui_json_error(403, "account is not available for apply")
            if is_rm_controller_bot(bot):
                ensure_rm_controller_db_config(cur, req.account_login)
                bot = RM_CONTROLLER_BOT
            columns = bot_config_param_catalog_columns(cur)
            has_min_numeric = "min_numeric" in columns
            has_max_numeric = "max_numeric" in columns

            for change in req.changes:
                change_bot = (change.bot_kind or bot).strip().lower()
                if change_bot != bot:
                    raise ValueError(f"{change.param_key}: bot_kind must stay {bot}")

                param_key = clean_required_text(change.param_key, "param_key")
                param_path = clean_required_text(change.param_path, f"{param_key}.param_path")
                value_type = clean_required_text(change.value_type, f"{param_key}.value_type").lower()
                input_param_name = clean_optional_text(change.input_param_name)
                allowed_values = normalize_catalog_allowed_values(change.allowed_values)
                user_editable = True if change.user_editable is None else bool(change.user_editable)

                cur.execute(
                    """
                    SELECT COALESCE(input_param_name, param_key) AS input_param
                      FROM bot_param.bot_config_param_catalog
                     WHERE bot_kind = %s
                       AND param_key = %s
                     FOR UPDATE
                    """,
                    (bot, param_key),
                )
                existing = cur.fetchone()
                if not existing:
                    raise ValueError(f"{param_key}: catalog row not found")
                old_input_param = existing["input_param"]
                new_input_param = input_param_name or param_key

                set_columns = [
                    ("section_name", clean_optional_text(change.section_name)),
                    ("input_param_name", input_param_name),
                    ("display_name", clean_optional_text(change.display_name)),
                    ("param_desc", clean_optional_text(change.param_desc)),
                    ("param_path", param_path),
                    ("value_type", value_type),
                    ("allowed_values", allowed_values),
                    ("sort_order", change.sort_order),
                    ("user_editable", user_editable),
                ]
                if has_min_numeric:
                    set_columns.insert(6, ("min_numeric", normalize_catalog_numeric(change.min_numeric, f"{param_key}.min_numeric")))
                if has_max_numeric:
                    insert_at = 7 if has_min_numeric else 6
                    set_columns.insert(insert_at, ("max_numeric", normalize_catalog_numeric(change.max_numeric, f"{param_key}.max_numeric")))

                set_sql = ", ".join(f"{column_name} = %s" for column_name, _ in set_columns)
                values = [value for _, value in set_columns]
                min_numeric_expr = "min_numeric" if has_min_numeric else "NULL::numeric AS min_numeric"
                max_numeric_expr = "max_numeric" if has_max_numeric else "NULL::numeric AS max_numeric"
                cur.execute(
                    f"""
                    UPDATE bot_param.bot_config_param_catalog
                       SET {set_sql}
                     WHERE bot_kind = %s
                       AND param_key = %s
                     RETURNING bot_kind,
                               section_name,
                               param_key,
                               input_param_name,
                               display_name,
                               param_desc,
                               param_path,
                               value_type,
                               {min_numeric_expr},
                               {max_numeric_expr},
                               allowed_values,
                               sort_order,
                               COALESCE(user_editable, true) AS user_editable
                    """,
                    tuple(values + [bot, param_key]),
                )
                row = dict(cur.fetchone())
                for numeric_field in ("min_numeric", "max_numeric"):
                    if row.get(numeric_field) is not None:
                        row[numeric_field] = str(row[numeric_field])
                if row.get("allowed_values") is None:
                    row["allowed_values"] = []
                updated.append(row)

                if old_input_param != new_input_param:
                    sync_catalog_allowed_value_rows(cur, bot, old_input_param, None)
                sync_catalog_allowed_value_rows(cur, bot, new_input_param, allowed_values)

            refreshed_accounts = catalog_refresh_accounts(cur, actor, req.account_login, bot)

        conn.commit()
        return {
            "ok": True,
            "actor": actor,
            "updated": updated,
            "updated_count": len(updated),
            "refreshed_accounts": refreshed_accounts,
            "version": CODE_VERSION,
        }
    except ValueError as exc:
        conn.rollback()
        return config_ui_json_error(400, str(exc))
    except Exception as exc:
        conn.rollback()
        return config_ui_json_error(500, first_error_line(exc))
    finally:
        conn.close()


@app.get("/config-ui/api/choices-bulk")
async def config_ui_choices_bulk(request: Request, bot: str):
    _, actor, auth = require_config_ui_api(request)
    if auth:
        return auth
    bot = (bot or "").strip().lower()
    if not bot:
        return config_ui_json_error(400, "bot is required")
    conn = config_ui_conn()
    try:
        with conn.cursor(cursor_factory=DictCursor) as cur:
            cur.execute(
                """
                SELECT input_param, allowed_value, value_desc
                  FROM bot_param.bot_config_allowed_value
                 WHERE bot = %s
                 ORDER BY input_param, sort_order, allowed_value
                """,
                (bot,),
            )
            choices_by_param = {}
            for row in cur.fetchall():
                input_param = row["input_param"]
                choices_by_param.setdefault(input_param, []).append({
                    "allowed_value": row["allowed_value"],
                    "value_desc": row["value_desc"],
                })
        return {"ok": True, "actor": actor, "choices_by_param": choices_by_param, "version": CODE_VERSION}
    except Exception as exc:
        return config_ui_json_error(500, first_error_line(exc))
    finally:
        conn.close()


@app.get("/config-ui/api/choices")
async def config_ui_choices(request: Request, bot: str, input_param: str):
    _, actor, auth = require_config_ui_api(request)
    if auth:
        return auth
    bot = (bot or "").strip().lower()
    input_param = (input_param or "").strip()
    if not bot or not input_param:
        return config_ui_json_error(400, "bot and input_param are required")
    conn = config_ui_conn()
    try:
        with conn.cursor(cursor_factory=DictCursor) as cur:
            cur.execute(
                """
                SELECT allowed_value, value_desc
                  FROM bot_param.bot_config_allowed_value
                 WHERE bot = %s
                   AND input_param = %s
                 ORDER BY sort_order, allowed_value
                """,
                (bot, input_param),
            )
            rows = [dict(row) for row in cur.fetchall()]
        return {"ok": True, "actor": actor, "choices": rows, "version": CODE_VERSION}
    except Exception as exc:
        return config_ui_json_error(500, first_error_line(exc))
    finally:
        conn.close()


@app.get("/config-ui/api/analytics/profit-by-source")
async def config_ui_analytics_profit_by_source(
    request: Request,
    year: Optional[int] = None,
    month: Optional[int] = None,
    filedate: Optional[str] = None,
):
    _, actor, auth = require_config_ui_api(request)
    if auth:
        return auth
    conn = config_ui_conn()
    try:
        with conn.cursor(cursor_factory=DictCursor) as cur:
            exact_filedate = parse_config_ui_date(filedate, "filedate")
            query_filedate = None
            period_label = None
            filter_mode = "latest"
            filedate_expr = config_ui_analytics_filedate_date_sql("cp")
            where_sql = ""
            params = []

            if exact_filedate is not None:
                query_filedate = exact_filedate
                period_label = exact_filedate.isoformat()
                filter_mode = "filedate"
                where_sql = "cp.filedate_date = %s"
                params = [query_filedate]
            elif year is not None:
                if year < 2000 or year > 2100:
                    raise ValueError("year must be between 2000 and 2100")
                if month is not None:
                    if month < 1 or month > 12:
                        raise ValueError("month must be between 1 and 12")
                    start_date = date(year, month, 1)
                    end_date = date(year + 1, 1, 1) if month == 12 else date(year, month + 1, 1)
                    period_label = f"{year:04d}-{month:02d}"
                    filter_mode = "month"
                else:
                    start_date = date(year, 1, 1)
                    end_date = date(year + 1, 1, 1)
                    period_label = f"{year:04d}"
                    filter_mode = "year"
                where_sql = "cp.filedate_date >= %s AND cp.filedate_date < %s"
                params = [start_date, end_date]
            elif month is not None:
                raise ValueError("year is required with month")
            else:
                cur.execute(
                    f"""
                    SELECT MAX(filedate_date) AS filedate
                      FROM (
                        SELECT {filedate_expr} AS filedate_date
                          FROM cust_positions cp
                      ) latest
                     WHERE filedate_date IS NOT NULL
                    """
                )
                latest = cur.fetchone()
                query_filedate = latest["filedate"] if latest else None
                period_label = config_ui_db_text(query_filedate)
                if query_filedate is None:
                    return {
                        "ok": True,
                        "actor": actor,
                        "filter_mode": filter_mode,
                        "filedate": None,
                        "period_label": None,
                        "total_profit": 0.0,
                        "total_deals": 0,
                        "rows": [],
                        "version": CODE_VERSION,
                    }
                where_sql = "cp.filedate_date = %s"
                params = [query_filedate]

            cur.execute(
                f"""
                WITH positions AS (
                    SELECT cp.source_id::text AS source_id,
                           cp.profit,
                           cp.position,
                           {filedate_expr} AS filedate_date
                      FROM cust_positions cp
                ),
                source_rows AS (
                    SELECT cp.source_id,
                           COALESCE(SUM(cp.profit), 0) AS profit,
                           COUNT(DISTINCT cp.position) AS deals
                      FROM positions cp
                     WHERE {where_sql}
                     GROUP BY cp.source_id
                )
                SELECT sr.source_id,
                       bc.bot_kind,
                       sr.profit,
                       sr.deals
                  FROM source_rows sr
                  LEFT JOIN LATERAL (
                    SELECT c.bot_kind
                      FROM bot_param.bot_catalog c
                     WHERE lower(c.source_id) = lower(sr.source_id)
                       AND COALESCE(c.enabled, true) = true
                     ORDER BY COALESCE(c.sort_order, 999999), c.bot_kind
                     LIMIT 1
                  ) bc ON true
                 ORDER BY sr.source_id ASC NULLS LAST
                 LIMIT 5000
                """,
                tuple(params),
            )
            rows = []
            total_profit = Decimal("0")
            total_deals = 0
            for row in cur.fetchall():
                profit = config_ui_db_decimal(row["profit"])
                deals = int(row["deals"] or 0)
                source_id = config_ui_db_text(row["source_id"])
                bot_kind = config_ui_db_text(row["bot_kind"]) or config_ui_analytics_source_fallback(source_id)
                total_profit += profit
                total_deals += deals
                rows.append(
                    {
                        "source_id": source_id,
                        "bot_kind": bot_kind,
                        "profit": float(profit),
                        "deals": deals,
                    }
                )
        return {
            "ok": True,
            "actor": actor,
            "filter_mode": filter_mode,
            "filedate": config_ui_db_text(query_filedate),
            "period_label": period_label,
            "total_profit": float(total_profit),
            "total_deals": total_deals,
            "rows": rows,
            "version": CODE_VERSION,
        }
    except ValueError as exc:
        return config_ui_json_error(400, str(exc))
    except Exception as exc:
        return config_ui_json_error(500, first_error_line(exc))
    finally:
        conn.close()


@app.get("/config-ui/api/copy-target-accounts")
async def config_ui_copy_target_accounts(request: Request, source_account_login: int, bot: str):
    _, actor, auth = require_config_ui_api(request)
    if auth:
        return auth
    bot = (bot or "").strip().lower()
    conn = config_ui_conn()
    try:
        with conn.cursor(cursor_factory=DictCursor) as cur:
            if not ensure_actor_account_access(cur, actor, source_account_login, can_apply=True):
                return config_ui_json_error(403, "source account is not available for apply")
            set_actor(cur, actor)
            if is_rm_controller_bot(bot):
                cur.execute(
                    """
                    SELECT oa.account_login
                      FROM bot_param.operator_account oa
                     WHERE oa.env = 'prod'
                       AND oa.db_user = %s
                       AND oa.enabled = true
                       AND oa.can_apply = true
                    """,
                    (actor,),
                )
                for row in cur.fetchall():
                    ensure_rm_controller_db_config(cur, row["account_login"])
            cur.execute(
                """
                SELECT oa.account_login,
                       oa.account_label,
                       EXISTS (
                           SELECT 1
                             FROM bot_param.bot_config_user_editor e
                             JOIN bot_param.bot_config_param_catalog pc
                               ON pc.bot_kind = e.bot
                              AND COALESCE(pc.input_param_name, pc.param_key) = e.input_param
                              AND COALESCE(pc.user_editable, true) = true
                            WHERE e.account_login = oa.account_login
                              AND e.bot = %s
                       ) AS has_bot_config
                  FROM bot_param.operator_account oa
                 WHERE oa.env = 'prod'
                   AND oa.db_user = %s
                   AND oa.enabled = true
                   AND oa.can_apply = true
                   AND oa.account_login <> %s
                 ORDER BY oa.account_label, oa.account_login
                """,
                (bot, actor, source_account_login),
            )
            rows = [dict(row) for row in cur.fetchall()]
        conn.commit()
        return {"ok": True, "accounts": rows, "version": CODE_VERSION}
    except Exception as exc:
        return config_ui_json_error(500, first_error_line(exc))
    finally:
        conn.close()


@app.get("/config-ui/api/runtime-status")
async def config_ui_runtime_status(request: Request, account_login: Optional[List[int]] = Query(None)):
    _, actor, auth = require_config_ui_api(request)
    if auth:
        return auth
    account_logins = []
    for account in account_login or []:
        account = int(account)
        if account not in account_logins:
            account_logins.append(account)
    conn = config_ui_conn()
    try:
        with conn.cursor(cursor_factory=DictCursor) as cur:
            for account in account_logins:
                if not ensure_actor_account_access(cur, actor, account, can_apply=False):
                    return config_ui_json_error(403, f"account {account} is not available")
            params = [actor]
            account_filter = ""
            if account_logins:
                account_filter = "AND r.account_login = ANY(%s)"
                params.append(account_logins)
            cur.execute(
                f"""
                SELECT r.account_login,
                       oa.account_label,
                       r.instance_id,
                       r.bot_kind,
                       r.bot_id,
                       r.source_id,
                       r.status,
                       r.allow_new_entries,
                       r.applied_version_no,
                       r.applied_config_hash,
                       r.last_seen_at,
                       r.last_error
                  FROM bot_param.bot_runtime_status r
                  JOIN bot_param.operator_account oa
                    ON oa.env = r.env
                   AND oa.account_login = r.account_login
                   AND oa.db_user = %s
                   AND oa.enabled = true
                 WHERE r.env = 'prod'
                   {account_filter}
                 ORDER BY r.account_login, r.bot_kind, r.bot_id, r.last_seen_at DESC
                 LIMIT 100
                """,
                tuple(params),
            )
            rows = [dict(row) for row in cur.fetchall()]
        return {"ok": True, "statuses": rows, "version": CODE_VERSION}
    except Exception as exc:
        return config_ui_json_error(500, first_error_line(exc))
    finally:
        conn.close()


@app.get("/config-ui/api/rm-state")
async def config_ui_rm_state(request: Request, account_login: Optional[List[int]] = Query(None)):
    _, actor, auth = require_config_ui_api(request)
    if auth:
        return auth
    account_logins = []
    for account in account_login or []:
        account = int(account)
        if account not in account_logins:
            account_logins.append(account)
    conn = config_ui_conn()
    try:
        with conn.cursor(cursor_factory=DictCursor) as cur:
            ensure_rm_state_schema(cur)
            params = [actor]
            account_filter = ""
            if account_logins:
                for account in account_logins:
                    if not ensure_actor_account_access(cur, actor, account, can_apply=False):
                        return config_ui_json_error(403, f"account {account} is not available")
                account_filter = "AND s.account_login = ANY(%s)"
                params.append(account_logins)
            cur.execute(
                f"""
                SELECT s.env,
                       s.account_login,
                       oa.account_label,
                       s.scope,
                       NULLIF(s.bot_kind, '') AS bot_kind,
                       s.active,
                       s.action,
                       s.reset_mode,
                       s.reset_at,
                       s.state_version,
                       s.source_command_id,
                       s.reason,
                       s.updated_by,
                       s.updated_source,
                       s.updated_at
                  FROM bot_param.rm_state s
                  JOIN bot_param.operator_account oa
                    ON oa.env = s.env
                   AND oa.account_login = s.account_login
                   AND oa.db_user = %s
                   AND oa.enabled = true
                 WHERE s.env = 'prod'
                   {account_filter}
                 ORDER BY s.account_login, s.scope, s.bot_kind
                """,
                tuple(params),
            )
            rows = [rm_state_row_dict(row) for row in cur.fetchall()]
        conn.commit()
        return {"ok": True, "rm_states": rows, "version": CODE_VERSION}
    except Exception as exc:
        conn.rollback()
        return config_ui_json_error(500, first_error_line(exc))
    finally:
        conn.close()


@app.post("/config-ui/api/rm-command")
async def config_ui_rm_command(req: RmControlCommandRequest, request: Request):
    _, actor, auth = require_config_ui_api(request)
    if auth:
        return auth

    account_logins = []
    for account in req.account_logins or []:
        account = int(account)
        if account not in account_logins:
            account_logins.append(account)
    if not account_logins:
        return config_ui_json_error(400, "choose at least one account")
    if len(account_logins) > 20:
        return config_ui_json_error(400, "too many accounts")

    conn = config_ui_conn()
    conn.autocommit = False
    try:
        commands = []
        with conn.cursor(cursor_factory=DictCursor) as cur:
            set_actor(cur, actor)
            for account in account_logins:
                if not ensure_actor_account_access(cur, actor, account, can_apply=True):
                    raise ValueError(f"account {account} is not available for RM control")
                command_text = build_rm_control_command(account, req)
                commands.append(
                    insert_rm_control_command(
                        cur,
                        account,
                        actor,
                        command_text,
                        (req.reason or f"RM control {(req.action or '').strip().lower()}").strip(),
                        (req.action or "").strip().lower(),
                    )
                )
        conn.commit()
        return {"ok": True, "commands": commands, "version": CODE_VERSION}
    except ValueError as exc:
        conn.rollback()
        return config_ui_json_error(400, str(exc))
    except Exception as exc:
        conn.rollback()
        return config_ui_json_error(500, first_error_line(exc))
    finally:
        conn.close()


@app.get("/config-ui/api/recommendations")
async def config_ui_recommendations(request: Request, account_login: int):
    _, actor, auth = require_config_ui_api(request)
    if auth:
        return auth
    conn = config_ui_conn()
    try:
        with conn.cursor(cursor_factory=DictCursor) as cur:
            if not ensure_actor_account_access(cur, actor, account_login, can_apply=False):
                return config_ui_json_error(403, "account is not available for actor")
            if not table_exists(cur, "bot_online", "bot_control_recommendation"):
                return {"ok": True, "recommendations": [], "version": CODE_VERSION}
            columns = table_columns(cur, "bot_online", "bot_control_recommendation")
            if "recommendation_id" not in columns or "account_login" not in columns:
                return config_ui_json_error(500, "bot_online.bot_control_recommendation is missing required columns")
            cur.execute(recommendation_select_sql(columns), (actor, account_login))
            rows = [dict(row) for row in cur.fetchall()]
        return {"ok": True, "recommendations": rows, "version": CODE_VERSION}
    except Exception as exc:
        return config_ui_json_error(500, first_error_line(exc))
    finally:
        conn.close()


@app.post("/config-ui/api/recommendations/{recommendation_id}/approve")
async def config_ui_recommendation_approve(recommendation_id: int, req: RecommendationDecisionRequest, request: Request):
    _, actor, auth = require_config_ui_api(request)
    if auth:
        return auth
    conn = config_ui_conn()
    conn.autocommit = False
    try:
        with conn.cursor(cursor_factory=DictCursor) as cur:
            if not table_exists(cur, "bot_online", "bot_control_recommendation"):
                return config_ui_json_error(404, "bot_online.bot_control_recommendation does not exist")
            columns = table_columns(cur, "bot_online", "bot_control_recommendation")
            if "recommendation_id" not in columns or "account_login" not in columns:
                return config_ui_json_error(500, "bot_online.bot_control_recommendation is missing required columns")
            if "status" not in columns:
                return config_ui_json_error(500, "bot_online.bot_control_recommendation.status is required")

            cur.execute(
                """
                SELECT account_login
                  FROM bot_online.bot_control_recommendation
                 WHERE recommendation_id = %s
                """,
                (recommendation_id,),
            )
            account_row = cur.fetchone()
            if not account_row:
                return config_ui_json_error(404, "recommendation not found")
            account_login = int(account_row["account_login"])
            if not ensure_actor_account_access(cur, actor, account_login, can_apply=True):
                return config_ui_json_error(403, "account is not available for apply")

            cur.execute(recommendation_select_sql(columns, for_update=True, by_id=True), (actor, account_login, recommendation_id))
            row = cur.fetchone()
            if not row:
                return config_ui_json_error(404, "recommendation not found")
            rec = dict(row)
            if str(rec.get("status") or "").lower() != "new":
                raise ValueError("only recommendation with status=new can be approved")
            if str(rec.get("decision_type") or "").lower() != "param_change":
                raise ValueError("only decision_type=param_change can be approved through Consul")

            bot = (rec.get("bot_kind") or rec.get("bot_id") or "").strip().lower()
            input_param = (rec.get("input_param") or "").strip()
            recommended_value = "" if rec.get("recommended_value") is None else str(rec.get("recommended_value")).strip()
            if not bot:
                raise ValueError("recommendation bot_kind is empty")
            if not input_param:
                raise ValueError("recommendation input_param/param_key is empty")
            if not recommended_value:
                raise ValueError("recommendation recommended_value is empty")

            cur.execute(
                """
                SELECT e.row_id, e.account_login, e.bot, e.input_param
                  FROM bot_param.bot_config_user_editor e
                  JOIN bot_param.bot_config_param_catalog pc
                    ON pc.bot_kind = e.bot
                   AND COALESCE(pc.input_param_name, pc.param_key) = e.input_param
                   AND COALESCE(pc.user_editable, true) = true
                 WHERE e.account_login = %s
                   AND e.bot = %s
                   AND e.input_param = %s
                 FOR UPDATE OF e
                """,
                (account_login, bot, input_param),
            )
            editor_row = cur.fetchone()
            if not editor_row:
                raise ValueError(f"config editor row not found for {bot}.{input_param}")
            new_choice, new_value = resolve_new_value_columns(cur, bot, input_param, recommended_value)
            reason = clean_optional_text(req.reason) or clean_optional_text(rec.get("reason")) or f"recommendation #{recommendation_id}"
            cur.execute(
                """
                UPDATE bot_param.bot_config_user_editor
                   SET new_choice = %s,
                       new_value = %s,
                       reason = %s
                 WHERE row_id = %s
                """,
                (new_choice, new_value, reason, editor_row["row_id"]),
            )

            update_sql, update_values = recommendation_status_update_sql(columns, "approved", actor, clean_optional_text(req.reason))
            cur.execute(update_sql, tuple(update_values + [recommendation_id]))
        conn.commit()
        return {
            "ok": True,
            "recommendation_id": recommendation_id,
            "status": "approved",
            "bot": bot,
            "input_param": input_param,
            "recommended_value": recommended_value,
            "version": CODE_VERSION,
        }
    except ValueError as exc:
        conn.rollback()
        return config_ui_json_error(400, str(exc))
    except Exception as exc:
        conn.rollback()
        return config_ui_json_error(500, first_error_line(exc))
    finally:
        conn.close()


@app.post("/config-ui/api/recommendations/{recommendation_id}/reject")
async def config_ui_recommendation_reject(recommendation_id: int, req: RecommendationDecisionRequest, request: Request):
    _, actor, auth = require_config_ui_api(request)
    if auth:
        return auth
    conn = config_ui_conn()
    conn.autocommit = False
    try:
        with conn.cursor(cursor_factory=DictCursor) as cur:
            if not table_exists(cur, "bot_online", "bot_control_recommendation"):
                return config_ui_json_error(404, "bot_online.bot_control_recommendation does not exist")
            columns = table_columns(cur, "bot_online", "bot_control_recommendation")
            if "recommendation_id" not in columns or "account_login" not in columns:
                return config_ui_json_error(500, "bot_online.bot_control_recommendation is missing required columns")
            if "status" not in columns:
                return config_ui_json_error(500, "bot_online.bot_control_recommendation.status is required")
            cur.execute(
                """
                SELECT account_login, status
                  FROM bot_online.bot_control_recommendation
                 WHERE recommendation_id = %s
                 FOR UPDATE
                """,
                (recommendation_id,),
            )
            row = cur.fetchone()
            if not row:
                return config_ui_json_error(404, "recommendation not found")
            if not ensure_actor_account_access(cur, actor, int(row["account_login"]), can_apply=True):
                return config_ui_json_error(403, "account is not available for apply")
            if str(row["status"] or "").lower() != "new":
                raise ValueError("only recommendation with status=new can be rejected")
            update_sql, update_values = recommendation_status_update_sql(columns, "rejected", actor, clean_optional_text(req.reason))
            cur.execute(update_sql, tuple(update_values + [recommendation_id]))
        conn.commit()
        return {"ok": True, "recommendation_id": recommendation_id, "status": "rejected", "version": CODE_VERSION}
    except ValueError as exc:
        conn.rollback()
        return config_ui_json_error(400, str(exc))
    except Exception as exc:
        conn.rollback()
        return config_ui_json_error(500, first_error_line(exc))
    finally:
        conn.close()


@app.post("/config-ui/api/save")
async def config_ui_save(req: ConfigUiSaveRequest, request: Request):
    _, actor, auth = require_config_ui_api(request)
    if auth:
        return auth
    bot = (req.bot or "").strip().lower()
    if not bot:
        return config_ui_json_error(400, "bot is required")
    if not req.changes:
        return config_ui_json_error(400, "changes is empty")
    if len(req.changes) > 100:
        return config_ui_json_error(400, "too many changes")

    conn = config_ui_conn()
    conn.autocommit = False
    try:
        applied = []
        target_accounts = []
        if req.copy_to_account_login is not None:
            target_accounts.append(int(req.copy_to_account_login))
        for account in (req.copy_to_account_logins or []):
            account = int(account)
            if account not in target_accounts:
                target_accounts.append(account)

        with conn.cursor(cursor_factory=DictCursor) as cur:
            set_actor(cur, actor)
            if not ensure_actor_account_access(cur, actor, req.account_login, can_apply=True):
                return config_ui_json_error(403, "source account is not available for apply")
            for target_account in target_accounts:
                if target_account == req.account_login:
                    return config_ui_json_error(400, "target account must differ from source account")
                if not ensure_actor_account_access(cur, actor, target_account, can_apply=True):
                    return config_ui_json_error(403, f"target account {target_account} is not available for apply")

            for change in req.changes:
                value = (change.value or "").strip()
                if not value:
                    raise ValueError(f"row {change.row_id}: value is empty")
                reason = (change.reason or "").strip() or None

                cur.execute(
                    """
                    SELECT e.row_id, e.account_login, e.bot, e.input_param
                      FROM bot_param.bot_config_user_editor e
                      JOIN bot_param.bot_config_param_catalog pc
                        ON pc.bot_kind = e.bot
                       AND COALESCE(pc.input_param_name, pc.param_key) = e.input_param
                       AND COALESCE(pc.user_editable, true) = true
                     WHERE e.row_id = %s
                       AND e.account_login = %s
                       AND e.bot = %s
                     FOR UPDATE OF e
                    """,
                    (change.row_id, req.account_login, bot),
                )
                source = cur.fetchone()
                if not source:
                    raise ValueError(f"row {change.row_id}: source row not found")

                new_choice, new_value = resolve_new_value_columns(cur, source["bot"], source["input_param"], value)
                cur.execute(
                    """
                    UPDATE bot_param.bot_config_user_editor
                       SET new_choice = %s,
                           new_value = %s,
                           reason = %s
                     WHERE row_id = %s
                     RETURNING row_id, account_login, bot, input_param, current_value
                    """,
                    (new_choice, new_value, reason, source["row_id"]),
                )
                applied.append({"source": dict(cur.fetchone())})

                for target_account in target_accounts:
                    cur.execute(
                        """
                        SELECT e.row_id, e.account_login, e.bot, e.input_param
                          FROM bot_param.bot_config_user_editor e
                          JOIN bot_param.bot_config_param_catalog pc
                            ON pc.bot_kind = e.bot
                           AND COALESCE(pc.input_param_name, pc.param_key) = e.input_param
                           AND COALESCE(pc.user_editable, true) = true
                         WHERE e.account_login = %s
                           AND e.bot = %s
                           AND e.input_param = %s
                         FOR UPDATE OF e
                        """,
                        (target_account, source["bot"], source["input_param"]),
                    )
                    target = cur.fetchone()
                    if not target:
                        raise ValueError(
                            f"row {change.row_id}: target row not found for account {target_account}"
                        )
                    cur.execute(
                        """
                        UPDATE bot_param.bot_config_user_editor
                           SET new_choice = %s,
                               new_value = %s,
                               reason = %s
                         WHERE row_id = %s
                         RETURNING row_id, account_login, bot, input_param, current_value
                        """,
                        (new_choice, new_value, reason, target["row_id"]),
                    )
                    if "targets" not in applied[-1]:
                        applied[-1]["targets"] = []
                    applied[-1]["targets"].append(dict(cur.fetchone()))

        conn.commit()
        return {
            "ok": True,
            "actor": actor,
            "applied": applied,
            "applied_count": sum(1 + len(row.get("targets", [])) for row in applied),
            "version": CODE_VERSION,
        }
    except ValueError as exc:
        conn.rollback()
        return config_ui_json_error(400, str(exc))
    except Exception as exc:
        conn.rollback()
        return config_ui_json_error(400, first_error_line(exc))
    finally:
        conn.close()


@app.post("/bot-runtime/recommendation/next-notification")
async def bot_runtime_recommendation_next_notification(req: BotRuntimeBaseRequest, request: Request):
    auth = require_bot_runtime_api(request)
    if auth:
        return auth
    try:
        env, account_login, bot_kind, bot_id, source_id, instance_id = normalize_runtime_identity(req)
    except ValueError as exc:
        return bot_runtime_json_error(400, str(exc))
    if bot_kind != RM_CONTROLLER_BOT or bot_id != RM_CONTROLLER_BOT_ID:
        return bot_runtime_json_error(400, "recommendation notifications are only for rm_controller")

    conn = config_ui_conn()
    conn.autocommit = False
    try:
        recommendation = None
        with conn.cursor(cursor_factory=DictCursor) as cur:
            if not table_exists(cur, "bot_online", "bot_control_recommendation"):
                conn.commit()
                return {"ok": True, "recommendation": None, "version": CODE_VERSION}
            columns = table_columns(cur, "bot_online", "bot_control_recommendation")
            required = {
                "recommendation_id",
                "env",
                "account_login",
                "bot_kind",
                "decision_type",
                "status",
                "created_at",
                "expires_at",
                "bot_id",
                "symbol",
                "tf",
                "input_param",
                "param_key",
                "old_value",
                "recommended_value",
                "reason",
                "confidence",
                "trend_strength",
                "min_sample_reached",
                "cooldown_until",
                "telegram_notified_at",
                "telegram_notified_by",
            }
            if not required.issubset(columns):
                conn.commit()
                return {"ok": True, "recommendation": None, "version": CODE_VERSION}

            cur.execute(
                """
                SELECT recommendation_id
                  FROM bot_online.bot_control_recommendation
                 WHERE env = %s
                   AND account_login = %s
                   AND status = 'new'
                   AND decision_type = 'param_change'
                   AND telegram_notified_at IS NULL
                   AND (expires_at IS NULL OR expires_at > now())
                 ORDER BY created_at ASC, recommendation_id ASC
                 LIMIT 1
                 FOR UPDATE SKIP LOCKED
                """,
                (env, account_login),
            )
            row = cur.fetchone()
            if row:
                recommendation_id = int(row["recommendation_id"])
                cur.execute(
                    """
                    UPDATE bot_online.bot_control_recommendation
                       SET telegram_notified_at = now(),
                           telegram_notified_by = %s
                     WHERE recommendation_id = %s
                    """,
                    (instance_id, recommendation_id),
                )
                cur.execute(
                    """
                    SELECT recommendation_id,
                           account_login,
                           bot_kind,
                           COALESCE(bot_id, bot_kind) AS bot_id,
                           symbol,
                           tf,
                           COALESCE(input_param, param_key) AS input_param,
                           old_value,
                           recommended_value,
                           reason,
                           confidence::text AS confidence,
                           trend_strength::text AS trend_strength,
                           min_sample_reached,
                           cooldown_until::text AS cooldown_until
                      FROM bot_online.bot_control_recommendation
                     WHERE recommendation_id = %s
                    """,
                    (recommendation_id,),
                )
                recommendation = dict(cur.fetchone())
        conn.commit()
        return {"ok": True, "recommendation": recommendation, "version": CODE_VERSION}
    except Exception as exc:
        conn.rollback()
        return bot_runtime_json_error(500, first_error_line(exc))
    finally:
        conn.close()


@app.post("/bot-runtime/config/current")
async def bot_runtime_config_current(req: BotRuntimeBaseRequest, request: Request):
    auth = require_bot_runtime_api(request)
    if auth:
        return auth
    try:
        env, account_login, bot_kind, bot_id, source_id, instance_id = normalize_runtime_identity(req)
    except ValueError as exc:
        return bot_runtime_json_error(400, str(exc))

    conn = config_ui_conn()
    conn.autocommit = False
    try:
        with conn.cursor(cursor_factory=DictCursor) as cur:
            cur.execute(
                """
                SELECT c.active_version_no,
                       c.active_config_id,
                       c.config_hash
                  FROM bot_param.bot_config_current c
                 WHERE c.env = %s
                   AND c.account_login = %s
                   AND c.bot_kind = %s
                   AND c.bot_id = %s
                """,
                (env, account_login, bot_kind, bot_id),
            )
            current = cur.fetchone()
            if not current:
                conn.rollback()
                return bot_runtime_json_error(404, "active config not found")

            cur.execute(
                """
                SELECT pc.input_param_name AS input_param,
                       pc.param_key,
                       pc.param_path,
                       pc.value_type,
                       v.config_json #>> string_to_array(pc.param_path, '.') AS value
                  FROM bot_param.bot_config_version v
                  JOIN bot_param.bot_config_param_catalog pc
                    ON pc.bot_kind = v.bot_kind
                 WHERE v.config_version_id = %s
                   AND pc.input_param_name IS NOT NULL
                   AND pc.param_path IS NOT NULL
                   AND COALESCE(pc.user_editable, true) = true
                 ORDER BY pc.sort_order, pc.input_param_name
                """,
                (current["active_config_id"],),
            )
            params = [dict(row) for row in cur.fetchall()]
            old_version_no = req.applied_version_no
            changed_params = bot_runtime_load_changed_params(
                cur,
                env,
                account_login,
                bot_kind,
                bot_id,
                old_version_no,
                current["active_version_no"],
                current["active_config_id"],
            )
            bot_runtime_touch(cur, env, account_login, instance_id, "last_config_check_at")
        conn.commit()
        return {
            "ok": True,
            "env": env,
            "account_login": account_login,
            "bot_kind": bot_kind,
            "bot_id": bot_id,
            "source_id": source_id,
            "instance_id": instance_id,
            "old_version_no": old_version_no,
            "version_no": current["active_version_no"],
            "config_hash": current["config_hash"],
            "params": params,
            "changed_params": changed_params,
            "version": CODE_VERSION,
        }
    except Exception as exc:
        conn.rollback()
        return bot_runtime_json_error(500, first_error_line(exc))
    finally:
        conn.close()


@app.post("/bot-runtime/config/resolve-on-init")
async def bot_runtime_config_resolve_on_init(req: BotRuntimeResolveOnInitRequest, request: Request):
    auth = require_bot_runtime_api(request)
    if auth:
        return auth
    try:
        env, account_login, bot_kind, bot_id, source_id, instance_id = normalize_runtime_identity(req)
    except ValueError as exc:
        return bot_runtime_json_error(400, str(exc))

    program_defaults = runtime_param_map(req.program_defaults)
    input_values = runtime_param_map(req.input_values)
    if not program_defaults and not input_values:
        return bot_runtime_json_error(400, "program_defaults or input_values are required")

    conn = config_ui_conn()
    conn.autocommit = False
    try:
        with conn.cursor(cursor_factory=DictCursor) as cur:
            catalog = load_runtime_param_catalog(cur, bot_kind)
            if not catalog:
                conn.rollback()
                return bot_runtime_json_error(404, f"param catalog not found for bot_kind={bot_kind}")

            current = load_runtime_current_config(cur, env, account_login, bot_kind, bot_id)
            old_version_no = req.applied_version_no
            current_config = dict(current["config_json"] or {}) if current else {}
            effective_config = json.loads(json.dumps(current_config, separators=(",", ":"), ensure_ascii=False)) if current_config else {}
            changed_params = []
            has_db_config = current is not None
            config_changed = not has_db_config

            for item in catalog:
                input_param = item["input_param"]
                param_path = item["param_path"]
                value_type = item.get("value_type")
                db_value = json_path_get(current_config, param_path)
                default_provided = input_param in program_defaults
                input_provided = input_param in input_values
                default_value = program_defaults.get(input_param)
                input_value = input_values.get(input_param)

                is_override = False
                if default_provided and input_provided:
                    is_override = (
                        normalize_runtime_param_value(input_value, value_type)
                        != normalize_runtime_param_value(default_value, value_type)
                    )

                if is_override:
                    effective_value_text = input_value
                    old_text = normalize_runtime_param_value(db_value, value_type) if db_value is not None else ""
                    new_text = normalize_runtime_param_value(effective_value_text, value_type)
                    if old_text != new_text:
                        config_changed = True
                        changed_params.append(
                            {
                                "input_param": input_param,
                                "old_value": old_text,
                                "new_value": new_text,
                                "reason": "mt5_input_override_on_init",
                            }
                        )
                    json_path_set(
                        effective_config,
                        param_path,
                        coerce_config_json_value(effective_value_text, value_type),
                    )
                elif db_value is not None:
                    json_path_set(effective_config, param_path, db_value)
                elif default_provided:
                    if has_db_config:
                        config_changed = True
                    json_path_set(
                        effective_config,
                        param_path,
                        coerce_config_json_value(default_value, value_type),
                    )
                elif input_provided and not has_db_config:
                    json_path_set(
                        effective_config,
                        param_path,
                        coerce_config_json_value(input_value, value_type),
                    )

            effective_hash = config_json_hash(effective_config)
            wrote_version = False
            active_version_no = int(current["active_version_no"]) if current else 0
            active_config_id = int(current["active_config_id"]) if current else 0
            current_hash = current["config_hash"] if current else ""

            if config_changed and current_hash != effective_hash:
                active_version_no, active_config_id = insert_runtime_config_version(
                    cur,
                    env,
                    account_login,
                    bot_kind,
                    bot_id,
                    effective_config,
                    effective_hash,
                    "mt5_input_override_on_init" if changed_params else "mt5_input_defaults_bootstrap",
                    "bot-runtime-resolve-on-init",
                )
                wrote_version = True
                refresh_bot_config_editors(cur, account_login, bot_kind)
            response_hash = effective_hash if wrote_version else (current_hash or effective_hash)

            bot_runtime_touch(cur, env, account_login, instance_id, "last_config_check_at")

        conn.commit()
        return {
            "ok": True,
            "env": env,
            "account_login": account_login,
            "bot_kind": bot_kind,
            "bot_id": bot_id,
            "source_id": source_id,
            "instance_id": instance_id,
            "old_version_no": old_version_no,
            "version_no": active_version_no,
            "active_config_id": active_config_id,
            "config_hash": response_hash,
            "wrote_version": wrote_version,
            "params": runtime_params_from_config(catalog, effective_config),
            "changed_params": changed_params,
            "version": CODE_VERSION,
        }
    except (InvalidOperation, ValueError) as exc:
        conn.rollback()
        return bot_runtime_json_error(400, str(exc))
    except Exception as exc:
        conn.rollback()
        return bot_runtime_json_error(500, first_error_line(exc))
    finally:
        conn.close()


@app.post("/bot-runtime/rm-state/update")
async def bot_runtime_rm_state_update(req: BotRuntimeRmStateUpdateRequest, request: Request):
    auth = require_bot_runtime_api(request)
    if auth:
        return auth
    try:
        env, account_login, bot_kind, bot_id, source_id, instance_id = normalize_runtime_identity(req)
        if bot_kind != RM_CONTROLLER_BOT or bot_id != RM_CONTROLLER_BOT_ID:
            raise ValueError("only rm_controller can update RM state")
        scope = normalize_rm_state_scope(req.scope)
        target_bot_kind = normalize_rm_state_bot_kind(scope, req.target_bot_kind)
        action = normalize_rm_state_action(req.action)
        reset_mode = normalize_rm_state_reset_mode(req.reset_mode)
        reset_at = rm_state_reset_at(req.reset_at_epoch)
    except ValueError as exc:
        return bot_runtime_json_error(400, str(exc))

    conn = config_ui_conn()
    conn.autocommit = False
    try:
        with conn.cursor(cursor_factory=DictCursor) as cur:
            row = upsert_rm_state(
                cur,
                env,
                account_login,
                scope,
                target_bot_kind,
                req.active,
                action,
                reset_mode,
                reset_at,
                req.source_command_id,
                (req.reason or "").strip() or None,
                instance_id,
                "rm_controller",
            )
            bot_runtime_touch(cur, env, account_login, instance_id, "last_command_check_at")
        conn.commit()
        return {"ok": True, "rm_state": row, "version": CODE_VERSION}
    except Exception as exc:
        conn.rollback()
        return bot_runtime_json_error(500, first_error_line(exc))
    finally:
        conn.close()


@app.post("/bot-runtime/rm-state")
async def bot_runtime_rm_state(req: BotRuntimeBaseRequest, request: Request):
    auth = require_bot_runtime_api(request)
    if auth:
        return auth
    try:
        env, account_login, bot_kind, bot_id, source_id, instance_id = normalize_runtime_identity(req)
    except ValueError as exc:
        return bot_runtime_json_error(400, str(exc))

    conn = config_ui_conn()
    conn.autocommit = False
    try:
        with conn.cursor(cursor_factory=DictCursor) as cur:
            account_state, bot_state, effective = load_effective_rm_state(cur, env, account_login, bot_kind)
            bot_runtime_touch(cur, env, account_login, instance_id, "last_command_check_at")
        conn.commit()
        return {
            "ok": True,
            "env": env,
            "account_login": account_login,
            "bot_kind": bot_kind,
            "bot_id": bot_id,
            "source_id": source_id,
            "instance_id": instance_id,
            "account_state": account_state,
            "bot_state": bot_state,
            "effective_state": effective,
            "version": CODE_VERSION,
        }
    except Exception as exc:
        conn.rollback()
        return bot_runtime_json_error(500, first_error_line(exc))
    finally:
        conn.close()


@app.post("/bot-runtime/rm-state/ack")
async def bot_runtime_rm_state_ack(req: BotRuntimeRmStateAckRequest, request: Request):
    auth = require_bot_runtime_api(request)
    if auth:
        return auth
    try:
        env, account_login, bot_kind, bot_id, source_id, instance_id = normalize_runtime_identity(req)
        scope = normalize_rm_state_scope(req.scope)
        target_bot_kind = normalize_rm_state_bot_kind(scope, req.target_bot_kind)
        decision = (req.decision or "").strip().lower()
        if decision not in RM_STATE_DECISIONS:
            raise ValueError("decision must be continue, halt_only, or flatten_and_halt")
    except ValueError as exc:
        return bot_runtime_json_error(400, str(exc))

    conn = config_ui_conn()
    conn.autocommit = False
    try:
        with conn.cursor(cursor_factory=DictCursor) as cur:
            ensure_rm_state_schema(cur)
            cur.execute(
                """
                INSERT INTO bot_param.rm_state_ack (
                    env, account_login, bot_kind, bot_id, instance_id,
                    scope, target_bot_kind, observed_state_version,
                    decision, last_error, applied_at
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, now())
                ON CONFLICT (env, account_login, bot_kind, bot_id, instance_id, scope, target_bot_kind)
                DO UPDATE SET
                    observed_state_version = EXCLUDED.observed_state_version,
                    decision = EXCLUDED.decision,
                    last_error = EXCLUDED.last_error,
                    applied_at = now()
                RETURNING env,
                          account_login,
                          bot_kind,
                          bot_id,
                          instance_id,
                          scope,
                          NULLIF(target_bot_kind, '') AS target_bot_kind,
                          observed_state_version,
                          decision,
                          last_error,
                          applied_at
                """,
                (
                    env,
                    account_login,
                    bot_kind,
                    bot_id,
                    instance_id,
                    scope,
                    target_bot_kind,
                    int(req.observed_state_version),
                    decision,
                    req.last_error,
                ),
            )
            row = dict(cur.fetchone())
            bot_runtime_touch(cur, env, account_login, instance_id, "last_command_check_at")
        conn.commit()
        return {"ok": True, "rm_state_ack": row, "version": CODE_VERSION}
    except Exception as exc:
        conn.rollback()
        return bot_runtime_json_error(500, first_error_line(exc))
    finally:
        conn.close()


@app.post("/bot-runtime/command/next")
async def bot_runtime_command_next(req: BotRuntimeBaseRequest, request: Request):
    auth = require_bot_runtime_api(request)
    if auth:
        return auth
    try:
        env, account_login, bot_kind, bot_id, source_id, instance_id = normalize_runtime_identity(req)
    except ValueError as exc:
        return bot_runtime_json_error(400, str(exc))

    conn = config_ui_conn()
    conn.autocommit = False
    try:
        with conn.cursor(cursor_factory=DictCursor) as cur:
            cur.execute(
                """
                WITH next_command AS (
                    SELECT command_id
                      FROM bot_param.bot_command
                     WHERE env = %s
                       AND account_login = %s
                       AND target_bot_kind = %s
                       AND target_bot_id = %s
                       AND (target_instance_id IS NULL OR target_instance_id = %s)
                       AND status = 'queued'
                     ORDER BY priority DESC, created_at ASC, command_id ASC
                     LIMIT 1
                     FOR UPDATE SKIP LOCKED
                )
                UPDATE bot_param.bot_command c
                   SET status = 'leased',
                       lease_owner = %s,
                       leased_at = now(),
                       ack_by_instance_id = %s,
                       ack_at = now()
                  FROM next_command n
                 WHERE c.command_id = n.command_id
                 RETURNING c.command_id,
                           c.command_type,
                           c.command_payload,
                           c.target_version_no,
                           c.status,
                           c.priority,
                           c.created_at
                """,
                (env, account_login, bot_kind, bot_id, instance_id, instance_id, instance_id),
            )
            row = cur.fetchone()
            bot_runtime_touch(cur, env, account_login, instance_id, "last_command_check_at")
        conn.commit()
        return {
            "ok": True,
            "env": env,
            "account_login": account_login,
            "bot_kind": bot_kind,
            "bot_id": bot_id,
            "source_id": source_id,
            "instance_id": instance_id,
            "command": dict(row) if row else None,
            "version": CODE_VERSION,
        }
    except Exception as exc:
        conn.rollback()
        return bot_runtime_json_error(500, first_error_line(exc))
    finally:
        conn.close()


@app.post("/bot-runtime/command/finish")
async def bot_runtime_command_finish(req: BotRuntimeFinishRequest, request: Request):
    auth = require_bot_runtime_api(request)
    if auth:
        return auth
    instance_id = (req.instance_id or "").strip()
    status = (req.status or "").strip().lower()
    if not instance_id:
        return bot_runtime_json_error(400, "instance_id is required")
    if status not in ("done", "error"):
        return bot_runtime_json_error(400, "status must be done or error")

    result_json = json.dumps(req.result_json or {}, separators=(",", ":"))
    conn = config_ui_conn()
    conn.autocommit = False
    try:
        with conn.cursor(cursor_factory=DictCursor) as cur:
            cur.execute(
                """
                UPDATE bot_param.bot_command
                   SET status = %s,
                       finished_at = now(),
                       result_json = %s::jsonb,
                       error_text = %s
                 WHERE command_id = %s
                   AND (lease_owner = %s OR ack_by_instance_id = %s)
                   AND status IN ('queued', 'leased')
                 RETURNING command_id,
                           env,
                           account_login,
                           target_bot_kind,
                           target_bot_id,
                           command_type,
                           target_version_no,
                           status
                """,
                (status, result_json, req.error_text, req.command_id, instance_id, instance_id),
            )
            row = cur.fetchone()
            if not row:
                conn.rollback()
                return bot_runtime_json_error(404, "command not found or not leased by instance")
            if status == "done" and row["command_type"] == "SOFT_REINIT":
                bot_runtime_touch(cur, row["env"], row["account_login"], instance_id, "last_reinit_at")
        conn.commit()
        return {"ok": True, "command": dict(row), "version": CODE_VERSION}
    except Exception as exc:
        conn.rollback()
        return bot_runtime_json_error(500, first_error_line(exc))
    finally:
        conn.close()


@app.post("/bot-runtime/status")
async def bot_runtime_status(req: BotRuntimeStatusRequest, request: Request):
    auth = require_bot_runtime_api(request)
    if auth:
        return auth
    try:
        env, account_login, bot_kind, bot_id, source_id, instance_id = normalize_runtime_identity(req)
    except ValueError as exc:
        return bot_runtime_json_error(400, str(exc))

    runtime_json = json.dumps(req.runtime_json or {}, separators=(",", ":"))
    conn = config_ui_conn()
    conn.autocommit = False
    try:
        with conn.cursor(cursor_factory=DictCursor) as cur:
            cur.execute(
                """
                INSERT INTO bot_param.bot_runtime_status (
                    env, account_login, instance_id,
                    bot_kind, bot_id, source_id,
                    status, allow_new_entries,
                    applied_version_no, applied_config_hash,
                    last_seen_at, last_error, runtime_json
                )
                VALUES (
                    %s, %s, %s,
                    %s, %s, %s,
                    %s, %s,
                    %s, %s,
                    now(), %s, %s::jsonb
                )
                ON CONFLICT (env, account_login, instance_id)
                DO UPDATE SET
                    bot_kind = EXCLUDED.bot_kind,
                    bot_id = EXCLUDED.bot_id,
                    source_id = EXCLUDED.source_id,
                    status = EXCLUDED.status,
                    allow_new_entries = EXCLUDED.allow_new_entries,
                    applied_version_no = EXCLUDED.applied_version_no,
                    applied_config_hash = EXCLUDED.applied_config_hash,
                    last_seen_at = now(),
                    last_error = EXCLUDED.last_error,
                    runtime_json = EXCLUDED.runtime_json
                RETURNING env,
                          account_login,
                          instance_id,
                          bot_kind,
                          bot_id,
                          source_id,
                          status,
                          allow_new_entries,
                          applied_version_no,
                          applied_config_hash,
                          last_seen_at
                """,
                (
                    env,
                    account_login,
                    instance_id,
                    bot_kind,
                    bot_id,
                    source_id,
                    req.status,
                    req.allow_new_entries,
                    req.applied_version_no,
                    req.applied_config_hash,
                    req.last_error,
                    runtime_json,
                ),
            )
            row = cur.fetchone()
        conn.commit()
        return {"ok": True, "runtime_status": dict(row), "version": CODE_VERSION}
    except Exception as exc:
        conn.rollback()
        return bot_runtime_json_error(500, first_error_line(exc))
    finally:
        conn.close()

@app.post("/db/read")
async def db_read(req: DbReadRequest, request: Request):
    auth = require_api_key(request)
    if auth:
        return auth
    if psycopg2 is None:
        return JSONResponse(status_code=500, content={"error": f"psycopg2 not available: {_PSYCOPG2_IMPORT_ERROR}", "version": CODE_VERSION})
    schema = (req.schema or "").lower()
    table = (req.table or "").lower()
    if not is_safe_ident(schema) or not is_safe_ident(table):
        return JSONResponse(status_code=400, content={"error": "Invalid schema/table", "version": CODE_VERSION})
    where = req.where or {}
    if where and not isinstance(where, dict):
        return JSONResponse(status_code=400, content={"error": "where must be object", "version": CODE_VERSION})

    sql = f"SELECT * FROM {schema}.{table}"
    params = []
    if where:
        clauses = []
        for k, v in where.items():
            if not is_safe_ident(k):
                return JSONResponse(status_code=400, content={"error": "Invalid column in where", "version": CODE_VERSION})
            if v is None:
                clauses.append(f"{k} IS NULL")
            else:
                clauses.append(f"{k} = %s")
                params.append(v)
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
    if req.order_by:
        order_by = (req.order_by or "").lower()
        if not is_safe_ident(order_by):
            return JSONResponse(status_code=400, content={"error": "Invalid order_by", "version": CODE_VERSION})
        order_dir = (req.order_dir or "desc").lower()
        if order_dir not in ("asc", "desc"):
            return JSONResponse(status_code=400, content={"error": "Invalid order_dir", "version": CODE_VERSION})
        sql += f" ORDER BY {order_by} {order_dir}"
    if req.limit:
        sql += " LIMIT %s"
        params.append(req.limit)

    db_url, db_label = resolve_db_url(db_mode=req.db_mode)
    print(f"[DB] read mode={req.db_mode or 'prod'} resolved={db_label}")
    conn = psycopg2.connect(db_url, sslmode="require")
    try:
        with conn.cursor(cursor_factory=DictCursor) as cur:
            cur.execute(sql, params)
            rows = [dict(r) for r in cur.fetchall()]
        return {"ok": True, "rows": rows, "count": len(rows), "version": CODE_VERSION}
    finally:
        conn.close()

@app.post("/db/delete")
async def db_delete(req: DbDeleteRequest, request: Request):
    auth = require_api_key(request)
    if auth:
        return auth
    if psycopg2 is None:
        return JSONResponse(status_code=500, content={"error": f"psycopg2 not available: {_PSYCOPG2_IMPORT_ERROR}", "version": CODE_VERSION})
    schema = (req.schema or "").lower()
    table = (req.table or "").lower()
    if not is_safe_ident(schema) or not is_safe_ident(table):
        return JSONResponse(status_code=400, content={"error": "Invalid schema/table", "version": CODE_VERSION})
    where = req.where or {}
    if not isinstance(where, dict) or not where:
        return JSONResponse(status_code=400, content={"error": "where must be non-empty object", "version": CODE_VERSION})

    clauses = []
    params = []
    for k, v in where.items():
        if not is_safe_ident(k):
            return JSONResponse(status_code=400, content={"error": "Invalid column in where", "version": CODE_VERSION})
        if v is None:
            clauses.append(f"{k} IS NULL")
        else:
            clauses.append(f"{k} = %s")
            params.append(v)
    sql = f"DELETE FROM {schema}.{table} WHERE " + " AND ".join(clauses)
    try:
        db_url, db_label = resolve_db_url(db_mode=req.db_mode)
        print(f"[DB] delete mode={req.db_mode or 'prod'} resolved={db_label}")
        conn = psycopg2.connect(db_url, sslmode="require")
        with conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                deleted = cur.rowcount
        return {"ok": True, "deleted": deleted, "version": CODE_VERSION}
    except Exception as exc:
        return JSONResponse(status_code=500, content={"error": str(exc), "version": CODE_VERSION})


@app.post("/db/write")
async def db_write(req: DbWriteRequest, request: Request):
    auth = require_api_key(request)
    if auth:
        return auth
    if psycopg2 is None:
        return JSONResponse(status_code=500, content={"error": f"psycopg2 not available: {_PSYCOPG2_IMPORT_ERROR}", "version": CODE_VERSION})
    schema = (req.schema or "").lower()
    table = (req.table or "").lower()
    if not is_safe_ident(schema) or not is_safe_ident(table):
        return JSONResponse(status_code=400, content={"error": "Invalid schema/table", "version": CODE_VERSION})
    rows = req.rows or []
    if not rows:
        return JSONResponse(status_code=400, content={"error": "rows is empty", "version": CODE_VERSION})
    if not isinstance(rows, list) or not isinstance(rows[0], dict):
        return JSONResponse(status_code=400, content={"error": "rows must be list of objects", "version": CODE_VERSION})

    columns = list(rows[0].keys())
    for col in columns:
        if not is_safe_ident(col):
            return JSONResponse(status_code=400, content={"error": "Invalid column name", "version": CODE_VERSION})
    for r in rows:
        if set(r.keys()) != set(columns):
            return JSONResponse(status_code=400, content={"error": "All rows must have same columns", "version": CODE_VERSION})

    db_url, db_label = resolve_db_url(db_mode=req.db_mode)
    print(f"[DB] write mode={req.db_mode or 'prod'} resolved={db_label}")
    conn = psycopg2.connect(db_url, sslmode="require")
    conn.autocommit = False
    try:
        ensure_table_for_rows(conn, schema, table, columns, rows[0])
        sql = f"INSERT INTO {schema}.{table} ({', '.join(columns)}) VALUES %s"
        values = [[r.get(c) for c in columns] for r in rows]
        with conn.cursor() as cur:
            execute_values(cur, sql, values, page_size=min(len(values), 1000))
        conn.commit()
        return {"ok": True, "inserted": len(values), "version": CODE_VERSION}
    except Exception as exc:
        conn.rollback()
        return JSONResponse(status_code=500, content={"error": str(exc), "version": CODE_VERSION})
    finally:
        conn.close()


@app.post("/neuro/refresh")
async def neuro_refresh(req: NeuroRefreshRequest):
    source_id = req.source_id or req.pid or NEURO_SOURCE_DEFAULT
    try:
        result = await asyncio.to_thread(run_neuro_refresh, req.pid, source_id, req.table, req.limit, req.db_mode)
    except Exception as exc:
        return JSONResponse(status_code=500, content={"error": str(exc), "version": CODE_VERSION})
    return result

@app.get("/", response_class=PlainTextResponse)
async def root():
    return f"OK: {CODE_VERSION}\n"

