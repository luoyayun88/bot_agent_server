from email.policy import default
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse, RedirectResponse
from openai import OpenAI, BadRequestError
import os, json, traceback, re, threading, asyncio, time, tempfile, base64, hashlib, hmac, secrets, html as html_lib
from urllib.parse import parse_qs
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime, timezone, timedelta
from pathlib import Path

try:
    import psycopg2
    from psycopg2.extras import DictCursor, execute_values
except Exception as _pg_exc:  # Allow app to start without psycopg2
    psycopg2 = None
    DictCursor = None
    _PSYCOPG2_IMPORT_ERROR = _pg_exc
from pydantic import BaseModel

app = FastAPI()

CODE_VERSION = "v1.03"
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
    reason: Optional[str] = None


class ConfigUiSaveRequest(BaseModel):
    account_login: int
    bot: str
    copy_to_account_login: Optional[int] = None
    copy_to_account_logins: Optional[List[int]] = None
    changes: List[ConfigUiChange]


class BotRuntimeBaseRequest(BaseModel):
    env: str = "prod"
    account_login: int
    bot_kind: str
    bot_id: str
    source_id: Optional[str] = None
    instance_id: str
    applied_version_no: Optional[int] = None
    command_id: Optional[int] = None


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


def config_ui_conn():
    db_url, _ = resolve_db_url()
    return psycopg2.connect(db_url, sslmode="require")


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
    select, input[type="text"], input[type="search"] { border: 1px solid #b8c2cc; border-radius: 6px; background: #fff; color: var(--text); min-height: 38px; padding: 8px 10px; width: 100%; }
    .copy-line { display: flex; gap: 8px; align-items: center; min-height: 38px; border: 1px solid var(--border); border-radius: 6px; background: #fff; padding: 8px 10px; }
    .copy-line input { width: auto; }
    .target-list { min-height: 86px; max-height: 112px; overflow-y: auto; border: 1px solid #b8c2cc; border-radius: 6px; background: #fff; padding: 4px; }
    .target-list.is-disabled { opacity: .65; background: #f8fafc; }
    .target-option { display: flex; align-items: center; gap: 8px; min-height: 26px; padding: 3px 5px; border-radius: 4px; color: var(--text); font-size: 13px; font-weight: 500; }
    .target-option:hover { background: #eef4fb; }
    .target-option.is-unavailable { color: var(--muted); }
    .target-option input { width: auto; flex: 0 0 auto; }
    .target-option span { min-width: 0; overflow-wrap: anywhere; }
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
    th:nth-child(1), td.group { width: 10%; }
    th:nth-child(2), td.param { width: 18%; }
    th:nth-child(3), td.desc { width: 24%; }
    th:nth-child(4), td.current { width: 18%; }
    th:nth-child(5), td.new-value { width: 17%; }
    th:nth-child(6), td.reason { width: 13%; }
    .group { color: #49566a; font-weight: 600; white-space: normal; overflow-wrap: anywhere; }
    .param { font-family: Consolas, Menlo, monospace; font-size: 12px; white-space: normal; overflow-wrap: anywhere; }
    .desc { overflow-wrap: anywhere; }
    .current { font-family: Consolas, Menlo, monospace; color: #147a3d; white-space: pre-wrap; overflow-wrap: anywhere; }
    .new-value, .reason { min-width: 0; }
    .new-value input, .new-value select, .reason input { width: 100%; min-width: 0; }
    .footer { position: sticky; bottom: 0; z-index: 15; margin-top: 12px; padding: 10px; display: flex; justify-content: space-between; align-items: center; gap: 12px; background: rgba(245, 247, 250, .96); border: 1px solid var(--border); border-radius: 8px; }
    .changed-count { color: var(--muted); font-size: 14px; }
    @media (max-width: 1100px) {
      th:first-child, td.group { display: none; }
      tr.group-break td:nth-child(2) { box-shadow: inset 4px 0 0 #3d7cae; }
      th:nth-child(2), td.param { width: 22%; }
      th:nth-child(3), td.desc { width: 28%; }
      th:nth-child(4), td.current { width: 20%; }
      th:nth-child(5), td.new-value { width: 18%; }
      th:nth-child(6), td.reason { width: 12%; }
    }
    @media (max-width: 760px) {
      .header-inner { align-items: flex-start; display: grid; grid-template-columns: minmax(0, 1fr) auto; }
      .logout-form { grid-column: 2; grid-row: 1; }
      .logout-form button { margin-top: 0; min-height: 36px; padding: 8px 10px; }
      .meta { grid-column: 1 / -1; }
      .toolbar { grid-template-columns: 1fr; }
      main { padding: 10px; }
      .table-wrap { border: 0; border-radius: 0; overflow-x: visible; background: transparent; }
      table, thead, tbody, tr, td { display: block; width: 100%; }
      table { min-width: 0; border-collapse: separate; }
      thead { display: none; }
      tr { display: grid; grid-template-columns: minmax(0, 1fr); gap: 12px; padding: 14px 12px; border: 1px solid var(--border); border-radius: 8px; margin-bottom: 10px; background: var(--panel); }
      tr.hidden { display: none; }
      tr.group-break { margin-top: 18px; border-top: 3px solid #3d7cae; }
      tr.group-break td { border-top: 0; }
      tr.group-break td:nth-child(2) { box-shadow: none; }
      td { border-bottom: 0; padding: 0; min-width: 0; width: 100% !important; }
      td.group, td.reason { display: none; }
      .param { grid-column: 1 / -1; font-family: Consolas, Menlo, monospace; font-size: 13px; font-weight: 700; white-space: normal; overflow-wrap: anywhere; }
      .param::after { content: attr(data-desc); display: block; margin-top: 4px; font-family: Inter, Segoe UI, Arial, sans-serif; font-weight: 500; color: var(--muted); }
      .desc { display: none; }
      .current, .new-value { grid-column: 1 / -1; min-width: 0; max-width: none; }
      .current::before, .new-value::before { display: block; margin-bottom: 5px; font-family: Inter, Segoe UI, Arial, sans-serif; font-size: 12px; font-weight: 700; color: var(--muted); white-space: nowrap; }
      .current::before { content: "current value"; color: #147a3d; }
      .new-value::before { content: "new"; }
      .current { font-family: Consolas, Menlo, monospace; white-space: pre-wrap; overflow-wrap: anywhere; overflow: visible; }
      .footer { align-items: stretch; flex-direction: column; }
      .footer button { width: 100%; min-height: 44px; }
      select, input[type="text"], input[type="search"] { min-height: 44px; font-size: 16px; }
      .target-list { max-height: 144px; }
      .target-option { min-height: 34px; font-size: 14px; }
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
    <section class="toolbar" aria-label="filters">
      <div class="field">
        <label for="accountSelect">Account</label>
        <select id="accountSelect"></select>
      </div>
      <div class="field">
        <label for="botSelect">Bot</label>
        <select id="botSelect"></select>
      </div>
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

    <div id="status" class="status"></div>

    <section class="table-wrap">
      <table>
        <thead>
          <tr>
            <th>group</th>
            <th>input_param</th>
            <th>param_desc</th>
            <th>current_value</th>
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
  </main>
  <script>
    const SESSION_USER = __SESSION_USER__;
    const CONFIG_ACTOR = __CONFIG_ACTOR__;
    const CODE_VERSION = __CODE_VERSION__;
    const choiceCache = new Map();

    const els = {
      sessionUser: document.getElementById('sessionUser'),
      actorName: document.getElementById('actorName'),
      codeVersion: document.getElementById('codeVersion'),
      accountSelect: document.getElementById('accountSelect'),
      botSelect: document.getElementById('botSelect'),
      groupFilter: document.getElementById('groupFilter'),
      searchInput: document.getElementById('searchInput'),
      copyToggle: document.getElementById('copyToggle'),
      targetAccountList: document.getElementById('targetAccountList'),
      status: document.getElementById('status'),
      paramsBody: document.getElementById('paramsBody'),
      changedCount: document.getElementById('changedCount'),
      saveBtn: document.getElementById('saveBtn')
    };

    els.sessionUser.textContent = SESSION_USER;
    els.actorName.textContent = CONFIG_ACTOR;
    els.codeVersion.textContent = CODE_VERSION;

    function setStatus(text, isError = false) {
      els.status.textContent = text || '';
      els.status.className = isError ? 'status error' : 'status';
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

    async function loadAccounts() {
      setStatus('Loading accounts...');
      const data = await api('/config-ui/api/accounts');
      fillSelect(els.accountSelect, data.accounts || [], 'account_login', 'account_label', '');
      if (!els.accountSelect.value) {
        setStatus('No accounts available for actor ' + CONFIG_ACTOR, true);
        return;
      }
      await loadBots();
    }

    async function loadBots() {
      const account = els.accountSelect.value;
      fillSelect(els.botSelect, [], 'bot', 'display_name', '');
      if (!account) return;
      setStatus('Loading bots...');
      const data = await api('/config-ui/api/bots?account_login=' + encodeURIComponent(account));
      fillSelect(els.botSelect, data.bots || [], 'bot', 'display_name', '');
      const n9 = Array.from(els.botSelect.options).find(o => o.value === 'n9');
      if (n9) els.botSelect.value = 'n9';
      await loadCopyTargets();
      await loadParams();
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

    function textCell(className, text) {
      const td = document.createElement('td');
      td.className = className || '';
      td.textContent = text == null ? '' : String(text);
      return td;
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

    async function loadParams() {
      const account = els.accountSelect.value;
      const bot = els.botSelect.value;
      els.paramsBody.innerHTML = '';
      if (!account || !bot) return;
      setStatus('Loading parameters...');
      const data = await api('/config-ui/api/params?account_login=' + encodeURIComponent(account) + '&bot=' + encodeURIComponent(bot));
      const rows = data.params || [];
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
      for (const row of rows) {
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

        const newTd = document.createElement('td');
        newTd.className = 'new-value';
        newTd.appendChild(await buildValueControl(row));
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
        els.paramsBody.appendChild(tr);
      }
      applyFilters();
      updateChangedCount();
      setStatus(rows.length + ' parameters loaded');
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
        changes.push({row_id: rowId, value, reason: reason ? reason.value.trim() || null : null});
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

    els.accountSelect.addEventListener('change', loadBots);
    els.botSelect.addEventListener('change', async () => { await loadCopyTargets(); await loadParams(); });
    els.groupFilter.addEventListener('change', applyFilters);
    els.searchInput.addEventListener('input', applyFilters);
    els.copyToggle.addEventListener('change', loadCopyTargets);
    els.saveBtn.addEventListener('click', saveChanges);

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
            cur.execute(
                """
                SELECT DISTINCT
                       e.bot,
                       COALESCE(c.display_name, e.bot) AS display_name,
                       COALESCE(c.sort_order, 9999) AS sort_order
                  FROM bot_param.bot_config_user_editor e
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
                    e.param_group,
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
                    ) AS has_choices
                  FROM bot_param.bot_config_user_editor e
                  JOIN bot_param.operator_account oa
                    ON oa.env = 'prod'
                   AND oa.account_login = e.account_login
                   AND oa.db_user = %s
                   AND oa.enabled = true
                   AND oa.can_edit = true
                 WHERE e.account_login = %s
                   AND e.bot = %s
                 ORDER BY e.param_group, e.input_param
                """,
                (actor, account_login, bot),
            )
            rows = [dict(row) for row in cur.fetchall()]
        return {"ok": True, "params": rows, "version": CODE_VERSION}
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
            cur.execute(
                """
                SELECT oa.account_login,
                       oa.account_label,
                       EXISTS (
                           SELECT 1
                             FROM bot_param.bot_config_user_editor e
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
        return {"ok": True, "accounts": rows, "version": CODE_VERSION}
    except Exception as exc:
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
                    SELECT row_id, account_login, bot, input_param
                      FROM bot_param.bot_config_user_editor
                     WHERE row_id = %s
                       AND account_login = %s
                       AND bot = %s
                     FOR UPDATE
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
                        SELECT row_id, account_login, bot, input_param
                          FROM bot_param.bot_config_user_editor
                         WHERE account_login = %s
                           AND bot = %s
                           AND input_param = %s
                         FOR UPDATE
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

