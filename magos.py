"""
MAGOS — Machine-Adept Gateway for LLM Servitors
====================================================
Broker LLM local pour P620, exposant une queue prioritaire
au-dessus de llama.cpp (API OpenAI-compatible). Conçu pour
FoetoPath Hub.

v0.5 — Backend llama.cpp :
  - Remplacement Ollama par llama.cpp (llama-server :8081)
  - Appels via /v1/chat/completions (SSE streaming)
  - Mapping options Ollama → format OpenAI (rétrocompat)
  - keep_alive ignoré (modèle chargé au démarrage du serveur)

v0.4 — Live preview :
  - Snapshot du thinking + response toutes les 3s pendant le job
  - Le dashboard peut afficher la progression en quasi-temps réel
  - tokens_out laissé à NULL pendant le run, valeur exacte à la fin

v0.3 :
  - Dashboard HTML servi à GET / (theme Adeptus Mechanicus)
  - Endpoint GET /jobs?status=X&limit=N&exclude_active=true pour historique
  - Champ `prompt` exposé dans JobOut quand include_content=true

v0.2 :
  - Cleanup des jobs 'running' orphelins au démarrage
  - keep_alive configurable par job (défaut "0")
  - Migration auto schema BDD
"""

import asyncio
import json
import os
import re
import signal
import sqlite3
import subprocess
import time
import uuid as uuidlib
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from threading import Event, Thread, Lock
from typing import Optional, List, Callable

import requests
import yaml
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

from react import is_react_job, react_loop


# ─── CONFIGURATION ──────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
LLAMA_PORT = int(os.environ.get("MAGOS_LLAMA_PORT", "8081"))
LLM_URL = os.environ.get("MAGOS_LLM_URL", f"http://127.0.0.1:{LLAMA_PORT}")
LLAMA_BIN = os.environ.get("MAGOS_LLAMA_BIN", "/home/mathevet/llama.cpp/build/bin/llama-server")
LLAMA_HOST = os.environ.get("MAGOS_LLAMA_HOST", "0.0.0.0")
LLAMA_READY_TIMEOUT = int(os.environ.get("MAGOS_LLAMA_READY_TIMEOUT", "600"))
MODELS_CONFIG = Path(os.environ.get("MAGOS_MODELS_CONFIG", str(SCRIPT_DIR / "magos_models.yaml")))
DB_PATH = Path(os.environ.get("MAGOS_DB", "/home/mathevet/Bureau/magos/magos.db"))
OUTPUT_DIR = Path(os.environ.get("MAGOS_OUTPUTS", "/home/mathevet/Bureau/magos/outputs"))
DASHBOARD_PATH = Path(os.environ.get("MAGOS_DASHBOARD", str(SCRIPT_DIR / "dashboard.html")))
LISTEN_HOST = os.environ.get("MAGOS_HOST", "0.0.0.0")
LISTEN_PORT = int(os.environ.get("MAGOS_PORT", "8200"))
DEFAULT_JOB_TIMEOUT = int(os.environ.get("MAGOS_DEFAULT_TIMEOUT", "600"))
WORKER_POLL_INTERVAL = 1.0
SNAPSHOT_EVERY_N_SEC = float(os.environ.get("MAGOS_SNAPSHOT_INTERVAL", "3.0"))

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

_cancel_events: dict[str, Event] = {}
_cancel_lock = Lock()


# ─── GESTION GPU (llama-server + BioLORD) ──────────────────────────────────
MODELS_REGISTRY: dict[str, dict] = {}  # alias -> {path, flags}
_llama_proc: Optional[subprocess.Popen] = None
_llama_current_model: Optional[str] = None
_gpu_lock = Lock()
_biolord_model = None


def load_models_registry() -> dict[str, dict]:
    if not MODELS_CONFIG.exists():
        raise RuntimeError(f"Config modèles introuvable : {MODELS_CONFIG}")
    with open(MODELS_CONFIG, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    for alias, cfg in data.items():
        if "path" not in cfg:
            raise RuntimeError(f"Modèle '{alias}' : clé 'path' manquante")
        cfg.setdefault("flags", [])
    return data


def kill_existing_llama_servers():
    """Tue tous les llama-server existants pour reprendre la main au démarrage."""
    try:
        out = subprocess.check_output(["pgrep", "-f", "llama-server"], text=True)
        pids = [int(p) for p in out.strip().split() if p.strip().isdigit()]
    except subprocess.CalledProcessError:
        return
    for pid in pids:
        if pid == os.getpid():
            continue
        try:
            os.kill(pid, signal.SIGTERM)
            print(f"[MAGOS] llama-server PID {pid} tué (SIGTERM)", flush=True)
        except ProcessLookupError:
            pass
    time.sleep(2)
    for pid in pids:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def stop_llama():
    """Stoppe le llama-server géré par magos (s'il y en a un)."""
    global _llama_proc, _llama_current_model
    if _llama_proc is None:
        return
    print(f"[MAGOS] Arrêt llama-server (modèle '{_llama_current_model}', PID {_llama_proc.pid})", flush=True)
    try:
        _llama_proc.terminate()
        try:
            _llama_proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            _llama_proc.kill()
            _llama_proc.wait(timeout=5)
    except Exception as e:
        print(f"[MAGOS] Erreur arrêt llama-server : {e}", flush=True)
    _llama_proc = None
    _llama_current_model = None


def wait_llama_ready(timeout: int) -> bool:
    """Poll /v1/models jusqu'à réponse 200 ou timeout."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = requests.get(f"{LLM_URL}/v1/models", timeout=2)
            if r.status_code == 200:
                return True
        except requests.RequestException:
            pass
        time.sleep(1)
    return False


def start_llama(model_alias: str):
    """Lance llama-server avec le modèle demandé. Bloquant jusqu'à readiness."""
    global _llama_proc, _llama_current_model
    if model_alias not in MODELS_REGISTRY:
        raise RuntimeError(f"Modèle inconnu : '{model_alias}'. Modèles disponibles : {list(MODELS_REGISTRY)}")
    cfg = MODELS_REGISTRY[model_alias]
    cmd = [LLAMA_BIN,
           "-m", cfg["path"],
           "--host", LLAMA_HOST,
           "--port", str(LLAMA_PORT),
           *cfg["flags"]]
    print(f"[MAGOS] Démarrage llama-server pour '{model_alias}' : {' '.join(cmd)}", flush=True)
    log_path = SCRIPT_DIR / "llama-server.log"
    log_fh = open(log_path, "ab", buffering=0)
    _llama_proc = subprocess.Popen(cmd, stdout=log_fh, stderr=subprocess.STDOUT)
    if not wait_llama_ready(LLAMA_READY_TIMEOUT):
        rc = _llama_proc.poll()
        stop_llama()
        raise RuntimeError(
            f"llama-server pas prêt après {LLAMA_READY_TIMEOUT}s "
            f"(exit code: {rc}). Voir {log_path}"
        )
    _llama_current_model = model_alias
    print(f"[MAGOS] llama-server prêt pour '{model_alias}' (PID {_llama_proc.pid})", flush=True)


def _unload_biolord():
    """Libère BioLORD du GPU pour faire place à llama-server."""
    global _biolord_model
    if _biolord_model is None:
        return
    print("[MAGOS] Déchargement BioLORD-2023...", flush=True)
    del _biolord_model
    _biolord_model = None
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
    import gc
    gc.collect()
    print("[MAGOS] BioLORD-2023 déchargé, GPU libéré.", flush=True)


def _load_biolord():
    """Charge BioLORD sur GPU. Appelé sous _gpu_lock."""
    global _biolord_model
    if _biolord_model is not None:
        return _biolord_model
    stop_llama()
    from sentence_transformers import SentenceTransformer
    print("[MAGOS] Chargement BioLORD-2023 sur GPU...", flush=True)
    _biolord_model = SentenceTransformer("FremyCompany/BioLORD-2023")
    print("[MAGOS] BioLORD-2023 prêt.", flush=True)
    return _biolord_model


def ensure_model_loaded(model_alias: str):
    """Garantit que llama-server tourne avec le bon modèle. Bloquant."""
    global _llama_current_model
    with _gpu_lock:
        if _llama_current_model == model_alias and _llama_proc and _llama_proc.poll() is None:
            return
        _unload_biolord()
        if _llama_proc is not None:
            stop_llama()
        start_llama(model_alias)


# ─── DATABASE ───────────────────────────────────────────────────────────────
SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    uuid         TEXT UNIQUE NOT NULL,
    priority     INTEGER NOT NULL DEFAULT 5,
    status       TEXT NOT NULL DEFAULT 'queued',
    model        TEXT NOT NULL,
    prompt       TEXT NOT NULL,
    system       TEXT,
    options      TEXT,
    timeout_s    INTEGER NOT NULL DEFAULT 1200,
    keep_alive   TEXT NOT NULL DEFAULT '0',
    thinking     TEXT,
    response     TEXT,
    error        TEXT,
    tokens_in    INTEGER,
    tokens_out   INTEGER,
    duration_s   REAL,
    client_id    TEXT,
    created_at   TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    started_at   TEXT,
    ended_at     TEXT,
    output_file  TEXT
);
CREATE INDEX IF NOT EXISTS idx_status_prio ON jobs(status, priority, created_at);
CREATE INDEX IF NOT EXISTS idx_uuid ON jobs(uuid);
CREATE INDEX IF NOT EXISTS idx_created ON jobs(created_at DESC);
"""


@contextmanager
def db():
    conn = sqlite3.connect(DB_PATH, timeout=10, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    with db() as conn:
        conn.executescript(SCHEMA)
        cols = [r["name"] for r in conn.execute("PRAGMA table_info(jobs)").fetchall()]
        if "keep_alive" not in cols:
            print("[MAGOS] Migration BDD : ajout colonne keep_alive...", flush=True)
            conn.execute("ALTER TABLE jobs ADD COLUMN keep_alive TEXT NOT NULL DEFAULT '0'")


# ─── MODÈLES PYDANTIC ───────────────────────────────────────────────────────
class JobSubmit(BaseModel):
    model: str
    prompt: str
    system: Optional[str] = None
    priority: int = Field(default=5, ge=1, le=9, description="1=top, 9=bottom")
    options: Optional[dict] = None
    timeout_s: int = Field(default=DEFAULT_JOB_TIMEOUT, ge=0, le=7200, description="0 = utilise le défaut")
    keep_alive: str = Field(default="0")
    client_id: Optional[str] = None


class JobOut(BaseModel):
    uuid: str
    status: str
    priority: int
    model: str
    created_at: str
    started_at: Optional[str] = None
    ended_at: Optional[str] = None
    duration_s: Optional[float] = None
    tokens_in: Optional[int] = None
    tokens_out: Optional[int] = None
    error: Optional[str] = None
    output_file: Optional[str] = None
    thinking: Optional[str] = None
    response: Optional[str] = None
    prompt: Optional[str] = None
    queue_position: Optional[int] = None


class EmbeddingRequest(BaseModel):
    input: str | list[str]
    model: str = "BioLORD-2023"


# ─── HELPERS ────────────────────────────────────────────────────────────────
def row_to_dict(row, include_content: bool = False) -> dict:
    if not row:
        return {}
    d = dict(row)
    for k in ("id", "system", "options", "timeout_s", "keep_alive", "client_id"):
        d.pop(k, None)
    if not include_content:
        d.pop("thinking", None)
        d.pop("response", None)
        d.pop("prompt", None)
    return d


def queue_position(job_uuid: str) -> Optional[int]:
    with db() as conn:
        rows = conn.execute(
            "SELECT uuid FROM jobs WHERE status='queued' "
            "ORDER BY priority ASC, created_at ASC"
        ).fetchall()
    for i, r in enumerate(rows, start=1):
        if r["uuid"] == job_uuid:
            return i
    return None


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ─── PARSING THINKING ───────────────────────────────────────────────────────
def split_thinking(text: str) -> tuple[str, str]:
    if not text:
        return "", ""
    thinking_parts = re.findall(r"<think>(.*?)</think>", text, re.DOTALL | re.IGNORECASE)
    response = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE).strip()
    thinking = "\n\n---\n\n".join(p.strip() for p in thinking_parts)
    return thinking, response


def split_thinking_partial(text: str) -> tuple[str, str]:
    """Variante de split_thinking() qui gère le cas d'un <think> non encore fermé.

    Pendant le streaming, on peut être en plein milieu d'un bloc <think>...</think>
    quand on prend un snapshot. On considère alors tout depuis <think> comme du
    thinking en cours, et on restitue les blocs déjà fermés normalement.
    """
    if not text:
        return "", ""
    # Blocs complets <think>...</think>
    completed = re.findall(r"<think>(.*?)</think>", text, re.DOTALL | re.IGNORECASE)
    # Reste après suppression des blocs complets
    rest = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    # Cherche un <think> non fermé dans ce qui reste
    open_match = re.search(r"<think>(.*)$", rest, re.DOTALL | re.IGNORECASE)
    if open_match:
        partial_thinking = open_match.group(1)
        completed.append(partial_thinking)
        response = rest[:open_match.start()].strip()
    else:
        response = rest.strip()
    thinking = "\n\n---\n\n".join(p.strip() for p in completed if p.strip())
    return thinking, response


# ─── OPTIONS OLLAMA → OPENAI MAPPING ───────────────────────────────────────
def _map_options(opts: dict) -> dict:
    """Traduit les options au format Ollama vers le format OpenAI."""
    mapped = {}
    direct = {"temperature", "top_p", "seed", "frequency_penalty",
              "presence_penalty", "stop"}
    for k in direct:
        if k in opts:
            mapped[k] = opts[k]
    if "num_predict" in opts:
        mapped["max_tokens"] = opts["num_predict"]
    if "top_k" in opts:
        mapped["top_k"] = opts["top_k"]
    return mapped


# ─── APPEL LLM EN STREAMING (OpenAI-compatible SSE) ───────────────────────
def stream_llm(job_row, cancel_event: Event,
               snapshot_callback: Optional[Callable[[str, str], None]] = None) -> dict:
    """Appelle llama.cpp via /v1/chat/completions en SSE streaming.

    Si snapshot_callback est fourni, il est appelé toutes les SNAPSHOT_EVERY_N_SEC
    secondes avec (thinking_partiel, response_partielle) pour permettre le live
    preview côté UI.
    """
    opts = json.loads(job_row["options"] or "{}")
    mapped_opts = _map_options(opts)

    messages = []
    if job_row["system"]:
        messages.append({"role": "system", "content": job_row["system"]})
    messages.append({"role": "user", "content": job_row["prompt"]})

    cfg = MODELS_REGISTRY.get(job_row["model"], {})
    llama_model_id = Path(cfg["path"]).name if cfg.get("path") else job_row["model"]
    payload = {
        "model": llama_model_id,
        "messages": messages,
        "stream": True,
        "stream_options": {"include_usage": True},
        **mapped_opts,
    }

    response_text = ""
    thinking_text = ""
    metadata = {}
    start = time.time()
    last_snapshot = start
    timeout_s = job_row["timeout_s"] if job_row["timeout_s"] and job_row["timeout_s"] > 0 else DEFAULT_JOB_TIMEOUT

    with requests.post(
        f"{LLM_URL}/v1/chat/completions",
        json=payload,
        stream=True,
        timeout=(10, None),
    ) as r:
        r.raise_for_status()
        r.encoding = "utf-8"
        for line in r.iter_lines(decode_unicode=True):
            if cancel_event.is_set():
                raise RuntimeError("Job annulé par l'utilisateur")
            if time.time() - start > timeout_s:
                raise RuntimeError(f"Timeout réflexion ({timeout_s}s dépassé) — modèle stoppé pour éviter une boucle")
            if not line:
                continue

            if not line.startswith("data: "):
                continue
            data_str = line[6:]
            if data_str.strip() == "[DONE]":
                break

            try:
                chunk = json.loads(data_str)
            except json.JSONDecodeError:
                continue

            delta = (chunk.get("choices") or [{}])[0].get("delta", {})

            # llama.cpp: reasoning_content = thinking, content = response
            reasoning = delta.get("reasoning_content")
            if reasoning:
                thinking_text += reasoning
            token = delta.get("content")
            if token:
                response_text += token

            usage = chunk.get("usage")
            if usage:
                metadata = usage

            # Snapshot périodique pour live preview
            if snapshot_callback and (time.time() - last_snapshot) >= SNAPSHOT_EVERY_N_SEC:
                try:
                    snapshot_callback(thinking_text.strip(), response_text.strip())
                except Exception as e:
                    print(f"[MAGOS] snapshot error: {e}", flush=True)
                last_snapshot = time.time()

    # Fallback: si pas de champ reasoning_content natif, parse les <think> tags
    if not thinking_text and "<think>" in response_text:
        parsed_thinking, parsed_response = split_thinking(response_text)
        return {
            "thinking": parsed_thinking,
            "response": parsed_response,
            "tokens_in": metadata.get("prompt_tokens"),
            "tokens_out": metadata.get("completion_tokens"),
            "metadata": metadata,
        }

    return {
        "thinking": thinking_text.strip(),
        "response": response_text.strip(),
        "tokens_in": metadata.get("prompt_tokens"),
        "tokens_out": metadata.get("completion_tokens"),
        "metadata": metadata,
    }


# ─── ÉCRITURE DU FICHIER .txt ───────────────────────────────────────────────
def write_output_file(job_uuid: str, job_row, result: dict, duration_s: float) -> str:
    out_path = OUTPUT_DIR / f"job_{job_uuid}.txt"
    sep = "=" * 72
    lines = [
        sep,
        f"JOB ID    : {job_uuid}",
        f"MODÈLE    : {job_row['model']}",
        f"DATE      : {now_iso()}",
        f"DURÉE     : {duration_s:.1f} s",
        f"PRIORITÉ  : {job_row['priority']}",
        f"KEEP_ALIVE: {job_row['keep_alive']}",
        f"CLIENT    : {job_row['client_id'] or 'unknown'}",
        f"TOKENS    : {result.get('tokens_in', '?')} in / {result.get('tokens_out', '?')} out",
        sep,
        "",
        "--- PROMPT SYSTÈME ---",
        job_row["system"] or "(aucun)",
        "",
        "--- PROMPT UTILISATEUR ---",
        job_row["prompt"],
        "",
        "--- THINKING ---",
        result["thinking"] or "(aucun thinking détecté)",
        "",
        "--- RÉPONSE ---",
        result["response"] or "(vide)",
        "",
        "--- MÉTADONNÉES LLM ---",
        json.dumps(result["metadata"], indent=2, ensure_ascii=False),
        "",
        sep,
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return str(out_path)


# ─── BOUCLE WORKER ──────────────────────────────────────────────────────────
def cleanup_orphan_jobs():
    with db() as conn:
        cur = conn.execute(
            "UPDATE jobs SET status='failed', "
            "error='Orphelin (MAGOS restarté pendant exécution)', "
            "ended_at=? WHERE status='running'",
            (now_iso(),)
        )
        return cur.rowcount


def make_snapshot_callback(job_uuid: str) -> Callable[[str, str], None]:
    """Crée une closure qui persiste les morceaux de thinking/response en BDD."""
    def cb(thinking: str, response: str):
        with db() as conn:
            conn.execute(
                "UPDATE jobs SET thinking=?, response=? WHERE uuid=?",
                (thinking, response, job_uuid),
            )
    return cb


def pick_next_job():
    """Choisit le prochain job. Préfère le modèle déjà chargé pour éviter les swaps.

    Stratégie :
      1. S'il y a des jobs queued sur le modèle actuellement chargé,
         on prend le plus prioritaire / le plus ancien de ce modèle.
      2. Sinon on prend le job le plus prioritaire toutes files confondues
         (== modèle qui a le job de plus haute prio en attente).
    """
    with db() as conn:
        row = None
        if _llama_current_model is not None:
            row = conn.execute(
                "SELECT * FROM jobs WHERE status='queued' AND model=? "
                "ORDER BY priority ASC, created_at ASC LIMIT 1",
                (_llama_current_model,),
            ).fetchone()
        if row is None:
            row = conn.execute(
                "SELECT * FROM jobs WHERE status='queued' "
                "ORDER BY priority ASC, created_at ASC LIMIT 1"
            ).fetchone()
        return row


def worker_loop():
    n_orphans = cleanup_orphan_jobs()
    if n_orphans > 0:
        print(f"[MAGOS] {n_orphans} job(s) orphelin(s) marqué(s) comme failed.", flush=True)
    print(f"[MAGOS] Worker démarré (snapshot toutes les {SNAPSHOT_EVERY_N_SEC}s).", flush=True)

    while True:
        job_uuid = None
        try:
            row = pick_next_job()
            if row:
                job_uuid = row["uuid"]
                try:
                    ensure_model_loaded(row["model"])
                except Exception as e:
                    with db() as conn:
                        conn.execute(
                            "UPDATE jobs SET status='failed', error=?, ended_at=? WHERE uuid=?",
                            (f"Échec chargement modèle: {e}", now_iso(), job_uuid),
                        )
                    print(f"[MAGOS] {job_uuid} failed (load model): {e}", flush=True)
                    continue
                with db() as conn:
                    conn.execute(
                        "UPDATE jobs SET status='running', started_at=? WHERE uuid=?",
                        (now_iso(), job_uuid),
                    )
                    conn.commit()

            if not job_uuid:
                time.sleep(WORKER_POLL_INTERVAL)
                continue

            with db() as conn:
                row = conn.execute("SELECT * FROM jobs WHERE uuid=?", (job_uuid,)).fetchone()

            cancel_event = Event()
            with _cancel_lock:
                _cancel_events[job_uuid] = cancel_event

            t0 = time.time()
            try:
                if is_react_job(row):
                    print(f"[MAGOS] {job_uuid} → ORACULUM ReAct mode", flush=True)
                    result = react_loop(
                        row, cancel_event, LLM_URL, MODELS_REGISTRY,
                        snapshot_callback=make_snapshot_callback(job_uuid),
                    )
                else:
                    result = stream_llm(
                        row, cancel_event,
                        snapshot_callback=make_snapshot_callback(job_uuid),
                    )
                duration = time.time() - t0
                output_file = write_output_file(job_uuid, row, result, duration)
                with db() as conn:
                    conn.execute(
                        "UPDATE jobs SET status='done', thinking=?, response=?, "
                        "tokens_in=?, tokens_out=?, duration_s=?, ended_at=?, output_file=? "
                        "WHERE uuid=?",
                        (result["thinking"], result["response"],
                         result["tokens_in"], result["tokens_out"],
                         duration, now_iso(), output_file, job_uuid),
                    )
                print(f"[MAGOS] {job_uuid} done in {duration:.1f}s "
                      f"({result.get('tokens_out', '?')} tok)", flush=True)
            except Exception as e:
                duration = time.time() - t0
                msg = str(e)
                status = "cancelled" if "annulé" in msg.lower() else "failed"
                with db() as conn:
                    conn.execute(
                        "UPDATE jobs SET status=?, error=?, duration_s=?, ended_at=? WHERE uuid=?",
                        (status, msg, duration, now_iso(), job_uuid),
                    )
                print(f"[MAGOS] {job_uuid} {status} ({duration:.1f}s): {msg}", flush=True)
            finally:
                with _cancel_lock:
                    _cancel_events.pop(job_uuid, None)

        except Exception as outer:
            print(f"[MAGOS] Erreur worker: {outer}", flush=True)
            time.sleep(2)


# ─── API FASTAPI ────────────────────────────────────────────────────────────
app = FastAPI(
    title="MAGOS",
    description="Machine-Adept Gateway for LLM Servitors — broker LLM local (llama.cpp)",
    version="0.5.0",
)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    body = await request.body()
    print(f"[MAGOS] 422 on {request.method} {request.url.path}")
    print(f"[MAGOS]   body: {body.decode('utf-8', errors='replace')[:500]}")
    print(f"[MAGOS]   errors: {exc.errors()}")
    return JSONResponse(status_code=422, content={"detail": exc.errors(), "body": body.decode('utf-8', errors='replace')[:500]})


@app.on_event("startup")
def startup():
    init_db()
    global MODELS_REGISTRY
    MODELS_REGISTRY = load_models_registry()
    print(f"[MAGOS] {len(MODELS_REGISTRY)} modèle(s) enregistré(s) : {list(MODELS_REGISTRY)}", flush=True)
    kill_existing_llama_servers()
    Thread(target=worker_loop, daemon=True, name="magos-worker").start()


@app.on_event("shutdown")
def shutdown():
    stop_llama()
    _unload_biolord()


@app.get("/", response_class=HTMLResponse)
def dashboard():
    if not DASHBOARD_PATH.exists():
        return HTMLResponse(
            f"<h1>MAGOS</h1><p>Dashboard introuvable. "
            f"Fichier attendu : <code>{DASHBOARD_PATH}</code></p>"
            f"<p>L'API reste accessible sur <a href='/docs'>/docs</a>.</p>",
            status_code=200,
        )
    return HTMLResponse(DASHBOARD_PATH.read_text(encoding="utf-8"))


@app.get("/health")
def health():
    try:
        r = requests.get(f"{LLM_URL}/v1/models", timeout=3)
        llm_ok = r.status_code == 200
    except Exception:
        llm_ok = False
    with db() as conn:
        queued = conn.execute("SELECT COUNT(*) FROM jobs WHERE status='queued'").fetchone()[0]
        running = conn.execute("SELECT COUNT(*) FROM jobs WHERE status='running'").fetchone()[0]
    return {"status": "ok", "llm": llm_ok, "biolord": _biolord_model is not None,
            "queued": queued, "running": running,
            "llm_url": LLM_URL, "snapshot_interval_s": SNAPSHOT_EVERY_N_SEC}


@app.get("/models")
def models():
    """Liste les modèles disponibles pour le dropdown du hub.

    Renvoie les trois formats que les clients pourraient parser :
      - models[] : format MAGOS étendu (name, path, flags, loaded)
      - data[]   : format OpenAI / llama-server (data[].id == alias)
      - tags[]   : forme Ollama (tags[].name == alias) pour les vieux clients
    """
    aliases = list(MODELS_REGISTRY.keys())
    now = int(time.time())
    return {
        "models": [
            {
                "name": alias,
                "model": alias,
                "id": alias,
                "path": cfg["path"],
                "flags": cfg["flags"],
                "loaded": alias == _llama_current_model,
            }
            for alias, cfg in MODELS_REGISTRY.items()
        ],
        "current": _llama_current_model,
        "object": "list",
        "data": [
            {"id": alias, "object": "model", "owned_by": "magos", "created": now}
            for alias in aliases
        ],
        "tags": [{"name": alias, "model": alias} for alias in aliases],
    }


@app.get("/api/tags")
def ollama_tags():
    """Compat Ollama : certains hubs/clients appellent /api/tags pour la liste."""
    return {"models": [
        {"name": alias, "model": alias, "modified_at": "", "size": 0, "digest": ""}
        for alias in MODELS_REGISTRY
    ]}


@app.get("/v1/models")
def openai_models():
    """Compat OpenAI : certains hubs/clients appellent /v1/models."""
    now = int(time.time())
    return {
        "object": "list",
        "data": [
            {"id": alias, "object": "model", "owned_by": "magos", "created": now}
            for alias in MODELS_REGISTRY
        ],
    }


# ─── EMBEDDINGS ────────────────────────────────────────────────────────────
@app.post("/embeddings")
@app.post("/v1/embeddings")
def create_embeddings(req: EmbeddingRequest):
    """Encode des textes avec BioLORD-2023. Swap GPU automatique avec llama-server."""
    texts = req.input if isinstance(req.input, list) else [req.input]
    if not texts:
        raise HTTPException(400, "input vide")
    with _gpu_lock:
        model = _load_biolord()
        t0 = time.time()
        embeddings = model.encode(texts, normalize_embeddings=True, batch_size=64,
                                  show_progress_bar=len(texts) > 10)
        elapsed = time.time() - t0
    n_tokens = sum(len(t.split()) for t in texts)
    print(f"[MAGOS] Embeddings: {len(texts)} texte(s), {elapsed:.2f}s", flush=True)
    return {
        "object": "list",
        "data": [
            {"object": "embedding", "embedding": emb.tolist(), "index": i}
            for i, emb in enumerate(embeddings)
        ],
        "model": "BioLORD-2023",
        "usage": {"prompt_tokens": n_tokens, "total_tokens": n_tokens},
    }


@app.post("/jobs", response_model=JobOut)
def submit_job(job: JobSubmit):
    if job.model not in MODELS_REGISTRY:
        raise HTTPException(
            status_code=400,
            detail=f"Modèle inconnu : '{job.model}'. Disponibles : {list(MODELS_REGISTRY)}",
        )
    job_uuid = uuidlib.uuid4().hex[:12]
    timeout_s = job.timeout_s if job.timeout_s > 0 else DEFAULT_JOB_TIMEOUT
    with db() as conn:
        conn.execute(
            "INSERT INTO jobs (uuid, priority, model, prompt, system, options, "
            "timeout_s, keep_alive, client_id) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (job_uuid, job.priority, job.model, job.prompt, job.system,
             json.dumps(job.options or {}), timeout_s, job.keep_alive,
             job.client_id),
        )
        row = conn.execute("SELECT * FROM jobs WHERE uuid=?", (job_uuid,)).fetchone()
    out = row_to_dict(row)
    out["queue_position"] = queue_position(job_uuid)
    return out


@app.get("/jobs", response_model=List[JobOut])
def list_jobs(
    status: Optional[str] = None,
    model: Optional[str] = None,
    client_id: Optional[str] = None,
    limit: int = 50,
    exclude_active: bool = False,
):
    limit = max(1, min(limit, 500))
    where = []
    params = []
    if status:
        where.append("status = ?")
        params.append(status)
    if model:
        where.append("model = ?")
        params.append(model)
    if client_id:
        where.append("client_id = ?")
        params.append(client_id)
    if exclude_active:
        where.append("status NOT IN ('queued', 'running')")
    where_sql = (" WHERE " + " AND ".join(where)) if where else ""
    query = f"SELECT * FROM jobs{where_sql} ORDER BY created_at DESC LIMIT ?"
    params.append(limit)
    with db() as conn:
        rows = conn.execute(query, params).fetchall()
    return [row_to_dict(r) for r in rows]


@app.get("/jobs/{job_uuid}", response_model=JobOut)
def get_job(job_uuid: str, include_content: bool = False):
    with db() as conn:
        row = conn.execute("SELECT * FROM jobs WHERE uuid=?", (job_uuid,)).fetchone()
    if not row:
        raise HTTPException(404, "Job inconnu")
    out = row_to_dict(row, include_content=include_content)
    if row["status"] == "queued":
        out["queue_position"] = queue_position(job_uuid)
    return out


@app.get("/jobs/{job_uuid}/wait", response_model=JobOut)
async def wait_job(job_uuid: str, timeout: int = 300, include_content: bool = True):
    timeout = min(timeout, 600)
    start = time.time()
    while time.time() - start < timeout:
        with db() as conn:
            row = conn.execute("SELECT * FROM jobs WHERE uuid=?", (job_uuid,)).fetchone()
        if not row:
            raise HTTPException(404, "Job inconnu")
        if row["status"] in ("done", "failed", "cancelled"):
            return row_to_dict(row, include_content=include_content)
        await asyncio.sleep(1)
    raise HTTPException(408, "Wait timeout (relancer la requête)")


@app.get("/jobs/{job_uuid}/file")
def get_job_file(job_uuid: str):
    with db() as conn:
        row = conn.execute(
            "SELECT output_file, status FROM jobs WHERE uuid=?", (job_uuid,)
        ).fetchone()
    if not row:
        raise HTTPException(404, "Job inconnu")
    if not row["output_file"] or not Path(row["output_file"]).exists():
        raise HTTPException(404, f"Fichier indisponible (status={row['status']})")
    return FileResponse(
        row["output_file"],
        media_type="text/plain; charset=utf-8",
        filename=f"job_{job_uuid}.txt",
    )


@app.delete("/jobs/{job_uuid}")
def cancel_job(job_uuid: str):
    with db() as conn:
        row = conn.execute("SELECT status FROM jobs WHERE uuid=?", (job_uuid,)).fetchone()
        if not row:
            raise HTTPException(404, "Job inconnu")
        if row["status"] == "queued":
            conn.execute(
                "UPDATE jobs SET status='cancelled', error='Annulé avant exécution', "
                "ended_at=? WHERE uuid=?",
                (now_iso(), job_uuid),
            )
            return {"uuid": job_uuid, "status": "cancelled", "method": "dequeued"}
        if row["status"] == "running":
            raise HTTPException(
                409,
                "Job en cours d'exécution — annulation interdite "
                "(le modèle est chargé, laisser terminer)"
            )
        raise HTTPException(409, f"Impossible d'annuler un job en status={row['status']}")


@app.get("/queue")
def get_queue(limit: int = 50):
    with db() as conn:
        queued = conn.execute(
            "SELECT * FROM jobs WHERE status='queued' "
            "ORDER BY priority ASC, created_at ASC LIMIT ?", (limit,)
        ).fetchall()
        running = conn.execute("SELECT * FROM jobs WHERE status='running'").fetchall()
    return {
        "running": [row_to_dict(r) for r in running],
        "queued": [row_to_dict(r) for r in queued],
        "queue_size": len(queued),
    }


@app.get("/stats")
def get_stats():
    with db() as conn:
        total = conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0]
        by_status = dict(conn.execute(
            "SELECT status, COUNT(*) FROM jobs GROUP BY status"
        ).fetchall())
        by_model = conn.execute(
            "SELECT model, COUNT(*) as n, AVG(duration_s) as avg_dur, "
            "AVG(tokens_out) as avg_tok_out FROM jobs WHERE status='done' GROUP BY model"
        ).fetchall()
    return {
        "total_jobs": total,
        "by_status": by_status,
        "by_model": [dict(r) for r in by_model],
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=LISTEN_HOST, port=LISTEN_PORT)
