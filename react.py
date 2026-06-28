"""
ORACULUM — Boucle ReAct pour raisonnement médical augmenté
============================================================
Module agentique pour MAGOS. Le modèle Qwen (via llama.cpp)
peut appeler des outils externes pendant son raisonnement :
SearXNG (web search), fetch URL, et à terme HPO/OMIM/Akinator.

Prérequis llama-server : --jinja (active le tool calling natif Qwen)

Usage dans MAGOS :
  POST /jobs avec options: {"react": true}
  → le worker route vers react_loop() au lieu de stream_llm()

Phase 1 : web_search + web_fetch via SearXNG
"""

import hashlib
import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from threading import Event
from typing import Callable, Optional


def _normalize_fts(text: str) -> str:
    return text.lower().replace("œ", "oe").replace("æ", "ae").replace("ß", "ss")


class _RoundTimeout(Exception):
    """Timeout d'un round avec thinking partiel récupérable."""
    def __init__(self, thinking: str = "", response: str = ""):
        self.thinking = thinking
        self.response = response


# ─── CONFIGURATION ──────────────────────────────────────────────────────────

SEARXNG_URL = os.environ.get("ORACULUM_SEARXNG_URL", "http://127.0.0.1:8888")
SYNDROME_DB_PATH = os.environ.get(
    "ORACULUM_SYNDROME_DB",
    str(Path(__file__).resolve().parent.parent / "foeto_base" / "syndromes_foetaux.db"),
)
MAX_ROUNDS = int(os.environ.get("ORACULUM_MAX_ROUNDS", "4"))
MAX_SEARCH_RESULTS = int(os.environ.get("ORACULUM_MAX_SEARCH_RESULTS", "3"))
MAX_SYNDROME_RESULTS = int(os.environ.get("ORACULUM_MAX_SYNDROME_RESULTS", "8"))
MAX_SYNDROME_SEARCHES = int(os.environ.get("ORACULUM_MAX_SYNDROME_SEARCHES", "2"))
OVERLAP_DEDUP_THRESHOLD = 0.6
MAX_SNIPPET_CHARS = int(os.environ.get("ORACULUM_MAX_SNIPPET_CHARS", "500"))
FETCH_MAX_CHARS = int(os.environ.get("ORACULUM_FETCH_MAX_CHARS", "4000"))
PROB_MODE = os.environ.get("ORACULUM_PROB_MODE", "augmented")
MAX_LOOKUP_RESULTS = int(os.environ.get("ORACULUM_MAX_LOOKUP_RESULTS", "8"))
RRF_K = 60
BIOLORD_MODEL_NAME = "FremyCompany/BioLORD-2023"
_biolord_model = None
ROUND_TIMEOUT = int(os.environ.get("ORACULUM_ROUND_TIMEOUT", "300"))
STALL_TIMEOUT = int(os.environ.get("ORACULUM_STALL_TIMEOUT", "60"))
THINKING_RUNAWAY_CHARS = int(os.environ.get("ORACULUM_THINKING_RUNAWAY", "12000"))


# ─── TOOL DEFINITIONS (format OpenAI, consommé par Qwen via --jinja) ───────

TOOL_DEFS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": (
                "Search the web via SearXNG. Use to find information about rare "
                "syndromes, differential diagnoses, gene-disease associations, "
                "clinical guidelines, or any medical knowledge you are unsure about. "
                "Prefer English queries for better coverage. Use 3-8 words."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query, 3-8 words, English preferred",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "syndrome_search",
            "description": (
                "Search the local fetal pathology syndrome database "
                "(syndromes_foetaux.db — 3194 syndromes, 70k HPO links). "
                "Pass clinical signs as a comma-separated list. Returns "
                "ranked candidate syndromes with matched HPO features, "
                "Bayesian scores, genes, and key discriminators. "
                "PREFER THIS over web_search for syndrome identification."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "signs": {
                        "type": "string",
                        "description": (
                            "Clinical signs, comma-separated. "
                            "French or English. E.g.: 'encéphalocèle, "
                            "polydactylie, reins polykystiques'"
                        ),
                    },
                },
                "required": ["signs"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "case_search",
            "description": (
                "Search the local case reports database for similar resolved cases. "
                "Pass clinical signs as a comma-separated list (same format as "
                "syndrome_search). Returns up to 5 similar cases with their gold "
                "diagnosis and clinical description. Use this to reason by analogy: "
                "'this case looks like case X which was diagnosed as Y'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "signs": {
                        "type": "string",
                        "description": (
                            "Clinical signs, comma-separated. "
                            "French or English. E.g.: 'micromélie, thorax étroit, "
                            "côtes courtes, polydactylie'"
                        ),
                    },
                },
                "required": ["signs"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "foetolookup_search",
            "description": (
                "Search the local FoetoLookup knowledge base — GeneReviews clinical "
                "descriptions and PubMed case reports, embedded with BioLORD-2023. "
                "Hybrid search: semantic (cosine on 768D vectors) + full-text (BM25). "
                "Use to find detailed clinical descriptions, genotype-phenotype "
                "correlations, and similar published cases for a suspected syndrome. "
                "Complements syndrome_search (which matches HPO codes) by providing "
                "rich narrative clinical text."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "Natural language clinical query. E.g.: "
                            "'encephalocele polydactyly polycystic kidneys ciliopathy' "
                            "or 'neonatal hypotonia with lactic acidosis'"
                        ),
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_fetch",
            "description": (
                "Fetch the text content of a specific URL. Use to read the full "
                "text of a page found via web_search — articles, OMIM entries, "
                "Orphanet pages, PubMed abstracts."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "Full URL to fetch",
                    }
                },
                "required": ["url"],
            },
        },
    },
]


# ─── TOOL EXECUTORS ─────────────────────────────────────────────────────────

def _strip_html(html: str) -> str:
    text = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL)
    text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL)
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def execute_web_search(params: dict) -> str:
    import requests

    query = params.get("query", "").strip()
    if not query:
        return "Error: empty query"
    try:
        r = requests.get(
            f"{SEARXNG_URL}/search",
            params={
                "q": query,
                "format": "json",
                "categories": "general,science",
            },
            timeout=15,
        )
        r.raise_for_status()
        results = r.json().get("results", [])[:MAX_SEARCH_RESULTS]
        if not results:
            return f"No results found for: {query}"
        lines = []
        for i, res in enumerate(results, 1):
            title = res.get("title", "")
            url = res.get("url", "")
            snippet = res.get("content", "")[:MAX_SNIPPET_CHARS]
            engine = res.get("engine", "?")
            lines.append(
                f"[{i}] {title}\n"
                f"    URL: {url}\n"
                f"    Source: {engine}\n"
                f"    {snippet}"
            )
        return "\n\n".join(lines)
    except Exception as e:
        return f"Search error: {e}"


def execute_web_fetch(params: dict) -> str:
    import requests

    url = params.get("url", "").strip()
    if not url:
        return "Error: empty URL"
    try:
        r = requests.get(
            url,
            timeout=15,
            headers={"User-Agent": "ORACULUM/1.0 (medical-reasoning-agent)"},
        )
        r.raise_for_status()
        text = _strip_html(r.text)
        if not text:
            return "Error: page returned empty content"
        if len(text) > FETCH_MAX_CHARS:
            text = text[:FETCH_MAX_CHARS] + f"\n[...truncated at {FETCH_MAX_CHARS} chars]"
        return text
    except Exception as e:
        return f"Fetch error: {e}"


_QUALIFIERS = {
    "sévère", "severe", "léger", "légère", "leger", "legere",
    "modéré", "modérée", "modere", "moderee", "très", "tres",
    "bilatéral", "bilatérale", "bilateral", "bilaterale",
    "unilatéral", "unilatérale", "unilateral", "unilaterale",
    "majeur", "majeure", "mineur", "mineure",
    "diffus", "diffuse", "massif", "massive",
    "complet", "complète", "complete", "partiel", "partielle",
    "isolé", "isolée", "isole", "isolee",
    "congénital", "congénitale", "congenital", "congenitale",
    "fœtal", "fœtale", "foetal", "foetale",
    "profond", "profonde", "grave", "important", "importante",
}

_STOP_WORDS = {
    "de", "du", "des", "le", "la", "les", "un", "une",
    "en", "et", "ou", "au", "aux", "par", "pour", "avec",
    "sur", "dans", "qui", "que", "est", "son", "ses",
    "ce", "cette", "ces", "the", "of", "and", "with", "in",
    "très", "tres", "non", "pas", "type",
}


def _hpo_like_query(conn, pattern: str):
    """Run LIKE query on hpo_terms labels/aliases."""
    prob_col = "sh.prob_augmented" if PROB_MODE == "augmented" else "sh.prob"
    w = f"%{pattern}%"
    return conn.execute(
        f"""
        SELECT DISTINCT sh.syndrome_id, {prob_col} as prob,
               h.label_en, h.label_fr, sh.frequency
        FROM syndrome_hpo sh
        JOIN hpo_terms h ON sh.hpo_id = h.hpo_id
        WHERE h.label_fr LIKE ? OR h.label_en LIKE ?
          OR h.aliases_fr LIKE ?
        """,
        (w, w, w),
    ).fetchall()


def _match_sign(conn, sign: str):
    """Three-tier matching cascade for a single clinical sign.

    Returns list of (syndrome_id, prob, label_en, label_fr, frequency, penalty)
    where penalty is 1.0 for exact, 0.85 for stripped, 0.7 for fuzzy.
    """
    rows = _hpo_like_query(conn, sign)
    if rows:
        return [(r["syndrome_id"], r["prob"], r["label_en"],
                 r["label_fr"], r["frequency"], 1.0) for r in rows]

    words = sign.split()
    stripped = [w for w in words if w not in _QUALIFIERS and w not in _STOP_WORDS]
    if stripped and len(stripped) < len(words):
        sign_stripped = " ".join(stripped)
        rows = _hpo_like_query(conn, sign_stripped)
        if rows:
            return [(r["syndrome_id"], r["prob"], r["label_en"],
                     r["label_fr"], r["frequency"], 0.85) for r in rows]

    tokens = stripped or words
    tokens = [t for t in tokens if len(t) > 3 and t not in _STOP_WORDS]
    best_per_syndrome: dict[str, tuple] = {}
    for token in tokens:
        stem = token[:min(len(token), 6)]
        rows = _hpo_like_query(conn, stem)
        for r in rows:
            sid = r["syndrome_id"]
            p = r["prob"] if r["prob"] else 0.5
            if sid not in best_per_syndrome or p > best_per_syndrome[sid][1]:
                best_per_syndrome[sid] = (
                    sid, p, r["label_en"], r["label_fr"], r["frequency"], 0.7,
                )
    return list(best_per_syndrome.values())


def _normalize_syndrome_args(params: dict) -> str:
    """Normalize syndrome_search args for dedup: sort signs, lowercase, strip."""
    signs_raw = params.get("signs", "").strip().lower()
    signs = sorted(
        s.strip() for s in re.split(r"[,;/\n]+", signs_raw) if s.strip()
    )
    return f"syndrome_search:{','.join(signs)}"


def _token_overlap(tokens_a: set[str], tokens_b: set[str]) -> float:
    """Jaccard-like overlap between two token sets."""
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = tokens_a & tokens_b
    smaller = min(len(tokens_a), len(tokens_b))
    return len(intersection) / smaller if smaller else 0.0


def execute_syndrome_search(params: dict) -> str:
    """Query syndromes_foetaux.db with clinical signs (text matching on HPO labels).

    Matching cascade:
    1. Exact LIKE on label_fr / label_en / aliases_fr
    2. Strip qualifiers (sévère, bilatéral, etc.) and retry
    3. Fuzzy: tokenize, stem to 6 chars, try each word (penalty ×0.7)

    TODO: à terme, passer par l'API database.pazuzu.uk (/api/foekinator/export)
    qui sera la source de vérité curée et à jour. Le query direct SQLite est un
    raccourci pour le prototype — il lit le .db local qui peut diverger de la
    version servie par le Flask.
    TODO: enrichir les aliases_fr dans hpo_terms avec les variantes cliniques
    courantes (ex: "poumons hypoplasiques" → alias de "Hypoplasie pulmonaire").
    """
    import sqlite3

    signs_raw = params.get("signs", "").strip()
    if not signs_raw:
        return "Error: empty signs"

    signs = [_normalize_fts(s.strip()) for s in re.split(r"[,;/\n]+", signs_raw) if s.strip()]
    if not signs:
        return "Error: could not parse any signs"

    try:
        conn = sqlite3.connect(SYNDROME_DB_PATH)
        conn.row_factory = sqlite3.Row
    except Exception as e:
        return f"DB error: {e}"

    scored: dict[str, dict] = {}

    for sign in signs:
        matches = _match_sign(conn, sign)

        for syndrome_id, prob, label_en, label_fr, frequency, penalty in matches:
            sid = syndrome_id
            if sid not in scored:
                scored[sid] = {"id": sid, "score": 0.0, "matched": []}
            p = (prob if prob else 0.5) * penalty
            scored[sid]["score"] += p
            label = label_fr or label_en
            freq = frequency or ""
            tag = "" if penalty == 1.0 else " ~fuzzy"
            scored[sid]["matched"].append(f"{label} ({freq}, p={prob}){tag}")

    if not scored:
        conn.close()
        return f"No syndromes found matching: {signs_raw}"

    ranked = sorted(scored.values(), key=lambda x: x["score"], reverse=True)[
        :MAX_SYNDROME_RESULTS
    ]

    lines = []
    for i, item in enumerate(ranked, 1):
        s = conn.execute(
            """
            SELECT id, name_fr, name_en, category, relevance, aliases,
                   description_md, prenatal_signs_summary,
                   key_discriminators, differential_diagnosis
            FROM syndromes WHERE id = ?
            """,
            (item["id"],),
        ).fetchone()
        if not s:
            continue

        genes_row = conn.execute(
            "SELECT GROUP_CONCAT(gene_symbol, ', ') as g FROM syndrome_genes WHERE syndrome_id = ?",
            (item["id"],),
        ).fetchone()
        genes = genes_row["g"] if genes_row and genes_row["g"] else "?"

        matched_str = "; ".join(item["matched"])
        name_en = f" ({s['name_en']})" if s["name_en"] else ""
        aliases = f"\n    Aliases: {s['aliases']}" if s["aliases"] else ""
        desc = ""
        if s["key_discriminators"]:
            desc = f"\n    Discriminateurs: {s['key_discriminators'][:300]}"
        elif s["prenatal_signs_summary"]:
            desc = f"\n    Signes prénataux: {s['prenatal_signs_summary'][:300]}"

        lines.append(
            f"[{i}] {s['name_fr']}{name_en}\n"
            f"    ID: {s['id']} | Catégorie: {s['category']} | Relevance: {s['relevance'] or '?'}\n"
            f"    Gènes: {genes}\n"
            f"    Score: {item['score']:.2f} ({len(item['matched'])} signes matchés)\n"
            f"    Matchés: {matched_str}"
            f"{aliases}{desc}"
        )

    conn.close()
    return "\n\n".join(lines)


MAX_CASE_RESULTS = int(os.environ.get("ORACULUM_MAX_CASE_RESULTS", "10"))


def execute_case_search(params: dict) -> str:
    """Search case_reports table via FTS5 + HPO tag matching.

    Two-tier search:
    1. FTS5 full-text on clinical_text + hpo_tags (fast, broad)
    2. Re-rank by HPO tag overlap with query signs
    """
    import sqlite3

    signs_raw = params.get("signs", "").strip()
    if not signs_raw:
        return "Error: empty signs"

    signs = [_normalize_fts(s.strip()) for s in re.split(r"[,;/\n]+", signs_raw) if s.strip()]
    if not signs:
        return "Error: could not parse any signs"

    try:
        conn = sqlite3.connect(SYNDROME_DB_PATH)
        conn.row_factory = sqlite3.Row
    except Exception as e:
        return f"DB error: {e}"

    table_exists = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='case_reports'"
    ).fetchone()
    if not table_exists:
        conn.close()
        return "case_reports table not found"

    count = conn.execute("SELECT COUNT(*) FROM case_reports").fetchone()[0]
    if count == 0:
        conn.close()
        return "No case reports in database yet"

    fts_query = " OR ".join(signs)
    try:
        rows = conn.execute(
            """SELECT cr.id, cr.gold_diagnosis, cr.clinical_text, cr.hpo_tags,
                      cr.syndrome_id, cr.format, cr.source,
                      rank
               FROM case_reports_fts fts
               JOIN case_reports cr ON cr.id = fts.rowid
               WHERE case_reports_fts MATCH ?
               ORDER BY rank
               LIMIT ?""",
            (fts_query, MAX_CASE_RESULTS * 3),
        ).fetchall()
    except Exception:
        rows = []

    if not rows:
        like_clauses = " OR ".join(
            "clinical_text LIKE ? OR hpo_tags LIKE ?" for _ in signs
        )
        like_params = []
        for s in signs[:5]:
            like_params.extend([f"%{s}%", f"%{s}%"])
        try:
            rows = conn.execute(
                f"""SELECT id, gold_diagnosis, clinical_text, hpo_tags,
                           syndrome_id, format, source, 0 as rank
                    FROM case_reports
                    WHERE {like_clauses}
                    LIMIT ?""",
                (*like_params, MAX_CASE_RESULTS * 3),
            ).fetchall()
        except Exception:
            rows = []

    if not rows:
        conn.close()
        return f"No similar cases found for: {signs_raw}"

    sign_tokens = set()
    for s in signs:
        sign_tokens.update(s.split())

    scored = []
    for row in rows:
        hpo_tags = (row["hpo_tags"] or "").lower()
        text = (row["clinical_text"] or "").lower()
        combined = hpo_tags + " " + text

        overlap = sum(1 for t in sign_tokens if t in combined)
        scored.append((overlap, row))

    scored.sort(key=lambda x: x[0], reverse=True)
    top = scored[:MAX_CASE_RESULTS]

    lines = []
    for i, (overlap, row) in enumerate(top, 1):
        text_preview = (row["clinical_text"] or "")[:400]
        if len(row["clinical_text"] or "") > 400:
            text_preview += "..."

        lines.append(
            f"[{i}] DIAGNOSTIC: {row['gold_diagnosis']}\n"
            f"    Source: {row['source']} | Format: {row['format']}\n"
            f"    Signes matchés: {overlap}/{len(sign_tokens)} tokens\n"
            f"    Extrait: {text_preview}"
        )

    conn.close()
    return "\n\n".join(lines)


def _get_biolord_model():
    global _biolord_model
    if _biolord_model is None:
        from sentence_transformers import SentenceTransformer
        _biolord_model = SentenceTransformer(BIOLORD_MODEL_NAME)
    return _biolord_model


def execute_foetolookup_search(params: dict) -> str:
    """Hybrid FTS5 + sqlite-vec search over GeneReviews and PubMed chunks."""
    import sqlite3
    import struct
    import sqlite_vec

    query = params.get("query", "").strip()
    if not query:
        return "Error: empty query"

    try:
        conn = sqlite3.connect(SYNDROME_DB_PATH)
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.row_factory = sqlite3.Row
    except Exception as e:
        return f"DB error: {e}"

    vec_exists = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='vec_chunks'"
    ).fetchone()
    if not vec_exists:
        conn.close()
        return "vec_chunks table not found — run embed_biolord.py first"

    # --- Semantic search via BioLORD ---
    try:
        model = _get_biolord_model()
        q_emb = model.encode([query], normalize_embeddings=True)[0]
        q_blob = struct.pack(f"{len(q_emb)}f", *q_emb)

        vec_rows = conn.execute(
            """SELECT v.rowid, v.distance, m.source_type, m.source_id,
                      m.title, m.chunk_index, m.chunk_text
               FROM vec_chunks v
               JOIN chunk_meta m ON m.rowid = v.rowid
               WHERE v.embedding MATCH ? AND k = 50
               ORDER BY v.distance""",
            (q_blob,),
        ).fetchall()
    except Exception as e:
        vec_rows = []

    # --- FTS5 search on chunk_meta.chunk_text ---
    fts_rows = []
    fts_table = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='chunk_meta_fts'"
    ).fetchone()
    if fts_table:
        tokens = [t.strip() for t in re.split(r"[,;/\s]+", query) if len(t.strip()) > 2]
        if tokens:
            fts_query = " OR ".join(tokens[:10])
            try:
                fts_rows = conn.execute(
                    """SELECT rowid, rank, source_type, source_id, title,
                              chunk_index, chunk_text
                       FROM chunk_meta_fts
                       WHERE chunk_meta_fts MATCH ?
                       ORDER BY rank
                       LIMIT 50""",
                    (fts_query,),
                ).fetchall()
            except Exception:
                fts_rows = []
    else:
        tokens = [t.strip() for t in re.split(r"[,;/\s]+", query) if len(t.strip()) > 2]
        if tokens:
            like_clauses = " OR ".join("chunk_text LIKE ?" for _ in tokens[:8])
            like_params = [f"%{t}%" for t in tokens[:8]]
            try:
                fts_rows = conn.execute(
                    f"""SELECT rowid, 0 as rank, source_type, source_id, title,
                               chunk_index, chunk_text
                        FROM chunk_meta
                        WHERE {like_clauses}
                        LIMIT 50""",
                    like_params,
                ).fetchall()
            except Exception:
                fts_rows = []

    # --- Reciprocal Rank Fusion ---
    rrf_scores: dict[int, float] = {}
    chunk_data: dict[int, dict] = {}

    for rank, row in enumerate(vec_rows):
        rid = row["rowid"]
        rrf_scores[rid] = rrf_scores.get(rid, 0) + 1.0 / (RRF_K + rank + 1)
        if rid not in chunk_data:
            chunk_data[rid] = {
                "source_type": row["source_type"],
                "source_id": row["source_id"],
                "title": row["title"],
                "chunk_index": row["chunk_index"],
                "chunk_text": row["chunk_text"],
                "cosine": 1.0 - row["distance"],
            }

    for rank, row in enumerate(fts_rows):
        rid = row["rowid"]
        rrf_scores[rid] = rrf_scores.get(rid, 0) + 1.0 / (RRF_K + rank + 1)
        if rid not in chunk_data:
            chunk_data[rid] = {
                "source_type": row["source_type"],
                "source_id": row["source_id"],
                "title": row["title"],
                "chunk_index": row["chunk_index"],
                "chunk_text": row["chunk_text"],
                "cosine": None,
            }

    conn.close()

    if not rrf_scores:
        return f"No results found for: {query}"

    ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:MAX_LOOKUP_RESULTS]

    lines = []
    for i, (rid, score) in enumerate(ranked, 1):
        d = chunk_data[rid]
        src_label = {"genereviews": "GeneReviews", "pubmed": "PubMed", "textbook": "Textbook"}.get(
            d["source_type"], d["source_type"]
        )
        cosine_str = f" | cosine={d['cosine']:.3f}" if d["cosine"] is not None else ""
        text_preview = d["chunk_text"][:500]
        if len(d["chunk_text"]) > 500:
            text_preview += "..."

        lines.append(
            f"[{i}] {d['title']}\n"
            f"    Source: {src_label} ({d['source_id']}) | chunk {d['chunk_index']}"
            f" | RRF={score:.4f}{cosine_str}\n"
            f"    {text_preview}"
        )

    header = f"FoetoLookup: {len(ranked)} results (from {len(vec_rows)} vec + {len(fts_rows)} FTS matches)"
    return header + "\n\n" + "\n\n".join(lines)


TOOL_EXECUTORS = {
    "web_search": execute_web_search,
    "web_fetch": execute_web_fetch,
    "syndrome_search": execute_syndrome_search,
    "case_search": execute_case_search,
    "foetolookup_search": execute_foetolookup_search,
}


# ─── REACT TRACE (pour le dashboard MAGOS) ─────────────────────────────────

@dataclass
class ReactTrace:
    rounds: list = field(default_factory=list)
    total_tokens_in: int = 0
    total_tokens_out: int = 0

    def add_round(
        self,
        round_num: int,
        tool_calls: list,
        results: list,
        duration_s: float,
    ):
        self.rounds.append({
            "round": round_num,
            "tool_calls": [
                {"name": tc["name"], "args": tc["args"]} for tc in tool_calls
            ],
            "results_preview": [r[:300] for r in results],
            "duration_s": round(duration_s, 2),
        })

    def summary(self) -> str:
        lines = [f"ORACULUM ReAct — {len(self.rounds)} round(s)"]
        for r in self.rounds:
            for tc in r["tool_calls"]:
                args_str = json.dumps(tc["args"], ensure_ascii=False)
                if len(args_str) > 80:
                    args_str = args_str[:80] + "..."
                lines.append(
                    f"  Round {r['round']}: {tc['name']}({args_str}) "
                    f"[{r['duration_s']}s]"
                )
        lines.append(
            f"  Tokens: {self.total_tokens_in} in / "
            f"{self.total_tokens_out} out"
        )
        return "\n".join(lines)


# ─── OPTIONS MAPPING (dupliqué de magos.py pour éviter l'import circulaire) ─

def _map_options(opts: dict) -> dict:
    mapped = {}
    for k in ("temperature", "top_p", "seed", "frequency_penalty",
              "presence_penalty", "stop"):
        if k in opts:
            mapped[k] = opts[k]
    if "num_predict" in opts:
        mapped["max_tokens"] = opts["num_predict"]
    if "top_k" in opts:
        mapped["top_k"] = opts["top_k"]
    return mapped


# ─── EXTRACTION JSON DEPUIS LE THINKING ───────────────────────────────────

def _extract_json_from_thinking(thinking: str) -> Optional[str]:
    """Extrait le dernier bloc JSON valide contenant 'diagnostics' du thinking.
    Cherche d'abord dans les blocs ```json```, puis tente un parsing brut."""
    blocks = re.findall(r"```json\s*(.*?)\s*```", thinking, re.DOTALL)
    best = None
    for block in blocks:
        try:
            d = json.loads(block)
            if isinstance(d, dict) and "diagnostics" in d:
                best = block.strip()
        except (json.JSONDecodeError, ValueError):
            pass
    if best:
        return best
    for m in re.finditer(r'\{"diagnostics"\s*:', thinking):
        start = m.start()
        depth = 0
        for i in range(start, min(start + 2000, len(thinking))):
            if thinking[i] == "{":
                depth += 1
            elif thinking[i] == "}":
                depth -= 1
                if depth == 0:
                    candidate = thinking[start:i + 1]
                    try:
                        d = json.loads(candidate)
                        if "diagnostics" in d:
                            best = candidate
                    except (json.JSONDecodeError, ValueError):
                        pass
                    break
    if best:
        return best
    # Fallback: numbered list "1. Diagnostic (N%)" in the thinking
    diags = []
    seen = set()
    for m in re.finditer(
        r"^\s*(\d+)\.\s+(.+?)\s*\((\d+)%\)", thinking, re.MULTILINE
    ):
        rang = int(m.group(1))
        name = m.group(2).strip().strip("*").strip()
        prob = int(m.group(3))
        if rang < 1 or rang > 5 or not name:
            continue
        key = (rang, name)
        if key not in seen:
            seen.add(key)
            diags.append({"rang": rang, "diagnostic": name, "probabilite": prob})
    if diags:
        by_rang = {}
        for d in diags:
            by_rang[d["rang"]] = d
        final = [by_rang[r] for r in sorted(by_rang) if r <= 5]
        if final:
            return json.dumps({"diagnostics": final}, ensure_ascii=False)
    return best


# --- PARSER FALLBACK : INTENTIONS IMPLICITES DANS LE THINKING -----------

_Q = '[“”„‘’"\'"]'

_IMPLICIT_SEARCH_PATTERNS = [
    re.compile(
        "(?:Let(?:[’']s| me) (?:search|look up|verify|check|google|query)"
        '(?:\\s+(?:for|about|on|if))?\\s+)'
        + _Q + '(.+?)' + _Q,
        re.IGNORECASE,
    ),
    re.compile(
        '(?:I (?:need to|should|will|want to) (?:search|look up|verify)'
        '\\s+(?:for\\s+)?)'
        + _Q + '(.+?)' + _Q,
        re.IGNORECASE,
    ),
    re.compile(
        'Search quer(?:y|ies):?\\s*\\n'
        '(?:\\s*[\\*\\-]\\s*' + _Q + '(.+?)' + _Q + ')',
        re.IGNORECASE,
    ),
    re.compile(
        'Search quer(?:y|ies):?\\s*\\n'
        '(?:\\s*[\\*\\-]\\s*(.+?)$)',
        re.MULTILINE | re.IGNORECASE,
    ),
]


_BOLD_SEARCH_PATTERNS = [
    re.compile(
        r"Let(?:['‘’]s| me) (?:verify|evaluate|check|consider|look up|search for)"
        r"\s+\*\*(.+?)\*\*",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:I (?:need to|should|will) (?:verify|check|consider|look up))"
        r"\s+\*\*(.+?)\*\*",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:One (?:more|last) check:?\s*)\*\*(.+?)\*\*",
        re.IGNORECASE,
    ),
]


def _extract_query_lists(thinking: str) -> list:
    """Parse les blocs 'Search queries:' suivis de listes a puces."""
    queries = []
    in_block = False
    for line in thinking.split("\n"):
        stripped = line.strip()
        if re.match(r"(?:Search|My)\s+quer(?:y|ies)", stripped, re.IGNORECASE):
            in_block = True
            continue
        if in_block:
            m = re.match(r"[\*\-\d.]+\s*[\"']?(.+?)[\"']?\s*$", stripped)
            if m:
                q = m.group(1).strip().strip("\"'").strip()
                if q and q not in queries:
                    queries.append(q)
            elif stripped == "":
                continue
            else:
                in_block = False
    return queries


def _extract_implicit_tool_calls(thinking: str) -> list:
    """Extrait des web_search implicites du CoT quand le modele
    ecrit ses intentions en langage naturel au lieu d'emettre
    des tool_calls structures."""
    queries = []
    for pat in _IMPLICIT_SEARCH_PATTERNS:
        for m in pat.finditer(thinking):
            q = m.group(1).strip().strip("\"'").strip()
            if 3 <= len(q.split()) <= 15 and q not in queries:
                queries.append(q)
    for q in _extract_query_lists(thinking):
        q = q.strip("\"'").strip()
        if 2 <= len(q.split()) <= 15 and q not in queries:
            queries.append(q)
    for pat in _BOLD_SEARCH_PATTERNS:
        for m in pat.finditer(thinking):
            q = m.group(1).strip()
            if 2 <= len(q.split()) <= 10 and q not in queries:
                queries.append(q)
    if not queries:
        return []
    tool_calls = []
    for i, q in enumerate(queries[:3]):
        tool_calls.append({
            "id": f"implicit_{i}",
            "name": "web_search",
            "args": {"query": q},
        })
    return tool_calls


# ─── PARSER FALLBACK POUR <tool_call> XML NATIF QWEN ───────────────────────

def _parse_xml_tool_calls(text: str) -> list:
    """Parse les <tool_call> XML que Qwen émet dans le thinking/response
    quand llama-server ne les capture pas en tool_calls structurés.

    Format Qwen natif:
      <tool_call>
      <function=web_search>
      <parameter=query>some query</parameter>
      </function>
      </tool_call>
    """
    blocks = re.findall(
        r"<tool_call>(.*?)</tool_call>", text, re.DOTALL
    )
    if not blocks:
        return []

    tool_calls = []
    for i, block in enumerate(blocks):
        fn_match = re.search(r"<function=(\w+)>", block)
        if not fn_match:
            continue
        name = fn_match.group(1)
        params = {}
        for pm in re.finditer(
            r"<parameter=(\w+)>\s*(.*?)\s*</parameter>", block, re.DOTALL
        ):
            params[pm.group(1)] = pm.group(2).strip()
        # Fallback: aussi parser le format JSON inline
        if not params:
            json_match = re.search(r"\{.*\}", block, re.DOTALL)
            if json_match:
                try:
                    params = json.loads(json_match.group(0))
                except json.JSONDecodeError:
                    pass
        tool_calls.append({
            "id": f"xmlcall_{i}",
            "name": name,
            "args": params,
        })
    return tool_calls


# ─── APPEL LLM AVEC TOOL CALLING ───────────────────────────────────────────

def _stream_round(
    messages: list,
    llm_url: str,
    model_id: str,
    mapped_opts: dict,
    cancel_event: Event,
    timeout_s: int,
    tools: Optional[list] = None,
) -> dict:
    """Un appel LLM unique avec support tool_calls en streaming."""
    import requests

    payload = {
        "model": model_id,
        "messages": messages,
        "stream": True,
        "stream_options": {"include_usage": True},
        **mapped_opts,
    }
    if tools:
        payload["tools"] = tools

    response_text = ""
    thinking_text = ""
    tool_calls_buf: dict[int, dict] = {}
    metadata = {}
    start = time.time()
    last_token_time = start

    with requests.post(
        f"{llm_url}/v1/chat/completions",
        json=payload,
        stream=True,
        timeout=(10, None),
    ) as r:
        r.raise_for_status()
        r.encoding = "utf-8"
        for line in r.iter_lines(decode_unicode=True):
            now = time.time()
            if cancel_event.is_set():
                raise RuntimeError("Job annulé par l'utilisateur")
            if now - start > timeout_s:
                raise _RoundTimeout(
                    thinking=thinking_text,
                    response=response_text,
                )
            if now - last_token_time > STALL_TIMEOUT:
                raise _RoundTimeout(
                    thinking=thinking_text,
                    response=response_text,
                )
            if not line or not line.startswith("data: "):
                continue
            data_str = line[6:]
            if data_str.strip() == "[DONE]":
                break
            try:
                chunk = json.loads(data_str)
            except json.JSONDecodeError:
                continue

            delta = (chunk.get("choices") or [{}])[0].get("delta", {})

            reasoning = delta.get("reasoning_content")
            if reasoning:
                thinking_text += reasoning
                last_token_time = now

            token = delta.get("content")
            if token:
                response_text += token
                last_token_time = now

            for tc in delta.get("tool_calls", []):
                idx = tc.get("index", 0)
                if idx not in tool_calls_buf:
                    tool_calls_buf[idx] = {
                        "id": tc.get("id", f"call_{idx}"),
                        "name": "",
                        "arguments": "",
                    }
                fn = tc.get("function", {})
                if fn.get("name"):
                    tool_calls_buf[idx]["name"] = fn["name"]
                if "arguments" in fn:
                    tool_calls_buf[idx]["arguments"] += fn["arguments"]
                last_token_time = now

            usage = chunk.get("usage")
            if usage:
                metadata = usage

    # Parse les tool_calls accumulés
    tool_calls = []
    for idx in sorted(tool_calls_buf):
        tc = tool_calls_buf[idx]
        try:
            args = json.loads(tc["arguments"]) if tc["arguments"] else {}
        except json.JSONDecodeError:
            args = {"_raw": tc["arguments"]}
        tool_calls.append({"id": tc["id"], "name": tc["name"], "args": args})

    # Fallback <think> tags si pas de reasoning_content natif
    if not thinking_text and "<think>" in response_text:
        parts = re.findall(
            r"<think>(.*?)</think>", response_text, re.DOTALL | re.IGNORECASE
        )
        response_text = re.sub(
            r"<think>.*?</think>", "", response_text,
            flags=re.DOTALL | re.IGNORECASE,
        ).strip()
        thinking_text = "\n\n---\n\n".join(p.strip() for p in parts)

    # Fallback 1: parser les <tool_call> XML natifs Qwen
    if not tool_calls:
        tool_calls = _parse_xml_tool_calls(thinking_text + "\n" + response_text)
        if tool_calls:
            thinking_text = re.sub(
                r"<tool_call>.*?</tool_call>",
                "", thinking_text, flags=re.DOTALL,
            ).strip()
            response_text = re.sub(
                r"<tool_call>.*?</tool_call>",
                "", response_text, flags=re.DOTALL,
            ).strip()

    # Fallback 2: extraire les intentions implicites du thinking
    if not tool_calls and thinking_text:
        tool_calls = _extract_implicit_tool_calls(thinking_text)
        if tool_calls:
            print(
                f"[ORACULUM] Implicit tool calls extracted from thinking: "
                f"{[tc['args'] for tc in tool_calls]}",
                flush=True,
            )

    return {
        "thinking": thinking_text.strip(),
        "response": response_text.strip(),
        "tool_calls": tool_calls,
        "tokens_in": metadata.get("prompt_tokens", 0),
        "tokens_out": metadata.get("completion_tokens", 0),
        "metadata": metadata,
    }


# ─── BOUCLE REACT PRINCIPALE ───────────────────────────────────────────────

def react_loop(
    job_row,
    cancel_event: Event,
    llm_url: str,
    models_registry: dict,
    snapshot_callback: Optional[Callable[[str, str], None]] = None,
) -> dict:
    """Boucle ReAct complète. Remplace stream_llm() pour les jobs avec outils.

    Retourne le même format que stream_llm() :
      {thinking, response, tokens_in, tokens_out, metadata}
    """
    opts = json.loads(job_row["options"] or "{}")
    react_opts = opts.pop("react", True)
    mapped_opts = _map_options(opts)

    cfg = models_registry.get(job_row["model"], {})
    model_id = Path(cfg["path"]).name if cfg.get("path") else job_row["model"]

    timeout_s = job_row["timeout_s"] or 600
    if timeout_s <= 0:
        timeout_s = 600

    # ── Extraction HPO structurée (pré-processing du prompt)
    hpo_block = ""
    try:
        from foeto_base.hpo_extractor import get_extractor
        hpo_block = get_extractor().format_for_prompt(job_row["prompt"])
    except Exception:
        try:
            import sys
            sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "foeto_base"))
            from hpo_extractor import get_extractor
            hpo_block = get_extractor().format_for_prompt(job_row["prompt"])
        except Exception:
            pass

    # ── Messages initiaux
    messages = []
    if job_row["system"]:
        messages.append({"role": "system", "content": job_row["system"]})

    user_content = job_row["prompt"]
    if hpo_block:
        user_content = f"{hpo_block}\n\n{user_content}"
    messages.append({"role": "user", "content": user_content})

    trace = ReactTrace()
    all_thinking = []
    seen_hashes: set[str] = set()
    syndrome_search_count = 0
    syndrome_query_tokens: list[set[str]] = []
    force_no_tools = False
    final_response = ""

    for round_num in range(1, MAX_ROUNDS + 1):
        if cancel_event.is_set():
            raise RuntimeError("Job annulé par l'utilisateur")

        print(
            f"[ORACULUM] Round {round_num}/{MAX_ROUNDS} — "
            f"{len(messages)} messages, lancement inférence...",
            flush=True,
        )

        t0 = time.time()
        round_tools = None if force_no_tools else TOOL_DEFS
        if force_no_tools:
            messages.append({
                "role": "user",
                "content": (
                    "You have already searched the syndrome database and the web. "
                    "Do NOT call any more tools. Provide your final JSON answer NOW."
                ),
            })
            print(
                f"[ORACULUM] Round {round_num} — FORCE NO TOOLS (anti-loop)",
                flush=True,
            )
        try:
            result = _stream_round(
                messages, llm_url, model_id, mapped_opts,
                cancel_event, min(ROUND_TIMEOUT, timeout_s),
                tools=round_tools,
            )
        except _RoundTimeout as e:
            elapsed = time.time() - t0
            print(
                f"[ORACULUM] Round {round_num} — timeout après "
                f"{elapsed:.0f}s, thinking: {len(e.thinking)} chars, "
                f"response: {len(e.response)} chars",
                flush=True,
            )
            # Try to extract implicit tool calls from runaway thinking
            if e.thinking and not e.response:
                implicit = _extract_implicit_tool_calls(e.thinking)
                if implicit:
                    print(
                        f"[ORACULUM] Runaway thinking rescue: "
                        f"{len(implicit)} implicit tool call(s) extracted",
                        flush=True,
                    )
                    all_thinking.append(
                        f"[Round {round_num} — timeout, rescued]\n"
                        f"{e.thinking}"
                    )
                    result = {
                        "thinking": e.thinking,
                        "response": "",
                        "tool_calls": implicit,
                        "tokens_in": 0,
                        "tokens_out": 0,
                        "metadata": {},
                    }
                    # Fall through to normal tool execution below
                else:
                    all_thinking.append(
                        f"[Round {round_num} — timeout]\n{e.thinking}"
                    )
                    break
            else:
                if e.thinking:
                    all_thinking.append(
                        f"[Round {round_num} — timeout]\n{e.thinking}"
                    )
                if e.response:
                    final_response = e.response
                break
        round_duration = time.time() - t0

        trace.total_tokens_in += result["tokens_in"]
        trace.total_tokens_out += result["tokens_out"]

        if result["thinking"]:
            all_thinking.append(f"[Round {round_num}]\n{result['thinking']}")

        # ── Pas de tool call → réponse finale
        if not result["tool_calls"]:
            final_response = result["response"]
            print(
                f"[ORACULUM] Round {round_num} — réponse finale "
                f"({len(final_response)} chars, {round_duration:.1f}s)",
                flush=True,
            )
            break

        # ── Message assistant avec tool_calls (format OpenAI)
        assistant_msg: dict = {"role": "assistant"}
        if result["response"]:
            assistant_msg["content"] = result["response"]
        else:
            assistant_msg["content"] = None
        assistant_msg["tool_calls"] = [
            {
                "id": tc["id"],
                "type": "function",
                "function": {
                    "name": tc["name"],
                    "arguments": json.dumps(tc["args"], ensure_ascii=False),
                },
            }
            for tc in result["tool_calls"]
        ]
        messages.append(assistant_msg)

        # ── Exécution des outils
        tool_results = []
        for tc in result["tool_calls"]:
            if tc["name"] == "syndrome_search":
                call_sig = _normalize_syndrome_args(tc["args"])
            else:
                call_sig = f"{tc['name']}:{json.dumps(tc['args'], sort_keys=True)}"
            call_hash = hashlib.md5(call_sig.encode()).hexdigest()

            is_near_dup = False
            if tc["name"] == "syndrome_search" and call_hash not in seen_hashes:
                new_tokens = set(
                    t for t in re.split(r"[,;/\s]+", tc["args"].get("signs", "").lower())
                    if len(t) > 2
                )
                for prev in syndrome_query_tokens:
                    if _token_overlap(new_tokens, prev) >= OVERLAP_DEDUP_THRESHOLD:
                        is_near_dup = True
                        break

            if call_hash in seen_hashes:
                tool_result = (
                    "[Duplicate query — already searched in a previous round. "
                    "Use the previous results or reformulate your query.]"
                )
                print(
                    f"[ORACULUM] Round {round_num} — {tc['name']} DEDUP",
                    flush=True,
                )
            elif is_near_dup:
                tool_result = (
                    "[Near-duplicate query — you already searched with very similar "
                    "signs. Use your previous results to formulate your answer.]"
                )
                print(
                    f"[ORACULUM] Round {round_num} — {tc['name']} NEAR-DEDUP "
                    f"(overlap >= {OVERLAP_DEDUP_THRESHOLD})",
                    flush=True,
                )
            elif tc["name"] in TOOL_EXECUTORS:
                tool_result = TOOL_EXECUTORS[tc["name"]](tc["args"])
                seen_hashes.add(call_hash)
                if tc["name"] == "syndrome_search":
                    syndrome_search_count += 1
                    new_tokens = set(
                        t for t in re.split(r"[,;/\s]+", tc["args"].get("signs", "").lower())
                        if len(t) > 2
                    )
                    syndrome_query_tokens.append(new_tokens)
                print(
                    f"[ORACULUM] Round {round_num} — {tc['name']}"
                    f"({json.dumps(tc['args'], ensure_ascii=False)}) "
                    f"→ {len(tool_result)} chars [{round_duration:.1f}s]",
                    flush=True,
                )
            else:
                tool_result = f"Unknown tool: {tc['name']}"

            tool_results.append(tool_result)
            messages.append({
                "role": "tool",
                "tool_call_id": tc["id"],
                "content": tool_result,
            })

        if syndrome_search_count >= MAX_SYNDROME_SEARCHES:
            force_no_tools = True
            print(
                f"[ORACULUM] Anti-loop: {syndrome_search_count} syndrome_search "
                f"calls reached (max={MAX_SYNDROME_SEARCHES}), forcing answer next round",
                flush=True,
            )

        trace.add_round(
            round_num, result["tool_calls"], tool_results, round_duration,
        )

        # ── Snapshot pour le dashboard (live preview)
        if snapshot_callback:
            try:
                thinking_preview = "\n\n---\n\n".join(all_thinking)
                status_line = (
                    f"[ORACULUM Round {round_num}/{MAX_ROUNDS}]\n"
                    f"{trace.summary()}"
                )
                snapshot_callback(thinking_preview, status_line)
            except Exception:
                pass

    # ── Final nudge : si pas de réponse JSON valide, on relance sans tools
    _has_valid_json = False
    if final_response.strip():
        try:
            d = json.loads(final_response.strip().removeprefix("```json").removesuffix("```").strip())
            _has_valid_json = bool(d.get("diagnostics"))
        except (json.JSONDecodeError, AttributeError):
            pass

    if not _has_valid_json:
        print(
            "[ORACULUM] Pas de JSON valide — round final sans tools...",
            flush=True,
        )
        messages.append({
            "role": "user",
            "content": (
                "STOP searching. You have enough information. "
                "Output your final answer NOW as JSON only. "
                "No commentary, no tool calls. Use EXACTLY this format:\n"
                '{"diagnostics": [{"rang": 1, "diagnostic": "...", '
                '"probabilite": N}, {"rang": 2, ...}, ... up to rang 5]}\n'
                "The 5 probabilities must sum to 100."
            ),
        })
        try:
            final_result = _stream_round(
                messages, llm_url, model_id, mapped_opts,
                cancel_event, min(ROUND_TIMEOUT, timeout_s),
                tools=None,
            )
            if final_result.get("response", "").strip():
                final_response = final_result["response"]
                trace.total_tokens_in += final_result["tokens_in"]
                trace.total_tokens_out += final_result["tokens_out"]
                if final_result.get("thinking"):
                    all_thinking.append(
                        f"[Round final]\n{final_result['thinking']}"
                    )
            print(
                f"[ORACULUM] Round final — réponse "
                f"({len(final_response)} chars)",
                flush=True,
            )
        except _RoundTimeout as e:
            print(
                f"[ORACULUM] Round final timeout — "
                f"thinking: {len(e.thinking)} chars",
                flush=True,
            )
            if e.thinking:
                all_thinking.append(
                    f"[Round final — timeout]\n{e.thinking}"
                )
            if e.response:
                final_response = e.response
        except Exception as e:
            print(f"[ORACULUM] Round final échoué: {e}", flush=True)

    # ── Fallback : réponse coincée dans le thinking
    if not final_response.strip() or final_response.startswith("[ORACULUM:"):
        combined = "\n\n---\n\n".join(all_thinking)
        extracted = _extract_json_from_thinking(combined)
        if extracted:
            final_response = extracted
            print(
                f"[ORACULUM] Fallback JSON: réponse extraite du thinking "
                f"({len(final_response)} chars)",
                flush=True,
            )
        elif not final_response.strip():
            final_response = (
                "[ORACULUM: maximum rounds reached without final answer. "
                f"Completed {MAX_ROUNDS} rounds of tool calling.]"
            )

    # ── Résultat final (même format que stream_llm)
    combined_thinking = "\n\n---\n\n".join(all_thinking)

    return {
        "thinking": combined_thinking,
        "response": final_response,
        "tokens_in": trace.total_tokens_in,
        "tokens_out": trace.total_tokens_out,
        "metadata": {
            "prompt_tokens": trace.total_tokens_in,
            "completion_tokens": trace.total_tokens_out,
            "react_rounds": len(trace.rounds),
            "total_tool_calls": sum(
                len(r["tool_calls"]) for r in trace.rounds
            ),
            "trace": trace.rounds,
        },
    }


# ─── HELPERS POUR MAGOS ────────────────────────────────────────────────────

def is_react_job(job_row) -> bool:
    """Détecte si un job doit passer par la boucle ReAct."""
    opts = json.loads(job_row["options"] or "{}")
    return bool(opts.get("react"))
