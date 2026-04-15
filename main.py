from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
import json

load_dotenv()

from app.parser import parse_log
from app.analyzer import analyze_log
from app.summarizer import generate_triage
from app.retriever import get_retriever

app = FastAPI(
    title="Surgical Log Triage Tool",
    description=(
        "Ingest structured JSON device logs. "
        "Returns strict JSON: schema_valid, schema_issues, anomalies, "
        "severity, root_cause, summary."
    ),
    version="2.0.0",
)


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/triage")
async def triage_log(file: UploadFile = File(...)):
    """
    Upload a JSON log file (clean or messy).

    Pipeline:
      Step 1 — Schema validation  (soft, parser.py)
      Step 2 — Anomaly detection  (rule-based, analyzer.py)
      Step 3 — Retrieval          (FAISS similar-log lookup, retriever.py)
      Step 4 — LLM inference      (instruction prompt → root_cause + summary)
      Step 5 — Index update       (store this log for future retrieval)

    Always returns HTTP 200 with strict JSON output, even for invalid schemas.
    HTTP 400/422 only for unparseable JSON or wrong file type.
    """
    if not file.filename.endswith(".json"):
        raise HTTPException(status_code=400, detail="Only .json files are supported.")

    try:
        raw      = await file.read()
        log_data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=422, detail=f"Invalid JSON: {str(e)}")

    # Step 1: soft schema validation — never raises, always returns a result
    parsed = parse_log(log_data)

    # Step 2: deterministic anomaly detection
    findings = analyze_log(parsed)

    # Step 3: retrieve similar past logs for RAG context
    retriever   = get_retriever()
    query_text  = (
        f"{findings['severity']} "
        + " ".join(a["detail"] for a in findings.get("anomalies", [])[:3])
    )
    similar_logs = retriever.search(query_text, k=3)

    # Step 4: LLM generates root_cause + summary from structured findings
    triage = generate_triage(findings, similar_logs)

    # Step 5: add this log to the retrieval index for future requests
    retriever.add(triage.get("summary", ""), findings)

    return JSONResponse(content={
        # ── Strict output contract ─────────────────────────────────────────────
        "schema_valid":  parsed["schema_valid"],
        "schema_issues": parsed["schema_issues"],
        "anomalies":     findings["anomalies"],
        "severity":      findings["severity"],
        "root_cause":    triage["root_cause"],
        "summary":       triage["summary"],
        # ── Extra context (for debugging / display) ────────────────────────────
        "device_id":     parsed.get("device_id"),
        "total_events":  len(parsed.get("events", [])),
        "time_analysis": parsed.get("time_analysis", {}),
    })