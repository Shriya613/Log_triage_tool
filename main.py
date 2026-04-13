from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
import json

load_dotenv()

from app.parser import parse_log
from app.analyzer import analyze_log
from app.summarizer import summarize_findings

app = FastAPI(
    title="Surgical Log Triage Tool",
    description="Ingest structured JSON device logs, detect anomalies, and generate AI triage summaries.",
    version="1.0.0"
)


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/triage")
async def triage_log(file: UploadFile = File(...)):
    """
    Upload a JSON log file. Returns validated findings and an AI-generated triage summary.
    """
    # 1. Read and decode uploaded file
    if not file.filename.endswith(".json"):
        raise HTTPException(status_code=400, detail="Only .json files are supported.")

    try:
        raw = await file.read()
        log_data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=422, detail=f"Invalid JSON: {str(e)}")

    # 2. Parse and validate schema
    try:
        parsed = parse_log(log_data)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=f"Schema validation error: {str(e)}")

    # 3. Analyze for anomalies and patterns
    findings = analyze_log(parsed)

    # 4. Generate LLM triage summary
    summary = summarize_findings(findings)

    return JSONResponse(content={
        "device_id": parsed.get("device_id"),
        "log_timestamp": parsed.get("timestamp"),
        "total_events": len(parsed.get("events", [])),
        "findings": findings,
        "triage_summary": summary
    })
