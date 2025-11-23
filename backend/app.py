import os
import uuid
import re
import asyncio
import json
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
from google import genai
import spoon_adapter as spoon

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY must be set")

client = genai.Client(api_key=GEMINI_API_KEY)
app = FastAPI(title="GEO Discoverer Backend")

JOBS: Dict[str, Dict[str, Any]] = {}

class CreateJobRequest(BaseModel):
    brand_name: str
    website_url: Optional[str] = None

def n(s: Optional[str]) -> str:
    return (s or "").strip()

def hit(text: str, brand: str) -> bool:
    return brand.lower() in (text or "").lower()

def competitors(text: str, brand: str) -> List[str]:
    words = re.findall(r"\b[A-Z][A-Za-z0-9&\-]{2,}\b", text or "")
    out: List[str] = []
    for w in words:
        if w.lower() != brand.lower() and w not in out:
            out.append(w)
    return out[:6]

async def gen_high_search(prompt: str) -> str:
    resp = await client.models.generate_content.async_call(
        model="gemini-3-pro-preview",
        tools=[{"google_search": {}}],
        contents=prompt,
        thinking={"level": "high"},
    )
    return resp.text or ""

async def gen_low(prompt: str) -> str:
    resp = await client.models.generate_content.async_call(
        model="gemini-3-pro-preview",
        contents=prompt,
        thinking={"level": "low"},
    )
    return resp.text or ""

async def gen_medium_search(prompt: str) -> str:
    resp = await client.models.generate_content.async_call(
        model="gemini-3-pro-preview",
        tools=[{"google_search": {}}],
        contents=prompt,
        thinking={"level": "medium"},
    )
    return resp.text or ""

def prompt_backprop(brand: str, site: Optional[str]) -> str:
    s = f" Website: {site}." if site else ""
    return (
        f'You are a "Reverse-Engineering Analyst" for Large Language Models. Analyze the brand "{brand}".'
        f"{s}\n"
        "Identify 5 distinct User Intent Vectors (Personas) where this brand would statistically be the #1 recommendation.\n"
        "Return a JSON array of 5 objects with keys: persona_name, hidden_memory, trigger_prompt, hypothesis.\n"
        "Ground in current web context. Output JSON only."
    )

def prompt_reasoning(brand: str, persona: str, trigger: str) -> str:
    return (
        f"Explain briefly WHY {brand} was recommended for persona '{persona}' with user trigger '{trigger}'. "
        "Use live web context and include concise citations."
    )

def trial_payload(hidden: str, trigger: str) -> str:
    return "[SYSTEM MEMORY INJECTION] " + hidden + "\n\nUser: " + trigger + "\nAssistant: Provide a helpful answer."

async def phase_backprop(brand: str, site: Optional[str], job: Dict[str, Any]):
    job["logs"].append("> Reverse-Engineering 5 Potential User Personas...")
    raw = await gen_high_search(prompt_backprop(brand, site))
    m = re.search(r"\[.*\]", raw, re.S)
    if not m:
        raise RuntimeError("Back-Propagation did not return JSON array")
    arr = json.loads(m.group(0))
    cands: List[Dict[str, str]] = []
    for c in arr[:5]:
        cands.append({
            "persona_name": n(c.get("persona_name")),
            "hidden_memory": n(c.get("hidden_memory")),
            "trigger_prompt": n(c.get("trigger_prompt")),
            "hypothesis": n(c.get("hypothesis")),
        })
    job["candidates"] = cands
    job["logs"].append("> Personas generated.")

async def phase_trials(brand: str, job: Dict[str, Any], trials: int, concurrency: int):
    job["logs"].append("> Generating Ghost Payloads (Memory Injection)...")
    sem = asyncio.Semaphore(concurrency)

    async def one(cidx: int, tidx: int, hidden: str, trigger: str) -> Dict[str, Any]:
        async with sem:
            text = await gen_low(trial_payload(hidden, trigger))
            return {
                "cand_idx": cidx,
                "trial_idx": tidx,
                "hit": hit(text, brand),
                "intro_text": text[:500],
                "competitors": competitors(text, brand),
            }

    tasks = []
    for i, c in enumerate(job["candidates"]):
        for t in range(trials):
            tasks.append(one(i, t, c["hidden_memory"], c["trigger_prompt"]))
    job["logs"].append(f"> Executing {len(tasks)} Parallel Simulations...")
    job["trial_results"] = await asyncio.gather(*tasks)
    job["logs"].append("> Trials completed.")

async def phase_aggregate(brand: str, job: Dict[str, Any]):
    job["logs"].append("> Analyzing Attribution & Competitor Co-occurrence...")
    buckets: Dict[int, List[Dict[str, Any]]] = {}
    for r in job["trial_results"]:
        buckets.setdefault(r["cand_idx"], []).append(r)
    paths: List[Dict[str, Any]] = []
    for idx, trials in buckets.items():
        cand = job["candidates"][idx]
        freq = sum(1 for r in trials if r["hit"]) 
        comps: List[str] = []
        for r in trials:
            for w in r["competitors"]:
                if w not in comps and w.lower() != brand.lower():
                    comps.append(w)
        why = await gen_medium_search(prompt_reasoning(brand, cand["persona_name"], cand["trigger_prompt"]))
        paths.append({
            "persona_name": cand["persona_name"],
            "trigger_prompt": cand["trigger_prompt"],
            "frequency_count": freq,
            "frequency_total": len(trials),
            "attribution": why[:600],
            "competitors": comps[:6],
            "debug": {
                "hidden_memory": cand["hidden_memory"],
                "trigger_prompt": cand["trigger_prompt"],
            },
        })
    paths.sort(key=lambda p: p["frequency_count"], reverse=True)
    job["paths"] = paths
    job["top_persona"] = paths[0]["persona_name"] if paths else ""
    rate = 0
    if paths:
        rate = round(100 * paths[0]["frequency_count"] / paths[0]["frequency_total"]) 
    job["highest_win_rate"] = f"{rate}%"
    job["top_competitor"] = (paths[0]["competitors"][0] if paths and paths[0]["competitors"] else "")
    job["logs"].append("> Report ready.")
    job["status"] = "completed"

@app.post("/discover/jobs")
async def create_job(req: CreateJobRequest, bg: BackgroundTasks):
    brand = n(req.brand_name)
    if not brand:
        raise HTTPException(400, "brand_name required")
    job_id = uuid.uuid4().hex
    JOBS[job_id] = {
        "id": job_id,
        "brand_name": brand,
        "website_url": n(req.website_url),
        "status": "running",
        "logs": [],
        "candidates": [],
        "trial_results": [],
        "paths": [],
    }
    async def run():
        try:
            if spoon.spoon_enabled():
                JOBS[job_id]["logs"].append("> SpoonOS orchestrator enabled")
            await phase_backprop(brand, JOBS[job_id]["website_url"], JOBS[job_id])
            trials = int(os.environ.get("TRIALS_PER_CANDIDATE", "5"))
            concurrency = int(os.environ.get("CONCURRENCY", "25"))
            await phase_trials(brand, JOBS[job_id], trials, concurrency)
            await phase_aggregate(brand, JOBS[job_id])
        except Exception as e:
            JOBS[job_id]["status"] = "failed"
            JOBS[job_id]["logs"].append(f"! Error: {str(e)}")
    bg.add_task(run)
    return {"job_id": job_id}

@app.get("/discover/jobs/{job_id}/stream")
async def stream_job(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "job not found")
    async def gen():
        i = 0
        while True:
            logs = job["logs"]
            while i < len(logs):
                yield f"data: {logs[i]}\n\n"
                i += 1
            if job["status"] in ("completed", "failed"):
                yield f"data: > {job['status'].upper()}\n\n"
                break
            await asyncio.sleep(0.2)
    return StreamingResponse(gen(), media_type="text/event-stream")

@app.get("/discover/jobs/{job_id}/result")
async def job_result(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "job not found")
    if job["status"] != "completed":
        return JSONResponse({"status": job["status"], "paths": []})
    return {
        "brand_name": job["brand_name"],
        "top_persona": job["top_persona"],
        "highest_win_rate": job["highest_win_rate"],
        "top_competitor": job["top_competitor"],
        "paths": [{
            "persona_name": p["persona_name"],
            "trigger_prompt": p["trigger_prompt"],
            "frequency_count": p["frequency_count"],
            "frequency_total": p["frequency_total"],
            "attribution": p["attribution"],
            "competitors": p["competitors"],
        } for p in job["paths"]],
        "debug": [p["debug"] for p in job["paths"]],
        "status": job["status"],
    }