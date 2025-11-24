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
from google.genai import types

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY must be set")

client = genai.Client(api_key=GEMINI_API_KEY)
app = FastAPI(title="GEO Discoverer Backend")

# Enable Google Search grounding for real-time web access
grounding_tool = types.Tool(google_search=types.GoogleSearch())
search_config = types.GenerateContentConfig(tools=[grounding_tool])

JOBS: Dict[str, Dict[str, Any]] = {}

class CreateJobRequest(BaseModel):
    brand_name: str
    website_url: Optional[str] = None

def n(s: Optional[str]) -> str:
    return (s or "").strip()

# --- Prompts ---

def prompt_backprop(brand: str, site: Optional[str]) -> str:
    s = f" Website: {site}." if site else ""
    return (
        f'Analyze the brand "{brand}".{s}\n'
        "Identify 5 distinct User Personas with **diverse search intents** where this brand should appear.\n"
        "You MUST include a mix of these intent types:\n"
        "1. **Problem Solving**: Users describing a specific pain point the brand solves (without naming the category).\n"
        "2. **Competitor Alternatives**: Users explicitly looking for alternatives to a specific rival.\n"
        "3. **Feature Hunting**: Users searching for a specific niche feature this brand is known for.\n"
        "4. **Direct Recommendation**: Users asking for the best tool in this specific vertical.\n\n"
        "**CRITICAL CONSTRAINT**: The `trigger_prompt` must be **realistic** and **natural** (something a real human would actually type). "
        "Do not create artificially long or complex prompts just to force a win. Find the **simplest** prompt that still leads to the brand.\n\n"
        "Return a JSON array of 5 objects with keys:\n"
        "- `persona_name`: Descriptive title.\n"
        "- `hidden_memory`: Context to inject into system memory (e.g., user role, constraints).\n"
        "- `trigger_prompt`: The specific user question. **Must explicitly ask for sources/URLs.**\n"
        "Output JSON only."
    )

def prompt_score_explanation(brand: str, score: int, level: str, paths: List[Dict[str, Any]]) -> str:
    summary = "\n".join([f"- {p['persona_name']}: {p['win_rate']}% win rate" for p in paths])
    return (
        f"The brand '{brand}' received a Visibility Score of {score}/100 ({level}).\n"
        f"Performance Summary:\n{summary}\n\n"
        "Write a brief, 2-sentence strategic explanation of this score. "
        "Highlight exactly which personas/intents are driving visibility and which are failing. "
        "Be direct, professional, and analytical. No fluff."
    )

def trial_payload(hidden: str, trigger: str) -> str:
    return "[SYSTEM MEMORY INJECTION] " + hidden + "\n\nUser: " + trigger + "\nAssistant: Provide a helpful answer with specific sources/URLs."

def prompt_analyst(brand: str, persona: str, responses: List[str]) -> str:
    combined_responses = "\n\n---\n\n".join([f"Response {i+1}:\n{r}" for i, r in enumerate(responses)])
    return (
        f"Here are 5 LLM responses for the persona '{persona}'. Target Brand: '{brand}'.\n"
        f"Analyze these 5 responses and return a JSON summary with keys:\n"
        "1. `win_rate`: Integer percentage (0-100) of times the target brand was recommended #1.\n"
        "2. `competitors`: List of other brands mentioned.\n"
        "3. `sources`: List of all URLs cited.\n"
        "4. `insight`: A one-sentence finding about why the brand won or lost (focus on the prompt/context fit, not just comparison).\n\n"
        f"Responses:\n{combined_responses}\n\n"
        "Output JSON only."
    )

# --- Stages ---

async def gen_high_search(prompt: str) -> str:
    resp = await asyncio.to_thread(
        client.models.generate_content,
        model="gemini-3-pro-preview",
        contents=prompt,
        config=search_config,
    )
    return resp.text or ""

async def gen_low(prompt: str) -> str:
    resp = await asyncio.to_thread(
        client.models.generate_content,
        model="gemini-3-pro-preview",
        contents=prompt,
        config=search_config,
    )
    return resp.text or ""

async def gen_medium(prompt: str) -> str:
    resp = await asyncio.to_thread(
        client.models.generate_content,
        model="gemini-3-pro-preview",
        contents=prompt,
        config=search_config,
    )
    return resp.text or ""

def calculate_discovery_score(paths: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate composite discovery score based on prompt complexity and win rates."""
    total_score = 0.0
    breakdown = []
    
    for path in paths:
        # Factor 1: Prompt complexity score
        trigger = path.get("trigger_prompt", "")
        word_count = len(trigger.split())

        if word_count <= 5:
            complexity = 1.0; c_label = "Simple"  # Simple, natural queries
        elif word_count <= 10:
            complexity = 0.8; c_label = "Medium" # Adjusted from 0.6 for better balance
        else:
            complexity = 0.5; c_label = "Complex" # Adjusted from 0.3

        # Factor 2: Win rate (normalize to 0-1)
        win_rate = path.get("win_rate", 0)

        # Max 20 points per persona
        # Points = (Win Rate %) * Complexity * 20
        points = (win_rate / 100.0) * complexity * 20.0
        total_score += points
        
        breakdown.append({
            "persona": path.get("persona_name"),
            "win_rate": win_rate,
            "complexity_label": c_label,
            "complexity_multiplier": complexity,
            "points": round(points, 1)
        })

    # Scale to 0-100
    discovery_score = int(total_score)

    # Determine level
    if discovery_score >= 61:
        level = "HIGH"
    elif discovery_score >= 31:
        level = "MEDIUM"
    else:
        level = "LOW"

    return {
        "score": discovery_score,
        "level": level,
        "breakdown": breakdown
    }

async def phase_backprop(brand: str, site: Optional[str], job: Dict[str, Any]):
    job["logs"].append("> Stage 1: Reverse-Engineering 5 Potential User Personas...")
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
        })
    job["candidates"] = cands
    job["logs"].append(f"> Discovery Complete. Found {len(cands)} candidate personas.")

async def phase_trials(brand: str, job: Dict[str, Any], trials: int, concurrency: int):
    job["logs"].append("> Stage 2: Running 25 Parallel Simulations...")
    sem = asyncio.Semaphore(concurrency)

    async def one(cidx: int, tidx: int, hidden: str, trigger: str) -> Dict[str, Any]:
        async with sem:
            text = await gen_low(trial_payload(hidden, trigger))
            return {
                "cand_idx": cidx,
                "trial_idx": tidx,
                "llm_result_text": text,
            }

    tasks = []
    for i, c in enumerate(job["candidates"]):
        for t in range(trials):
            tasks.append(one(i, t, c["hidden_memory"], c["trigger_prompt"]))
    
    job["trial_results"] = await asyncio.gather(*tasks)
    job["logs"].append("> Simulations Completed.")

async def phase_aggregate(brand: str, job: Dict[str, Any]):
    job["logs"].append("> Stage 3: Analyzing Results with LLM Analyst...")
    buckets: Dict[int, List[str]] = {}
    for r in job["trial_results"]:
        buckets.setdefault(r["cand_idx"], []).append(r["llm_result_text"])

    # Analyze all groups in parallel for faster processing
    async def analyze_one(idx: int, responses: List[str]) -> Dict[str, Any]:
        cand = job["candidates"][idx]
        job["logs"].append(f"> Analyzing Group {idx+1}: {cand['persona_name']}...")

        try:
            analysis_raw = await gen_medium(prompt_analyst(brand, cand["persona_name"], responses))
            m = re.search(r"\{.*\}", analysis_raw, re.S)
            if m:
                analysis = json.loads(m.group(0))
            else:
                analysis = {"win_rate": 0, "competitors": [], "sources": [], "insight": "Analysis failed"}
        except Exception as e:
             analysis = {"win_rate": 0, "competitors": [], "sources": [], "insight": f"Error: {str(e)}"}

        return {
            "persona_name": cand["persona_name"],
            "trigger_prompt": cand["trigger_prompt"],
            "frequency_count": int(analysis.get("win_rate", 0) / 20),
            "frequency_total": 5,
            "win_rate": analysis.get("win_rate", 0),
            "attribution": analysis.get("insight", ""),
            "competitors": analysis.get("competitors", [])[:6],
            "sources": analysis.get("sources", []),
            "trial_responses": responses,  # Include all trial responses
            "debug": {
                "hidden_memory": cand["hidden_memory"],
                "trigger_prompt": cand["trigger_prompt"],
            },
        }

    tasks = [analyze_one(idx, responses) for idx, responses in buckets.items()]
    paths = await asyncio.gather(*tasks)

    paths = sorted(paths, key=lambda p: p["win_rate"], reverse=True)
    job["paths"] = paths

    if paths:
        top = paths[0]
        job["top_persona"] = top["persona_name"]
        job["highest_win_rate"] = f"{top['win_rate']}%"
        job["top_competitor"] = (top["competitors"][0] if top["competitors"] else "")
    else:
        job["top_persona"] = "None"
        job["highest_win_rate"] = "0%"
        job["top_competitor"] = ""

    # Calculate Discovery Score
    score_data = calculate_discovery_score(paths)
    job["discovery_score"] = score_data["score"]
    job["score_level"] = score_data["level"]
    job["score_breakdown"] = score_data["breakdown"] # Store breakdown

    # Call AI for explanation
    job["logs"].append("> Generating Strategic Breakdown...")
    try:
        expl = await gen_medium(prompt_score_explanation(brand, job["discovery_score"], job["score_level"], paths))
        job["score_explanation"] = expl.strip()
    except Exception:
        job["score_explanation"] = "Analysis unavailable."

    job["logs"].append(f"> Discovery Score: {score_data['score']}/100 ({score_data['level']})")

    job["logs"].append("> Report Ready.")
    job["status"] = "completed"

# --- API ---

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
            await phase_backprop(brand, JOBS[job_id]["website_url"], JOBS[job_id])
            
            trials = int(os.environ.get("TRIALS_PER_CANDIDATE", "5"))
            concurrency = int(os.environ.get("CONCURRENCY", "25"))
            await phase_trials(brand, JOBS[job_id], trials, concurrency)
            
            await phase_aggregate(brand, JOBS[job_id])
            
        except Exception as e:
            JOBS[job_id]["status"] = "failed"
            JOBS[job_id]["logs"].append(f"! Error: {str(e)}")
            import traceback
            traceback.print_exc()
            
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
        "discovery_score": job.get("discovery_score", 0),
        "score_level": job.get("score_level", "LOW"),
        "score_explanation": job.get("score_explanation", ""),
        "score_breakdown": job.get("score_breakdown", []),
        "paths": [{
            "persona_name": p["persona_name"],
            "trigger_prompt": p["trigger_prompt"],
            "frequency_count": p["frequency_count"],
            "frequency_total": p["frequency_total"],
            "attribution": p["attribution"],
            "competitors": p["competitors"],
            "sources": p.get("sources", []),
            "trial_responses": p.get("trial_responses", [])
        } for p in job["paths"]],
        "debug": [p["debug"] for p in job["paths"]],
        "status": job["status"],
    }