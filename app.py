import os
import uuid
import re
import asyncio
import json
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google import genai
from google.genai import types

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY must be set")

client = genai.Client(api_key=GEMINI_API_KEY)
app = FastAPI(title="GEO Discoverer Backend")

# Add CORS middleware to allow direct connections from frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://geo-discoverer.vercel.app",
        "https://geo-discoverer-*.vercel.app",
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Enable Google Search grounding for real-time web access
grounding_tool = types.Tool(google_search=types.GoogleSearch())
search_config = types.GenerateContentConfig(tools=[grounding_tool])

JOBS: Dict[str, Dict[str, Any]] = {}

# --- Job Cancellation Helper ---
class JobCancelledException(Exception):
    """Raised when a job has been cancelled."""
    pass

def check_cancelled(job: Dict[str, Any]):
    """Raise exception if job was cancelled."""
    if job.get("cancelled"):
        raise JobCancelledException("Job cancelled by user")

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



def trial_payload(hidden: str, trigger: str) -> str:
    return "[SYSTEM MEMORY INJECTION] " + hidden + "\n\nUser: " + trigger + "\nAssistant: Provide a helpful answer with specific sources/URLs."

def prompt_score_explanation(brand: str, score: int, level: str, paths: List[Dict[str, Any]]) -> str:
    summary = "\n".join([f"- {p['persona_name']}: {p['win_rate']}% win rate" for p in paths])
    return (
        f"The brand '{brand}' received a Visibility Score of {score}/100 ({level}).\n"
        f"Performance Summary:\n{summary}\n\n"
        "Write a brief, 2-sentence strategic explanation of this score. "
        "Highlight exactly which personas/intents are driving visibility and which are failing. "
        "Be direct, professional, and analytical. No fluff."
    )
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

def prompt_scoring_tribunal(brand: str, paths: List[Dict[str, Any]]) -> str:
    """Generate prompt for LLM to evaluate realism and reach of each persona."""
    personas_text = "\n".join([
        f"Persona {i+1}:\n"
        f"  - Name: {p['persona_name']}\n"
        f"  - Trigger Prompt: \"{p['trigger_prompt']}\"\n"
        f"  - Win Rate: {p.get('win_rate', 0)}%"
        for i, p in enumerate(paths)
    ])
    
    return f"""You are an expert in AI search behavior and market research evaluating brand visibility.

Brand being evaluated: "{brand}"

Here are 5 user personas that were tested:

{personas_text}

For EACH persona, evaluate two dimensions:

1. **Prompt Realism** (0.0 - 1.0): How likely is a real human to type this exact query into ChatGPT/Claude/Gemini?
   - 1.0 = Very natural, common query (e.g., "best crm for small business")
   - 0.7 = Natural but specific query
   - 0.5 = Somewhat realistic but overly detailed
   - 0.2 = Robotic, overly specific, or unnatural phrasing that no human would type
   
2. **Persona Reach** (0.0 - 1.0): How large is this user segment in the real world?
   - 1.0 = Universal/Mass Market (e.g., "Student", "Small Business Owner", "Job Seeker", "Parent")
   - 0.7 = Broad Vertical (e.g., "SaaS CTO", "Marketing Freelancer", "E-commerce Seller")
   - 0.4 = Niche/Long Tail (e.g., "Cobol Mainframe Maintainer", "Dental Practice CFO")

Return a JSON array with exactly 5 objects (one per persona, in the same order):
[
  {{
    "persona_name": "exact name from input",
    "prompt_realism": 0.0-1.0,
    "realism_reasoning": "One sentence explanation",
    "persona_reach": 0.0-1.0,
    "reach_category": "Universal" or "Broad Vertical" or "Niche",
    "reach_reasoning": "One sentence explanation"
  }}
]

Be strict and objective. A query asking for "sources" or "URLs" is still realistic if the core question is natural.
Output valid JSON only, no markdown."""

# --- Stages ---

# Timeout for Gemini API calls (in seconds)
GEMINI_TIMEOUT = 90.0

async def gen_high_search(prompt: str) -> str:
    """Generate content with Google Search grounding. 90s timeout. Uses gemini-2.0-flash for stability."""
    try:
        resp = await asyncio.wait_for(
            asyncio.to_thread(
                client.models.generate_content,
                model="gemini-2.0-flash",
                contents=prompt,
                config=search_config,
            ),
            timeout=GEMINI_TIMEOUT
        )
        return resp.text or ""
    except asyncio.TimeoutError:
        raise RuntimeError(f"Gemini API timed out after {GEMINI_TIMEOUT}s (gen_high_search)")

async def gen_low(prompt: str) -> str:
    """Generate content for trial simulations. 90s timeout. Uses gemini-2.0-flash for stability."""
    try:
        resp = await asyncio.wait_for(
            asyncio.to_thread(
                client.models.generate_content,
                model="gemini-2.0-flash",
                contents=prompt,
                config=search_config,
            ),
            timeout=GEMINI_TIMEOUT
        )
        return resp.text or ""
    except asyncio.TimeoutError:
        raise RuntimeError(f"Gemini API timed out after {GEMINI_TIMEOUT}s (gen_low)")

async def gen_medium(prompt: str) -> str:
    """Generate content for analysis. 90s timeout. Uses gemini-2.0-flash for stability."""
    try:
        resp = await asyncio.wait_for(
            asyncio.to_thread(
                client.models.generate_content,
                model="gemini-2.0-flash",
                contents=prompt,
                config=search_config,
            ),
            timeout=GEMINI_TIMEOUT
        )
        return resp.text or ""
    except asyncio.TimeoutError:
        raise RuntimeError(f"Gemini API timed out after {GEMINI_TIMEOUT}s (gen_medium)")

async def gen_judge(prompt: str) -> str:
    """Use gemini-2.5-flash for fast, accurate scoring judgments in the tribunal. 90s timeout."""
    try:
        resp = await asyncio.wait_for(
            asyncio.to_thread(
                client.models.generate_content,
                model="gemini-2.5-flash-preview-05-20",
                contents=prompt,
            ),
            timeout=GEMINI_TIMEOUT
        )
        return resp.text or ""
    except asyncio.TimeoutError:
        raise RuntimeError(f"Gemini API timed out after {GEMINI_TIMEOUT}s (gen_judge)")

def calculate_discovery_score(paths: List[Dict[str, Any]], use_tribunal: bool = False) -> Dict[str, Any]:
    """Calculate discovery score using LLM-evaluated realism and reach (if available)."""
    total_score = 0.0
    breakdown = []
    
    for path in paths:
        win_rate = path.get("win_rate", 0) / 100.0  # Normalize to 0-1
        
        if use_tribunal and "prompt_realism" in path:
            # New formula: Win Rate × Realism × Reach × 20
            realism = path.get("prompt_realism", 0.5)
            reach = path.get("persona_reach", 0.5)
            reach_category = path.get("reach_category", "Broad Vertical")
            realism_reasoning = path.get("realism_reasoning", "")
            reach_reasoning = path.get("reach_reasoning", "")
            
            points = win_rate * realism * reach * 20.0
            
            breakdown.append({
                "persona": path.get("persona_name"),
                "win_rate": path.get("win_rate", 0),
                "prompt_realism": realism,
                "realism_reasoning": realism_reasoning,
                "persona_reach": reach,
                "reach_category": reach_category,
                "reach_reasoning": reach_reasoning,
                "points": round(points, 1)
            })
        else:
            # Fallback to old word-count based complexity
            trigger = path.get("trigger_prompt", "")
            word_count = len(trigger.split())

            if word_count <= 5:
                complexity = 1.0; c_label = "Simple"
            elif word_count <= 10:
                complexity = 0.8; c_label = "Medium"
            else:
                complexity = 0.5; c_label = "Complex"

            points = win_rate * complexity * 20.0
            
            breakdown.append({
                "persona": path.get("persona_name"),
                "win_rate": path.get("win_rate", 0),
                "prompt_realism": complexity,  # Use complexity as fallback
                "realism_reasoning": f"Word count: {word_count} ({c_label})",
                "persona_reach": 0.7,  # Default to Broad Vertical
                "reach_category": "Broad Vertical",
                "reach_reasoning": "Default estimate (tribunal not run)",
                "points": round(points, 1)
            })
        
        total_score += points

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
    check_cancelled(job)  # Check before starting
    job["logs"].append("Stage 1: Reverse-Engineering 5 Potential User Personas...")
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
    job["logs"].append(f"Discovery Complete. Found {len(cands)} candidate personas.")

async def phase_trials(brand: str, job: Dict[str, Any], trials: int, concurrency: int):
    check_cancelled(job)  # Check before starting
    job["logs"].append("Stage 2: Running 25 Parallel tests...")
    sem = asyncio.Semaphore(concurrency)

    async def one(cidx: int, tidx: int, hidden: str, trigger: str) -> Dict[str, Any]:
        async with sem:
            check_cancelled(job)  # Check before each API call
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
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    # Filter out cancelled/failed results (exceptions become the result when return_exceptions=True)
    job["trial_results"] = [r for r in results if isinstance(r, dict)]
    
    check_cancelled(job)  # Check after completion
    job["logs"].append("Tests Completed.")

async def phase_aggregate(brand: str, job: Dict[str, Any]):
    check_cancelled(job)  # Check before starting
    job["logs"].append("Stage 3: Analyzing Results with LLM Analyst...")
    job["status"] = "analyzing"  # Intermediate status for progressive loading
    
    buckets: Dict[int, List[str]] = {}
    for r in job["trial_results"]:
        buckets.setdefault(r["cand_idx"], []).append(r["llm_result_text"])

    # Initialize paths with pending state for progressive display
    job["paths"] = []
    for idx, cand in enumerate(job["candidates"]):
        job["paths"].append({
            "persona_name": cand["persona_name"],
            "trigger_prompt": cand["trigger_prompt"],
            "frequency_count": 0,
            "frequency_total": 5,
            "win_rate": 0,
            "attribution": "Analyzing...",
            "competitors": [],
            "sources": [],
            "trial_responses": buckets.get(idx, []),
            "analysis_complete": False,
            "debug": {
                "hidden_memory": cand["hidden_memory"],
                "trigger_prompt": cand["trigger_prompt"],
            },
        })

    # Analyze each persona and update paths progressively
    async def analyze_one(idx: int, responses: List[str]):
        check_cancelled(job)  # Check before each analysis
        cand = job["candidates"][idx]
        job["logs"].append(f"Analyzing Persona {idx+1}: {cand['persona_name']}...")

        try:
            analysis_raw = await gen_medium(prompt_analyst(brand, cand["persona_name"], responses))
            m = re.search(r"\{.*\}", analysis_raw, re.S)
            if m:
                analysis = json.loads(m.group(0))
            else:
                analysis = {"win_rate": 0, "competitors": [], "sources": [], "insight": "Analysis failed"}
        except JobCancelledException:
            raise  # Re-raise cancellation
        except Exception as e:
            analysis = {"win_rate": 0, "competitors": [], "sources": [], "insight": f"Error: {str(e)}"}

        # Update the path in place for progressive updates
        job["paths"][idx].update({
            "frequency_count": int(analysis.get("win_rate", 0) / 20),
            "win_rate": analysis.get("win_rate", 0),
            "attribution": analysis.get("insight", ""),
            "competitors": analysis.get("competitors", [])[:6],
            "sources": analysis.get("sources", []),
            "analysis_complete": True,
        })
        
        job["logs"].append(f"✓ Persona {idx+1} Complete: {analysis.get('win_rate', 0)}% win rate")

    # Run analyses in parallel but update progressively
    tasks = [analyze_one(idx, responses) for idx, responses in buckets.items()]
    await asyncio.gather(*tasks)

    # Sort paths by win_rate after all complete
    job["paths"] = sorted(job["paths"], key=lambda p: p["win_rate"], reverse=True)

    if job["paths"]:
        top = job["paths"][0]
        job["top_persona"] = top["persona_name"]
        job["highest_win_rate"] = f"{top['win_rate']}%"
        job["top_competitor"] = (top["competitors"][0] if top["competitors"] else "")
    else:
        job["top_persona"] = "None"
        job["highest_win_rate"] = "0%"
        job["top_competitor"] = ""

    job["logs"].append("Stage 3 Complete. Proceeding to Scoring Tribunal...")

async def phase_tribunal(brand: str, job: Dict[str, Any]):
    """Stage 4: LLM Scoring Tribunal - evaluate prompt realism and persona reach."""
    check_cancelled(job)
    job["logs"].append("Stage 4: Scoring Tribunal evaluating prompt quality...")
    
    tribunal_success = False
    try:
        raw = await gen_judge(prompt_scoring_tribunal(brand, job["paths"]))
        m = re.search(r"\[.*\]", raw, re.S)
        if m:
            evaluations = json.loads(m.group(0))
            
            # Update paths with tribunal scores (match by index since order is preserved)
            for i, eval_data in enumerate(evaluations):
                if i < len(job["paths"]):
                    job["paths"][i]["prompt_realism"] = float(eval_data.get("prompt_realism", 0.5))
                    job["paths"][i]["realism_reasoning"] = eval_data.get("realism_reasoning", "")
                    job["paths"][i]["persona_reach"] = float(eval_data.get("persona_reach", 0.5))
                    job["paths"][i]["reach_category"] = eval_data.get("reach_category", "Broad Vertical")
                    job["paths"][i]["reach_reasoning"] = eval_data.get("reach_reasoning", "")
                    
                    job["logs"].append(f"✓ {job['paths'][i]['persona_name']}: Realism={eval_data.get('prompt_realism', 0.5)}, Reach={eval_data.get('reach_category', 'N/A')}")
            
            tribunal_success = True
            job["logs"].append("Tribunal evaluation complete.")
        else:
            raise RuntimeError("Tribunal did not return valid JSON array")
            
    except JobCancelledException:
        raise
    except Exception as e:
        # Fallback to default values if tribunal fails
        job["logs"].append(f"! Tribunal error: {str(e)}. Using fallback scoring.")
        for p in job["paths"]:
            p["prompt_realism"] = 0.5
            p["realism_reasoning"] = "Tribunal unavailable - using default"
            p["persona_reach"] = 0.7
            p["reach_category"] = "Broad Vertical"
            p["reach_reasoning"] = "Tribunal unavailable - using default"
    
    # Calculate Discovery Score with tribunal data
    score_data = calculate_discovery_score(job["paths"], use_tribunal=tribunal_success)
    job["discovery_score"] = score_data["score"]
    job["score_level"] = score_data["level"]
    job["score_breakdown"] = score_data["breakdown"]

    # Call AI for explanation
    job["logs"].append("Generating Strategic Breakdown...")
    try:
        expl = await gen_medium(prompt_score_explanation(brand, job["discovery_score"], job["score_level"], job["paths"]))
        job["score_explanation"] = expl.strip()
    except Exception:
        job["score_explanation"] = "Analysis unavailable."

    job["logs"].append(f"Discovery Score: {score_data['score']}/100 ({score_data['level']})")
    job["logs"].append("Report Ready.")
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
        "cancelled": False,  # Cancellation flag
        "current_stage": "",  # Track which stage is running
        "logs": [],
        "candidates": [],
        "trial_results": [],
        "paths": [],
    }
    async def run():
        try:
            JOBS[job_id]["current_stage"] = "Stage 1: Persona Discovery"
            await phase_backprop(brand, JOBS[job_id]["website_url"], JOBS[job_id])
            
            JOBS[job_id]["current_stage"] = "Stage 2: Running Tests"
            trials = int(os.environ.get("TRIALS_PER_CANDIDATE", "5"))
            concurrency = int(os.environ.get("CONCURRENCY", "25"))
            await phase_trials(brand, JOBS[job_id], trials, concurrency)
            
            JOBS[job_id]["current_stage"] = "Stage 3: Analyzing Results"
            await phase_aggregate(brand, JOBS[job_id])
            
            JOBS[job_id]["current_stage"] = "Stage 4: Scoring Tribunal"
            await phase_tribunal(brand, JOBS[job_id])  # Stage 4: LLM Scoring Tribunal
            
        except JobCancelledException:
            # Job was cancelled - status already set by cancel endpoint
            JOBS[job_id]["logs"].append("> Job stopped (cancelled)")
        except Exception as e:
            stage = JOBS[job_id].get("current_stage", "Unknown Stage")
            JOBS[job_id]["status"] = "failed"
            JOBS[job_id]["failed_stage"] = stage
            JOBS[job_id]["logs"].append(f"! FAILED at {stage}: {str(e)}")
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
        heartbeat_counter = 0
        while True:
            logs = job["logs"]
            while i < len(logs):
                yield f"data: {logs[i]}\n\n"
                i += 1
                heartbeat_counter = 0  # Reset on real message
            if job["status"] in ("completed", "failed", "cancelled"):
                yield f"data: > {job['status'].upper()}\n\n"
                break
            
            # Send heartbeat every 5 seconds to keep connection alive (25 * 0.2s = 5s)
            heartbeat_counter += 1
            if heartbeat_counter >= 25:
                yield f"data: > ...\n\n"
                heartbeat_counter = 0
                
            await asyncio.sleep(0.2)
    return StreamingResponse(gen(), media_type="text/event-stream")

@app.get("/discover/jobs/{job_id}/partial")
async def job_partial(job_id: str):
    """Return partial results as they become available for progressive loading."""
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "job not found")
    
    # Return current state regardless of completion
    return {
        "status": job["status"],
        "brand_name": job["brand_name"],
        "candidates": job.get("candidates", []),
        "paths": [{
            "persona_name": p["persona_name"],
            "trigger_prompt": p["trigger_prompt"],
            "frequency_count": p.get("frequency_count", 0),
            "frequency_total": p.get("frequency_total", 5),
            "win_rate": p.get("win_rate", 0),
            "attribution": p.get("attribution", ""),
            "competitors": p.get("competitors", []),
            "sources": p.get("sources", []),
            "trial_responses": p.get("trial_responses", []),
            "analysis_complete": p.get("analysis_complete", False)
        } for p in job.get("paths", [])],
        "top_persona": job.get("top_persona", ""),
        "highest_win_rate": job.get("highest_win_rate", ""),
        "top_competitor": job.get("top_competitor", ""),
        "discovery_score": job.get("discovery_score"),
        "score_level": job.get("score_level"),
        "score_explanation": job.get("score_explanation"),
        "score_breakdown": job.get("score_breakdown", []),
    }

@app.post("/discover/jobs/{job_id}/cancel")
async def cancel_job(job_id: str):
    """Cancel a running job. Returns success even if job doesn't exist (idempotent)."""
    job = JOBS.get(job_id)
    if not job:
        # Job doesn't exist (already completed, failed, or server restarted) - that's OK
        return {"status": "not_found", "job_id": job_id, "message": "Job not found (may have already completed)"}
    job["cancelled"] = True
    job["status"] = "cancelled"
    job["logs"].append("Job cancelled by user")
    return {"status": "cancelled", "job_id": job_id}

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