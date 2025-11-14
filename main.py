# main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os
import openai
from typing import Dict

app = FastAPI()

# Enable CORS for demo (restrict later)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Environment variables (set these in Render or your host)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")

# Initialize OpenAI (do not hardcode keys)
openai.api_key = OPENAI_API_KEY

def call_openai_system(user_prompt: str, model: str = "gpt-4o-mini") -> str:
    """Call OpenAI chat completion and return text response."""
    resp = openai.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are Astra — a concise assistant that returns useful output for product and strategy generation."},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.2,
        max_tokens=1200
    )
    return resp.choices[0].message.content

async def _generate_for_mode(niche: str, mode: str, extra: Dict = None):
    extra = extra or {}
    if mode == "problem_generation":
        prompt = f"Generate 10 monetizable problems, solutions, and monetization strategies in the niche '{niche}'. Return a JSON-like textual result labeled clearly so frontend can display it."
    elif mode == "product_build":
        experience = extra.get("experience", "")
        prompt = (
            f"Create a product build plan for niche '{niche}' with user experience: '{experience}'."
            " Provide step-by-step deliverables, resources needed, pricing tiers, and a 30/60/90 day launch checklist."
            " Return a JSON-like textual result."
        )
    elif mode == "monetization_tips":
        product_desc = extra.get("product", "")
        prompt = (
            f"For product/service: '{product_desc}' in niche '{niche}', create 8 actionable monetization strategies."
            " Return a JSON-like textual result."
        )
    else:
        prompt = f"Generate suggestions for mode '{mode}' in niche '{niche}'."

    return call_openai_system(prompt)

@app.get("/")
def root():
    return {"message": "Astra backend running"}

@app.post("/generate-problem")
async def generate_problem(payload: Dict):
    niche = payload.get("niche", "general")
    try:
        result_text = await _generate_for_mode(niche, "problem_generation")
        return {"ok": True, "result": result_text}
    except Exception as e:
        return {"ok": False, "error": str(e)}

@app.post("/generate-product")
async def generate_product(payload: Dict):
    niche = payload.get("niche", "general")
    experience = payload.get("experience", "")
    try:
        result_text = await _generate_for_mode(niche, "product_build", {"experience": experience})
        return {"ok": True, "result": result_text}
    except Exception as e:
        return {"ok": False, "error": str(e)}

@app.post("/generate-strategy")
async def generate_strategy(payload: Dict):
    niche = payload.get("niche", "general")
    product = payload.get("product", "")
    try:
        result_text = await _generate_for_mode(niche, "monetization_tips", {"product": product})
        return {"ok": True, "result": result_text}
    except Exception as e:
        return {"ok": False, "error": str(e)}
