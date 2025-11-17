from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
import os
import json
import textwrap

app = FastAPI()

# Allow CORS from your frontend (adjust origins if needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

OPENAI_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_KEY:
    raise RuntimeError("Set OPENAI_API_KEY environment variable before running the backend.")

client = OpenAI(api_key=OPENAI_KEY)

class Payload(BaseModel):
    mode: str
    inputs: dict = {}

def call_model(prompt: str, system: str = "You are Astra, a business-product building AI."):
    # Use gpt-4o-mini or whichever model you have access to.
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt}
        ],
        max_tokens=4000,
        temperature=0.2,
    )
    return response.choices[0].message.content

@app.post("/api/generate")
async def generate(payload: Payload):
    mode = payload.mode
    inputs = payload.inputs or {}

    # MODE 1: Problem generation from a niche
    if mode == "mode1":
        niche = inputs.get("niche", "general")
        prompt = f"""
        Generate a JSON object containing:
        - problems: array of 10 monetizable problem statements for the niche '{niche}'.
        - solutions: short actionable solutions for each problem.
        - quick_tips: 5 short tips to start solving/monetizing these problems.
        Return valid JSON only.
        """
        out = call_model(textwrap.dedent(prompt))
        # try to parse JSON, but if not valid return raw
        try:
            parsed = json.loads(out)
            return {"result": json.dumps(parsed, indent=2)}
        except Exception:
            return {"result": out}

    # MODE 2: Main locked mode -> pick niche and create 3-5 complete products
    if mode == "mode2":
        userType = inputs.get("userType", "")
        strengths = inputs.get("strengths", "")
        prompt = f"""
        You are Astra: an AI that BUILDS complete, ready-to-sell digital products (not ideas). 

        Task:
        1) Choose the single most profitable niche (based on global demand and user's context).
        2) Create 3 to 5 COMPLETE digital products for that niche. 

        For EACH product produce the following sections explicitly and fully written out (no placeholders):
        - product_title
        - short_product_description (2-3 lines)
        - full_product_content: a finished, publishable text/manual (equivalent to the content of a 10-20 page PDF or templates/workbook)
        - modules_or_chapters: list with full text for each module (headings + paragraphs)
        - templates_checklists: actual checklist/template text that a user can copy & paste
        - target_audience: who to sell to, with demographics and psychographics
        - main_pain_points: 3 core pain points
        - value_proposition and sales_angles (full paragraphs)
        - landing_page_copy (headline, subhead, bullets, CTA, 200-400 words body text)
        - pricing_recommendation and packaging (single price + upsells)
        - 7_day_launch_plan (day-by-day actionable plan)
        - email_sequence (5 emails fully written)
        - social_media_posts (5 captions / post texts)
        - facebook/google_ads_copy (3 short ad variations)
        - suggested_bonuses and upgrades
        - quick_creation_checklist (step-by-step actions the user must do to finalize product)

        User context:
        - userType: {userType}
        - strengths: {strengths}

        Output:
        - Return a single JSON object with:
          {{"niche": "...", "products": [ {{product object}}, ... ]}}

        IMPORTANT: outputs must be fully written, no placeholders like [TEXT], and each product must be usable by a human to publish and sell.
        """
        out = call_model(textwrap.dedent(prompt), system="You are Astra, a product-building AI that writes full content.")
        try:
            parsed = json.loads(out)
            return {"result": json.dumps(parsed, indent=2)}
        except Exception:
            # If model returned text (not strict JSON), just return raw text
            return {"result": out}

    # MODE 3: Expert mode - user provides exact idea to be fully built
    if mode == "mode3":
        idea = inputs.get("idea", "")
        prompt = f"""
        You are Astra. The user gave this request: {idea}

        Task: Build a full, ready-to-publish digital product from that exact idea.
        Provide:
        - product_title
        - full_product_content (complete text, long form)
        - modules and chapter text
        - templates and checklists
        - landing page copy
        - email sequence (5 emails)
        - ads copy and social captions
        - pricing and launch plan

        Return as JSON with keys: title, content, modules, templates, landing, emails, ads, price, launch_plan.
        """
        out = call_model(textwrap.dedent(prompt))
        try:
            parsed = json.loads(out)
            return {"result": json.dumps(parsed, indent=2)}
        except Exception:
            return {"result": out}

    return {"result": f"Unknown mode: {mode}"}

