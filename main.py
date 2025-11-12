from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from supabase import create_client
from openai import OpenAI
import os

app = FastAPI()

# Allow CORS for your frontend (Claude UI later)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Environment variables
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Clients
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
openai = OpenAI(api_key=OPENAI_API_KEY)

@app.get("/")
def root():
    return {"message": "Astra backend is running successfully!"}

@app.post("/api/generate")
async def generate(data: dict):
    niche = data.get("niche", "general")
    completion = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that generates monetizable niche problems and solutions."},
            {"role": "user", "content": f"Generate 10 monetizable problems, solutions, and strategies for the niche: {niche}"}
        ]
    )
    return {"result": completion.choices[0].message.content}
