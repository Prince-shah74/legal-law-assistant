from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import os
from dotenv import load_dotenv
from openai import OpenAI
import asyncio
from functools import lru_cache

load_dotenv()

# Init OpenAI client
client = OpenAI(
    api_key="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",
    base_url="https://openrouter.ai/api/v1"
)

# --- FastAPI setup ---
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Load laws CSV once at startup ---
laws_df = pd.read_csv("laws.csv")

# --- Build keyword index for fast lookup ---
laws_index = {}
for _, row in laws_df.iterrows():
    for tag in str(row["tags"]).split(","):
        tag = tag.lower().strip()
        if tag not in laws_index:
            laws_index[tag] = []
        laws_index[tag].append({
            "code_ref": row["code_ref"],
            "title": row["title"],
            "summary": row["summary"]
        })

# --- Request model ---
class CaseRequest(BaseModel):
    user_text: str
    language: str = "auto"

# --- Role detection ---
def detect_role(text: str) -> str:
    t = text.lower()
    if any(p in t for p in ["mene nahi kiya", "maine nahi kiya", "pakda gaya", "arre mujhe pakda", "maine nahi kiya"]):
        return "accused"
    if any(p in t for p in ["mujhe mara", "mujhe chot", "injured", "rape hua", "mujhe humne", "mere saath hua"]):
        return "victim"
    if any(p in t for p in ["maine dekha", "witness", "dekha maine", "maine dekha"]):
        return "witness"
    return "unknown"

# --- Fast keyword match using index ---
def match_laws(user_input: str):
    user_words = user_input.lower().split()
    matched = []
    for word in user_words:
        if word in laws_index:
            for law in laws_index[word]:
                if law not in matched:
                    matched.append(law)
    return matched

# --- Async AI suggestion with optional caching ---
@lru_cache(maxsize=128)
def get_ai_suggestion_cached(system_prompt: str, user_prompt: str):
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        max_tokens=500,
        temperature=0.2
    )
    return resp.choices[0].message.content.strip()

async def generate_ai_suggestion(context: dict) -> str:
    user_text = context["user_text"]
    role = context["role"]
    matched = context["matched_sections"]

    system_prompt = """Aap ek legal assistant ho. Respond strictly in Hinglish (latin script), short and practical.
    
Produce a structured, numbered roadmap with clear sections and line breaks.
Include:
Step-by-Step Legal Guidance
1.	Immediate Actions (First 24 Hours): Ensure safety, seek medical aid, preserve evidence, and inform trusted contacts.
2.	Assessment & Emotional Support: Listen actively, validate feelings, provide reassurance, maintain confidentiality.
3.	Police & Official Procedures: File FIR/complaint promptly, provide accurate statements, know your rights, and keep copies of all documents.
4.	Legal Rights & Steps
•	If accused: Understand charges, constitutional protections, arrest, bail, and trial rights.
•	If victim: Know rights to protection, compensation, and legal remedies.
5.	Evidence & Documentation: Collect and securely store evidence, maintain a detailed timeline, record witness details.
6.	Timeline & Follow-up: Track deadlines (FIR, medical exam, hearings), set reminders, and regularly check case status.
7.	Precautions: Stay calm, document everything, avoid public disclosures, never tamper with evidence.
8.	Consulting a Lawyer: Seek legal advice early, choose specialized counsel, prepare documents, consider legal aid if needed.
9.	Costs, Charges & Legal Provisions: Estimate lawyer/court fees, explain charges if accused, outline victim costs and compensation, consider pros/cons of filing FIR.
10.	Ongoing Support: Access counseling, legal aid, victim services, and reliable information; encourage patience through the process.

Tailor advice to the user's role: victim => protection & evidence; accused => rights, lawyer, bail; witness => how to preserve evidence.
If matched_sections is not 'No relevant', mention those sections briefly and how they affect steps.
Always keep each numbered step on a new line and keep answers concise but complete. Do NOT give legal conclusions — give practical roadmap only.
"""

    user_prompt = (
        f"User case: {user_text}\n"
        f"Detected role: {role}\n"
        f"Matched sections:\n{matched}\n\n"
        "Now give the roadmap as described in system instruction."
    )

    # Run AI call in thread to not block async loop
    loop = asyncio.get_event_loop()
    try:
        suggestion = await loop.run_in_executor(None, get_ai_suggestion_cached, system_prompt, user_prompt)
        return suggestion
    except Exception as e:
        return f"AI Suggestion error: {str(e)}"

# --- Async endpoint ---
@app.post("/analyze-case")
async def analyze_case(request: CaseRequest):
    user_input = request.user_text
    role = detect_role(user_input)

    matched_laws = match_laws(user_input)
    sections_text = "\n".join(
        [f"- {law['code_ref']}: {law['title']} → {law['summary']}" for law in matched_laws]
    ) if matched_laws else "No relevant section found in dataset."

    ai_input = {
        "user_text": user_input,
        "role": role,
        "matched_sections": sections_text
    }
    suggestion = await generate_ai_suggestion(ai_input)

    return {
        "summary": f"According to the given case:\n{user_input}\n\nApplicable Sections:\n{sections_text}",
        "suggestion": suggestion,
        "disclaimer": "⚠️ Ye sirf informational hai. Hamesha ek licensed lawyer ki salah lein."
    }
