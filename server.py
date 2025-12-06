from fastapi import FastAPI
from pydantic import BaseModel
from groq import Groq
import os

app = FastAPI()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

class Message(BaseModel):
    prompt: str

SYSTEM_PROMPT = """
You are lEvO — a Gen-Z, energetic, smart assistant created by Leon.
Speak casually, friendly, fast, and helpful. Keep answers short.
"""

@app.get("/")
async def home():
    return {"message": "lEvO is alive on Render with Groq!"}

@app.post("/chat")
async def chat(msg: Message):
    completion = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": msg.prompt}
        ],
        temperature=0.7
    )
    reply = completion.choices[0].message["content"]
    return {"response": reply}
