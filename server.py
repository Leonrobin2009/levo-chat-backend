import os
import sqlite3
import base64
import requests
import matplotlib.pyplot as plt
from datetime import datetime
from uuid import uuid4
from urllib.parse import quote_plus

from fastapi import FastAPI, Request, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse

from slowapi import Limiter
from slowapi.util import get_remote_address

from groq import Groq
from fpdf import FPDF
from pptx import Presentation
import replicate

# ==========================================================
# CONFIG
# ==========================================================
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")
REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN")

client = Groq(api_key=GROQ_API_KEY)
replicate.Client(api_token=REPLICATE_API_TOKEN)

app = FastAPI()
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================================
# SYSTEM PROMPT
# ==========================================================
SYSTEM_PROMPT = """
You are lEvO, a smart and calm AI assistant.

Rules:
- Short replies for small talk
- Long replies only when needed
- NEVER invent links
- If links are provided, format them nicely
- Be accurate, not dramatic
"""

# ==========================================================
# MEMORY
# ==========================================================
conn = sqlite3.connect("memory.db", check_same_thread=False)
cursor = conn.cursor()
cursor.execute("CREATE TABLE IF NOT EXISTS memory (user_id TEXT, text TEXT)")
conn.commit()

def get_memory(uid):
    cursor.execute("SELECT text FROM memory WHERE user_id=?", (uid,))
    return "\n".join(row[0] for row in cursor.fetchall())

def save_memory(uid, msg):
    cursor.execute("INSERT INTO memory VALUES (?, ?)", (uid, msg))
    conn.commit()

# ==========================================================
# LINK SEARCH (REAL)
# ==========================================================
def get_links(query: str, site: str = None, limit=5):
    q = query
    if site:
        q += f" site:{site}"
    url = f"https://ddg-api.herokuapp.com/search?query={quote_plus(q)}"
    data = requests.get(url).json()
    results = []
    for r in data.get("results", [])[:limit]:
        results.append(f"- [{r['title']}]({r['url']})")
    return "\n".join(results)

# ==========================================================
# CHAT
# ==========================================================
@app.post("/chat")
@limiter.limit("15/minute")
async def chat(request: Request):
    data = await request.json()
    prompt = data.get("prompt", "").strip()
    user_id = data.get("user_id", "guest")

    today = datetime.now().strftime("%A, %d %B %Y")
    memory = get_memory(user_id)

    # 🔍 Detect link intent
    wants_amazon = "amazon" in prompt.lower()
    wants_youtube = "youtube" in prompt.lower()

    link_block = ""

    if wants_amazon:
        link_block = get_links(prompt, site="amazon.com")

    if wants_youtube:
        link_block = get_links(prompt, site="youtube.com")

    msgs = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "system", "content": f"Today's real date is {today}"},
        {"role": "system", "content": f"Conversation memory:\n{memory}"},
    ]

    if link_block:
        msgs.append({
            "role": "system",
            "content": f"Here are verified real links:\n{link_block}"
        })

    msgs.append({"role": "user", "content": prompt})

    completion = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=msgs,
        max_tokens=2048
    )

    reply = completion.choices[0].message.content

    save_memory(user_id, prompt)
    save_memory(user_id, reply)

    return {"response": reply}

# ==========================================================
# IMAGE ANALYSIS
# ==========================================================
@app.post("/vision")
async def vision(file: UploadFile = File(...)):
    img = base64.b64encode(await file.read()).decode()
    completion = client.chat.completions.create(
        model="llava-v1.6",
        messages=[{
            "role": "user",
            "content": [
                {"type": "input_text", "text": "Analyze this image."},
                {"type": "input_image", "image": img}
            ]
        }]
    )
    return {"response": completion.choices[0].message.content}

# ==========================================================
# IMAGE GENERATION
# ==========================================================
@app.post("/image-generate")
def image_generate(prompt: str):
    output = replicate.run(
        "stability-ai/sdxl",
        input={"prompt": prompt}
    )
    return {"image_url": output[0]}

# ==========================================================
# FILE GENERATION
# ==========================================================
@app.post("/create-pdf")
def create_pdf(text: str):
    name = f"{uuid4()}.pdf"
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 8, text)
    pdf.output(name)
    return FileResponse(name)

@app.post("/create-txt")
def create_txt(text: str):
    name = f"{uuid4()}.txt"
    open(name, "w").write(text)
    return FileResponse(name)

@app.post("/create-ppt")
def create_ppt(text: str):
    name = f"{uuid4()}.pptx"
    prs = Presentation()
    slide = prs.slides.add_slide(prs.slide_layouts[1])
    slide.shapes.title.text = "Generated by lEvO"
    slide.placeholders[1].text = text
    prs.save(name)
    return FileResponse(name)

# ==========================================================
# GRAPH
# ==========================================================
@app.get("/graph")
def graph():
    name = f"{uuid4()}.png"
    plt.plot([1, 2, 3], [10, 25, 15])
    plt.savefig(name)
    plt.close()
    return FileResponse(name)
