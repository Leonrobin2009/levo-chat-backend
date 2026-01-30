from flask import Flask, request, jsonify
import os
from groq import Groq

app = Flask(__name__)

# ENV
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

client = Groq(api_key=GROQ_API_KEY)

SYSTEM_PROMPT = "You are lEvO, a calm and accurate AI assistant."

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    user_msg = data.get("message", "")

    completion = client.chat.completions.create(
        model="llama-3.1-8b-instant",  # ✅ FREE
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg}
        ],
        max_tokens=1024
    )

    reply = completion.choices[0].message.content
    return jsonify({"reply": reply})

@app.route("/")
def home():
    return "lEvO chat server running (Groq free model)"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
