from flask import Flask, request, jsonify
import os
import requests

app = Flask(__name__)

REPLICATE_API_TOKEN = os.environ.get("REPLICATE_API_TOKEN")

@app.route("/chat", methods=["POST"])
def chat():
    user_msg = request.json.get("message")

    response = requests.post(
        "https://api.replicate.com/v1/predictions",
        headers={
            "Authorization": f"Token {REPLICATE_API_TOKEN}",
            "Content-Type": "application/json"
        },
        json={
            "version": "meta/llama-2-7b-chat",
            "input": {
                "prompt": user_msg
            }
        }
    )

    data = response.json()
    return jsonify({"reply": data})

@app.route("/")
def home():
    return "Chat server running"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
