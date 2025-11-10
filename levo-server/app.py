from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from transformers import BlipProcessor, BlipForConditionalGeneration
import requests, torch, os

app = Flask(__name__)

print("🔹 Loading lEvO v1.0 (CPU mode)...")

# Hugging Face models
TEXT_MODEL = "microsoft/Phi-3-mini-4k-instruct"
IMAGE_MODEL = "Salesforce/blip-image-captioning-base"

# ✅ Load text model
tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL)
model = AutoModelForCausalLM.from_pretrained(
    TEXT_MODEL,
    torch_dtype=torch.float32,
    low_cpu_mem_usage=True
)
chat_pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device_map=None)

# ✅ Load image model
print("🖼️ Loading image processor...")
blip_processor = BlipProcessor.from_pretrained(IMAGE_MODEL)
blip_model = BlipForConditionalGeneration.from_pretrained(IMAGE_MODEL)

# ✅ Free News API (get one from newsdata.io)
NEWS_API_KEY = os.environ.get("NEWSDATA_API", "pub_12345_demo")

@app.route("/")
def home():
    return jsonify({"message": "✅ lEvO v1.0 backend is running!"})

# 🗨️ Chat endpoint
@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    prompt = data.get("prompt", "")
    if not prompt:
        return jsonify({"error": "Empty prompt"}), 400

    print(f"🗨️ User: {prompt}")
    response = chat_pipe(prompt, max_new_tokens=200, temperature=0.7, do_sample=True)[0]["generated_text"]
    return jsonify({"response": response.strip()})

# 🖼️ Image caption
@app.route("/image", methods=["POST"])
def image_caption():
    data = request.get_json()
    if "image_url" not in data:
        return jsonify({"error": "Missing image_url"}), 400

    image_url = data["image_url"]
    image = requests.get(image_url, stream=True).raw
    inputs = blip_processor(images=image, return_tensors="pt")
    out = blip_model.generate(**inputs)
    caption = blip_processor.decode(out[0], skip_special_tokens=True)
    return jsonify({"caption": caption})

# 📰 Live news
@app.route("/news", methods=["GET"])
def get_news():
    topic = request.args.get("topic", "ai")
    url = f"https://newsdata.io/api/1/news?apikey={NEWS_API_KEY}&q={topic}&language=en"
    r = requests.get(url)
    data = r.json()
    if "results" in data:
        headlines = [item["title"] for item in data["results"][:5]]
        return jsonify({"topic": topic, "headlines": headlines})
    return jsonify({"error": "No news found."})

# 🔎 Web search
@app.route("/search", methods=["GET"])
def search():
    query = request.args.get("q")
    if not query:
        return jsonify({"error": "Missing query"}), 400

    url = f"https://api.duckduckgo.com/?q={query}&format=json"
    r = requests.get(url).json()
    result = r.get("AbstractText") or "No direct answer found."
    return jsonify({"query": query, "result": result})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    app.run(host="0.0.0.0", port=port)
