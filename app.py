from flask import Flask, render_template, jsonify, request
from chat import run_group_chat, normalize
import threading
import time
import json
import os

app = Flask(__name__)

# --- File paths ---
jsonl_path = "group_history.jsonl"
rendered_path = "rendered_messages.json"

# --- JSONL helpers ---
def load_jsonl(filepath):
    if not os.path.exists(filepath):
        return []
    with open(filepath, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def load_rendered(filepath=rendered_path):
    if not os.path.exists(filepath):
        return set()
    with open(filepath, "r", encoding="utf-8") as f:
        return set(json.load(f))

def save_rendered(rendered_set, filepath=rendered_path):
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(list(rendered_set), f)

# --- Filter unrendered messages ---
def get_new_messages():
    all_messages = load_jsonl(jsonl_path)
    rendered = load_rendered()

    new_messages = []
    for msg in all_messages:
        norm = normalize(msg['text'])
        if norm not in rendered:
            new_messages.append(msg)
            rendered.add(norm)

    save_rendered(rendered)
    return new_messages

# --- Bot loop ---
def bot_loop():
    while True:
        run_group_chat(rounds=1)
        time.sleep(5)

# --- Routes ---
@app.route("/")
def index():
    return render_template("chat.html")

@app.route("/generate")
def generate_chat():
    chat_output = run_group_chat(rounds=3)
    return jsonify(chat_output)

@app.route("/history")
def get_history():
    return jsonify(get_new_messages())

@app.route("/interact")
def interact():
    user_message = request.args.get("message", "")
    bot_name = request.args.get("bot", "")
    chat_output = run_group_chat(human_input=user_message, bot_override=bot_name)
    return jsonify(chat_output)

# --- Start app ---
if __name__ == "__main__":
    threading.Thread(target=bot_loop, daemon=True).start()
    app.run(debug=True)
