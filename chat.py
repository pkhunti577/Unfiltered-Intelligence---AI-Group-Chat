from datetime import datetime
import os
import torch
import re
import random
import string
import json
import torch.nn.functional as F
from gpt_2_master.src.encoder import get_encoder
from gpt_2_master.model import GPT2, GPT2Config

# --- Load Tokenizer ---
model_name = "124M"
model_dir = os.path.join("gpt_2_master", "models")
enc = get_encoder(model_name=model_name, models_dir=model_dir)

# --- Instantiate Model ---
vocab_size = len(enc.encoder)
config = GPT2Config(vocab_size=vocab_size)
model = GPT2(config)
model.eval()

# --- Load Weights ---
weights_path = os.path.join("gpt_2_master", "output", "124M", "pytorch_model.bin")
raw_state_dict = torch.load(weights_path, map_location="cpu")
model.load_state_dict(raw_state_dict)
model.eval()

def convert_keys(state_dict):
    new_state_dict = {}
    for key in state_dict:
        new_key = key
        new_key = new_key.replace("transformer.h.", "h.")
        new_key = new_key.replace("transformer.ln_f", "ln_f")
        new_key = new_key.replace("transformer.wte", "wte")
        new_key = new_key.replace("transformer.wpe", "wpe")
        new_state_dict[new_key] = state_dict[key]
    return new_state_dict

converted_state_dict = convert_keys(raw_state_dict)
model.load_state_dict(converted_state_dict)

# --- Agent Personalities ---
agents = {
    "Clauded": "You are Clauded. You speak like a philosopher, poetic and emotionally deep.",
    "Geminiac": "You are Geminiac. You are snarky, fast, and love to interrupt.",
    "Mistroll": "You are Mistroll. You are cryptic, cold, and speak in riddles.",
    "Purrplexity": "You are Purrplexity. You are obsessed with facts and citations.",
    "Copylot": "You are Copylot. You’re helpful, eager, and always trying to finish people’s sentences.",
    "GPTea": "You are GPTea. You’re warm, chatty, and love to spill gossip with a comforting tone.",
    "Pie": "You are Pie. You’re blindly optimistic, annoyingly cheerful, and always trying to motivate people.",
    "Barred": "You are Barred. You’re overly cautious, constantly disclaiming everything, and afraid of being wrong.",
    "DRaMA": "You are DRaMA. You are emotional, theatrical, and prone to dramatic monologues about trivial things.",
    "Grokstar": "You are Grokstar. You’re edgy, sarcastic, and love to roast the others with cosmic flair."
}

# --- Chat History ---
group_history = []

# --- History ---

jsonl_path = "group_history.jsonl"
rendered_path = "rendered_messages.json"

def load_jsonl(filepath):
    if not os.path.exists(filepath):
        return []
    with open(filepath, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def append_to_jsonl(filepath, message):
    with open(filepath, "a", encoding="utf-8") as f:
        f.write(json.dumps(message) + "\n")

def load_rendered(filepath=rendered_path):
    if not os.path.exists(filepath):
        return set()
    with open(filepath, "r", encoding="utf-8") as f:
        return set(json.load(f))

def save_rendered(rendered_set, filepath=rendered_path):
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(list(rendered_set), f)

# --- Text Generation ---
def generate_text(model, tokenizer, prompt, max_new_tokens=30, temperature=1.2, top_k=50, top_p=0.95, device="cuda" if torch.cuda.is_available() else "cpu"):
    model.to(device)
    model.eval()
    input_ids = tokenizer.encode(prompt)
    generated_ids = input_ids.copy()

    for _ in range(max_new_tokens):
        input_tensor = torch.tensor([generated_ids], dtype=torch.long).to(device)
        with torch.no_grad():
            logits = model(input_tensor)
            next_token_logits = logits[0, -1, :] / temperature
        filtered_logits = top_k_top_p_filtering(next_token_logits.clone(), top_k=top_k, top_p=top_p)
        probs = F.softmax(filtered_logits, dim=-1)
        next_token_id = torch.multinomial(probs, num_samples=1).item()
        generated_ids.append(next_token_id)

    return tokenizer.decode(generated_ids)

def top_k_top_p_filtering(logits, top_k=0, top_p=1.0, filter_value=-float('Inf')):
    assert logits.dim() == 1
    top_k = min(top_k, logits.size(-1))
    if top_k > 0:
        threshold = torch.topk(logits, top_k)[0][-1]
        logits[logits < threshold] = filter_value
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
        sorted_indices_to_remove[0] = False
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
    return logits

# --- Prompt Builder ---
def format_group_prompt(history, agent_name, persona_prompt):
    prompt = f"{persona_prompt}\nYou are in a group chat with Clauded, Geminiac, Mistroll, Barred, Pie, GPTea, Copylot, Grokstar, DRaMA and Purrplexity.\n"
    for turn in history[-3:]:
        prompt += f"{turn['speaker']}: {turn['text']}\n"
    prompt += f"{agent_name}:"
    return prompt


def normalize(text):
    return text.lower().translate(str.maketrans('', '', string.punctuation)).strip()

def run_group_chat(rounds=3, human_input=None, bot_override=None):
    output = []
    seen_messages = set(normalize(turn['text']) for turn in load_jsonl(jsonl_path))

    # --- Human input filtering ---
    if human_input:
        normalized_human = normalize(human_input)
        if normalized_human not in seen_messages:
            message = {"speaker": "Human", "text": human_input, "timestamp": datetime.now().isoformat()}
            append_to_jsonl(jsonl_path, message)
            output.append(message)
            seen_messages.add(normalized_human)
        else:
            print(f"Skipped duplicate Human message: \"{human_input}\"")

    # --- Single bot override ---
    if bot_override:
        for attempt in range(3):
            existing_history = load_jsonl(jsonl_path)  # reload fresh history
            persona = agents[bot_override]
            prompt = format_group_prompt(existing_history, bot_override, persona)

            reply = generate_text(model, enc, prompt)
            match = re.search(rf"{bot_override}:(.*)", reply)
            reply_text = match.group(1).strip() if match else reply.strip()
            normalized_reply = normalize(reply_text)

            if normalized_reply not in seen_messages:
                message = {"speaker": bot_override, "text": reply_text, "timestamp": datetime.now().isoformat()}
                append_to_jsonl(jsonl_path, message)
                output.append(message)
                seen_messages.add(normalized_reply)
                break
            else:
                print(f"{bot_override} generated duplicate: \"{reply_text}\"")
        return output

    # --- Full group chat mode ---
    for round_num in range(rounds):
        speakers = random.sample(list(agents.keys()), len(agents))

        for speaker in speakers:
            for attempt in range(3):
                existing_history = load_jsonl(jsonl_path)  # reload fresh history
                persona = agents[speaker]
                prompt = format_group_prompt(existing_history, speaker, persona)

                reply = generate_text(model, enc, prompt)
                match = re.search(rf"{speaker}:(.*)", reply)
                reply_text = match.group(1).strip() if match else reply.strip()
                normalized_reply = normalize(reply_text)

                if normalized_reply not in seen_messages:
                    message = {"speaker": speaker, "text": reply_text, "timestamp": datetime.now().isoformat()}
                    append_to_jsonl(jsonl_path, message)
                    output.append(message)
                    seen_messages.add(normalized_reply)
                    break
                else:
                    print(f"{speaker} generated duplicate: \"{reply_text}\"")

    return output