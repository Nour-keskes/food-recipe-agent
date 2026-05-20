import sys
import os
import json
from datetime import datetime

import gradio as gr

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
try:
    from agent import chat as agent_chat, chat_with_image, chat_history
except ImportError:
    def agent_chat(msg, temperature=0.7): return f"Delicious response to: {msg}"
    def chat_with_image(img, temperature=0.7): return "That looks like an incredible Tunisian dish!"
    chat_history = []

BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, '..', 'Data')
os.makedirs(DATA_DIR, exist_ok=True)
CHATS_FILE = os.path.join(DATA_DIR, 'saved_chats.json')

SYSTEM_PROMPT = (
    "You are a helpful food recipe assistant specializing in Tunisian and international cuisine.\n"
    "- Always call RecipeSearch first when the user asks about recipes to query the FAISS vector space.\n"
    "- Call IngredientSubstitute when looking for cooking alternatives.\n"
    "- Call WebSearch only if the local vector lookup yields irrelevant contexts.\n"
    "- Be friendly, detailed, and format your markdown lists clearly."
)


def load_saved_chats():
    if os.path.exists(CHATS_FILE):
        try:
            with open(CHATS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_chats(chats):
    with open(CHATS_FILE, "w", encoding="utf-8") as f:
        json.dump(chats, f, ensure_ascii=False, indent=2)


def get_chat_choices():
    chats = load_saved_chats()
    if not chats:
        return []
    sorted_chats = sorted(chats.items(), key=lambda x: x[0], reverse=True)
    return [(f"🍽️ {data.get('title', 'New Chat')[:24]} ({data.get('date', '')})", id_) for id_, data in sorted_chats]


def new_chat():
    new_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    if isinstance(chat_history, list):
        chat_history.clear()
        chat_history.append({"role": "system", "content": SYSTEM_PROMPT})
    return [], gr.update(choices=get_chat_choices(), value=None), new_id


def load_selected_chat(chat_id):
    if not chat_id:
        return [], gr.skip()
    chats = load_saved_chats()
    chat_data = chats.get(chat_id, {})
    messages = chat_data.get("messages", [])
    formatted = [
        {"role": m["role"], "content": m["content"]}
        for m in messages
        if m.get("role") in ("user", "assistant") and isinstance(m.get("content"), str)
    ]
    return formatted, chat_id


def respond(message, history, chat_id, temperature):
    if not message.strip():
        return "", history, gr.update(), chat_id

    response = agent_chat(message, temperature=temperature)

    history.append({"role": "user",      "content": message})
    history.append({"role": "assistant", "content": response})

    active_id = chat_id if chat_id else datetime.now().strftime("%Y%m%d_%H%M%S")

    chats = load_saved_chats()
    chats[active_id] = {
        "title": message[:30] + "..." if len(message) > 30 else message,
        "date": datetime.now().strftime("%b %d"),
        "messages": history,
    }
    save_chats(chats)

    return "", history, gr.update(choices=get_chat_choices(), value=active_id), active_id


def handle_image(image_path, history, chat_id, temperature):
    if image_path is None:
        return history, gr.update(), chat_id

    response = chat_with_image(image_path, temperature=temperature)

    history.append({"role": "user",      "content": "📸 Analysing custom dish photo..."})
    history.append({"role": "assistant", "content": response})

    active_id = chat_id if chat_id else datetime.now().strftime("%Y%m%d_%H%M%S")

    chats = load_saved_chats()
    chats[active_id] = {
        "title": "Image Recipe Analysis 📸",
        "date": datetime.now().strftime("%b %d"),
        "messages": history,
    }
    save_chats(chats)

    return history, gr.update(choices=get_chat_choices(), value=active_id), active_id


css = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

html, body {
    width: 100%;
    height: 100%;
    overflow: hidden;
}

body {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
    background: linear-gradient(135deg, #C41E3A 0%, #DC143C 25%, #FF1744 50%, #E53935 75%, #B71C1C 100%) !important;
    overflow: hidden;
}

body::before {
    content: "";
    position: fixed;
    inset: 0;
    background: radial-gradient(circle at 20% 50%, rgba(196, 30, 58, 0.2), transparent 50%),
                radial-gradient(circle at 80% 80%, rgba(229, 57, 53, 0.15), transparent 50%),
                linear-gradient(180deg, rgba(196, 30, 58, 0.7) 0%, rgba(220, 20, 60, 0.75) 50%, rgba(229, 57, 53, 0.8) 100%) !important;
    z-index: -1;
    pointer-events: none;
}

.gradio-container {
    max-width: 100% !important;
    width: 100% !important;
    height: 100vh !important;
    margin: 0 !important;
    padding: 0 !important;
}

.gradio-block {
    background: transparent !important;
    border: none !important;
}

.main-layout {
    display: flex !important;
    gap: 16px !important;
    height: 100vh !important;
    padding: 16px !important;
    width: 100% !important;
    max-width: 100% !important;
}

.sidebar-panel {
    flex: 0 0 280px !important;
    background: linear-gradient(135deg, rgba(139, 0, 0, 0.85), rgba(178, 34, 52, 0.85)) !important;
    backdrop-filter: blur(20px) !important;
    -webkit-backdrop-filter: blur(20px) !important;
    border: 2px solid rgba(220, 20, 60, 0.3) !important;
    border-radius: 20px !important;
    padding: 20px !important;
    display: flex !important;
    flex-direction: column !important;
    gap: 16px !important;
    box-shadow: 0 25px 50px rgba(0, 0, 0, 0.6), 
                0 0 40px rgba(220, 20, 60, 0.2),
                inset 0 1px 0 rgba(255, 200, 200, 0.15) !important;
    overflow-y: auto !important;
}

.sidebar-panel::-webkit-scrollbar {
    width: 6px !important;
}

.sidebar-panel::-webkit-scrollbar-track {
    background: transparent !important;
}

.sidebar-panel::-webkit-scrollbar-thumb {
    background: rgba(220, 20, 60, 0.4) !important;
    border-radius: 3px !important;
}

.sidebar-panel::-webkit-scrollbar-thumb:hover {
    background: rgba(220, 20, 60, 0.6) !important;
}

.content-panel {
    flex: 1 !important;
    background: linear-gradient(135deg, rgba(178, 34, 52, 0.85), rgba(139, 0, 0, 0.85)) !important;
    backdrop-filter: blur(20px) !important;
    -webkit-backdrop-filter: blur(20px) !important;
    border: 2px solid rgba(220, 20, 60, 0.35) !important;
    border-radius: 20px !important;
    padding: 20px !important;
    display: flex !important;
    flex-direction: column !important;
    gap: 12px !important;
    box-shadow: 0 25px 50px rgba(0, 0, 0, 0.6), 
                0 0 50px rgba(220, 20, 60, 0.2),
                inset 0 1px 0 rgba(255, 200, 200, 0.1) !important;
}

.brand-container {
    text-align: center !important;
    padding-bottom: 16px !important;
    border-bottom: 2px solid rgba(220, 20, 60, 0.4) !important;
    margin-bottom: 8px !important;
}

.brand-title {
    color: #FFE0E6 !important;
    font-weight: 800 !important;
    font-size: 1.8rem !important;
    letter-spacing: -0.5px !important;
    margin: 0 !important;
    display: block !important;
    line-height: 1 !important;
    text-shadow: 0 3px 10px rgba(0, 0, 0, 0.5) !important;
}

.brand-subtitle {
    color: #FFCCCC !important;
    font-size: 0.8rem !important;
    font-weight: 600 !important;
    margin-top: 4px !important;
    letter-spacing: 0.3px !important;
    text-shadow: 0 2px 5px rgba(0, 0, 0, 0.3) !important;
}

.history-radio-list {
    flex: 1 !important;
}

.history-radio-list span.wrap {
    flex-direction: column !important;
    gap: 8px !important;
}

.history-radio-list label {
    background: rgba(220, 20, 60, 0.12) !important;
    border: 1.5px solid rgba(220, 20, 60, 0.3) !important;
    color: #FFCCCC !important;
    border-radius: 12px !important;
    padding: 12px 14px !important;
    transition: all 0.3s cubic-bezier(0.34, 1.56, 0.64, 1) !important;
    cursor: pointer !important;
    font-weight: 500 !important;
    font-size: 0.9rem !important;
}

.history-radio-list label:hover {
    background: rgba(220, 20, 60, 0.2) !important;
    border-color: #FFE0E6 !important;
    transform: translateX(4px) !important;
    box-shadow: 0 5px 15px rgba(220, 20, 60, 0.25) !important;
}

.history-radio-list label.selected {
    background: rgba(220, 20, 60, 0.3) !important;
    border-color: #FFE0E6 !important;
    color: #FFE0E6 !important;
    box-shadow: 0 0 25px rgba(220, 20, 60, 0.4), inset 0 0 10px rgba(255, 200, 200, 0.15) !important;
}

button {
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
    border: none !important;
    border-radius: 12px !important;
    transition: all 0.3s cubic-bezier(0.34, 1.56, 0.64, 1) !important;
    cursor: pointer !important;
    font-size: 0.95rem !important;
    padding: 11px 20px !important;
}

.primary-btn {
    background: linear-gradient(135deg, #DC143C 0%, #C41E3A 100%) !important;
    color: #fff !important;
    box-shadow: 0 10px 30px rgba(220, 20, 60, 0.4), 0 0 20px rgba(196, 30, 58, 0.25) !important;
    border: 1px solid rgba(255, 200, 200, 0.3) !important;
}

.primary-btn:hover {
    transform: translateY(-3px) !important;
    box-shadow: 0 15px 40px rgba(220, 20, 60, 0.6), 0 0 30px rgba(196, 30, 58, 0.4) !important;
    filter: brightness(1.1) !important;
}

.primary-btn:active {
    transform: translateY(-1px) !important;
}

.secondary-btn {
    background: rgba(220, 20, 60, 0.15) !important;
    color: #FFCCCC !important;
    border: 1.5px solid rgba(220, 20, 60, 0.4) !important;
    box-shadow: 0 4px 15px rgba(220, 20, 60, 0.15) !important;
}

.secondary-btn:hover {
    background: rgba(220, 20, 60, 0.25) !important;
    border-color: #FFE0E6 !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(220, 20, 60, 0.3), 0 0 15px rgba(220, 20, 60, 0.2) !important;
    color: #FFE0E6 !important;
}

textarea, input[type="text"] {
    background: rgba(50, 10, 20, 0.7) !important;
    border: 1.5px solid rgba(220, 20, 60, 0.3) !important;
    color: #FFE0E6 !important;
    border-radius: 12px !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.95rem !important;
    padding: 12px 16px !important;
    transition: all 0.3s ease !important;
}

textarea::placeholder, input[type="text"]::placeholder {
    color: rgba(220, 20, 60, 0.4) !important;
}

textarea:focus, input[type="text"]:focus {
    border-color: #FFE0E6 !important;
    background: rgba(50, 10, 20, 0.9) !important;
    box-shadow: 0 0 0 3px rgba(220, 20, 60, 0.15), inset 0 0 0 1px rgba(220, 20, 60, 0.3), 0 0 20px rgba(220, 20, 60, 0.2) !important;
    outline: none !important;
    color: #FFF0F5 !important;
}

.gradio-chatbot {
    background: #E8F0F8 !important;
    border: 1.5px solid rgba(220, 20, 60, 0.25) !important;
    border-radius: 16px !important;
    padding: 20px !important;
    flex: 1 !important;
    overflow-y: auto !important;
    box-shadow: inset 0 0 20px rgba(220, 20, 60, 0.08) !important;
    min-height: 300px !important;
}

.gradio-chatbot::-webkit-scrollbar {
    width: 8px !important;
}

.gradio-chatbot::-webkit-scrollbar-track {
    background: transparent !important;
}

.gradio-chatbot::-webkit-scrollbar-thumb {
    background: rgba(220, 20, 60, 0.3) !important;
    border-radius: 4px !important;
}

.gradio-chatbot::-webkit-scrollbar-thumb:hover {
    background: rgba(220, 20, 60, 0.5) !important;
}

.gradio-chatbot .message.user {
    background: #B3D9F2 !important;
    color: #1a1a1a !important;
    border-radius: 16px 16px 4px 16px !important;
    font-weight: 500 !important;
    padding: 14px 18px !important;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1) !important;
    border: 1px solid rgba(179, 217, 242, 0.6) !important;
    margin: 8px 0 !important;
}

.gradio-chatbot .message.bot {
    background: #FFFFFF !important;
    color: #1a1a1a !important;
    border: 1.5px solid #D0D0D0 !important;
    border-radius: 16px 16px 16px 4px !important;
    padding: 14px 18px !important;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08) !important;
    font-weight: 500 !important;
    margin: 8px 0 !important;
}

.gradio-image {
    border: 2px dashed rgba(220, 20, 60, 0.5) !important;
    border-radius: 12px !important;
    background: rgba(220, 20, 60, 0.1) !important;
}

.gradio-image:hover {
    border-color: #FFE0E6 !important;
    background: rgba(220, 20, 60, 0.2) !important;
}

.gradio-row {
    gap: 10px !important;
    margin: 0 !important;
}

.gradio-column {
    gap: 10px !important;
    margin: 0 !important;
}

@media (max-width: 1024px) {
    .main-layout {
        flex-direction: column !important;
        padding: 12px !important;
    }
    
    .sidebar-panel {
        flex: 0 0 auto !important;
        max-height: 150px !important;
    }
    
    .content-panel {
        flex: 1 !important;
    }
}
"""

with gr.Blocks(title="🍲 YummyBot") as demo:
    active_chat_id = gr.State(None)

    with gr.Row(elem_classes="main-layout"):

        # ── Sidebar ──────────────────────────────
        with gr.Column(elem_classes="sidebar-panel", scale=1):
            gr.HTML("""
                <div class="brand-container">
                    <span class="brand-title">🍲 YummyBot</span>
                    <span class="brand-subtitle">Gourmet Companion</span>
                </div>
                <div style="font-size:0.75rem;color:#FFCCCC;font-weight:700;letter-spacing:0.5px;">📜 RECENT CHATS</div>
            """)

            chat_selector = gr.Radio(
                choices=get_chat_choices(),
                label="",
                interactive=True,
                elem_classes="history-radio-list",
            )

            new_chat_btn = gr.Button(
                "➕ New Conversation",
                elem_classes="secondary-btn",
                variant="secondary",
            )

            # ── Temperature control ───────────────
            gr.HTML("""
                <div style="margin-top:8px;padding-top:12px;border-top:1.5px solid rgba(220,20,60,0.2);">
                    <div style="font-size:0.75rem;color:#FFCCCC;font-weight:700;letter-spacing:0.5px;margin-bottom:6px;">
                        🌡️ CREATIVITY (TEMPERATURE)
                    </div>
                </div>
            """)
            temperature_slider = gr.Slider(
                minimum=0.0,
                maximum=1.5,
                value=0.7,
                step=0.05,
                label="0 = precise  ·  1.5 = creative",
                elem_classes="temp-slider",
                interactive=True,
            )

            gr.HTML("""
                <div style="margin-top:auto;padding-top:12px;text-align:center;font-size:0.7rem;
                            color:#FFB3B3;border-top:1.5px solid rgba(220,20,60,0.2);">
                    🔥 Powered by Advanced AI
                </div>
            """)

        # ── Main chat panel ───────────────────────
        with gr.Column(elem_classes="content-panel", scale=3):
            chatbot = gr.Chatbot(
                label="",
                show_label=False,
                height=500,
            )

            with gr.Row():
                msg = gr.Textbox(
                    placeholder="Ask for recipes, techniques, ingredients, or cooking tips...",
                    scale=4,
                    show_label=False,
                    container=False,
                    lines=1,
                )
                img_input = gr.Image(
                    type="filepath",
                    sources=["upload"],
                    show_label=False,
                    scale=1,
                    container=False,
                    height=50,
                )

            with gr.Row():
                send = gr.Button("Send ✨", elem_classes="primary-btn", scale=2)

    # ── Event wiring ──────────────────────────────
    chat_selector.change(
        load_selected_chat,
        inputs=[chat_selector],
        outputs=[chatbot, active_chat_id],
    )

    send.click(
        respond,
        inputs=[msg, chatbot, active_chat_id, temperature_slider],
        outputs=[msg, chatbot, chat_selector, active_chat_id],
    )

    msg.submit(
        respond,
        inputs=[msg, chatbot, active_chat_id, temperature_slider],
        outputs=[msg, chatbot, chat_selector, active_chat_id],
    )

    new_chat_btn.click(
        new_chat,
        outputs=[chatbot, chat_selector, active_chat_id],
    )

    img_input.change(
        handle_image,
        inputs=[img_input, chatbot, active_chat_id, temperature_slider],
        outputs=[chatbot, chat_selector, active_chat_id],
    )

if __name__ == "__main__":
    demo.launch(css=css)