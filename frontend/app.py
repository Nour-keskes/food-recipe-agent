import sys
sys.path.append(r"F:\ProjetAgentAI\agent")

import gradio as gr
import json
import os
from datetime import datetime
from agent import chat as agent_chat, chat_with_image, chat_history

# ─────────────────────────────────────────
# SAVED CHATS
# ─────────────────────────────────────────
CHATS_FILE = r"F:\ProjetAgentAI\data\saved_chats.json"

def load_saved_chats():
    if os.path.exists(CHATS_FILE):
        with open(CHATS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_chats(chats):
    with open(CHATS_FILE, "w", encoding="utf-8") as f:
        json.dump(chats, f, ensure_ascii=False, indent=2)

def render_sidebar():
    chats = load_saved_chats()
    html = """
    <div class="sidebar-header">
        <span class="sidebar-title">🫕 Recipe Agent</span>
    </div>
    <div class="chat-list">
    """
    if not chats:
        html += "<p class='no-chats'>No saved chats yet.<br>Start a conversation!</p>"
    else:
        for chat_id, chat_data in sorted(chats.items(), reverse=True):
            title = chat_data.get("title", "New Chat")[:28]
            date = chat_data.get("date", "")
            html += f"""
            <div class="chat-item">
                <div class="chat-info">
                    <span class="chat-title">{title}</span>
                    <span class="chat-date">{date}</span>
                </div>
            </div>
            """
    html += "</div>"
    return html

# ─────────────────────────────────────────
# STATE
# ─────────────────────────────────────────
current_chat_id = None

def new_chat():
    global current_chat_id
    chat_history.clear()
    chat_history.append({
        "role": "system",
        "content": (
            "You are a helpful food recipe assistant specializing in Tunisian and international cuisine.\n"
            "- Always call RecipeSearch first when the user asks about a recipe.\n"
            "- Call IngredientSubstitute when the user asks for a substitute.\n"
            "- Call WebSearch only if RecipeSearch returns no results.\n"
            "- Be friendly, detailed, and format recipes clearly with ingredients and steps."
        )
    })
    current_chat_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    return [], gr.update(value=render_sidebar())

def respond(message, history):
    global current_chat_id
    if not message.strip():
        return "", history, gr.update()
    response = agent_chat(message)
    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": response})
    if current_chat_id is None:
        current_chat_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    chats = load_saved_chats()
    chats[current_chat_id] = {
        "title": message[:40],
        "date": datetime.now().strftime("%b %d, %Y"),
        "messages": history
    }
    save_chats(chats)
    return "", history, gr.update(value=render_sidebar())

def handle_image(image_path, history):
    global current_chat_id
    if image_path is None:
        return history, gr.update()
    response = chat_with_image(image_path)
    history.append({"role": "user", "content": "📸 I uploaded a food photo — what dish is this?"})
    history.append({"role": "assistant", "content": response})
    if current_chat_id is None:
        current_chat_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    chats = load_saved_chats()
    chats[current_chat_id] = {
        "title": "📸 Photo identification",
        "date": datetime.now().strftime("%b %d, %Y"),
        "messages": history
    }
    save_chats(chats)
    return history, gr.update(value=render_sidebar())

# ─────────────────────────────────────────
# CSS
# ─────────────────────────────────────────
css = """
@import url('https://fonts.googleapis.com/css2?family=Lora:wght@400;500;600;700&family=Nunito:wght@300;400;500;600&display=swap');

:root {
    --spice: #c2410c;
    --terracotta: #d97706;
    --saffron: #f59e0b;
    --cream: #fef3c7;
    --dark: #1c1a17;
    --dark2: #27241e;
    --dark3: #322e26;
    --border: #3d3830;
    --text: #f5e6c8;
    --text-muted: #9a8a6a;
    --text-dim: #5a4e38;
}

* { box-sizing: border-box; margin: 0; padding: 0; }
body, .gradio-container {
    background: var(--dark) !important;
    font-family: 'Nunito', sans-serif !important;
}
.gradio-container { max-width: 100% !important; padding: 0 !important; }
.main-layout { display: flex !important; height: 100vh !important; overflow: hidden !important; }

/* SIDEBAR */
.sidebar {
    width: 270px !important; min-width: 270px !important;
    background: var(--dark2) !important;
    border-right: 1px solid var(--border) !important;
    display: flex !important; flex-direction: column !important;
    height: 100vh !important; overflow: hidden !important;
}
.sidebar-header { padding: 20px 16px 14px !important; border-bottom: 1px solid var(--border) !important; }
.sidebar-title { font-family: 'Lora', serif !important; font-size: 1.1rem !important; font-weight: 700 !important; color: var(--saffron) !important; }
.new-chat-btn { padding: 12px 16px !important; border-bottom: 1px solid var(--border) !important; }
.new-chat-btn button {
    width: 100% !important; background: var(--spice) !important;
    color: #fff !important; border: none !important; border-radius: 10px !important;
    font-family: 'Nunito', sans-serif !important; font-weight: 600 !important;
    font-size: 0.88rem !important; padding: 10px !important; cursor: pointer !important;
    transition: all 0.2s !important;
}
.new-chat-btn button:hover { background: #b91c0c !important; }
.chat-list { overflow-y: auto !important; flex: 1 !important; padding: 8px !important; }
.no-chats { color: var(--text-dim) !important; font-size: 0.8rem !important; text-align: center !important; padding: 24px 16px !important; line-height: 1.6 !important; }
.chat-item { display: flex !important; align-items: center !important; gap: 10px !important; padding: 10px 12px !important; border-radius: 10px !important; cursor: pointer !important; transition: background 0.15s !important; margin-bottom: 4px !important; }
.chat-item:hover { background: var(--dark3) !important; }
.chat-info { display: flex !important; flex-direction: column !important; overflow: hidden !important; flex: 1 !important; }
.chat-title { font-size: 0.82rem !important; color: var(--text-muted) !important; white-space: nowrap !important; overflow: hidden !important; text-overflow: ellipsis !important; font-weight: 500 !important; }
.chat-date { font-size: 0.7rem !important; color: var(--text-dim) !important; margin-top: 2px !important; }
.sidebar-stats { padding: 12px 16px !important; border-top: 1px solid var(--border) !important; display: flex !important; justify-content: space-around !important; }
.stat-num { display: block !important; font-family: 'Lora', serif !important; font-size: 1rem !important; font-weight: 700 !important; color: var(--saffron) !important; text-align: center !important; }
.stat-lbl { display: block !important; font-size: 0.62rem !important; color: var(--text-dim) !important; text-transform: uppercase !important; letter-spacing: 0.8px !important; text-align: center !important; }

/* MAIN */
.main-content { flex: 1 !important; display: flex !important; flex-direction: column !important; height: 100vh !important; overflow: hidden !important; }
.top-header { padding: 16px 28px !important; border-bottom: 1px solid var(--border) !important; display: flex !important; align-items: center !important; gap: 14px !important; background: var(--dark2) !important; }
.main-title { font-family: 'Lora', serif !important; font-size: 1.4rem !important; font-weight: 700 !important; color: var(--cream) !important; margin: 0 !important; }
.main-title span { color: var(--saffron) !important; }
.header-badges { margin-left: auto !important; display: flex !important; gap: 8px !important; }
.badge { font-size: 0.72rem !important; padding: 4px 10px !important; border-radius: 20px !important; font-weight: 600 !important; }
.badge-tools { background: rgba(194,65,12,0.2) !important; color: #fb923c !important; border: 1px solid rgba(194,65,12,0.4) !important; }
.badge-recipes { background: rgba(77,124,15,0.2) !important; color: #a3e635 !important; border: 1px solid rgba(77,124,15,0.4) !important; }

/* CHAT */
.chat-area { flex: 1 !important; overflow: hidden !important; }
.chat-area .chatbot { background: transparent !important; border: none !important; height: 100% !important; }

/* INPUT */
.input-section { padding: 14px 24px 18px !important; border-top: 1px solid var(--border) !important; }
.input-section textarea { background: var(--dark2) !important; border: 1px solid var(--border) !important; border-radius: 14px !important; color: var(--text) !important; font-family: 'Nunito', sans-serif !important; font-size: 0.95rem !important; padding: 13px 18px !important; resize: none !important; transition: border-color 0.2s !important; }
.input-section textarea:focus { border-color: var(--saffron) !important; outline: none !important; box-shadow: 0 0 0 3px rgba(245,158,11,0.12) !important; }
.input-section textarea::placeholder { color: var(--text-dim) !important; }
.send-btn button { background: linear-gradient(135deg, var(--spice), var(--terracotta)) !important; color: #fff !important; border: none !important; border-radius: 14px !important; font-family: 'Nunito', sans-serif !important; font-weight: 700 !important; font-size: 0.92rem !important; padding: 13px 28px !important; cursor: pointer !important; transition: all 0.2s !important; }
.send-btn button:hover { transform: translateY(-2px) !important; box-shadow: 0 6px 20px rgba(194,65,12,0.4) !important; }

/* EXAMPLES */
.examples-wrap { padding: 8px 24px 10px !important; background: var(--dark) !important; }
.examples-wrap .examples button { background: var(--dark2) !important; color: var(--text-muted) !important; border: 1px solid var(--border) !important; border-radius: 20px !important; font-size: 0.78rem !important; padding: 5px 13px !important; cursor: pointer !important; transition: all 0.2s !important; margin: 3px !important; }
.examples-wrap .examples button:hover { border-color: var(--saffron) !important; color: var(--saffron) !important; }

footer { display: none !important; }
.photo-btn { width: 50px !important; height: 50px !important; border-radius: 12px !important; border: 1px dashed var(--saffron) !important; background: var(--dark2) !important; cursor: pointer !important; overflow: hidden !important; }
"""

# ─────────────────────────────────────────
# UI
# ─────────────────────────────────────────
with gr.Blocks(title="🫕 Recipe Agent") as demo:

    with gr.Row(elem_classes="main-layout"):

        # ── SIDEBAR ──
        with gr.Column(elem_classes="sidebar", scale=0, min_width=270):
            sidebar_html = gr.HTML(value=render_sidebar())
            with gr.Row(elem_classes="new-chat-btn"):
                new_chat_btn = gr.Button("＋ New Chat")
            gr.HTML("""
            <div class="sidebar-stats">
                <div><span class="stat-num">522K</span><span class="stat-lbl">Recipes</span></div>
                <div><span class="stat-num">15</span><span class="stat-lbl">Tunisian</span></div>
                <div><span class="stat-num">3</span><span class="stat-lbl">AI Tools</span></div>
            </div>
            """)

        # ── MAIN ──
        with gr.Column(elem_classes="main-content", scale=1):

            # Header
            gr.HTML("""
            <div class="top-header">
                <h1 class="main-title">🫕 Recipe <span>Agent</span></h1>
                <div class="header-badges">
                    <span class="badge badge-tools">⚡ 3 AI Tools</span>
                    <span class="badge badge-recipes">🌿 522K+ Recipes</span>
                </div>
            </div>
            """)

            # Chat
            with gr.Column(elem_classes="chat-area"):
                chatbot = gr.Chatbot(
                    height=360,
                    show_label=False,
                    render_markdown=True,
                    placeholder="<div style='text-align:center;color:#5a4e38;padding:60px 20px;font-family:Lora,serif;font-size:1rem;'>Ask me anything about Tunisian & international recipes 🍽️<br><br>Or upload a food photo to identify a dish!</div>"
                )

            # Examples
            with gr.Column(elem_classes="examples-wrap"):
                gr.Examples(
                    examples=[
                        ["How do I make Brik?"],
                        ["Tunisian recipe with tuna"],
                        ["Substitute for harissa?"],
                        ["How to make Lablabi?"],
                        ["Easy Italian pasta"],
                        ["Popular Tunisian desserts"],
                        ["What can replace merguez?"],
                    ],
                    inputs=gr.Textbox(visible=False),
                    label=""
                )

            # Input
            with gr.Row(elem_classes="input-section"):
                msg = gr.Textbox(
                    placeholder="Ask me about any recipe... or upload a photo 📸",
                    show_label=False,
                    lines=1,
                    scale=5
                )
                image_input = gr.Image(
                    type="filepath",
                    show_label=False,
                    sources=["upload"],
                    height=50,
                    scale=1,
                    elem_classes="photo-btn"
                )
                with gr.Column(scale=1, elem_classes="send-btn"):
                    send_btn = gr.Button("Send 🚀")
                    identify_btn = gr.Button("🔍 Identify", variant="secondary")

    # ── ACTIONS ──
    send_btn.click(respond, [msg, chatbot], [msg, chatbot, sidebar_html])
    msg.submit(respond, [msg, chatbot], [msg, chatbot, sidebar_html])
    new_chat_btn.click(new_chat, outputs=[chatbot, sidebar_html])
    identify_btn.click(handle_image, [image_input, chatbot], [chatbot, sidebar_html])

if __name__ == "__main__":
    demo.launch(share=False, css=css)