import os
import json
import base64
from dotenv import load_dotenv
from groq import Groq
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# ─────────────────────────────────────────
# ENVIRONMENT & CLIENT INITIALIZATION
# ─────────────────────────────────────────
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError("GROQ_API_KEY is not set. Please add it to your .env file.")
print(f"✅ API Key loaded: {api_key[:10]}...")

client = Groq(api_key=api_key)
MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

# ─────────────────────────────────────────
# CHUNKING HELPERS
# ─────────────────────────────────────────
CHUNK_SIZE = 200        # max characters per chunk
CHUNK_OVERLAP = 100      # characters overlapping between consecutive chunks

def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split a long string into overlapping character-level chunks."""
    if not text or len(text) <= size:
        return [text] if text else []
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        start += size - overlap
    return chunks

# ─────────────────────────────────────────
# FAISS VECTOR INDEX (RAG SETUP WITH CHUNKING)
# ─────────────────────────────────────────
print("🧠 Loading Sentence Transformer embedding model...")
encoder = SentenceTransformer("all-MiniLM-L6-v2")

RECIPE_PATH = os.path.join(os.path.dirname(__file__), '..', 'Data', 'tunisian_recipes.json')
with open(RECIPE_PATH, encoding="utf-8") as f:
    recipes = json.load(f)
print(f"✅ Loaded {len(recipes)} recipes — chunking for FAISS indexing...")

# Each entry in chunk_store: {"text": str, "recipe_idx": int}
chunk_store: list[dict] = []

for recipe_idx, r in enumerate(recipes):
    # Build a full context string per recipe then chunk it
    full_text = (
        f"Recipe: {r.get('name', '')}. "
        f"Category: {r.get('category', '')}. "
        f"Cuisine: {r.get('cuisine', '')}. "
        f"Ingredients: {r.get('ingredients', '')}. "
        f"Steps: {r.get('steps', '')}. "
        f"Cook time: {r.get('cook_time', '')}."
    )
    for chunk in chunk_text(full_text):
        chunk_store.append({"text": chunk, "recipe_idx": recipe_idx})

print(f"⚡ {len(chunk_store)} chunks created — generating embeddings and building FAISS index...")
chunk_texts = [c["text"] for c in chunk_store]
embeddings = encoder.encode(chunk_texts, show_progress_bar=False)
dimension = embeddings.shape[1]

faiss_index = faiss.IndexFlatL2(dimension)
faiss_index.add(np.array(embeddings).astype("float32"))
print(f"✅ FAISS index ready ({faiss_index.ntotal} vectors).")

# ─────────────────────────────────────────
# SEMANTIC RAG TOOL FUNCTION
# ─────────────────────────────────────────
def RecipeSearch(query: str, top_k: int = 5) -> str:
    """
    Semantic FAISS search over chunked recipe text.
    Retrieves top_k chunks, deduplicates by parent recipe,
    and returns up to 3 unique full recipes.
    """
    query_vector = encoder.encode([query]).astype("float32")
    k = min(top_k, faiss_index.ntotal)
    distances, indices = faiss_index.search(query_vector, k)

    seen_recipes: set[int] = set()
    output = "Semantic Search Results (FAISS Chunk Match):\n"
    found = 0

    for idx in indices[0]:
        if idx < 0 or idx >= len(chunk_store):
            continue
        recipe_idx = chunk_store[idx]["recipe_idx"]
        if recipe_idx in seen_recipes:
            continue  # deduplicate — only show each recipe once
        seen_recipes.add(recipe_idx)
        found += 1

        r = recipes[recipe_idx]
        output += f"\n🍽️ {r.get('name', 'Unknown')}\n"
        output += f"   Ingredients: {r.get('ingredients', 'N/A')}\n"
        output += f"   Steps: {r.get('steps', 'N/A')}\n"
        output += f"   Cook time: {r.get('cook_time', 'N/A')}\n---"

        if found >= 3:
            break

    return output if found > 0 else f"No matching recipe found for: '{query}'"

# ─────────────────────────────────────────
# ADDITIONAL TOOLS
# ─────────────────────────────────────────
def WebSearch(query: str) -> str:
    try:
        from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            results = list(ddgs.text(query + " recipe", max_results=3))
        if results:
            output = "Web results:\n"
            for r in results:
                output += f"\n🌐 {r['title']}\n{r['body']}\n---"
            return output
        return "No web results found."
    except Exception as e:
        return f"Web search failed: {str(e)}"

def IngredientSubstitute(ingredient: str) -> str:
    substitutes = {
        "harissa": "chili paste or sriracha",
        "merguez": "spicy lamb sausage",
        "tabil": "coriander + caraway mix",
        "preserved lemon": "fresh lemon zest",
        "smen": "aged butter or ghee",
        "frik": "bulgur wheat or freekeh",
        "malsouka": "filo pastry sheets",
        "brik pastry": "filo pastry sheets",
        "ras el hanout": "cumin + coriander + cinnamon + ginger",
        "orange blossom water": "rose water or vanilla extract"
    }
    result = substitutes.get(ingredient.lower().strip())
    if result:
        return f"✅ Substitute for '{ingredient}': {result}"
    return f"No substitute found for '{ingredient}'. Try WebSearch!"

# ─────────────────────────────────────────
# TOOL DECLARATIONS FOR THE AGENT
# ─────────────────────────────────────────
tools = [
    {
        "type": "function",
        "function": {
            "name": "RecipeSearch",
            "description": "Search semantically for food recipes using chunked FAISS vector retrieval. Use this first for any recipe requests.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The concept or dish query, e.g. 'spicy tuna starter', 'Brik', 'chicken'"
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "WebSearch",
            "description": "Search the web for recipes not found in the local vector database.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "The search query"}},
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "IngredientSubstitute",
            "description": "Get an ingredient alternative if the target item is missing.",
            "parameters": {
                "type": "object",
                "properties": {"ingredient": {"type": "string", "description": "Ingredient name"}},
                "required": ["ingredient"]
            }
        }
    }
]

def run_tool(name, args):
    if name == "RecipeSearch":
        return RecipeSearch(args.get("query", ""))
    elif name == "WebSearch":
        return WebSearch(args.get("query", ""))
    elif name == "IngredientSubstitute":
        return IngredientSubstitute(args.get("ingredient", ""))
    return f"Unknown tool: {name}"

# ─────────────────────────────────────────
# CONVERSATIONAL STATE
# ─────────────────────────────────────────
SYSTEM_PROMPT = (
    "You are a helpful food recipe assistant specializing in Tunisian and international cuisine.\n"
    "- Always call RecipeSearch first when the user asks about recipes to query the FAISS vector space.\n"
    "- Call IngredientSubstitute when looking for cooking alternatives.\n"
    "- Call WebSearch only if the local vector lookup yields irrelevant contexts.\n"
    "- Be friendly, detailed, and format your markdown lists clearly."
)

chat_history = [{"role": "system", "content": SYSTEM_PROMPT}]

# ─────────────────────────────────────────
# CHAT LOOP  (temperature-aware)
# ─────────────────────────────────────────
def chat(user_input: str, temperature: float = 0.7) -> str:
    """
    Run the agentic chat loop.
    temperature: 0.0 = deterministic/precise, 1.5 = very creative.
    """
    chat_history.append({"role": "user", "content": user_input})
    max_iterations = 5

    for _ in range(max_iterations):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=chat_history,
                tools=tools,
                tool_choice="auto",
                max_tokens=1024,
                temperature=temperature,
            )
        except Exception as e:
            error_msg = f"Error processing agent loop: {str(e)}"
            chat_history.append({"role": "assistant", "content": error_msg})
            return error_msg

        message = response.choices[0].message

        if not message.tool_calls:
            reply = message.content or "I couldn't generate a response."
            chat_history.append({"role": "assistant", "content": reply})
            return reply

        chat_history.append({
            "role": "assistant",
            "content": message.content or "",
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments}
                } for tc in message.tool_calls
            ]
        })

        for tc in message.tool_calls:
            try:
                args = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                args = {}
            tool_result = run_tool(tc.function.name, args)
            chat_history.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": str(tool_result)
            })

    return "Processing timeout reached."

# ─────────────────────────────────────────
# VISION PROCESSING  (temperature-aware)
# ─────────────────────────────────────────
def identify_dish_from_image(image_path: str, temperature: float = 0.3) -> str:
    with open(image_path, "rb") as f:
        image_data = base64.b64encode(f.read()).decode("utf-8")

    ext = image_path.split(".")[-1].lower()
    mime = f"image/{'png' if ext == 'png' else 'jpeg'}"

    response = client.chat.completions.create(
        model=VISION_MODEL,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{image_data}"}},
                    {"type": "text", "text": "Identify this dish, its visible components, and its national cuisine origins. Respond strictly using lines stating 'Dish: [name]', 'Ingredients: [list]', and 'Cuisine: [type]'."}
                ]
            }
        ],
        max_tokens=300,
        temperature=temperature,
    )
    return response.choices[0].message.content.strip()

def chat_with_image(image_path: str, temperature: float = 0.7) -> str:
    try:
        # Use lower temp for identification (factual), user temp for recipe generation
        identification = identify_dish_from_image(image_path, temperature=max(0.1, temperature - 0.3))
    except Exception as e:
        return f"❌ Analysis fault: {str(e)}"

    dish_name = "this dish"
    for line in identification.split("\n"):
        if line.startswith("Dish:"):
            dish_name = line.replace("Dish:", "").strip()
            break

    recipe_result = chat(
        f"Give me the full detailed recipe for {dish_name} with ingredients and steps",
        temperature=temperature
    )
    return f"## 🔍 Image Analysis\n{identification}\n\n---\n\n## 📖 Recipe for {dish_name}\n{recipe_result}"
