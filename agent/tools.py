import os
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

try:
    from duckduckgo_search import DDGS
except ImportError:
    DDGS = None

# ─────────────────────────────────────────
# LOAD MERGED DATASET
# ─────────────────────────────────────────
print("Loading final combined recipes dataset...")
BASE_DIR = os.path.dirname(__file__)
RECIPE_PATH = os.path.join(BASE_DIR, "..", "Data", "tunisian_recipes.json")

try:
    with open(RECIPE_PATH, encoding="utf-8") as f:
        recipes = json.load(f)
    print(f"✅ Loaded {len(recipes)} total recipes successfully!")
except Exception as e:
    print(f"❌ Error loading JSON file at {RECIPE_PATH}: {e}")
    recipes = []


# ─────────────────────────────────────────
# FAISS VECTOR SELECTION ENGINE SETUP
# ─────────────────────────────────────────
print("🧠 Initializing Sentence Transformer embedding model for tools...")
encoder = SentenceTransformer("all-MiniLM-L6-v2")

recipe_texts = []
for r in recipes:
    combined_text = (
        f"Recipe Name: {r.get('name', '')}. "
        f"Description: {r.get('description', '')}. "
        f"Ingredients: {r.get('ingredients', '')}. "
        f"Category: {r.get('category', '')}. "
        f"Cuisine: {r.get('cuisine', '')}."
    ).lower()
    recipe_texts.append(combined_text)

print("⚡ Building FAISS structural matrix indices...")
if recipe_texts:
    embeddings = encoder.encode(recipe_texts, show_progress_bar=False)
    dimension = embeddings.shape[1]
    faiss_index = faiss.IndexFlatL2(dimension)
    faiss_index.add(np.array(embeddings).astype("float32"))
    print("✅ Tool-level FAISS Semantic Database successfully operational.")
else:
    faiss_index = None
    print("⚠️ Warning: FAISS index empty because no recipes were loaded.")


# ─────────────────────────────────────────
# TOOL 1 — SEMANTIC RECIPE SEARCH
# ─────────────────────────────────────────
def search_recipe(query):
    """Searches tunisian_recipes.json using semantic FAISS vector search."""
    if not recipes or faiss_index is None:
        return "The local recipe database is currently offline or empty."

    query_vector = encoder.encode([query.lower()]).astype("float32")
    k = min(3, len(recipes))
    distances, indices = faiss_index.search(query_vector, k)

    valid_results = 0
    output = f"Semantic Vector Match (FAISS) found the following top results for '{query}':\n"

    for idx in indices[0]:
        if 0 <= idx < len(recipes):
            valid_results += 1
            r = recipes[idx]
            cuisine_tag = r.get('cuisine', 'International').upper()
            output += f"\n 🍽️ {r.get('name')} [{cuisine_tag}]\n"
            output += f"   Description: {r.get('description', '')}\n"
            output += f"   Ingredients: {r.get('ingredients', '')}\n"
            output += f"   Quantities: {r.get('quantities', '')}\n"
            output += f"   Steps: {r.get('steps', '')}\n"
            output += f"   Cook time: {r.get('cook_time', 'N/A')}\n"
            output += "   ---"

    if valid_results > 0:
        return output

    return f"No contextually mapping recipe discovered for query: {query}"


# ─────────────────────────────────────────
# TOOL 2 — WEB SEARCH
# ─────────────────────────────────────────
def web_search(query):
    """Searches the internet for recipes or cooking info."""
    if DDGS is None:
        return "Web search unavailable: duckduckgo_search package not installed."
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query + " recipe", max_results=3))

        if results:
            output = "Web search results:\n"
            for r in results:
                output += f"\n🌐 {r['title']}\n"
                output += f"   {r['body']}\n"
                output += "   ---"
            return output

        return "No web results found."

    except Exception as e:
        return f"Web search failed: {str(e)}"


# ─────────────────────────────────────────
# TOOL 3 — INGREDIENT SUBSTITUTE
# ─────────────────────────────────────────
def get_substitute(ingredient):
    """Suggests a substitute for a missing ingredient."""
    substitutes = {
        "harissa": "chili paste or sriracha",
        "merguez": "spicy lamb sausage",
        "tabil": "coriander + caraway mix",
        "preserved lemon": "fresh lemon zest",
        "smen": "aged butter or ghee",
        "frik": "bulgur wheat or freekeh",
        "malsouka": "filo pastry sheets",
        "brik pastry": "filo pastry sheets",
        "ras el hanout": "cumin + coriander + cinnamon + ginger mix",
        "orange blossom water": "rose water or vanilla extract",
        "lamb": "beef or chicken",
        "merguez sausage": "chorizo or spicy sausage"
    }

    ingredient = ingredient.lower().strip()
    result = substitutes.get(ingredient)

    if result:
        return f"✅ Substitute for '{ingredient}': {result}"
    else:
        return f"No substitute found for '{ingredient}' in my list. Try the web search tool!"


# ─────────────────────────────────────────
# EXPORT ALL TOOLS
# ─────────────────────────────────────────
tools = [search_recipe, web_search, get_substitute]