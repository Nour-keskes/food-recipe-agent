import json
from langchain_core.tools import Tool
from ddgs import DDGS

# ─────────────────────────────────────────
# LOAD DATASET
# ─────────────────────────────────────────
print("Loading recipes dataset...")
with open(r"F:\ProjetAgentAI\data\recipes_final.json", encoding="utf-8") as f:
    recipes = json.load(f)
print(f"✅ Loaded {len(recipes)} recipes")


# ─────────────────────────────────────────
# TOOL 1 — RECIPE SEARCH
# ─────────────────────────────────────────
def search_recipe(query):
    """Searches recipes_final.json by name or ingredient"""
    query = query.lower()
    
    results = [
        r for r in recipes
        if query in str(r.get("name", "")).lower()
        or query in str(r.get("ingredients", "")).lower()
        or query in str(r.get("category", "")).lower()
    ]
    
    if results:
        top3 = results[:3]  # return only top 3 results
        output = f"Found {len(results)} recipes. Here are the top 3:\n"
        for r in top3:
            output += f"\n {r.get('name')}\n"
            output += f"   Ingredients: {r.get('ingredients')}\n"
            output += f"   Steps: {r.get('steps')}\n"
            output += f"   Cook time: {r.get('cook_time')}\n"
            output += "   ---"
        return output
    
    return f"No recipe found for: {query}"


recipe_tool = Tool(
    name="RecipeSearch",
    func=search_recipe,
    description="""Use this tool to search for food recipes by dish name or ingredient.
    Input: a dish name like 'Brik' or an ingredient like 'chicken' or 'tuna'."""
)


# ─────────────────────────────────────────
# TOOL 2 — WEB SEARCH
# ─────────────────────────────────────────
def web_search(query):
    """Searches the internet for recipes or cooking info"""
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


web_tool = Tool(
    name="WebSearch",
    func=web_search,
    description="""Use this tool to search the internet for recipes or cooking tips
    that are not found in the database.
    Input: a dish name or cooking question."""
)


# ─────────────────────────────────────────
# TOOL 3 — INGREDIENT SUBSTITUTE
# ─────────────────────────────────────────
def get_substitute(ingredient):
    """Suggests a substitute for a missing ingredient"""
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


substitute_tool = Tool(
    name="IngredientSubstitute",
    func=get_substitute,
    description="""Use this tool when the user doesn't have an ingredient and needs a substitute.
    Input: a single ingredient name like 'harissa' or 'merguez'."""
)


# ─────────────────────────────────────────
# EXPORT ALL TOOLS
# ─────────────────────────────────────────
tools = [recipe_tool, web_tool, substitute_tool]