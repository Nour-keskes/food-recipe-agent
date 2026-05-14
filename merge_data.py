import json
import pandas as pd

# Opens your big Kaggle JSON file and loads it into a table (like Excel). The result is stored in kaggle_df.
kaggle_df = pd.read_json(r"F:\ProjetAgentAI\data\recipes_clean.json")
#Converts that table into a Python list where each recipe is a dictionary
kaggle_data = kaggle_df.to_dict(orient="records") #one recipe = one row"

# Load Tunisian recipes
with open(r"F:\ProjetAgentAI\data\tunisian_recipes.json", encoding="utf-8") as f:
    tunisian_data = json.load(f)

# Merge both
final_data = kaggle_data + tunisian_data

# Save final dataset    
with open(r"F:\ProjetAgentAI\data\recipes_final.json", "w", encoding="utf-8") as f:
    json.dump(final_data, f, indent=2)

print(f" Kaggle recipes: {len(kaggle_data)}")
print(f" Tunisian recipes: {len(tunisian_data)}")
print(f" Total final dataset: {len(final_data)} recipes")
print(" Saved to data/recipes_final.json")