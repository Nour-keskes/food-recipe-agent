# cleandata.py
import os
import pandas as pd

# Get the directory where this script lives
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# FORCE CAPITALIZED "Data" FOLDER
DATA_DIR = os.path.join(BASE_DIR, "Data")
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR, exist_ok=True)

CSV_PATH = os.path.join(DATA_DIR, "recipes.csv")
OUTPUT_PATH = os.path.join(DATA_DIR, "recipes_clean.json")

print(f"Reading Kaggle data from: {CSV_PATH}")
if not os.path.exists(CSV_PATH):
    print(f"❌ Error: Could not find 'recipes.csv' inside {DATA_DIR}!")
else:
    df = pd.read_csv(CSV_PATH)

    # Select target columns
    df = df[[
        "Name",
        "RecipeIngredientParts",      
        "RecipeIngredientQuantities", 
        "RecipeInstructions",          
        "CookTime",
        "PrepTime",
        "RecipeCategory",
        "Calories",
        "RecipeServings",
        "AggregatedRating",
        "Description"
    ]]

    # Remove missing entries
    df = df.dropna(subset=["Name", "RecipeIngredientParts", "RecipeInstructions"])

    # Mark dataset source tags
    df["cuisine"] = "international"

    # Map to standard JSON nomenclature
    df = df.rename(columns={
        "Name": "name",
        "RecipeIngredientParts": "ingredients",
        "RecipeIngredientQuantities": "quantities",
        "RecipeInstructions": "steps",
        "CookTime": "cook_time",
        "PrepTime": "prep_time",
        "RecipeCategory": "category",
        "Calories": "calories",
        "RecipeServings": "servings",
        "AggregatedRating": "rating",
        "Description": "description"
    })

    print(f"Total processed Kaggle recipes: {len(df)}")

    # Export clean intermediate block
    df.to_json(OUTPUT_PATH, orient="records", indent=2)
    print(f"✅ Saved clean intermediate file to: {OUTPUT_PATH}")