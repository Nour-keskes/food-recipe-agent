import pandas as pd   # library nekhedmou beha data tables w manipulation taa csv files
import json

# Load kaggle csv file fi dataframe (df=dataframe=A data table in pandas kima Excel)
df = pd.read_csv("data/recipes.csv")

#  nkhalliw ken columns li hachetna bihom . The double brackets [[...]] mean "select these specific columns"
df = df[[
    "Name",
    "RecipeIngredientParts",      # ingredients
    "RecipeIngredientQuantities", # quantities
    "RecipeInstructions",         # steps
    "CookTime",
    "PrepTime",
    "RecipeCategory",
    "Calories",
    "RecipeServings",
    "AggregatedRating",
    "Description"
]]

# Remove rows with missing values in essential columns
df = df.dropna(subset=["Name", "RecipeIngredientParts", "RecipeInstructions"])

# Add cuisine column
df["cuisine"] = "international"

# Rename columns to cleaner names
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

# Print info
print(f" Total recipes after cleaning: {len(df)}")
print(df.head(2))

# Save to JSON orient="records" means each recipe becomes one object
df.to_json("data/recipes_clean.json", orient="records", indent=2)
print("Saved to data/recipes_clean.json")