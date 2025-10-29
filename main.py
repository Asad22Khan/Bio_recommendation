from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------
# Initialize FastAPI App
# -----------------------
app = FastAPI(
    title="Bio Recommendation API",
    description="Suggest top-k similar user bios based on input text.",
    version="1.0.0"
)

# -----------------------
# Load Model and Data

with open("bio_recommendation_system.pkl", "rb") as f: 
    data = pickle.load(f) 
    
model = data["model"] 
df = data["df"]


# -----------------------
# Request Schema
# -----------------------
class RecommendRequest(BaseModel):
    query: str
    top_k: int = 10

# -----------------------
# Recommendation Function
# -----------------------
def recommend_bio(query: str, top_k: int = 10):
    query_vec = model.encode([query])
    bio_matrix = np.vstack(df["embeddings"].values)
    similarities = cosine_similarity(query_vec, bio_matrix)[0]
    
    df["similarity"] = similarities
    results = df.sort_values(by="similarity", ascending=False).head(top_k)
    
    recommendations = []
    for _, row in results.iterrows():
        recommendations.append({
            "user_id": str(row["id"]),
            "Bio": row["bio"],
            "Similarity": round(float(row["similarity"]), 4)
        })
    return recommendations

# -----------------------
# API Endpoint
# -----------------------
@app.post("/recommend")
def get_recommendations(request: RecommendRequest):
    """
    Get top-k user bios similar to the provided bio text.
    """
    recs = recommend_bio(request.query, request.top_k)
    return {
        "query": request.query,
        "top_k": request.top_k,
        "recommendations": recs
    }

# -----------------------
# Root Endpoint
# -----------------------
@app.get("/")
def home():
    return {"message": "Welcome to the Bio Recommendation API! Use /docs to test the endpoint."}
