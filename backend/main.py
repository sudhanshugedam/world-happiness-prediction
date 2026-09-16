# ============================================================
# WORLD HAPPINESS PREDICTION - FASTAPI BACKEND
# Complete ML Dashboard Backend
# ============================================================

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import joblib
import json
import os
import numpy as np
import pandas as pd


# ============================================================
# 1. CREATE FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="World Happiness Prediction API",
    description="Machine Learning API for World Happiness Score Prediction",
    version="2.0.0"
)


# ============================================================
# 2. ENABLE CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# 3. PROJECT PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

MODEL_FOLDER = os.path.join(
    BASE_DIR,
    "model"
)

MODEL_PATH = os.path.join(
    MODEL_FOLDER,
    "happiness_model.pkl"
)

INFO_PATH = os.path.join(
    MODEL_FOLDER,
    "model_info.json"
)

COMPARISON_PATH = os.path.join(
    MODEL_FOLDER,
    "model_comparison.csv"
)

PLOTS_FOLDER = os.path.join(
    MODEL_FOLDER,
    "plots"
)


# ============================================================
# 4. LOAD MACHINE LEARNING MODEL
# ============================================================

try:

    model = joblib.load(MODEL_PATH)

    print("Machine Learning model loaded successfully!")

except Exception as e:

    model = None

    print("ERROR loading model:", e)


# ============================================================
# 5. LOAD MODEL INFORMATION
# ============================================================

model_info = {}

try:

    with open(INFO_PATH, "r") as file:

        model_info = json.load(file)

    print("Model information loaded successfully!")

except Exception as e:

    print("ERROR loading model information:", e)


# ============================================================
# 6. MOUNT GENERATED PLOTS
# ============================================================

if os.path.exists(PLOTS_FOLDER):

    app.mount(
        "/plots",
        StaticFiles(directory=PLOTS_FOLDER),
        name="plots"
    )


# ============================================================
# 7. INPUT DATA MODEL
# ============================================================

class HappinessInput(BaseModel):

    gdp_per_capita: float = Field(
        ...,
        description="Log GDP per capita"
    )

    social_support: float = Field(
        ...,
        description="Social support"
    )

    healthy_life_expectancy: float = Field(
        ...,
        description="Healthy life expectancy"
    )

    freedom: float = Field(
        ...,
        description="Freedom to make life choices"
    )

    generosity: float = Field(
        ...,
        description="Generosity"
    )

    corruption: float = Field(
        ...,
        description="Perceptions of corruption"
    )


# ============================================================
# 8. HOME
# ============================================================

@app.get("/")
def home():

    return {
        "message": "World Happiness Prediction API is running!",
        "status": "success"
    }


# ============================================================
# 9. HEALTH CHECK
# ============================================================

@app.get("/health")
def health_check():

    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "api": "World Happiness Prediction API"
    }


# ============================================================
# 10. MODEL INFORMATION
# ============================================================

@app.get("/model-info")
def get_model_info():

    if not model_info:

        raise HTTPException(
            status_code=500,
            detail="Model information is not available."
        )

    return model_info


# ============================================================
# 11. HAPPINESS PREDICTION
# ============================================================

@app.post("/predict")
def predict_happiness(data: HappinessInput):

    if model is None:

        raise HTTPException(
            status_code=500,
            detail="Machine Learning model is not loaded."
        )

    try:

        input_data = np.array([[
            data.gdp_per_capita,
            data.social_support,
            data.healthy_life_expectancy,
            data.freedom,
            data.generosity,
            data.corruption
        ]])

        prediction = model.predict(input_data)[0]

        prediction = float(
            np.clip(prediction, 0, 10)
        )

        return {

            "success": True,

            "predicted_happiness_score":
                round(prediction, 3),

            "model":
                model_info.get(
                    "model_name",
                    "Gradient Boosting Regression"
                )

        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Prediction error: {str(e)}"
        )


# ============================================================
# 12. ANALYTICS DATA
# ============================================================

@app.get("/analytics")
def get_analytics():

    try:

        # ----------------------------------------------------
        # MODEL COMPARISON
        # ----------------------------------------------------

        comparison = []

        if os.path.exists(COMPARISON_PATH):

            comparison_df = pd.read_csv(
                COMPARISON_PATH
            )

            comparison = comparison_df.to_dict(
                orient="records"
            )


        # ----------------------------------------------------
        # FEATURE IMPORTANCE
        # ----------------------------------------------------

        features = model_info.get(
            "features",
            [
                "Explained by: Log GDP per capita",
                "Explained by: Social support",
                "Explained by: Healthy life expectancy",
                "Explained by: Freedom to make life choices",
                "Explained by: Generosity",
                "Explained by: Perceptions of corruption"
            ]
        )

        feature_names = [
            "Log GDP per capita",
            "Social Support",
            "Healthy Life Expectancy",
            "Freedom",
            "Generosity",
            "Perceptions of Corruption"
        ]

        feature_importance = []

        if model is not None and hasattr(
            model,
            "feature_importances_"
        ):

            values = model.feature_importances_

            for name, value in zip(
                feature_names,
                values
            ):

                feature_importance.append({
                    "feature": name,
                    "importance": round(
                        float(value),
                        4
                    )
                })

            feature_importance.sort(
                key=lambda x: x["importance"],
                reverse=True
            )


        # ----------------------------------------------------
        # DATASET STATISTICS
        # ----------------------------------------------------

        total_samples = model_info.get(
            "total_clean_samples",
            1013
        )

        training_samples = model_info.get(
            "training_samples",
            810
        )

        testing_samples = model_info.get(
            "testing_samples",
            203
        )

        r2_score = model_info.get(
            "r2_score",
            0
        )

        rmse = model_info.get(
            "rmse",
            0
        )

        mae = model_info.get(
            "mae",
            0
        )


        # ----------------------------------------------------
        # RETURN ALL ANALYTICS
        # ----------------------------------------------------

        return {

            "success": True,

            "model": {
                "name": model_info.get(
                    "model_name",
                    "Gradient Boosting Regression"
                ),
                "r2": r2_score,
                "rmse": rmse,
                "mae": mae
            },

            "dataset": {

                "original_samples": 2116,

                "clean_samples":
                    total_samples,

                "features":
                    len(features),

                "training_samples":
                    training_samples,

                "testing_samples":
                    testing_samples
            },

            "model_comparison":
                comparison,

            "feature_importance":
                feature_importance,

            "plots": {

                "correlation":
                    "/plots/correlation_heatmap.png",

                "distribution":
                    "/plots/happiness_distribution.png",

                "actual_predicted":
                    "/plots/actual_vs_predicted.png",

                "feature_importance":
                    "/plots/feature_importance.png"
            }

        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Analytics error: {str(e)}"
        )


# ============================================================
# END
# ============================================================

print("=" * 60)
print("World Happiness Prediction Backend")
print("=" * 60)
print("API ready")
print("Model:", model_info.get("model_name", "Unknown"))
print("=" * 60)