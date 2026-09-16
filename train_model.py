# ============================================================
# WORLD HAPPINESS REPORT - HAPPINESS SCORE PREDICTION
# Teacher Assessment Examination (TAE)
# Machine Learning Project
# ============================================================

import os
import json
import warnings

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")


# ============================================================
# 1. PROJECT CONFIGURATION
# ============================================================

DATASET_PATH = "dataset/WHR26_Data_Figure_2.1 (1).xlsx"
MODEL_FOLDER = "model"
PLOT_FOLDER = "model/plots"

os.makedirs(MODEL_FOLDER, exist_ok=True)
os.makedirs(PLOT_FOLDER, exist_ok=True)


# ============================================================
# 2. LOAD DATASET
# ============================================================

print("=" * 70)
print("WORLD HAPPINESS REPORT - ML PROJECT")
print("=" * 70)

print("\n[1] Loading dataset...")

df = pd.read_excel(DATASET_PATH)

print("Dataset loaded successfully!")
print("Original dataset shape:", df.shape)


# ============================================================
# 3. DEFINE FEATURES AND TARGET
# ============================================================

features = [
    "Explained by: Log GDP per capita",
    "Explained by: Social support",
    "Explained by: Healthy life expectancy",
    "Explained by: Freedom to make life choices",
    "Explained by: Generosity",
    "Explained by: Perceptions of corruption"
]

target = "Life evaluation (3-year average)"


print("\n[2] Machine Learning Features:")

for i, feature in enumerate(features, 1):
    print(f"{i}. {feature}")

print("\nTarget variable:")
print(target)


# ============================================================
# 4. SELECT REQUIRED DATA
# ============================================================

data = df[features + [target]].copy()

print("\n[3] Selected data shape:", data.shape)


# ============================================================
# 5. CHECK MISSING VALUES
# ============================================================

print("\n[4] Missing values before cleaning:")

missing_values = data.isnull().sum()

print(missing_values)


# ============================================================
# 6. REMOVE INCOMPLETE OBSERVATIONS
# ============================================================
# The selected WHR dataset contains many missing values in
# the six explanatory factors.
#
# Instead of replacing a large part of the dataset with the
# same artificial median values, only complete observations
# are used for model training.


print("\n[5] Cleaning dataset...")

data_clean = data.dropna().copy()

print("Rows before cleaning:", len(data))
print("Rows after cleaning :", len(data_clean))
print("Rows removed        :", len(data) - len(data_clean))


# ============================================================
# 7. CREATE X AND y
# ============================================================

X = data_clean[features]
y = data_clean[target]

print("\n[6] Final ML dataset:")
print("Input features (X):", X.shape)
print("Target (y):", y.shape)


# ============================================================
# 8. DISPLAY CLEAN DATA
# ============================================================

print("\n[7] First 5 cleaned records:")
print(data_clean.head())


# ============================================================
# 9. STATISTICAL SUMMARY
# ============================================================

print("\n[8] Statistical summary:")
print(X.describe())


print("\nHappiness score summary:")
print(y.describe())


# ============================================================
# 10. EXPLORATORY DATA ANALYSIS
# ============================================================

print("\n[9] Creating data visualizations...")


# ------------------------------------------------------------
# 10.1 Correlation Heatmap
# ------------------------------------------------------------

plt.figure(figsize=(11, 8))

correlation_data = data_clean[features + [target]].corr()

sns.heatmap(
    correlation_data,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    linewidths=0.5
)

plt.title("Correlation Matrix - World Happiness Dataset")
plt.tight_layout()

plt.savefig(
    os.path.join(PLOT_FOLDER, "correlation_heatmap.png"),
    dpi=300
)

plt.close()


# ------------------------------------------------------------
# 10.2 Happiness Score Distribution
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

plt.hist(
    y,
    bins=20,
    edgecolor="black"
)

plt.xlabel("Happiness Score")
plt.ylabel("Number of Observations")
plt.title("Distribution of Happiness Scores")

plt.tight_layout()

plt.savefig(
    os.path.join(PLOT_FOLDER, "happiness_distribution.png"),
    dpi=300
)

plt.close()


# ------------------------------------------------------------
# 10.3 Feature vs Happiness Scatter Plots
# ------------------------------------------------------------

short_names = {
    "Explained by: Log GDP per capita": "Log GDP per capita",
    "Explained by: Social support": "Social Support",
    "Explained by: Healthy life expectancy": "Healthy Life Expectancy",
    "Explained by: Freedom to make life choices": "Freedom",
    "Explained by: Generosity": "Generosity",
    "Explained by: Perceptions of corruption": "Perceptions of Corruption"
}


for feature in features:

    plt.figure(figsize=(8, 5))

    plt.scatter(
        X[feature],
        y,
        alpha=0.6
    )

    plt.xlabel(short_names[feature])
    plt.ylabel("Happiness Score")
    plt.title(
        f"{short_names[feature]} vs Happiness Score"
    )

    plt.tight_layout()

    filename = short_names[feature].lower().replace(" ", "_") + "_vs_happiness.png"

    plt.savefig(
        os.path.join(PLOT_FOLDER, filename),
        dpi=300
    )

    plt.close()


# ============================================================
# 11. TRAIN-TEST SPLIT
# ============================================================

print("\n[10] Splitting dataset into training and testing sets...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("Training samples:", len(X_train))
print("Testing samples :", len(X_test))


# ============================================================
# 12. CREATE MACHINE LEARNING MODELS
# ============================================================

print("\n[11] Creating regression models...")

models = {

    "Multiple Linear Regression": LinearRegression(),

    "Random Forest Regression": RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        max_depth=None,
        min_samples_split=2,
        n_jobs=-1
    ),

    "Gradient Boosting Regression": GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        random_state=42
    )
}


# ============================================================
# 13. TRAIN AND EVALUATE MODELS
# ============================================================

print("\n[12] Training and evaluating models...")

results = []
trained_models = {}

for model_name, model in models.items():

    print(f"\nTraining: {model_name}")

    # Train
    model.fit(X_train, y_train)

    # Predict
    predictions = model.predict(X_test)

    # Metrics
    mae = mean_absolute_error(y_test, predictions)

    mse = mean_squared_error(y_test, predictions)

    rmse = np.sqrt(mse)

    r2 = r2_score(y_test, predictions)

    # Store model
    trained_models[model_name] = model

    # Store results
    results.append({
        "Model": model_name,
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2 Score": r2
    })

    print(f"MAE      : {mae:.4f}")
    print(f"MSE      : {mse:.4f}")
    print(f"RMSE     : {rmse:.4f}")
    print(f"R2 Score : {r2:.4f}")


# ============================================================
# 14. MODEL COMPARISON
# ============================================================

results_df = pd.DataFrame(results)

# Sort by R2 score from highest to lowest
results_df = results_df.sort_values(
    by="R2 Score",
    ascending=False
).reset_index(drop=True)


print("\n" + "=" * 70)
print("MODEL COMPARISON")
print("=" * 70)

print(
    results_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# ============================================================
# 15. SELECT BEST MODEL
# ============================================================

best_model_name = results_df.iloc[0]["Model"]

best_model = trained_models[best_model_name]

print("\n" + "=" * 70)
print("BEST MODEL")
print("=" * 70)

print("Selected model:", best_model_name)

best_r2 = results_df.iloc[0]["R2 Score"]
best_rmse = results_df.iloc[0]["RMSE"]
best_mae = results_df.iloc[0]["MAE"]

print(f"R2 Score: {best_r2:.4f}")
print(f"RMSE    : {best_rmse:.4f}")
print(f"MAE     : {best_mae:.4f}")


# ============================================================
# 16. BEST MODEL PREDICTIONS
# ============================================================

best_predictions = best_model.predict(X_test)


# ============================================================
# 17. ACTUAL VS PREDICTED GRAPH
# ============================================================

print("\n[13] Creating Actual vs Predicted graph...")

plt.figure(figsize=(8, 6))

plt.scatter(
    y_test,
    best_predictions,
    alpha=0.7
)

# Perfect prediction line
minimum = min(y_test.min(), best_predictions.min())
maximum = max(y_test.max(), best_predictions.max())

plt.plot(
    [minimum, maximum],
    [minimum, maximum],
    linestyle="--"
)

plt.xlabel("Actual Happiness Score")
plt.ylabel("Predicted Happiness Score")

plt.title(
    f"Actual vs Predicted Happiness Score\n{best_model_name}"
)

plt.tight_layout()

plt.savefig(
    os.path.join(PLOT_FOLDER, "actual_vs_predicted.png"),
    dpi=300
)

plt.close()


# ============================================================
# 18. FEATURE IMPORTANCE
# ============================================================

print("\n[14] Creating feature importance analysis...")

if hasattr(best_model, "feature_importances_"):

    importance_values = best_model.feature_importances_

    importance_df = pd.DataFrame({
        "Feature": features,
        "Importance": importance_values
    })

    importance_df = importance_df.sort_values(
        by="Importance",
        ascending=True
    )

    print("\nFeature Importance:")
    print(
        importance_df.sort_values(
            by="Importance",
            ascending=False
        ).to_string(index=False)
    )

    plt.figure(figsize=(9, 6))

    plt.barh(
        importance_df["Feature"].map(short_names),
        importance_df["Importance"]
    )

    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.title("Feature Importance - Best Model")

    plt.tight_layout()

    plt.savefig(
        os.path.join(PLOT_FOLDER, "feature_importance.png"),
        dpi=300
    )

    plt.close()

else:

    # Linear Regression coefficients
    coefficients = best_model.coef_

    coefficient_df = pd.DataFrame({
        "Feature": features,
        "Coefficient": coefficients
    })

    print("\nLinear Regression Coefficients:")
    print(coefficient_df.to_string(index=False))


# ============================================================
# 19. SAVE BEST MODEL
# ============================================================

print("\n[15] Saving trained model...")

model_path = os.path.join(
    MODEL_FOLDER,
    "happiness_model.pkl"
)

joblib.dump(
    best_model,
    model_path
)

print("Model saved successfully:")
print(model_path)


# ============================================================
# 20. SAVE FEATURE INFORMATION
# ============================================================

model_info = {
    "model_name": best_model_name,
    "features": features,
    "target": target,
    "training_samples": int(len(X_train)),
    "testing_samples": int(len(X_test)),
    "total_clean_samples": int(len(data_clean)),
    "mae": float(best_mae),
    "rmse": float(best_rmse),
    "r2_score": float(best_r2)
}

info_path = os.path.join(
    MODEL_FOLDER,
    "model_info.json"
)

with open(info_path, "w") as file:
    json.dump(
        model_info,
        file,
        indent=4
    )

print("Model information saved:")
print(info_path)


# ============================================================
# 21. SAVE MODEL COMPARISON
# ============================================================

results_path = os.path.join(
    MODEL_FOLDER,
    "model_comparison.csv"
)

results_df.to_csv(
    results_path,
    index=False
)

print("Model comparison saved:")
print(results_path)


# ============================================================
# 22. TEST WITH A SAMPLE INPUT
# ============================================================

print("\n" + "=" * 70)
print("SAMPLE HAPPINESS PREDICTION")
print("=" * 70)

# Use the first testing sample as an example
sample_input = X_test.iloc[[0]]

sample_actual = y_test.iloc[0]

sample_prediction = best_model.predict(
    sample_input
)[0]

print("\nSample input:")
print(sample_input.to_string(index=False))

print(f"\nActual happiness score   : {sample_actual:.3f}")
print(f"Predicted happiness score: {sample_prediction:.3f}")


# ============================================================
# 23. FINAL PROJECT SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("PROJECT TRAINING COMPLETED SUCCESSFULLY")
print("=" * 70)

print(f"Original observations : {len(df)}")
print(f"Clean observations    : {len(data_clean)}")
print(f"Number of features    : {len(features)}")
print(f"Training samples      : {len(X_train)}")
print(f"Testing samples       : {len(X_test)}")

print("\nModels trained:")
for model_name in models:
    print("-", model_name)

print("\nBest model:")
print(best_model_name)

print("\nBest model performance:")
print(f"MAE      : {best_mae:.4f}")
print(f"RMSE     : {best_rmse:.4f}")
print(f"R2 Score : {best_r2:.4f}")

print("\nFiles created:")
print("-", model_path)
print("-", info_path)
print("-", results_path)
print("-", PLOT_FOLDER)

print("\nThe trained model is now ready to be connected")
print("to the FastAPI backend and web dashboard.")

print("=" * 70)