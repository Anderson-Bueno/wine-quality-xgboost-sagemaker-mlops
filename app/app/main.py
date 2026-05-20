from pathlib import Path
from typing import Literal

import xgboost as xgb
from fastapi import FastAPI
from pydantic import BaseModel, Field


MODEL_PATH = Path("models/xgboost-model")


app = FastAPI(
    title="Wine Quality Prediction API",
    description="API for predicting wine quality using an optimized XGBoost model.",
    version="1.0.0"
)


class WineFeatures(BaseModel):
    fixed_acidity: float = Field(..., example=7.4)
    volatile_acidity: float = Field(..., example=0.70)
    citric_acid: float = Field(..., example=0.00)
    residual_sugar: float = Field(..., example=1.9)
    chlorides: float = Field(..., example=0.076)
    free_sulfur_dioxide: float = Field(..., example=11.0)
    total_sulfur_dioxide: float = Field(..., example=34.0)
    density: float = Field(..., example=0.9978)
    pH: float = Field(..., example=3.51)
    sulphates: float = Field(..., example=0.56)
    alcohol: float = Field(..., example=9.4)
    wine_type: Literal["red", "white"] = Field(..., example="red")


def load_model() -> xgb.Booster:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model artifact not found at {MODEL_PATH}. "
            "Make sure the trained model is available in the models/ folder."
        )

    model = xgb.Booster()
    model.load_model(str(MODEL_PATH))
    return model


model = load_model()


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "model": "xgboost",
        "model_path": str(MODEL_PATH)
    }


@app.post("/predict")
def predict(features: WineFeatures):
    wine_type_red = 1 if features.wine_type == "red" else 0
    wine_type_white = 1 if features.wine_type == "white" else 0

    input_data = [[
        features.fixed_acidity,
        features.volatile_acidity,
        features.citric_acid,
        features.residual_sugar,
        features.chlorides,
        features.free_sulfur_dioxide,
        features.total_sulfur_dioxide,
        features.density,
        features.pH,
        features.sulphates,
        features.alcohol,
        wine_type_red,
        wine_type_white
    ]]

    dmatrix = xgb.DMatrix(input_data)

    prediction = float(model.predict(dmatrix)[0])

    return {
        "predicted_quality": round(prediction, 3),
        "model": "xgboost",
        "status": "success"
    }
