from fastapi import FastAPI, HTTPException, File, UploadFile
from pydantic import BaseModel
from typing import List
import pandas as pd
import joblib
import lightgbm as lgb
import io

# Load model and preprocessing artifacts
model = lgb.Booster(model_file="lightgbm_model.txt")
encoder = joblib.load("encoder.pkl")
scaler = joblib.load("scaler.pkl")

# App
app = FastAPI(title="Sales Forecast API", version="1.0")

# Input schema for single/batch prediction
class SalesInput(BaseModel):
    unit_price: float
    quantity: int
    age: int
    discount: float
    customer_rating: float
    stock: int
    category_id: int
    category_avg_price: float
    category_total_revenue: float
    category_popularity: int
    year: int
    month: int
    day: int
    weekday: int
    color: str
    size: str
    category: str
    holiday_type: str

# Reusable preprocessing
def preprocess(df: pd.DataFrame):
    # Auto-handle date column if present
    if 'order_date' in df.columns:
        df['order_date'] = pd.to_datetime(df['order_date'])
        df['year'] = df['order_date'].dt.year
        df['month'] = df['order_date'].dt.month
        df['day'] = df['order_date'].dt.day
        df['weekday'] = df['order_date'].dt.weekday
        df.drop(columns=['order_date'], inplace=True)
    # Handle missing values
    df.fillna({
        'unit_price': 0,
        'quantity': 0,
        'age': 0,
        'discount': 0,
        'customer_rating': 0,
        'stock': 0,
        'category_id': 0,
        'category_avg_price': 0,
        'category_total_revenue': 0,
        'category_popularity': 0
    }, inplace=True)
    # Convert categorical columns to string if not already

    cat_cols = ['color', 'size', 'category', 'holiday_type']
    num_cols = ['unit_price', 'quantity', 'age', 'discount', 'customer_rating','stock', 'category_id', 'category_avg_price', 'category_total_revenue',
                'category_popularity', 'year', 'month', 'day', 'weekday']

    # Transform
    cat_encoded = encoder.transform(df[cat_cols])
    cat_df = pd.DataFrame(cat_encoded.toarray(), columns=encoder.get_feature_names_out(cat_cols))

    num_scaled = scaler.transform(df[num_cols])
    num_df = pd.DataFrame(num_scaled, columns=num_cols)

    final_df = pd.concat([num_df, cat_df], axis=1)

    # Add missing columns (if any)
    expected_cols = model.feature_name()
    for col in expected_cols:
        if col not in final_df.columns:
            final_df[col] = 0

    # Align column order
    return final_df[expected_cols]

# Route 1: Welcome message
@app.get("/")
def home():
    return {"message": "✅ Sales Prediction API is live! Visit /docs to try it out."}

# Route 2: Single prediction
@app.post("/predict")
def predict(data: SalesInput):
    try:
        df = pd.DataFrame([data.dict()])
        processed = preprocess(df)
        prediction = model.predict(processed.values)
        return {
            "status": "success",
            "prediction": float(prediction[0]),
            "currency": "USD",
            "input_features": data.dict(),
            "model_features_used": processed.columns.tolist()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Route 3: Batch prediction
@app.post("/predict-batch")
def predict_batch(data: List[SalesInput]):
    try:
        df = pd.DataFrame([item.dict() for item in data])
        processed = preprocess(df)
        prediction = model.predict(processed.values)
        return {
            "status": "success",
            "predictions": prediction.tolist(),
            "records": len(prediction)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Route 4: CSV upload
from fastapi.responses import FileResponse
import tempfile

@app.post("/upload-csv")
def upload_csv(file: UploadFile = File(...)):
    try:
        content = file.file.read()
        df = pd.read_csv(io.BytesIO(content))
        processed = preprocess(df)
        prediction = model.predict(processed.values)
        return {
            "status": "success",
            "rows": len(df),
            "predictions": prediction.tolist()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))