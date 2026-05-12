py
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import io
from scipy import stats
import numpy as np

app = FastAPI()

# -------------------------
# CORS (ALLOW FRONTEND)
# -------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------
# SPSS LOGIC FUNCTION
# -------------------------
def run_spss_logic(df):
    # 1. CLEANING: Remove %, $, and commas so "5.0%" becomes 5.0
    for col in df.columns:
        if df[col].dtype == 'object':
            # Remove common non-numeric symbols
            df[col] = df[col].astype(str).str.replace(r'[%\$,]', '', regex=True)
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # 2. DESCRIPTIVE STATS: SPSS Table Metrics
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    descriptive = {}

    for col in numeric_cols:
        series = df[col].dropna()
        if len(series) > 0:
            descriptive[col] = {
                "n": int(series.count()),
                "min": float(series.min()),
                "max": float(series.max()),
                "mean": float(series.mean()),
                "median": float(series.median()),
                "std": float(series.std()),
                "skew": float(series.skew()),
                "kurt": float(series.kurtosis()) # Fisher’s Kurtosis (Normal = 0)
            }

    result = {"descriptive": descriptive}

    # 3. T-TEST: Independent Samples
    if "Gender" in df.columns and "Annual Salary" in df.columns:
        # Normalize gender strings (handle 'male' vs 'Male')
        df["Gender"] = df["Gender"].astype(str).str.strip().str.capitalize()
        
        male_sal = df[df["Gender"] == "Male"]["Annual Salary"].dropna()
        female_sal = df[df["Gender"] == "Female"]["Annual Salary"].dropna()

        if len(male_sal) > 1 and len(female_sal) > 1:
            t_stat, p_val = stats.ttest_ind(male_sal, female_sal, nan_policy="omit")
            result["t_test"] = {
                "t_stat": float(t_stat),
                "p_value": float(p_val),
                "significant": bool(p_val < 0.05)
            }
        else:
            result["t_test"] = {"error": "Insufficient group data for T-Test"}

    return result

# -------------------------
# API ENDPOINT
# -------------------------
@app.post("/spss/analyze")
async def analyze_file(file: UploadFile = File(...)):
    contents = await file.read()

    try:
        if file.filename.endswith(".csv"):
            df = pd.read_csv(io.BytesIO(contents))
        else:
            df = pd.read_excel(io.BytesIO(contents))
            
        return run_spss_logic(df)
    except Exception as e:
        return {"error": f"Failed to process file: {str(e)}"}
