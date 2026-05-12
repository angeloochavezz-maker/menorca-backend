from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import io
from scipy import stats
import numpy as np

app = FastAPI()

# --- CORS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- SPSS LOGIC (The Brain) ---
def run_spss_logic(df):
    # Fix 1: Auto-Align Headers (Removes spaces and handles case)
    df.columns = df.columns.str.strip()

    # Fix 2: Auto-Clean Data (Removes %, $, and commas)
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].astype(str).str.replace(r'[%\$,]', '', regex=True)
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Fix 3: Fuzzy Variable Mapping
    gender_col = next((c for c in df.columns if 'gender' in c.lower()), None)
    salary_col = next((c for c in df.columns if 'salary' in c.lower() or 'income' in c.lower()), None)

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    descriptive = {col: {
        "n": int(df[col].count()),
        "mean": float(df[col].mean()),
        "median": float(df[col].median()),
        "std": float(df[col].std()),
        "min": float(df[col].min()),
        "max": float(df[col].max()),
        "skew": float(df[col].skew()),
        "kurt": float(df[col].kurtosis())
    } for col in numeric_cols}

    result = {"descriptive": descriptive}

    # Fix 4: Robust T-Test Alignment
    if gender_col and salary_col:
        df['temp_gender'] = df[gender_col].astype(str).str.strip().str.lower()
        male_grp = df[df['temp_gender'].str.startswith('m', na=False)][salary_col].dropna()
        female_grp = df[df['temp_gender'].str.startswith('f', na=False)][salary_col].dropna()
        
        if len(male_grp) > 1 and len(female_grp) > 1:
            t_stat, p_val = stats.ttest_ind(male_grp, female_grp, nan_policy='omit')
            result["t_test"] = {
                "t_stat": float(t_stat),
                "p_value": float(p_val),
                "significant": bool(p_val < 0.05)
            }
    return result

# --- API ENDPOINT ---
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
