
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import io
import json
from scipy import stats
import numpy as np

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def run_universal_logic(df, options):
    df.columns = df.columns.str.strip()
    
    # Clean Symbols
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].astype(str).str.replace(r'[%\$,]', '', regex=True)
            df[col] = pd.to_numeric(df[col], errors='coerce')

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    results = {}

    # ALWAYS calculate descriptives by default
    results["descriptive"] = {col: {
        "n": int(df[col].count()),
        "mean": float(df[col].mean()),
        "std": float(df[col].std()),
        "min": float(df[col].min()),
        "max": float(df[col].max()),
        "skew": float(df[col].skew())
    } for col in numeric_cols}

    # Optional Inferential Tests (Only if requested)
    if options.get("ttest") or options.get("inferential"):
        gender_col = next((c for c in df.columns if 'gender' in c.lower()), None)
        salary_col = next((c for c in df.columns if 'salary' in c.lower()), None)
        if gender_col and salary_col:
            df['tmp_g'] = df[gender_col].astype(str).str.strip().str.lower()
            m = df[df['tmp_g'].str.startswith('m', na=False)][salary_col].dropna()
            f = df[df['tmp_g'].str.startswith('f', na=False)][salary_col].dropna()
            if len(m) > 1 and len(f) > 1:
                t_stat, p_val = stats.ttest_ind(m, f, nan_policy='omit')
                results["t_test"] = {"t_stat": float(t_stat), "p_value": float(p_val), "significant": bool(p_val < 0.05), "df": int(len(m)+len(f)-2)}

    return results

@app.post("/spss/analyze")
async def analyze(file: UploadFile = File(...), options: str = Form("{}")):
    contents = await file.read()
    # Handle both empty and filled options
    try:
        opt_dict = json.loads(options) if options else {}
    except:
        opt_dict = {}
        
    try:
        if file.filename.endswith(".csv"):
            df = pd.read_csv(io.BytesIO(contents))
        else:
            df = pd.read_excel(io.BytesIO(contents))
        return run_universal_logic(df, opt_dict)
    except Exception as e:
        return {"error": str(e)}
