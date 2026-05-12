py
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

def run_research_suite(df, options):
    df.columns = df.columns.str.strip()
    # Auto-Clean Symbols
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].astype(str).str.replace(r'[%\$,]', '', regex=True)
            df[col] = pd.to_numeric(df[col], errors='coerce')

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    results = {}

    # 1. DESCRIPTIVE & NORMALITY
    if options.get("descriptive"):
        results["descriptive"] = {col: {
            "n": int(df[col].count()), "mean": float(df[col].mean()),
            "std": float(df[col].std()), "min": float(df[col].min()), "max": float(df[col].max()),
            "skew": float(df[col].skew()), "kurt": float(df[col].kurtosis())
        } for col in numeric_cols}
        
    if options.get("normality") and len(numeric_cols) > 0:
        results["normality"] = {col: {
            "w_stat": float(stats.shapiro(df[col].dropna())[0]),
            "p_value": float(stats.shapiro(df[col].dropna())[1])
        } for col in numeric_cols if len(df[col].dropna()) > 3}

    # 2. INFERENTIAL TESTS
    gender_col = next((c for c in df.columns if 'gender' in c.lower()), None)
    salary_col = next((c for c in df.columns if 'salary' in c.lower() or 'income' in c.lower()), None)
    
    if options.get("ttest") and gender_col and salary_col:
        df['tmp_g'] = df[gender_col].astype(str).str.strip().str.lower()
        m = df[df['tmp_g'].str.startswith('m', na=False)][salary_col].dropna()
        f = df[df['tmp_g'].str.startswith('f', na=False)][salary_col].dropna()
        if len(m) > 1 and len(f) > 1:
            t, p = stats.ttest_ind(m, f)
            results["ttest"] = {"t_stat": float(t), "p_value": float(p), "df": int(len(m)+len(f)-2)}

    # 3. REGRESSION & PREDICTION
    if options.get("regression") and len(numeric_cols) >= 2:
        x, y = df[numeric_cols[0]].dropna(), df[numeric_cols[-1]].dropna()
        idx = x.index.intersection(y.index)
        slope, intercept, r, p, err = stats.linregress(x.loc[idx], y.loc[idx])
        results["regression"] = {"r_squared": float(r**2), "p_value": float(p), "slope": float(slope)}

    # 4. SPECIALIZED
    if options.get("alpha") and len(numeric_cols) > 1:
        items = df[numeric_cols].dropna()
        k = items.shape[1]
        alpha = (k / (k-1)) * (1 - (items.var(axis=0).sum() / items.sum(axis=1).var()))
        results["alpha"] = float(alpha)

    if options.get("ecological"):
        c_col = next((c for c in df.columns if 'count' in c.lower() or 'abund' in c.lower()), None)
        if c_col:
            counts = df[c_col].dropna().values
            p = counts / np.sum(counts)
            h = -np.sum(p[p > 0] * np.log(p[p > 0]))
            results["ecological"] = {"shannon_h": float(h), "evenness": float(h/np.log(len(counts)) if len(counts) > 1 else 0)}

    return results

@app.post("/spss/analyze")
async def analyze(file: UploadFile = File(...), options: str = Form("{}")):
    contents = await file.read()
    opt_dict = json.loads(options)
    try:
        df = pd.read_csv(io.BytesIO(contents)) if file.filename.endswith(".csv") else pd.read_excel(io.BytesIO(contents))
        return run_research_suite(df, opt_dict)
    except Exception as e:
        return {"error": str(e)}
