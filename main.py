
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import io
from scipy import stats
import numpy as np

app = FastAPI()

# --- CORS (Connects to your website) ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def run_spss_logic(df):
    # Fix 1: Auto-Align Headers (Removes spaces)
    df.columns = df.columns.str.strip()

    # Fix 2: Auto-Clean Data (Removes %, $, commas)
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].astype(str).str.replace(r'[%\$,]', '', regex=True)
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Fix 3: Fuzzy Variable Mapping
    gender_col = next((c for c in df.columns if 'gender' in c.lower()), None)
    salary_col = next((c for c in df.columns if 'salary' in c.lower() or 'income' in c.lower()), None)
    group_col = next((c for c in df.columns if 'group' in c.lower()), None)

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    # 1. DESCRIPTIVE STATISTICS
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

    # 2. CORRELATION MATRIX (Pearson r)
    if len(numeric_cols) > 1:
        corr_matrix = df[numeric_cols].corr().to_dict()
        result["correlation"] = corr_matrix

    # 3. INDEPENDENT T-TEST (Gender vs Salary)
    if gender_col and salary_col:
        df['tmp_g'] = df[gender_col].astype(str).str.strip().str.lower()
        m_grp = df[df['tmp_g'].str.startswith('m', na=False)][salary_col].dropna()
        f_grp = df[df['tmp_g'].str.startswith('f', na=False)][salary_col].dropna()
        
        if len(m_grp) > 1 and len(f_grp) > 1:
            t_stat, p_val = stats.ttest_ind(m_grp, f_grp, nan_policy='omit')
            result["t_test"] = {
                "t_stat": float(t_stat), "p_value": float(p_val),
                "significant": bool(p_val < 0.05), "df": int(len(m_grp) + len(f_grp) - 2)
            }

    # 4. ONE-WAY ANOVA (Group Analysis)
    if group_col and salary_col:
        # Get groups (e.g., 'A', 'B', 'C')
        unique_groups = df[group_col].dropna().unique()
        if len(unique_groups) > 2:
            grps_data = [df[df[group_col] == g][salary_col].dropna() for g in unique_groups]
            f_stat, p_val = stats.f_oneway(*grps_data)
            result["anova"] = {
                "f_stat": float(f_stat), "p_value": float(p_val),
                "significant": bool(p_val < 0.05)
            }

    return result

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
        return {"error": str(e)}
