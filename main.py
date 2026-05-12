from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import io
import json
import numpy as np
import pingouin as pg  # High-precision stats
from scipy import stats
from docx import Document # Word support

app = FastAPI()

# --- CORS (Handshake with your website) ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def run_elite_engine(df, options):
    df.columns = df.columns.str.strip()
    
    # 1. ELITE CLEANING: Fixes symbol errors (%, $, commas)
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].astype(str).str.replace(r'[%\$,]', '', regex=True)
            df[col] = pd.to_numeric(df[col], errors='coerce')

    num_cols = df.select_dtypes(include=[np.number]).columns
    results = {}

    # 2. DESCRIPTIVE STATISTICS
    if options.get("descriptive", True):
        results["descriptive"] = {col: {
            "n": int(df[col].count()), "mean": float(df[col].mean()),
            "std": float(df[col].std()), "skew": float(df[col].skew()),
            "kurt": float(df[col].kurtosis())
        } for col in num_cols}

    # 3. NORMALITY (Shapiro-Wilk)
    if options.get("normality") and len(num_cols) > 0:
        results["normality"] = {col: {
            "p_value": float(stats.shapiro(df[col].dropna())[1])
        } for col in num_cols if len(df[col].dropna()) > 3}

    # 4. PINGOUIN T-TEST (Fuzzy Gender/Salary matching)
    if options.get("ttest"):
        try:
            g_col = next((c for c in df.columns if 'gender' in c.lower()), None)
            s_col = next((c for c in df.columns if 'salary' in c.lower() or 'income' in c.lower()), None)
            if g_col and s_col:
                m_grp = df[df[g_col].astype(str).str.lower().str.startswith('m', na=False)][s_col].dropna()
                f_grp = df[df[g_col].astype(str).str.lower().str.startswith('f', na=False)][s_col].dropna()
                if len(m_grp) > 1 and len(f_grp) > 1:
                    t_res = pg.ttest(m_grp, f_grp)
                    results["t_test"] = {
                        "t_stat": float(t_res['T'].iloc[0]), "p_value": float(t_res['p-val'].iloc[0]),
                        "significant": bool(t_res['p-val'].iloc[0] < 0.05), "df": int(t_res['dof'].iloc[0])
                    }
        except: pass

    # 5. CORRELATION MATRIX (Pearson)
    if options.get("correlation") and len(num_cols) >= 2:
        results["correlation"] = df[num_cols].corr().to_dict()

    # 6. REGRESSION
    if options.get("regression") and len(num_cols) >= 2:
        try:
            slope, intercept, r, p, std = stats.linregress(df[num_cols[0]].dropna(), df[num_cols[-1]].dropna())
            results["regression"] = {"r_squared": float(r**2), "p_value": float(p), "slope": float(slope), "intercept": float(intercept)}
        except: pass

    # 7. RELIABILITY (Cronbach's Alpha)
    if options.get("reliability") and len(num_cols) > 2:
        try:
            results["reliability"] = {"alpha": float(pg.cronbach_alpha(data=df[num_cols].dropna())[0])}
        except: pass

    return results

@app.post("/spss/analyze")
async def analyze(file: UploadFile = File(...), options: str = Form("{}")):
    contents = await file.read()
    try:
        opt_dict = json.loads(options)
        if file.filename.endswith(".docx"):
            doc = Document(io.BytesIO(contents))
            if not doc.tables: return {"error": "No table found in Word document."}
            data = [[cell.text for cell in row.cells] for row in doc.tables[0].rows]
            df = pd.DataFrame(data[1:], columns=data[0])
        elif file.filename.endswith(".csv"):
            df = pd.read_csv(io.BytesIO(contents))
        else:
            df = pd.read_excel(io.BytesIO(contents))
        return run_elite_engine(df, opt_dict)
    except Exception as e:
        return {"error": f"Menorca AI Error: {str(e)}"}
