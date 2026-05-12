py
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import io
import json
import numpy as np
import pingouin as pg  # High-precision statistics
from docx import Document # Support for Word/WPS

app = FastAPI()

# --- CONNECT TO YOUR WEBSITE ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- THE CALCULATION ENGINE ---
def run_elite_research_suite(df, options):
    # Standardize column headers (remove spaces)
    df.columns = df.columns.str.strip()
    
    # 1. CLEANING: Convert symbols like % and $ into numbers
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].astype(str).str.replace(r'[%\$,]', '', regex=True)
            df[col] = pd.to_numeric(df[col], errors='coerce')

    num_cols = df.select_dtypes(include=[np.number]).columns
    results = {}

    # 2. DESCRIPTIVES: Full SPSS Table
    if options.get("descriptive"):
        results["descriptive"] = {col: {
            "n": int(df[col].count()), 
            "mean": float(df[col].mean()),
            "std": float(df[col].std()), 
            "skew": float(df[col].skew()),
            "kurt": float(df[col].kurtosis())
        } for col in num_cols}

    # 3. PINGOUIN T-TEST: PhD-Level Accuracy
    if options.get("ttest"):
        try:
            # Fuzzy match 'gender' and 'salary'
            g_col = next((c for c in df.columns if 'gender' in c.lower()), None)
            s_col = next((c for c in df.columns if 'salary' in c.lower() or 'income' in c.lower()), None)
            
            if g_col and s_col:
                # Use Pingouin for Bayes Factor and Effect Size (Cohen's d)
                t_res = pg.ttest(df[df[g_col].astype(str).str.lower().str.startswith('m')][s_col], 
                                 df[df[g_col].astype(str).str.lower().str.startswith('f')][s_col])
                
                results["t_test"] = {
                    "t_stat": float(t_res['T'].iloc[0]),
                    "p_value": float(t_res['p-val'].iloc[0]),
                    "cohens_d": float(t_res['cohen-d'].iloc[0]),
                    "bayes_factor": float(t_res['BF10'].iloc[0]),
                    "significant": bool(t_res['p-val'].iloc[0] < 0.05),
                    "df": int(t_res['dof'].iloc[0])
                }
        except: pass

    # 4. CORRELATION MATRIX
    if options.get("correlation") and len(num_cols) >= 2:
        results["correlation"] = df[num_cols].corr().to_dict()

    return results

# --- THE API ENDPOINT ---
@app.post("/spss/analyze")
async def analyze(file: UploadFile = File(...), options: str = Form("{}")):
    contents = await file.read()
    try:
        opt_dict = json.loads(options)
        
        # Word/WPS Support (.docx)
        if file.filename.endswith(".docx"):
            doc = Document(io.BytesIO(contents))
            if not doc.tables: return {"error": "No table found in Word document."}
            table = doc.tables[0]
            data = [[cell.text for cell in row.cells] for row in table.rows]
            df = pd.DataFrame(data[1:], columns=data[0])
        # Excel/CSV Support
        elif file.filename.endswith(".csv"):
            df = pd.read_csv(io.BytesIO(contents))
        else:
            df = pd.read_excel(io.BytesIO(contents))
            
        return run_elite_research_suite(df, opt_dict)
    except Exception as e:
        return {"error": f"Menorca AI Error: {str(e)}"}
