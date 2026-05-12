
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import io
import json
from scipy import stats
import numpy as np
from docx import Document # New library for Word docs

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

def run_research_suite(df, options):
    df.columns = df.columns.str.strip()
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].astype(str).str.replace(r'[%\$,]', '', regex=True)
            df[col] = pd.to_numeric(df[col], errors='coerce')
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    results = {}
    if options.get("descriptive"):
        results["descriptive"] = {col: {"n": int(df[col].count()), "mean": float(df[col].mean()), "std": float(df[col].std())} for col in numeric_cols}
    return results

@app.post("/spss/analyze")
async def analyze(file: UploadFile = File(...), options: str = Form("{}")):
    contents = await file.read()
    opt_dict = json.loads(options)
    
    try:
        # NEW: Handle Word/WPS/Docs (.docx)
        if file.filename.endswith(".docx"):
            doc = Document(io.BytesIO(contents))
            if len(doc.tables) == 0:
                return {"error": "No table found inside the Word document."}
            # Extract first table into a DataFrame
            table = doc.tables[0]
            data = [[cell.text for cell in row.cells] for row in table.rows]
            df = pd.DataFrame(data[1:], columns=data[0])
        
        # Handle Excel/CSV
        elif file.filename.endswith(".csv"):
            df = pd.read_csv(io.BytesIO(contents))
        else:
            df = pd.read_excel(io.BytesIO(contents))
            
        return run_research_suite(df, opt_dict)
    except Exception as e:
        return {"error": f"Document Processing Error: {str(e)}"}
