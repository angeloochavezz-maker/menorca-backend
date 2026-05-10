from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import io
from scipy import stats

app = FastAPI()

# -------------------------
# CORS (ALLOW FRONTEND)
# -------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # change later to your domain for security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------
# SPSS ANALYTICS ENDPOINT
# -------------------------
@app.post("/spss/analyze")
async def analyze(file: UploadFile = File(...)):

    contents = await file.read()

    # Detect file type
    if file.filename.endswith(".csv"):
        df = pd.read_csv(io.BytesIO(contents))
    else:
        df = pd.read_excel(io.BytesIO(contents))

    result = {}

    # -------------------------
    # DESCRIPTIVE STATS
    # -------------------------
    numeric_cols = df.select_dtypes(include="number").columns

    result["descriptive"] = {
        col: {
            "mean": float(df[col].mean()),
            "median": float(df[col].median()),
            "std": float(df[col].std())
        }
        for col in numeric_cols
    }

    # -------------------------
    # T-TEST (Gender vs Salary)
    # -------------------------
    if "Gender" in df.columns and "Annual Salary" in df.columns:

        male = df[df["Gender"] == "Male"]["Annual Salary"]
        female = df[df["Gender"] == "Female"]["Annual Salary"]

        t_stat, p_val = stats.ttest_ind(male, female, nan_policy="omit")

        result["t_test"] = {
            "t_stat": float(t_stat),
            "p_value": float(p_val),
            "significant": bool(p_val < 0.05)
        }

    return result