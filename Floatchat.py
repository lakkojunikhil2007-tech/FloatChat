import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import io, base64, os
import re
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
import subprocess

# --- Load dataset ---

df = pd.read_csv(r"C:\Users\vrish\OneDrive\Desktop\argo_monthly\testing.csv")

# ✅ Detect datetime column

datetime_col = None

for col in ["DATETIME", "TIME", "DATE"]:
    if col in df.columns:
        datetime_col = col
        break

if datetime_col is None:
    raise KeyError("No datetime column found (expected DATETIME, TIME, DATE)")

df[datetime_col] = pd.to_datetime(df[datetime_col], errors="coerce")

# Init FastAPI

app = FastAPI()
# Init OpenAI client

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class Query(BaseModel):
    question: str

# Synonym mapping

COLUMN_SYNONYMS = {
    "temperature": "TEMP", "temp": "TEMP",
    "salinity": "PSAL", "salt": "PSAL",
    "pressure": "PRES", "depth": "PRES",
    "oxygen": "DOXY", "nitrate": "NITRATE",
    "ph": "PH_IN_SITU_TOTAL", "chlorophyll": "CHLA",
    "bbp": "BBP700", "latitude": "LATITUDE",
    "longitude": "LONGITUDE", "time": datetime_col,
    "date": "DATE", "day": "DATE",
    "hour": "TIME_ONLY", "time_only": "TIME_ONLY",
    "float": "PLATFORM_NUMBER", "cycle": "CYCLE_NUMBER"
}

def normalize_question(q: str) -> str:
    """Replace user terms with dataset column names"""
    q = q.lower()
    if "daily" in q or "per day" in q or "by date" in q:
        q = q.replace("time", "DATE")
    elif "hourly" in q or "per hour" in q or "diurnal" in q:
        q = q.replace("time", "TIME_ONLY")
    else:
        q = q.replace("time", str(datetime_col))
    for synonym, col in COLUMN_SYNONYMS.items():
        if synonym in q:
            q = q.replace(synonym, col)
    return q

# ---- 🎨 PowerBI-like styling ----

def style_plot():
    plt.style.use("dark_background")
    plt.rcParams.update({
        "font.size": 13,
        "axes.titlesize": 17,
        "axes.labelsize": 14,
        "axes.edgecolor": "white",
        "axes.linewidth": 1.2,
        "grid.color": "gray",
        "grid.linestyle": "--",
        "grid.alpha": 0.3,
        "legend.frameon": False,
        "figure.figsize": (10, 6)
    })
    sns.set_palette("Spectral") # vibrant, non-clumsy palette

# ---- Utility: smart annotations + clean look ----

def finalize_plot(ax):
    # Grid + legend
    ax.grid(True, alpha=0.3)
    handles, labels = ax.get_legend_handles_labels()
    if len(handles) > 0:
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3)
    # Annotate if it's a line plot with one line
    if len(ax.lines) == 1:
        line = ax.lines[0]
        xdata, ydata = line.get_xdata(), line.get_ydata()
        if len(xdata) > 0:
            ax.annotate(f"Max: {ydata.max():.2f}",
                xy=(xdata[np.argmax(ydata)], ydata.max()),
                xytext=(10, 10), textcoords="offset points",
                arrowprops=dict(arrowstyle="->", color="yellow"),
                color="yellow", fontsize=11)
            ax.annotate(f"Min: {ydata.min():.2f}",
                xy=(xdata[np.argmin(ydata)], ydata.min()),
                xytext=(-30, -20), textcoords="offset points",
                arrowprops=dict(arrowstyle="->", color="cyan"),
                color="cyan", fontsize=11)
    plt.tight_layout()

# ---- Execute plotting code ----

def execute_plot(code: str):
    code = code.strip()
    if code.startswith("```"):
        code = code.split("```")[1]
    if code.startswith("python"):
        code = code[len("python"):].strip()
    safe_globals = {"df": df, "plt": plt, "sns": sns, "np": np, "pd": pd}
    safe_locals = {}
    try:
        style_plot()
        fig, ax = plt.subplots()
        safe_globals["ax"] = ax
        exec(code, safe_globals, safe_locals)
        finalize_plot(ax)
        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight", dpi=250)
        buf.seek(0)
        img_b64 = base64.b64encode(buf.read()).decode("utf-8")
        plt.close()
        return img_b64, None
    except Exception as e:
        return None, str(e)

# ---- Fallback: Ollama ----

def query_ollama(prompt: str, model: str = "llama3"):
    try:
        result = subprocess.run(
            ["ollama", "run", model],
            input=prompt.encode("utf-8"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True
        )
        return result.stdout.decode("utf-8").strip()
    except Exception as e:
        return f"Ollama failed: {str(e)}"

# ---- Text/statistics query helper functions ----

def detect_text_query(q: str):
    plot_keywords = ["plot", "graph", "scatter", "visualize", "trend", "profile", "heatmap"]
    return not any(pk in q.lower() for pk in plot_keywords)

def handle_statistical_text_query(q: str, df):
    q = q.lower()
    col = None
    for syn, colname in COLUMN_SYNONYMS.items():
        if syn.lower() in q:
            col = colname
            break
    # 1. Descriptive statistics
    if "mean" in q and col:
        val = df[col].mean()
        return f"The average {col} is {val:.3f}"
    if "median" in q and col:
        val = df[col].median()
        return f"The median {col} is {val:.3f}"
    if "max" in q and col:
        val = df[col].max()
        return f"The maximum {col} is {val:.3f}"
    if "min" in q and col:
        val = df[col].min()
        return f"The minimum {col} is {val:.3f}"
    if "std" in q and col:
        val = df[col].std()
        return f"The standard deviation of {col} is {val:.3f}"
    if ("count" in q or "number of" in q or "how many" in q) and col:
        val = df[col].count()
        return f"The count of {col} is {val}"
    # 2. Unique values
    if "unique" in q and col:
        vals = df[col].unique()
        # Show at most first 20 unique values
        return f"The unique values of {col} include: {vals[:20]}"
    # 3. Describe data columns
    if "columns" in q or "fields" in q or "show columns" in q or "list columns" in q:
        return f"The dataset columns are: {list(df.columns)}"
    # 4. Describe variable
    if "describe" in q and col:
        desc = df[col].describe()
        return f"Description of {col}:\n{desc}"
    # 5. Sample lookup (example)
    m = re.search(r"sample (\d+)", q)
    if m and col:
        idx = int(m.group(1))
        if idx < len(df):
            return f"Sample {idx} {col} value is {df.iloc[idx][col]}"
    # 6. General info
    if "info" in q or "summary" in q:
        buffer = io.StringIO()
        df.info(buf=buffer)
        return buffer.getvalue()
    # 7. Fallback
    return "Sorry, I could not interpret your text question. Please specify the statistic or field you want."

# ---- Chat endpoint ----

@app.post("/chat")
async def chat(query: Query):
    user_q = normalize_question(query.question)

    # Handle text/statistical/data questions first
    if detect_text_query(user_q):
        text_answer = handle_statistical_text_query(user_q, df)
        return {"type": "text", "answer": text_answer}

    system_prompt = f"""
    You are a scientific data assistant for ARGO ocean dataset.
    - If user asks for daily trends, use DATE.
    - If user asks for hourly/diurnal trends, use TIME_ONLY.
    - Otherwise, use {datetime_col} (full datetime).
    - If user asks a statistical question: compute it directly (use pandas/numpy).
    - If user asks for a visualization: return ONLY valid Python plotting code (no markdown fences).
    - Always label axes and add a title to plots.
    - Use 'ax' instead of plt.gca() when plotting.
    - For heatmaps of variables vs pressure, always:
      * Use {datetime_col} or CYCLE_NUMBER on the x-axis,
      * Use PRES on the y-axis (inverted so surface is at top),
      * Use the requested variable (e.g., TEMP, PSAL) as the color,
      * Use seaborn.heatmap or matplotlib pcolormesh.
    - Dataset columns available: {', '.join(df.columns)}
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a helpful assistant that generates Python code for analyzing "
                        "ARGO ocean data. Always assume the dataset variable is called df. "
                        "Never use data, dataset, or any other variable name for the DataFrame."
                    )
                },
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_q},
            ]
        )
        output = response.choices[0].message.content.strip()
    except Exception:
        ollama_prompt = f"{system_prompt}\nUser: {user_q}\nAnswer in Python if visualization is needed."
        output = query_ollama(ollama_prompt)

    if "plt." in output or "sns." in output or "ax." in output:
        img_b64, error = execute_plot(output)
        if img_b64:
            return {"type": "visualization", "code": output, "image": img_b64}
        else:
            return {"type": "error", "error": error, "code": output}
    return {"type": "text", "answer": output}


