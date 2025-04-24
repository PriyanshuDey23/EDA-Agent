import os, io, tempfile
import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
import google.generativeai as genai
import matplotlib, streamlit as st

matplotlib.use('Agg')
from dotenv import load_dotenv
load_dotenv()

# — Model Setup
GEMINI_MODEL   = "gemini-2.0-flash"
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise EnvironmentError("Missing GOOGLE_API_KEY")

genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel(GEMINI_MODEL)

# — Utilities
def save_fig(fig):
    """Save a matplotlib Figure to a temp PNG and return its path."""
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    fig.savefig(tmp.name, bbox_inches="tight")
    plt.close(fig)
    return tmp.name

def convert_df_to_string(df):
    buf = io.StringIO()
    df.info(buf=buf)
    schema = buf.getvalue()
    head   = df.head().to_markdown(index=False)
    missing = df.isnull().sum()
    misstxt = "No Missing Values" if missing.eq(0).all() else missing.to_string()
    return f"### Schema\n```\n{schema}\n```\n### Preview\n```\n{head}\n```\n### Missing\n```\n{misstxt}\n```"

# — AI Analysis
def ai_text_analysis(prompt_type, df_str):
    prompts = {
        "plan":  f"As a data analyst, outline a concise analysis plan based on this dataset:\n{df_str}",
        "final": f"Summarize key insights from the dataset:\n{df_str}"
    }
    if prompt_type not in prompts:
        raise ValueError("Unknown prompt_type")
    resp = model.generate_content(
        prompts[prompt_type],
        generation_config=genai.types.GenerationConfig(temperature=0.3)
    )
    return resp.text


# — Visualization
def generate_visuals(df):
    visuals, files = [], []
    num_cols = df.select_dtypes(include=[np.number]).columns
    cat_cols = df.select_dtypes(include=[object]).columns

    # Correlation heatmap & pairplot if >1 numeric
    if len(num_cols) > 1:
        fig, ax = plt.subplots(figsize=(8,6))
        corr = df[num_cols].corr()
        sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
        files.append(save_fig(fig)); visuals.append(("Correlation Heatmap", files[-1]))

        pp = sns.pairplot(df[num_cols].dropna())
        pp.fig.suptitle("Pair Plot", y=1.02)
        files.append(save_fig(pp.fig)); visuals.append(("Pair Plot", files[-1]))

    # Histograms
    for c in num_cols:
        fig, ax = plt.subplots()
        sns.histplot(df[c].dropna(), kde=True, ax=ax)
        ax.set_title(f"Distribution of {c}")
        files.append(save_fig(fig)); visuals.append((f"Distribution of {c}", files[-1]))

    # Counts & Boxplots
    for cat in cat_cols:
        fig, ax = plt.subplots()
        sns.countplot(x=cat, data=df, ax=ax)
        ax.set_title(f"Count of {cat}"); plt.xticks(rotation=45)
        files.append(save_fig(fig)); visuals.append((f"Count Plot: {cat}", files[-1]))

        for num in num_cols:
            fig, ax = plt.subplots()
            sns.boxplot(x=cat, y=num, data=df, ax=ax)
            ax.set_title(f"Box Plot of {num} by {cat}"); plt.xticks(rotation=45)
            files.append(save_fig(fig)); visuals.append((f"Box: {num} by {cat}", files[-1]))

    return visuals, files

def cleanup(files):
    for f in files:
        try: os.remove(f)
        except: pass

# — Streamlit UI
st.title("📊 Data Analysis Assistant")
uploaded = st.file_uploader("Upload CSV", type="csv")
if uploaded:
    try:
        df = pd.read_csv(io.StringIO(uploaded.getvalue().decode("utf-8", "replace")))
        if df.empty:
            st.error("Empty CSV"); st.stop()

        df_str = convert_df_to_string(df)
        st.markdown(df_str)

        # Text analysis
        plan = ai_text_analysis("plan", df_str)
        st.subheader("📝 Analysis Plan"); st.write(plan)

        # Final summary
        final = ai_text_analysis("final", df_str)
        st.subheader("📊 Final Insights"); st.write(final)


        # Visuals & display
        visuals, files = generate_visuals(df)
        for title, path in visuals:
            st.subheader(title)
            st.image(path, caption=title)


    finally:
        cleanup(files)

