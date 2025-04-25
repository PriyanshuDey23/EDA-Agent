# Standard Library Imports
import os, io, tempfile

# Data & Visualization Libraries
import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns

# AI Model & UI Libraries
import google.generativeai as genai
import matplotlib, streamlit as st

# Set Matplotlib backend for non-GUI environments (like Streamlit Cloud)
matplotlib.use('Agg')

# Load environment variables (for API keys, etc.)
from dotenv import load_dotenv
load_dotenv()

# Model Setup
GEMINI_MODEL   = "gemini-2.0-flash"
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise EnvironmentError("Missing GOOGLE_API_KEY")

# Configure AI Model
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel(GEMINI_MODEL)

# Utility: Save a matplotlib Figure to a temporary PNG file and return its path
def save_fig(fig):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    fig.savefig(tmp.name, bbox_inches="tight")
    plt.close(fig)
    return tmp.name

# Utility: Convert a DataFrame summary into a markdown-like string for AI analysis
def convert_df_to_string(df):
    buf = io.StringIO()
    df.info(buf=buf)
    schema = buf.getvalue()
    head   = df.head().to_markdown(index=False)
    missing = df.isnull().sum()
    misstxt = "No Missing Values" if missing.eq(0).all() else missing.to_string()
    return f"### Schema\n```\n{schema}\n```\n### Preview\n```\n{head}\n```\n### Missing\n```\n{misstxt}\n```"

# AI: Generate text analysis based on dataset description
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

# AI: Analyze chart titles and numerical summaries for insights
def ai_chart_analysis(chart_summaries):
    prompt = (
        f"You are a data analyst reviewing the following chart summaries:\n\n"
        f"{chart_summaries}\n\n"
        f"Please identify notable patterns, anomalies, or key takeaways. Be concise and data-driven."
    )
    resp = model.generate_content(
        prompt,
        generation_config=genai.types.GenerationConfig(temperature=0.4)
    )
    return resp.text

# Generate charts based on data and collect their titles & file paths
def generate_visuals(df):
    visuals= [] # Stores information about charts for AI analysis
    files = [] # used to clean up temp image files after displaying them
    num_cols = df.select_dtypes(include=[np.number]).columns
    cat_cols = df.select_dtypes(include=[object]).columns

    # Correlation heatmap & pairplot if more than one numeric column
    if len(num_cols) > 1:
        fig, ax = plt.subplots(figsize=(8,6))
        corr = df[num_cols].corr()
        sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
        title = "Correlation Heatmap"
        files.append(save_fig(fig))
        visuals.append((title, files[-1], f"Correlation coefficients:\n{corr.to_string()}"))

        pp = sns.pairplot(df[num_cols].dropna())
        pp.fig.suptitle("Pair Plot", y=1.02)
        files.append(save_fig(pp.fig))
        visuals.append(("Pair Plot", files[-1], "Pairwise scatterplots for numeric columns."))

    # Histograms for numerical columns
    for c in num_cols:
        fig, ax = plt.subplots()
        sns.histplot(df[c].dropna(), kde=True, ax=ax)
        ax.set_title(f"Distribution of {c}")
        files.append(save_fig(fig))
        visuals.append((f"Distribution of {c}", files[-1], f"Histogram and KDE for {c}"))

    # Count plots & boxplots for categorical vs numerical
    for cat in cat_cols:
        fig, ax = plt.subplots()
        sns.countplot(x=cat, data=df, ax=ax)
        ax.set_title(f"Count of {cat}")
        plt.xticks(rotation=45)
        files.append(save_fig(fig))
        visuals.append((f"Count Plot: {cat}", files[-1], f"Counts for each category in {cat}"))

        for num in num_cols:
            fig, ax = plt.subplots()
            sns.boxplot(x=cat, y=num, data=df, ax=ax)
            ax.set_title(f"Box Plot of {num} by {cat}")
            plt.xticks(rotation=45)
            files.append(save_fig(fig))
            visuals.append((f"Box: {num} by {cat}", files[-1], f"Boxplot of {num} grouped by {cat}"))

    return visuals, files

#  Utility: Clean up temp image files
def cleanup(files):
    for f in files:
        try:
            os.remove(f)
        except:
            pass

# Streamlit UI Layout
st.title("📊 Data Analysis Assistant")
uploaded = st.file_uploader("Upload CSV", type="csv")

if uploaded:
    try:
        # Load uploaded CSV
        df = pd.read_csv(io.StringIO(uploaded.getvalue().decode("utf-8", "replace")))
        if df.empty:
            st.error("Empty CSV"); st.stop()

        # Convert DataFrame summary for AI text analysis
        df_str = convert_df_to_string(df)
        st.markdown(df_str)

        # AI: Analysis plan and summary insights
        plan = ai_text_analysis("plan", df_str)
        st.subheader("📝 Analysis Plan")
        st.write(plan)

        final = ai_text_analysis("final", df_str)
        st.subheader("📊 Final Insights")
        st.write(final)

        # Generate and display visuals
        visuals, files = generate_visuals(df)

        # Compile chart summaries for AI chart analysis
        chart_summaries = "\n\n".join([f"{title}: {desc}" for title, _, desc in visuals])

        for title, path, _ in visuals:
            st.subheader(title)
            st.image(path, caption=title)

        # AI: Chart insights
        chart_analysis = ai_chart_analysis(chart_summaries)
        st.subheader("📈 Chart Analysis")
        st.write(chart_analysis)

    finally:
        # Clean up temp files
        cleanup(files)
