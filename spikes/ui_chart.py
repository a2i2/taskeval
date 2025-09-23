"""
Chart Extraction Eval — Clean Consistent Layout (Single Page)

Run:
    streamlit run chart_eval_ui_enhanced.py
"""

import streamlit as st
import pandas as pd
from pathlib import Path
from PIL import Image
import html

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="TASKEVAL - Chart Analysis",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# -----------------------------
# CONFIG
# -----------------------------
CONFIG = {
    "TASK_PROMPT": "Extract the data points in the chart image and provide the output as a table.",
    "ORIG_IMAGE_PATH": "./data/new_chart_images/1.png",
    "GEN_IMAGE_PATH": "./outputs/1_redrawn.png",
    "EXTRACTED_DATA": {
        "PKA Energy (eV)": [250, 500, 1000, 1500, 2000, 2500],
        "Number of Collisions": [3, 5, 8, 12, 17, 22],
    },
}

# -----------------------------
# CSS (larger font sizes throughout)
# -----------------------------
st.markdown(
    """
<style>
  :root{
    --text:#000000;       /* black text everywhere */
    --border:#cbd5e1;
    --accent:#2563eb;     /* blue for dividers */
    --table-max: 95%;
  }

  #MainMenu, footer, header {visibility: hidden;}
  .block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 95%;
    font-size: 2.2rem !important;      /* Increased base font size */
    color: var(--text);
  }

  /* All text elements bigger */
  .stMarkdown, .stText, p, div, span {
    font-size: 2.2rem !important;
    color: var(--text) !important;
  }

  h1 {
    font-size: 4.5rem !important;      /* Increased from 3.2rem */
    font-weight: 900;
    text-align: center;
    color: var(--text);
    margin-bottom: 1rem;
  }
  
  h2, h3, h4 {
    font-size: 2.8rem !important;      /* Increased from 2rem */
    font-weight: 800;
    color: var(--text);   
    margin-top: 1.5rem;
  }

  /* Streamlit specific text elements */
  .stMarkdown h4 {
    font-size: 2.8rem !important;
    font-weight: 800;
    color: var(--text) !important;
  }

  .card {
    background: #f9fafb;   
    border: 2px solid #e5e7eb;
    border-radius: 12px;
    box-shadow: 0 3px 8px rgba(0,0,0,0.04);
    padding: 1.8rem;                   /* Increased padding */
    font-size: 2.2rem !important;     /* Increased from 1.5rem */
    color: var(--text);
  }
  
  .card-header {
    font-weight: 800; 
    color: var(--text);
    margin-bottom: 1rem;               /* Increased spacing */
    padding-bottom: 0.6rem;            /* Increased padding */ 
    border-bottom: 2px solid #d1d5db;
    font-size: 2.5rem !important;     /* Increased from 1.7rem */
  }

  .prompt {
    background:#fef3c7;
    border: 2px solid #facc15;
    border-radius: 10px; 
    padding: 1.8rem 2rem;              /* Increased padding */
    font-size: 2.4rem !important;     /* Increased from 1.6rem */
    font-weight: 600;
    color: var(--text);
    line-height: 1.6;                  /* Better line spacing */
  }

  hr { 
    margin: 2.2rem 0;                  /* Increased margins */
    border: none; 
    height: 3px;                       /* Thicker line */
    background: var(--accent); 
    border-radius: 2px;
  }

  .section-title { 
    margin-bottom: 1.2rem;             /* Increased spacing */
    font-size: 2.8rem !important;     /* Increased from 1.8rem */
    font-weight: 800;
    color: var(--text);
  }

  /* TABLE: bigger text and spacing */
  .table-wrap {
    width: 100%;
    display: flex;
    justify-content: center;
    margin: 25px 0 15px;               /* Increased margins */
  }
  
  .table-shell {
    width: var(--table-max);
  }
  
  table.grid {
    border-collapse: collapse;
    width: 100%;
    table-layout: fixed;
    background: #ffffff;
    border: 3px solid var(--border);   /* Thicker border */
    border-radius: 15px;               /* Larger radius */
    overflow: hidden;
    box-shadow: 0 4px 12px rgba(0,0,0,0.08);
    font-size: 2.2rem !important;     /* Increased from 1.6rem */
  }
  
  table.grid th, table.grid td {
    border: 2px solid var(--border);   /* Thicker cell borders */
    padding: 22px 25px !important;     /* Increased from 16px 20px */
    text-align: center;
    color: var(--text);
    font-size: 2.2rem !important;     /* Explicit font size */
  }
  
  table.grid th {
    font-weight: 800;
    background: #f3f4f6;
    font-size: 2.5rem !important;     /* Increased from 1.7rem */
    color: var(--text);
  }

  .table-note {
    text-align: center; 
    color: #475569; 
    font-size: 1.8rem !important;     /* Increased from 1.2rem */ 
    margin-top: 15px;                  /* Increased spacing */
  }

  /* Error messages and warnings bigger */
  .stAlert {
    font-size: 2rem !important;
  }
  
  /* Streamlit column content */
  .element-container {
    font-size: 2.2rem !important;
  }
</style>
""",
    unsafe_allow_html=True,
)


# -----------------------------
# Helpers
# -----------------------------
def load_image(path_str: str):
    """Open an image or return an error message."""
    if not path_str:
        return None, "No path provided."
    p = Path(path_str)
    if not p.exists():
        return None, f"File not found: {p}"
    try:
        img = Image.open(p)
        return img, None
    except Exception as e:
        return None, f"Failed to open image: {e}"


def render_table_bordered(df: pd.DataFrame) -> str:
    """Return a clean, centered HTML table with bigger text."""
    cols = list(df.columns)
    thead = "".join(f"<th>{html.escape(str(c))}</th>" for c in cols)

    rows_html = []
    for r in range(len(df)):
        tds = []
        for c_idx, c in enumerate(cols):
            v = df.iloc[r, c_idx]
            if isinstance(v, float):
                v = f"{v:.3g}".rstrip("0").rstrip(".")
            tds.append(f"<td>{html.escape(str(v))}</td>")
        rows_html.append(f"<tr>{''.join(tds)}</tr>")

    return f"""
    <div class="table-wrap">
      <div class="table-shell">
        <table class="grid">
          <thead><tr>{thead}</tr></thead>
          <tbody>{''.join(rows_html)}</tbody>
        </table>
        <div class="table-note"></div>
      </div>
    </div>
    """


# -----------------------------
# UI
# -----------------------------
st.markdown("<h1>TASKEVAL</h1>", unsafe_allow_html=True)

st.markdown("#### Task Instruction")
st.markdown(
    f"<div class='prompt'>{CONFIG['TASK_PROMPT']}</div>", unsafe_allow_html=True
)

# --- Image comparison ---
st.markdown("#### Image Comparison")
col1, col2 = st.columns(2, gap="large")

with col1:
    st.markdown(
        "<div class='card'><div class='card-header'>Original Image</div>",
        unsafe_allow_html=True,
    )
    img, err = load_image(CONFIG.get("ORIG_IMAGE_PATH", ""))
    if err:
        st.error(err)
    elif img is not None:
        st.image(img, use_column_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown(
        "<div class='card'><div class='card-header'>Regenerated Image</div>",
        unsafe_allow_html=True,
    )
    img, err = load_image(CONFIG.get("GEN_IMAGE_PATH", ""))
    if err:
        st.error(err)
    elif img is not None:
        st.image(img, use_column_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# --- Table ---
st.markdown("<hr/>", unsafe_allow_html=True)
st.markdown(
    "<h4 class='section-title'>Extracted Table Data</h4>", unsafe_allow_html=True
)

df = pd.DataFrame(CONFIG["EXTRACTED_DATA"])
st.markdown(render_table_bordered(df), unsafe_allow_html=True)
