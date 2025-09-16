"""
TASKEVAL — General LLM Task Evaluation (Single Page) with Multi-Color Keyword Highlighting
                                                                                                                         
Run:
    streamlit run taskeval_ui.py
"""

import streamlit as st
import pandas as pd
import html
import re
from typing import List, Set, Dict, Tuple

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="TASKEVAL - LLM Task Evaluation",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# -----------------------------
# CONFIG (replace with your values)
# -----------------------------
CONFIG = {
    "TASK_INSTRUCTION": "Based on the pdf, how do they perform multilingual training? Provide as prose.",
    "LLM_OUTPUT": (
        "In the paper, multilingual training is carried out through a shared auxiliary decoder and joint training across languages. "
        "The system adds an auxiliary task of predicting the morphosyntactic description (MSD) of the target form, and the parameters "
        "of this MSD decoder are shared across multiple languages. Instead of grouping languages strictly by family (which would leave "
        "some languages isolated, like Russian), the authors randomly group sets of two to three languages. During training, minibatches "
        "are drawn from different languages in random order, so the model alternates between them without being told explicitly which language "
        "the data comes from. This is based on the assumption that abstract morphosyntactic features can be shared across languages. After this "
        "multilingual pretraining phase (20 epochs), the model undergoes monolingual fine-tuning for each language individually, with a reduced "
        "learning rate. This ensures that the multilingual knowledge is retained while still adapting specifically to each language. The results "
        "show that this multilingual training strategy improves accuracy substantially—on average, multilingual models outperform monolingual ones "
        "by about 7.96%, and the combination of multilingual training with fine-tuning yields the best results for most languages."
    ),
    "EXTRACTED_SECTIONS": [
        "The parameters of the entire MSD (auxiliary-task) decoder are shared across languages.",
        "We experiment with random groupings of two to three languages.",
        "Multilingual training is performed by randomly alternating between languages for every new minibatch",
        "We do not pass any information to the auxiliary decoder as to the source language…",
        "After 20 epochs of multilingual training, we perform 5 epochs of monolingual finetuning for each language",
        "…we reduce the learning rate to a tenth of the original learning rate, i.e. 0.0001…",
        "…multilingual results [are] 7.96% higher than monolingual ones on average.",
        "Monolingual finetuning improves accuracy across the board… by 2.72% on average.",
        "…the multi-tasking approach paired with multilingual training and subsequent monolingual finetuning outperforms… for five out of seven languages."
    ],
    "DEEPEVAL_SCORE": {
        "Answer Relevancy ": 1.0,
        "Faithfulness ": 1.0,
        "Contextual Precision": 1.0,
        "Contextual Recall": 1.0
    }
}

# -----------------------------
# HIGHLIGHT COLORS FOR EACH SECTION
# -----------------------------
HIGHLIGHT_COLORS = [
    "#fef08a",  # Yellow
    "#bfdbfe",  # Light Blue
    "#c7d2fe",  # Light Purple
    "#fed7d7",  # Light Pink
    "#bbf7d0",  # Light Green
    "#fed7aa",  # Light Orange
    "#e9d5ff",  # Light Violet
    "#fde68a",  # Light Amber
    "#fecaca",  # Light Red
]

# -----------------------------
# KEYWORD EXTRACTION AND HIGHLIGHTING FUNCTIONS
# -----------------------------
def extract_keywords_and_phrases(text: str, min_word_length: int = 3) -> Set[str]:
    """Extract meaningful keywords and phrases from text."""
    keywords = set()
    
    # Clean text and convert to lowercase
    cleaned_text = re.sub(r'[^\w\s]', ' ', text.lower())
    words = cleaned_text.split()
    
    # Single words (filter out common words)
    stop_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from', 'up', 'about', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'between', 'among', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'shall', 'can', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'them', 'their', 'there', 'where', 'when', 'why', 'how', 'what', 'which', 'who', 'whom', 'whose', 'a', 'an'}
    
    for word in words:
        if len(word) >= min_word_length and word not in stop_words:
            keywords.add(word)
    
    # Extract 2-word phrases
    for i in range(len(words) - 1):
        if len(words[i]) >= min_word_length and len(words[i+1]) >= min_word_length:
            phrase = f"{words[i]} {words[i+1]}"
            if words[i] not in stop_words or words[i+1] not in stop_words:
                keywords.add(phrase)
    
    # Extract 3-word phrases
    for i in range(len(words) - 2):
        phrase = f"{words[i]} {words[i+1]} {words[i+2]}"
        keywords.add(phrase)
    
    # Extract specific numbers and technical terms
    numbers = re.findall(r'\d+\.?\d*%?', text)
    for num in numbers:
        keywords.add(num.lower())
    
    return keywords

def get_section_specific_matches(llm_output: str, extracted_sections: List[str]) -> Dict[int, Set[str]]:
    """Get keywords that match between LLM output and each specific extracted section."""
    llm_keywords = extract_keywords_and_phrases(llm_output)
    
    section_matches = {}
    
    for i, section in enumerate(extracted_sections):
        section_keywords = extract_keywords_and_phrases(section)
        
        # Find matches for this specific section
        matches = set()
        for llm_kw in llm_keywords:
            for sect_kw in section_keywords:
                if llm_kw == sect_kw or (len(llm_kw) > 4 and llm_kw in sect_kw) or (len(sect_kw) > 4 and sect_kw in llm_kw):
                    matches.add(llm_kw)
        
        if matches:
            section_matches[i] = matches
    
    return section_matches

def highlight_text_multicolor(text: str, section_matches: Dict[int, Set[str]]) -> str:
    """Highlight keywords in text with different colors for each section."""
    highlighted_text = text
    
    # Create a list of all keywords with their associated colors
    keyword_color_map = {}
    for section_idx, keywords in section_matches.items():
        color = HIGHLIGHT_COLORS[section_idx % len(HIGHLIGHT_COLORS)]
        for keyword in keywords:
            if len(keyword) >= 3:  # Only meaningful keywords
                keyword_color_map[keyword] = color
    
    # Sort keywords by length (longest first) to avoid partial replacements
    sorted_keywords = sorted(keyword_color_map.keys(), key=len, reverse=True)
    
    for keyword in sorted_keywords:
        color = keyword_color_map[keyword]
        # Case-insensitive replacement
        pattern = re.compile(re.escape(keyword), re.IGNORECASE)
        highlighted_text = pattern.sub(
            f'<span style="background-color: {color}; color: #000; font-weight: 600; padding: 3px 6px; border-radius: 4px; border: 1px solid rgba(0,0,0,0.2);">{keyword}</span>', 
            highlighted_text
        )
    
    return highlighted_text

def highlight_section_text(text: str, section_keywords: Set[str], color: str) -> str:
    """Highlight keywords in a specific section with its assigned color."""
    highlighted_text = text
    
    # Sort keywords by length (longest first)
    sorted_keywords = sorted(section_keywords, key=len, reverse=True)
    
    for keyword in sorted_keywords:
        if len(keyword) >= 3:
            pattern = re.compile(re.escape(keyword), re.IGNORECASE)
            highlighted_text = pattern.sub(
                f'<span style="background-color: {color}; color: #000; font-weight: 600; padding: 3px 6px; border-radius: 4px; border: 1px solid rgba(0,0,0,0.2);">{keyword}</span>', 
                highlighted_text
            )
    
    return highlighted_text

# -----------------------------
# CSS
# -----------------------------
st.markdown("""
<style>
  #MainMenu, footer, header {visibility: hidden;}
  
  .block-container { 
    padding-top: 2rem; 
    padding-bottom: 2.5rem; 
    font-size: 2.2rem !important;
    max-width: 95%;
  }

  .stMarkdown, .stText, p, div, span {
    font-size: 2.2rem !important;
    color: #000000 !important;
    line-height: 1.6 !important;
  }

  h1 {
    font-size: 5rem !important;
    font-weight: 900 !important;
    color: #000000 !important;
    margin-bottom: 2rem !important;
    text-align: center;
  }

  h4, .stMarkdown h4 {
    font-size: 3rem !important;
    font-weight: 800 !important;
    color: #000000 !important;
    margin-bottom: 1.5rem !important;
    margin-top: 2rem !important;
  }

  .card {
    background: #fff;
    border: 3px solid #eef1f5;
    border-radius: 15px;
    box-shadow: 0 6px 20px rgba(0,0,0,0.08);
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    font-size: 2.2rem !important;
    line-height: 1.7 !important;
  }

  .card-header {
    font-weight: 800 !important; 
    color: #000000 !important; 
    margin-bottom: 1.5rem !important;
    padding-bottom: 0.8rem !important;
    border-bottom: 4px solid #dbeafe;
    font-size: 2.5rem !important;
  }

  .simple-card {
    background: #fff;
    border: 3px solid #eef1f5;
    border-radius: 15px;
    box-shadow: 0 6px 20px rgba(0,0,0,0.08);
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    font-size: 2.2rem !important;
    line-height: 1.7 !important;
  }

  .prompt { 
    background: #fef3c7 !important;
    border: 3px solid #facc15 !important;
    border-radius: 12px !important; 
    padding: 2.5rem 3rem !important;
    font-size: 2.8rem !important;
    font-weight: 600 !important;
    color: #000000 !important;
    line-height: 1.6 !important;
    margin-bottom: 2.5rem !important;
  }

  /* Wide, compact metrics grid */
  .metric-grid { 
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 12px;
    margin: 0.8rem 0;
    width: 100%;
    max-width: 1200px;
    margin-left: auto;
    margin-right: auto;
  }

  /* Short, wide metric boxes */
  .metric {
    text-align: center;
    background: #f8fafc; 
    border: 2px solid #e2e8f0;
    border-radius: 8px;
    padding: 0.6rem 1.8rem;
    min-height: 65px;
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
  }

  /* Compact metric scores */
  .metric span { 
    font-size: 1.9rem !important;
    font-weight: 800 !important; 
    color: #000000 !important;
    margin-bottom: 0.2rem;
    line-height: 1;
  }

  /* Compact metric labels */
  .metric-label {
    font-size: 1.1rem !important;
    color: #475569 !important;
    font-weight: 600;
    line-height: 1.2;
    text-align: center;
  }

  hr { 
    margin: 1.8rem 0;
    height: 3px;
    background: #2563eb;
    border: none;
    border-radius: 2px;
  }
</style>
""", unsafe_allow_html=True)

# -----------------------------
# UI
# -----------------------------
st.markdown("<h1>TASKEVAL</h1>", unsafe_allow_html=True)

# Task Instruction
st.markdown("#### Task Instruction")
st.markdown(f"<div class='prompt'>{CONFIG['TASK_INSTRUCTION']}</div>", unsafe_allow_html=True)

# Evaluation Metrics - using standard h4 heading
st.markdown("#### Evaluation Metrics")
scores = CONFIG["DEEPEVAL_SCORE"]
col_html = "".join(
    f"<div class='metric'><span>{v:.1f}</span><div class='metric-label'>{k}</div></div>"
    for k, v in scores.items()
)
st.markdown(f"<div class='metric-grid'>{col_html}</div>", unsafe_allow_html=True)

st.markdown("<hr/>", unsafe_allow_html=True)

# Get section-specific matches
section_matches = get_section_specific_matches(CONFIG['LLM_OUTPUT'], CONFIG['EXTRACTED_SECTIONS'])

# Side-by-side layout: LLM Output vs Extracted Sections
col1, col2 = st.columns(2, gap="large")

with col1:
    st.markdown("#### Extracted Output")
    
    # Highlight the LLM output with different colors
    highlighted_output = highlight_text_multicolor(CONFIG['LLM_OUTPUT'], section_matches)
    
    st.markdown(f"""
    <div class='card'>
        <div class='card-header'>LLM Generated Output</div>
        {highlighted_output}
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("#### Extracted Sections from Content")
    for i, sec in enumerate(CONFIG["EXTRACTED_SECTIONS"]):
        color = HIGHLIGHT_COLORS[i % len(HIGHLIGHT_COLORS)]
        
        # Highlight keywords in this section if it has matches
        if i in section_matches:
            highlighted_section = highlight_section_text(sec.strip(), section_matches[i], color)
        else:
            highlighted_section = sec.strip()
        
        st.markdown(f"""
        <div class='simple-card'>
            {highlighted_section}
        </div>
        """, unsafe_allow_html=True)