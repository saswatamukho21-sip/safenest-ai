"""
SafeNest AI - Streamlit Dashboard Edition
Run: streamlit run app_streamlit.py
"""

import os
import streamlit as st
from core.ml_engine import SafeNestRecommenderEngine

st.set_page_config(
    page_title="SafeNest AI - Accommodation Matcher",
    page_icon="🛡️",
    layout="wide"
)

# Load engine
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "data", "pg_dataset.json")

@st.cache_resource
def get_engine():
    return SafeNestRecommenderEngine(DATASET_PATH)

engine = get_engine()

st.title("🛡️ SafeNest AI: Intelligent Student Living Discovery")
st.caption("AI/ML Prototype: Natural Language Semantic Preference Matching & Multi-Criteria Safety Index")

# Sidebar Controls
st.sidebar.header("⚙️ Model Parameters & Filters")
budget = st.sidebar.slider("Maximum Monthly Budget (₹)", min_value=5000, max_value=15000, value=12000, step=500)
min_safety = st.sidebar.slider("Minimum Safety Score (%)", min_value=30, max_value=95, value=50, step=5)

semantic_weight = st.sidebar.slider("Semantic Match Weight (α)", min_value=0.0, max_value=1.0, value=0.6, step=0.1)
safety_weight = 1.0 - semantic_weight
st.sidebar.caption(f"Safety Weight (1-α): **{round(safety_weight, 1)}**")

# Query Input
query = st.text_input(
    "💬 Search with Natural Language Preferences:",
    placeholder="e.g. Quiet single room with biometric lock and home cooked meals near campus"
)

col_quick1, col_quick2, col_quick3 = st.columns(3)
with col_quick1:
    if st.button("🎧 Study & Biometric"):
        query = "quiet study with biometric lock"
with col_quick2:
    if st.button("🍲 Budget & Home Food"):
        query = "budget twin sharing with home food"
with col_quick3:
    if st.button("🛡️ 24/7 Night Security"):
        query = "24x7 security guard CCTV close to gate"

# Run Recommender
results = engine.recommend(
    query=query,
    max_budget=budget,
    min_safety_score=min_safety,
    alpha_semantic=semantic_weight,
    alpha_safety=safety_weight
)

st.divider()
st.subheader(f"Matched Accommodations ({len(results)} found)")

if not results:
    st.warning("No listings found matching these constraints. Try increasing your budget or lowering the safety threshold.")
else:
    for item in results:
        with st.container():
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown(f"### {item['name']}")
                st.write(f"📍 **Location:** {item['location']} | **Curfew:** {item['curfew_time']}")
                st.write(f"_{item['description']}_")
                st.write("**Amenities:** " + ", ".join([f"`{f}`" for f in item["features"]]))
            with col2:
                st.metric(label="Overall AI Match", value=f"{item['composite_match_score']}%")
                st.metric(label="Safety Index", value=f"{item['safety_score']}/100")
                st.write(f"💰 **₹{item['monthly_rent']} / month**")
                st.progress(item['safety_score'] / 100.0)
            
            with st.expander("🔍 View Safety Breakdown & Semantic Score"):
                st.write(f"- **Semantic Similarity:** {item['semantic_similarity_pct']}%")
                st.write("- **Factor Contributions:**")
                for factor, val in item["safety_breakdown"].items():
                    st.write(f"  - {factor}: `{round(val, 1)} pts`")
            st.divider()
