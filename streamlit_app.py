import streamlit as st
import pandas as pd
from pathlib import Path
import os
from dotenv import load_dotenv

from src.core.recommendation_engine import UserPreference, get_candidate_restaurants
from src.llm.orchestrator import generate_llm_recommendations

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Zomato AI Recommendation",
    page_icon="🍴",
    layout="wide",
)

# Custom CSS for Zomato branding
st.markdown("""
    <style>
    .main {
        background-color: #f8f8f8;
    }
    .stButton>button {
        background-color: #e23744;
        color: white;
        border-radius: 8px;
        border: none;
        font-weight: 600;
    }
    .stButton>button:hover {
        background-color: #b0202b;
        color: white;
    }
    .restaurant-card {
        background-color: white;
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #e8e8e8;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        margin-bottom: 20px;
    }
    .rating-badge {
        background-color: #267e3e;
        color: white;
        padding: 2px 8px;
        border-radius: 6px;
        font-weight: bold;
        font-size: 0.9em;
    }
    </style>
    """, unsafe_allow_html=True)

# App Header
st.title("🍴 Zomato AI Recommendation")
st.subheader("AI-powered restaurant suggestions using real Zomato data.")

@st.cache_data
def load_data():
    path = Path("data/processed/zomato_clean.parquet")
    if not path.exists():
        return None
    return pd.read_parquet(path)

# Initialize data session state
if "data_initialized" not in st.session_state:
    st.session_state.data_initialized = False

df = load_data()

if df is None and not st.session_state.data_initialized:
    st.info("👋 Welcome! This application requires a processed dataset to provide recommendations.")
    col1, col2 = st.columns([1, 2])
    with col1:
        if st.button("🚀 Initialize Dataset", use_container_width=True):
            path = Path("data/processed/zomato_clean.parquet")
            with st.status("Downloading and processing data (this may take a minute)..."):
                try:
                    # Heavy import delayed until button click
                    from src.data_access.phase1_preprocessing import run_phase1_preprocessing
                    run_phase1_preprocessing(output_path=path)
                    st.session_state.data_initialized = True
                    st.rerun()
                except Exception as e:
                    st.error(f"Error during preprocessing: {e}")
    st.stop()
elif df is not None:
    # Sidebar Filters
    st.sidebar.header("Filters")
    
    # Location Filter
    loc_col = None
    for candidate in ("location", "listed_in(city)", "city", "address"):
        if candidate in df.columns:
            loc_col = candidate
            break
    
    locations = sorted(df[loc_col].dropna().unique()) if loc_col else []
    selected_location = st.sidebar.selectbox("Location", ["All"] + list(locations))
    
    # Cuisine Filter
    all_cuisines = set()
    if "cuisines_clean" in df.columns:
        for cuisines in df["cuisines_clean"].dropna():
            if isinstance(cuisines, (list, tuple)):
                all_cuisines.update(cuisines)
            elif isinstance(cuisines, str):
                all_cuisines.update([c.strip() for c in cuisines.split(",")])
    
    selected_cuisines = st.sidebar.multiselect("Preferred Cuisines", sorted(list(all_cuisines)))
    
    # Rating filter
    min_rating = st.sidebar.slider("Minimum Rating", 0.0, 5.0, 3.5, 0.1)
    
    # Price filter
    max_price = st.sidebar.number_input("Maximum Price for Two", min_value=0, value=2000, step=100)
    
    # AI Toggle
    use_ai = st.sidebar.checkbox("Use AI for Personalized Explanations", value=True)

    # Main search button
    if st.button("Find Restaurants"):
        prefs = UserPreference(
            location=None if selected_location == "All" else selected_location,
            min_rating=min_rating,
            cuisines=selected_cuisines,
            max_price=float(max_price)
        )
        
        with st.spinner("Finding the best options for you..."):
            candidates = get_candidate_restaurants(df, prefs, top_n=10)
            
            if not candidates:
                st.warning("No restaurants found matching your criteria. Try relaxing your filters!")
            else:
                if use_ai:
                    try:
                        results = generate_llm_recommendations(prefs, candidates, max_results=5)
                        display_results = results
                    except Exception as e:
                        st.error(f"AI Error: {e}. Falling back to standard recommendations.")
                        display_results = candidates[:5]
                else:
                    display_results = candidates[:5]
                
                st.write(f"### Found {len(display_results)} Recommendations")
                
                for res in display_results:
                    with st.container():
                        st.markdown(f"""
                        <div class="restaurant-card">
                            <div style="display: flex; justify-content: space-between; align-items: start;">
                                <div>
                                    <h3 style="margin: 0;">{res.name}</h3>
                                    <p style="color: #686b78; margin: 4px 0;">{res.location or ''}</p>
                                </div>
                                <span class="rating-badge">{res.rating if res.rating else 'N/A'} ★</span>
                            </div>
                            <p style="font-size: 0.9em; margin: 8px 0;">
                                <b>Cuisines:</b> {", ".join(res.cuisines) if res.cuisines else 'N/A'} | 
                                <b>Approx. Price:</b> ₹{res.price if res.price else 'N/A'}
                            </p>
                            <p style="color: #3d4152; font-style: italic;">
                                {getattr(res, 'reason', 'High quality option based on your preferences.')}
                            </p>
                        </div>
                        """, unsafe_allow_html=True)

else:
    st.info("Please ensure Phase 1 has been run and data is available at `data/processed/zomato_clean.parquet`.")
