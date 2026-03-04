import os
import streamlit as st
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env (for local dev)
load_dotenv()

# Load GEMINI key from Streamlit secrets or environment
os.environ["GEMINI_API_KEY"] = st.secrets.get("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY")
if not os.environ.get("GEMINI_API_KEY"):
    st.error("⚠️ GEMINI_API_KEY is not set. Add it to Streamlit Secrets or .env")
    st.stop()

# Page config
st.set_page_config(
    page_title="Zomato AI Recommendation",
    page_icon="🍴",
    layout="wide",
)

# Custom CSS
st.markdown("""
    <style>
    .main { background-color: #f8f8f8; }
    .stButton>button {
        background-color: #e23744; color: white; border-radius: 8px; border: none; font-weight: 600;
    }
    .stButton>button:hover { background-color: #b0202b; color: white; }
    .restaurant-card { background-color: white; padding: 20px; border-radius: 12px; border: 1px solid #e8e8e8; box-shadow: 0 2px 4px rgba(0,0,0,0.05); margin-bottom: 20px; }
    .rating-badge { background-color: #267e3e; color: white; padding: 2px 8px; border-radius: 6px; font-weight: bold; font-size: 0.9em; }
    </style>
""", unsafe_allow_html=True)

# App Header
st.title("🍴 Zomato AI Recommendation")
st.subheader("AI-powered restaurant suggestions using real Zomato data.")

# Load data
@st.cache_data
def load_data():
    import pandas as pd
    path = Path("data/processed/zomato_clean.parquet")
    if not path.exists():
        return None
    essential_cols = ["name", "location", "cuisines", "price", "rating", "cuisines_clean"]
    try:
        return pd.read_parquet(path, columns=essential_cols)
    except Exception:
        return pd.read_parquet(path)

# Session state to track dataset initialization
if "data_initialized" not in st.session_state:
    st.session_state.data_initialized = False

try:
    df = load_data()
    if df is not None:
        st.write(f"✅ Dataset loaded. Number of rows: {len(df)}")
        st.write(df.head())
    else:
        st.info("👋 Welcome! Dataset not found.")
    
    # Initialize dataset if missing
    if df is None and not st.session_state.data_initialized:
        if st.button("🚀 Initialize Dataset"):
            path = Path("data/processed/zomato_clean.parquet")
            from src.data_access.phase1_preprocessing import run_phase1_preprocessing
            with st.spinner("Processing data..."):
                run_phase1_preprocessing(output_path=path)
                st.session_state.data_initialized = True
                st.cache_data.clear()
                st.rerun()

    elif df is not None:
        # Sidebar Filters
        st.sidebar.header("Filters")
        
        # Location Filter
        loc_col = next((c for c in ("location", "listed_in(city)", "city", "address") if c in df.columns), None)
        locations = sorted(df[loc_col].dropna().unique()) if loc_col else []
        selected_location = st.sidebar.selectbox("Location", ["All"] + list(locations))
        
        # Cuisine Filter (case-insensitive)
        all_cuisines = set()
        for cuisines in df.get("cuisines_clean", []):
            if isinstance(cuisines, str):
                all_cuisines.update([c.strip().lower() for c in cuisines.split(",")])
            elif isinstance(cuisines, (list, tuple)):
                all_cuisines.update([c.strip().lower() for c in cuisines])
        selected_cuisines = [c.lower() for c in st.sidebar.multiselect("Preferred Cuisines", sorted(all_cuisines))]
        
        # Rating and Price
        min_rating = st.sidebar.slider("Minimum Rating", 0.0, 5.0, 3.5, 0.1)
        max_price = st.sidebar.number_input("Maximum Price for Two", min_value=0, value=2000, step=100)
        
        # AI Toggle
        use_ai = st.sidebar.checkbox("Use AI for Personalized Explanations", value=True)
        
        # Main search
        if st.button("Find Restaurants"):
            from src.core.recommendation_engine import UserPreference, get_candidate_restaurants
            
            prefs = UserPreference(
                location=None if selected_location == "All" else selected_location,
                min_rating=min_rating,
                cuisines=selected_cuisines,
                max_price=float(max_price)
            )

            # Debug: show prefs
            st.write("Filter preferences:", prefs)
            
            with st.spinner("Finding the best options for you..."):
                candidates = get_candidate_restaurants(df, prefs, top_n=10)
                st.write("Candidates found:", len(candidates))
                if len(candidates) == 0:
                    st.warning("No restaurants found matching your criteria. Try relaxing your filters!")
                else:
                    # AI recommendations
                    display_results = candidates[:5]
                    if use_ai:
                        try:
                            from src.llm.orchestrator import generate_llm_recommendations
                            results = generate_llm_recommendations(prefs, candidates, max_results=5)
                            if results:
                                display_results = results
                        except Exception as e:
                            st.error(f"AI Error: {e}. Showing standard recommendations.")
                    
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

except Exception as app_error:
    import streamlit.runtime.scriptrunner as sr
    if isinstance(app_error, (sr.StopException, sr.RerunException)):
        raise app_error
    st.error("⚠️ An unexpected error occurred while running the application.")
    st.exception(app_error)
