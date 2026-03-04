## AI Restaurant Recommendation Service – Architecture

### 1. Goal & Scope

- **Objective**: Build an AI-powered restaurant recommendation service that takes user preferences (price, place/location, rating, cuisine), uses the Zomato dataset from Hugging Face, calls an LLM, and returns clear, human-friendly recommendations.
- **Out of scope (initially)**: Full production deployment, advanced personalization based on user history, multi-tenant support, payments, user accounts/identity.

---

### 2. High-Level System Overview

- **Client layer**: Any consumer (CLI, web UI, mobile app, or Postman) that sends a recommendation request and displays results.
- **API layer**: A backend service exposing HTTP endpoints to receive user preferences and return recommendations.
- **Frontend layer**: A **Streamlit** application (`streamlit_app.py`) that provides a user-friendly interface for restaurant discovery and AI-powered recommendations.
- **Recommendation engine**: Core logic that:
  - Validates and normalizes user preferences.
  - Filters and scores restaurants from the Zomato dataset.
  - Prepares a concise context and prompt for the LLM.
  - Calls the LLM and post-processes its responses.
- **Data layer**:
  - Local or cached copy of the Hugging Face Zomato dataset.
  - Optional vector index or search index for efficient retrieval.
- **LLM integration layer**:
  - A thin client that wraps external LLM APIs (e.g., OpenAI, Anthropic, etc.).
  - Handles prompt construction, safety controls, and retry logic.

---

### 3. Key Data Source

- **Hugging Face dataset**: `ManikaSaini/zomato-restaurant-recommendation` (from `https://huggingface.co/datasets/ManikaSaini/zomato-restaurant-recommendation`).
- **Data ingestion strategy**:
  - Access via Hugging Face `datasets` library or HTTP download during an offline data preparation step.
  - Persist processed data in a local store (e.g., Parquet/CSV in a `data/` folder) or lightweight DB (SQLite/Postgres) for fast querying.
  - Optionally build a separate index (e.g., by city, cuisine, price range) for quick filtering.

---

### 4. Core Components

#### 4.1 API Layer

- **Responsibilities**:
  - Expose REST endpoints (e.g., `POST /api/v1/recommendations`).
  - Handle request parsing, authentication (if any), and response formatting.
  - Map raw HTTP input to internal DTOs / domain objects (e.g., `UserPreference`).
- **Possible technologies**: Any modern web framework (e.g., FastAPI/Flask/Express/NestJS). Exact choice can be decided later.

#### 4.2 Preference Parsing & Validation

- **Input fields**:
  - `price`: numeric or categorical range (e.g., low/medium/high or 1–4).
  - `place`: city/area or free-text location.
  - `rating`: minimum rating threshold.
  - `cuisine`: one or multiple cuisine tags.
- **Responsibilities**:
  - Validate required fields and types.
  - Normalize values into internal canonical forms (e.g., standardized cuisine labels, numeric price bands).
  - Provide defaults when fields are missing (e.g., reasonable rating threshold, any cuisine).

#### 4.3 Dataset Access Layer

- **Responsibilities**:
  - Abstract direct access to the Zomato dataset (load, refresh, query).
  - Provide query methods such as:
    - `get_restaurants_by_filters(location, cuisines, price_range, min_rating)`
    - `get_top_n_restaurants(filters, n)`
  - Support indexing or caching to avoid full scans on every request.
- **Implementation idea**:
  - Use an offline preprocessing phase to:
    - Clean and normalize raw columns.
    - Derive computed columns (price bands, cuisine tags, standardized locations).
    - Persist the processed dataset to a file or DB.

#### 4.4 Ranking & Filtering Engine

- **Responsibilities**:
  - Apply user preferences to filter the dataset (hard constraints like location, min rating).
  - Define a scoring function that considers:
    - Rating (primary factor).
    - Price alignment with user preference.
    - Cuisine match (exact or partial).
    - Popularity signals if available (e.g., number of votes/reviews).
  - Return a candidate list of top N restaurants (e.g., 10–20) for the LLM to reason over.

#### 4.5 LLM Orchestration Layer

- **Responsibilities**:
  - Convert candidate restaurants and user preferences into a concise, structured context.
  - Construct robust prompts that:
    - List candidate restaurants with key attributes (name, address, rating, price, cuisines, highlights).
    - Clearly state user preferences and constraints.
    - Ask the LLM to select and explain the best options.
  - Call the external LLM API with appropriate parameters (temperature, max tokens, etc.).
  - Handle errors, rate limits, retries, and timeouts.
  - Optionally run simple safety checks (e.g., content filters) on the LLM’s response.

#### 4.6 Response Formatting Layer

- **Responsibilities**:
  - Parse the LLM’s response into a structured internal format (e.g., `RecommendedRestaurant[]`).
  - Enforce schema: each recommendation must include at least name, location, cuisine, price band, rating, and reasoning.
  - Convert structured results to the API’s response schema.
  - Ensure responses are deterministic and machine-readable in addition to being human-friendly (e.g., JSON with `explanations` fields).

---

### 5. Project Phases

#### Phase 0 – Requirements & Environment Setup

- **Goals**:
  - Clarify target runtime environment (local only vs. also cloud).
  - Decide tech stack (language/framework, LLM provider).
  - Initialize project structure and basic tooling.
- **Deliverables**:
  - Project skeleton (e.g., `src/`, `tests/`, `data/`, `docs/`).
  - Basic README explaining the problem and dataset.
  - Environment configuration (e.g., `.env.example` for API keys).

#### Phase 1 – Data Ingestion & Preprocessing

- **Goals**:
  - Download and explore the Zomato dataset from Hugging Face.
  - Clean and normalize fields needed for recommendations.
- **Key steps**:
  - Ingest dataset using the Hugging Face `datasets` library or direct download.
  - Explore and profile data (missing values, outliers, inconsistent categories).
  - Normalize key fields:
    - Standardize city/place names.
    - Define price bands from raw price fields.
    - Clean and tokenize cuisine fields.
  - Persist cleaned dataset in a fast-to-load format.
- **Deliverables**:
  - Data processing script or module.
  - Processed dataset stored locally (or in a simple DB).
  - Basic data documentation describing important columns and transformations.

#### Phase 2 – Core Recommendation Engine (Non-LLM)

- **Goals**:
  - Implement a deterministic recommendation core that can work without the LLM.
  - Provide a clear candidate selection mechanism.
- **Key steps**:
  - Implement preference parsing & validation.
  - Implement dataset query functions and index loading.
  - Implement scoring and ranking logic.
  - Expose a simple internal API (e.g., `get_candidate_restaurants(preferences)`) returning top N candidates.
- **Deliverables**:
  - `UserPreference` and `Restaurant` domain models.
  - Reusable core recommendation module.
  - Unit tests for filtering and ranking logic.

#### Phase 3 – API Layer & Basic Service

- **Goals**:
  - Stand up the backend service and public endpoints.
  - Wire the core recommendation engine into the API (still without LLM, initially).
- **Key steps**:
  - Implement `POST /api/v1/recommendations` endpoint.
  - Map JSON request body to `UserPreference`.
  - Invoke recommendation core and return structured JSON responses.
  - Add basic input validation and error handling.
- **Deliverables**:
  - Running local API server.
  - API contract documented (e.g., OpenAPI/Swagger or a markdown doc).
  - Integration tests for end-to-end non-LLM recommendation flow.

#### Phase 4 – LLM Integration & Orchestration

- **Goals**:
  - Integrate **Google Gemini** (Gemini LLM) to generate human-friendly and context-aware explanations and final selections.
- **Key steps**:
  - Implement Gemini client module (thin wrapper around Gemini API).
  - Design prompt templates (system and user prompts) for:
    - Selecting the best subset of candidate restaurants.
    - Explaining why each recommendation matches the preferences.
  - Implement orchestration function, e.g., `generate_llm_recommendations(preferences, candidates)`.
  - Ensure the LLM is constrained to only recommend from provided candidates (no hallucinated restaurants).
  - Parse and validate LLM output into structured data.
- **Deliverables**:
  - Gemini client and orchestration layer.
  - Prompt templates versioned and documented.
  - Tests with mocked LLM responses.

#### Phase 5 – Quality, Evaluation & UX Refinement

- **Goals**:
  - Evaluate the quality of recommendations and refine heuristics and prompts.
  - Improve the user-facing format of recommendations.
  - Build a **Streamlit** dashboard for entering preferences and viewing recommendations.
- **Key steps**:
  - Create `streamlit_app.py` with:
    - Sidebar filters (Location, Cuisine, Rating, Price).
    - AI toggle for personalized explanations.
    - Result cards with restaurant details and AI-generated reasoning.
  - Call the backend core/LLM logic directly within the Streamlit process or via the API.
  - Display results as readable cards (restaurant details + short reasoning).
  - Define simple evaluation metrics:
    - Coverage (how often we can return recommendations).
    - Alignment with filters (no off-constraint results).
    - User satisfaction via manual review or small survey.
  - Log requests and recommendations (with anonymization) for analysis.
  - Iterate on:
    - Ranking weights.
    - Prompt wording and structure.
    - Default behaviors for edge cases (e.g., no results found).
- **Deliverables**:
  - UI page (basic frontend) integrated with the API.
  - Evaluation scripts or notebooks.
  - Updated prompt templates and ranking configurations.
  - Documentation of evaluation findings.

#### Phase 6 – Deployment, Monitoring & Future Enhancements

- **Goals**:
  - Make the service deployable and observable.
  - Outline optional future improvements.
- **Key steps**:
  - Containerize the application (e.g., Docker) and define environment configs.
  - Add basic health checks and logging.
  - Define minimal monitoring: request counts, error rates, latency, LLM usage.
  - Plan for scaling (stateless API + shared data store).
  - Identify future enhancements:
    - User profiles and personalization across sessions.
    - Feedback loops (thumbs up/down) to improve ranking.
    - Multi-language support for prompts and responses.
    - More sophisticated retrieval (e.g., vector search over restaurant descriptions/reviews).
    - Rate limiting and request throttling for the public API.
    - Integration with external rating platforms (e.g., TripAdvisor, Yelp) for data enrichment.
- **Deliverables**:
    - Deployment configuration (Dockerfile, health checks).
    - Operations documentation (`DEPLOYMENT.md`).
    - Backlog of future feature ideas.

---

### 6. Directory Structure (Proposed)

- **`/docs`**: Architecture docs, API specs, evaluation notes.
- **`/data`**: Raw and processed datasets (ignored by git if large).
- **`/src`**:
  - **`api/`**: HTTP controllers/routers, request/response schemas.
  - **`core/`**: Preference models, filtering, ranking logic.
  - **`data_access/`**: Dataset loading, querying, caching.
  - **`llm/`**: LLM client, prompt templates, orchestration.
  - **`config/`**: Configuration loading (env vars, settings).
  - **`utils/`**: Shared helpers (logging, error handling).
- **`/tests`**: Unit and integration tests.

---

### 7. Non-Functional Considerations

- **Performance**: Use preprocessed dataset and in-memory/indexed queries to keep response times low even before LLM call; set sensible limits on candidate set size and context length sent to the LLM.
- **Cost**: Minimize LLM usage by pre-filtering with deterministic logic; allow configuration of max tokens and temperature; enable using a local or cheaper model in development.
- **Reliability**: Implement graceful degradation (if LLM is unavailable, return deterministic recommendations only); include timeouts and fallbacks.
- **Security**: Store API keys securely via environment variables; never log sensitive credentials; apply basic rate limiting if exposed publicly.
- **Maintainability**: Keep LLM integration, data access, and core logic decoupled so each can evolve independently (e.g., swapping the dataset or LLM provider without rewriting the API).

