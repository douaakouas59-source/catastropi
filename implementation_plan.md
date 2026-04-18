# Implementation Plan: CATASTROPI Complete Overhaul

This plan details the comprehensive fixes and new features requested for the CATASTROPI platform, segmented by functionalities for Responsable, Agent, Citoyen, Chatbot, and the AI Module.

## User Review Required

> [!WARNING]
> **Performance Impact of CamemBERT**: The integration of HuggingFace's `transformers` library with CamemBERT will significantly increase the memory footprint of the Flask backend and the startup time. I will use the base Distil-CamemBERT pipeline which is smaller, but please be aware of the performance cost if running locally.
> **Dependency Additions**: I will add `transformers`, `torch`, `shap`, and `requests` to your `requirements.txt`.

## Proposed Changes

### 1. Module IA (`module_ia.py`)
- **[MODIFY] module_ia.py**
  - **Accuracy Overfitting**: Refactor the Random Forest training (`_entrainer_modele_risque` & `_entrainer_modele_impact`) to use `train_test_split(test_size=0.2)` and report accuracy only on the test set.
  - **Fake News Dataset**: Replace the loops generating fake/real datasets with actual, diverse string samples (100 real Algerian official phrases, 50 local rumors in Algerian context).
  - **Continuous Learning**: Swap the Fake News learning mechanism to `MultinomialNB` (Naive Bayes) and ensure `partial_fit` is properly receiving the text vector features from validation/rejection routes.
  - **Time Aggravation Factor**: Add `heure_actuelle = datetime.utcnow().hour` feature to the ML risk models. Update `_analyser_facteurs_risque` and Random Forest inputs.
  - **Deep NLP (CamemBERT)**: Integrate HuggingFace `pipeline("text-classification", model="cmarkea/distilcamembert-base-sentiment")` (or similar compact CamemBERT model) inside `detecter_fake_news` to complement or replace the basic TF-IDF context extraction.
  - **SHAP Explicability**: Integrate `shap.TreeExplainer` on the Random Forest Risk model to dynamically generate exact percentage-based reasons for the predicted risk.
  - **External APIs**: Integrate `requests` to fetch data from:
    - **OpenWeatherMap API**: Fetch temperature, wind, and rain based on the Wilaya (zone/localisation) dynamically when predicting impact.
    - **Nominatim / OSRM**: Replace random evacuation logic with an API call to OSRM mapping using city coordinates to estimate real evacuation route times.

### 2. Backend API (`app.py`)
- **[MODIFY] app.py**
  - Update `/api/declarer` and `/api/analyze_signal` to pass the correct parameters (including current hour) to `module_ia`.
  - Add logic to retrieve external API data (weather + OSRM) and serve it in the `analyze_signal` payload.
  - **Agents Assignment Workflow**: Enhance `/api/catastrophes/<id>/agents` to save custom user instructions.
  - **Victim & Needs Workflows**: Enhance endpoints (`/api/victimes`) to attach an explicit `catastrophe_id`. Add a new route `/api/besoins/<id>/valider` to approve/reject agent resource requests.
  - **Chatbot Route**: Add a new route `/api/chatbot` that processes text commands (e.g., matching Wilaya names, "consignes", "traitement", victim names) relying on the SQLite DB.

### 3. Frontend Web Application (`index.html` and assets)
- **[MODIFY] index.html**
  - **UI Responsable**:
    - Build "Historique IA" sub-view using the existing `/api/get_analyses` endpoint.
    - Enhance "Validation des Signalements" table with `<span class="badge">` color-coding (Green >70%, Orange 40-70%, Red <40%).
    - Build the requested Comprehensive Modal (`#modal-ia-analysis`) containing: SHAP explicit reasons, CamemBERT text analysis score, risks, weather factors, OSRM evacuation times, and two buttons "Accepter et déclarer" / "Refuser".
    - Enhance "Connecter Agent" modal: Display Agents in a multi-select table with mock 'Distance to catastrophe', plus a textarea for custom missions.
  - **UI Agent**:
    - Add a dropdown for `catastrophe_id` in the Victim declaration form.
    - Build an archive view mapping Victims to their associated Catastrophe, enabling CSV/PDF export.
    - Complete workflow forms for Agent -> Responsable Needs (Requests).
    - "Mes Missions": Tabbed UI mapping to missions assigned to `agent_id`.
    - "Catastrophes Publiées": General view with map integration for Agents.
  - **UI Citoyen**:
    - Overhaul the reporting form: Add Wilaya dropdown (listing all 69 Algerian wilayas), precise localizations, photo upload element preview.
    - Table mapping Citoyen histories (`/api/catastrophes`).
  - **Chatbot Window**:
    - Construct an interactive Floating Chat Interface with JS logic fetching from `/api/chatbot`.

## Open Questions

- Ensure API Keys: Since I will add OpenWeatherMap and OSRM logic, do you have API Keys for OpenWeatherMap, or should I use a generic free placeholder key/mock it via backend if the API blocks request?
- CamemBERT Model Size: CamemBERT base can be ~400MB to download locally on the first run. Are you okay with this slight delay when you start the server the first time?

## Verification Plan

### Automated Tests
- N/A - Focusing on end-to-end integration mapping over rigorous unit tests due to time frame.

### Manual Verification
- Log in as the Admin/Responsable to test the updated IA continuous training routes.
- Trigger fake signalement in Citadel to evaluate the new UI color codings and SHAP factors.
- Use Chatbot as Citoyen with exact keywords (e.g. "Oran catastrophe") to ensure valid JSON replies.
