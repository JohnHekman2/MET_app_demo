import streamlit as st
import pandas as pd
import altair as alt
from io import StringIO
from datetime import datetime
import re
import html
import json
import requests
import time

st.set_page_config(page_title="Beleidsoptie Milieu-impact Scoping (2-pagina)", layout="wide")


# --- Helpers -----------------------------------------------------------------
@st.cache_data
def load_indicators(uploaded_file):
    if uploaded_file is None:
        return [
            "Luchtkwaliteit",
            "Broeikasgasuitstoot",
            "Waterkwaliteit",
            "Biodiversiteit",
            "Landgebruik",
            "Energieverbruik",
            "Afvalgeneratie",
            "Geluidsoverlast",
            "Bodemverontreiniging",
            "Uitputting natuurlijke hulpbronnen"
        ]
    else:
        try:
            df = pd.read_csv(uploaded_file, header=0)
            first_col = df.columns[0]
            items = df[first_col].dropna().astype(str).tolist()
            return items
        except Exception as e:
            st.sidebar.error(f"Kon geüploade CSV niet lezen: {e}")
            return []


def sanitize_key(s):
    s2 = s.strip().lower()
    s2 = re.sub(r"\s+", "_", s2)
    s2 = re.sub(r"[^a-z0-9_]", "", s2)
    return s2 or "indicator"


def sanitize_filename(s, maxlen=30):
    s2 = re.sub(r"\s+", "_", s.strip())
    s2 = re.sub(r"[^A-Za-z0-9_\-]", "", s2)
    return (s2[:maxlen] or "beleid").rstrip("_")


# --- Session State Initialization (More robust, key-by-key) ---
# Initialize all essential keys if they don't already exist
if "history" not in st.session_state:
    st.session_state.history = []
if "policy_desc" not in st.session_state:
    st.session_state.policy_desc = ""
if "indicator_list" not in st.session_state:
    st.session_state.indicator_list = load_indicators(None)
if "selected_inds" not in st.session_state:
    st.session_state.selected_inds = st.session_state.indicator_list
if "render_inds" not in st.session_state:
    st.session_state.render_inds = st.session_state.indicator_list

# Initialize AI-specific session state keys
if "gemini_api_key" not in st.session_state:
    # Use st.secrets to retrieve the API key securely
    st.session_state.gemini_api_key = st.secrets.get("GEMINI_API_KEY", "")
if "ai_suggestions" not in st.session_state:
    st.session_state.ai_suggestions = {}
if "show_ai_columns" not in st.session_state:
    st.session_state.show_ai_columns = False

# --- Sidebar: inputs and controls --------------------------------------------
st.sidebar.title("Beleidsoptie Milieu-impact Scoping")

st.sidebar.markdown("---")
st.sidebar.header("Stap 1 — Beschrijf de beleidsoptie")
st.sidebar.markdown("Geef een duidelijke toelichting op wat de beleidsoptie inhoudt.")
st.sidebar.text_area(
    "Beleidsoptie toelichting",
    value=st.session_state.policy_desc,
    placeholder="Beschrijf het beleid (wat het doet, wie het beïnvloedt, tijdsbestek, grenzen, belangrijke aannames)...",
    height=260,
    key="policy_desc"
)

st.sidebar.markdown("---")
st.sidebar.header("Indicatorbron")
uploaded = st.sidebar.file_uploader("Upload een CSV met indicatoren (eerste kolom wordt gebruikt)", type=["csv"],
                                    key="indicator_uploader")

if uploaded:
    st.session_state.indicator_list = load_indicators(uploaded)
    st.session_state.selected_inds = st.session_state.indicator_list
    st.session_state.render_inds = st.session_state.indicator_list
st.sidebar.write(f"Geladen indicatoren: {len(st.session_state.indicator_list)}")

with st.sidebar.expander("Kies indicatoren om te tonen"):
    st.multiselect(
        "Kies indicatoren (laat leeg om alles te tonen)",
        options=st.session_state.indicator_list,
        default=st.session_state.selected_inds,
        key="selected_inds"
    )
    st.session_state.render_inds = st.session_state.selected_inds if st.session_state.selected_inds else st.session_state.indicator_list

impact_options = ["Negatief", "Geen impact", "Positief"]

# Initialize per-indicator session_state defaults (do this before widget creation)
for ind in st.session_state.render_inds:
    sk = sanitize_key(ind)
    key_imp = f"imp__{sk}"
    key_exp = f"exp__{sk}"
    # Initialize keys if they don't already exist
    if key_imp not in st.session_state:
        st.session_state[key_imp] = "Geen impact"
    if key_exp not in st.session_state:
        st.session_state[key_exp] = ""

# --- Sidebar AI Integration --------------------------------------------------
st.sidebar.markdown("---")
st.sidebar.header("AI-integratie (Gemini)")
st.sidebar.markdown("Ontvang een snelle beoordeling van Gemini voor de impacts.")

# Check if the API key is available
api_key_loaded = bool(st.session_state.gemini_api_key)
if api_key_loaded:
    st.sidebar.success("Gemini API-sleutel geladen uit secrets!")
else:
    st.sidebar.warning("Gemini API-sleutel niet gevonden. Voeg deze toe aan uw Streamlit secrets.")

if st.sidebar.button("Ontvang AI-suggesties", disabled=not api_key_loaded):
    if not api_key_loaded:
        st.sidebar.error("Kan geen AI-suggesties ophalen zonder een geldige API-sleutel. Configureer uw secrets.")
    else:
        st.session_state.ai_suggestions = {}
        with st.spinner("AI-suggesties aan het genereren..."):
            for ind in st.session_state.render_inds:
                try:
                    # The prompt to the Gemini API is now in Dutch.
                    # This tells the model to respond in Dutch.
                    prompt = f"Voeg een vlag en een korte (maximaal 3 zinnen) toelichting toe of de beleidsoptie '{st.session_state.policy_desc}' de milieu-indicator '{ind}' negatief, positief, of geen impact heeft. De context zijn beleidsopties in Noordwest-Europa. Herhaal de beleidsoptie niet in uw toelichting, geef alleen beknopte hints over positieve of negatieve impacts. De vlag moet een van deze drie opties zijn: 'Positief', 'Negatief', of 'Geen impact'."

                    # Generation config for structured JSON output
                    payload = {
                        "contents": [
                            {
                                "role": "user",
                                "parts": [
                                    {"text": prompt}
                                ]
                            }
                        ],
                        "generationConfig": {
                            "responseMimeType": "application/json",
                            "responseSchema": {
                                "type": "OBJECT",
                                "properties": {
                                    "flag": {
                                        "type": "STRING",
                                        "enum": ["Positief", "Negatief", "Geen impact"]
                                    },
                                    "explanation": {"type": "STRING"}
                                }
                            }
                        }
                    }

                    # API key is used here
                    apiUrl = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-05-20:generateContent?key={st.session_state.gemini_api_key}"

                    response = requests.post(apiUrl, json=payload, headers={'Content-Type': 'application/json'})

                    # Exponential backoff retry logic
                    retries = 0
                    while response.status_code == 429 and retries < 5:
                        delay = 2 ** retries
                        time.sleep(delay)
                        response = requests.post(apiUrl, json=payload, headers={'Content-Type': 'application/json'})
                        retries += 1

                    response.raise_for_status()  # Raise an exception for bad status codes

                    ai_response_json = response.json()

                    ai_output = ai_response_json['candidates'][0]['content']['parts'][0]['text']
                    ai_data = json.loads(ai_output)

                    st.session_state.ai_suggestions[ind] = {
                        "flag": ai_data.get("flag", "Geen impact"),
                        "explanation": ai_data.get("explanation", "Geen toelichting gegeven.")
                    }

                except Exception as e:
                    st.error(f"Fout voor indicator '{ind}': {e}")
                    st.session_state.ai_suggestions[ind] = {
                        "flag": "Fout",
                        "explanation": f"API-aanroep mislukt: {e}"
                    }
        st.session_state.show_ai_columns = True
        st.success("AI-suggesties gegenereerd!")
        st.rerun()

# --- Main Page: Assess impacts ----------------------------------------------
st.title("Stap 2 — Beoordeel de milieu-impact")
if not st.session_state.policy_desc.strip():
    st.info("Voer een beleidsbeschrijving in de zijbalk in om de beoordeling te starten.")
else:
    st.subheader("Beleidsbeschrijving")
    st.write(st.session_state.policy_desc)

    # Custom CSS to style the AI columns
    st.markdown("""
        <style>
        .ai-column {
            background-color: #e6e6ff; /* A light purple */
            border-radius: 5px;
            padding: 10px;
        }
        .stTextArea [data-baseweb=base-input] {
            background-color: #e6e6ff;
        }
        </style>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown(
        "Voor elke indicator hieronder, selecteer de impact (Negatief / Geen impact / Positief) en geef een toelichting op uw oordeel.")

    # Determine the columns to display based on whether AI suggestions have been run
    if st.session_state.show_ai_columns:
        # Show all 5 columns
        cols_config = [3, 2, 2, 2, 3]
        col_names = ["Indicator", "Impact", "Uw Toelichting", "AI-oordeel", "AI-toelichting"]
    else:
        # Show only the original 3 columns
        cols_config = [3, 2, 5]
        col_names = ["Indicator", "Impact", "Toelichting"]

    # Display headers
    cols = st.columns(cols_config)
    for i, name in enumerate(col_names):
        with cols[i]:
            st.write(f"**{name}**")

    # Display rows
    for i, ind in enumerate(st.session_state.render_inds):
        sk = sanitize_key(ind)
        key_imp = f"imp__{sk}"
        key_exp = f"exp__{sk}"

        cols = st.columns(cols_config)
        with cols[0]:
            st.write(f"**{ind}**")
        with cols[1]:
            try:
                init_idx = impact_options.index(st.session_state.get(key_imp, "Geen impact"))
            except ValueError:
                init_idx = 1
            cols[1].selectbox(f"Impact {i + 1}", options=impact_options, index=init_idx, key=key_imp,
                              label_visibility="collapsed")
        with cols[2]:
            cols[2].text_area(f"Toelichting {i + 1}", value=st.session_state.get(key_exp, ""), key=key_exp, height=90,
                              label_visibility="collapsed")

        # Conditionally render AI columns
        if st.session_state.show_ai_columns:
            ai_data = st.session_state.ai_suggestions.get(ind, {})
            ai_flag = ai_data.get('flag', 'Niet Uitgevoerd')
            ai_explanation = ai_data.get('explanation', '')

            with cols[3]:
                st.markdown(f'<div class="ai-column">**{ai_flag}**</div>', unsafe_allow_html=True)
            with cols[4]:
                cols[4].text_area(f"AI Suggestion {i + 1}", value=ai_explanation, key=f"ai_exp_{i}", height=90,
                                  label_visibility="collapsed", help="Dit is een AI-gegenereerde suggestie.")

    rows = []
    for ind in st.session_state.render_inds:
        sk = sanitize_key(ind)
        key_imp = f"imp__{sk}"
        key_exp = f"exp__{sk}"
        ai_data = st.session_state.ai_suggestions.get(ind, {})
        rows.append({
            "Indicator": ind,
            "Impact": st.session_state.get(key_imp, "Geen impact"),
            "Toelichting": st.session_state.get(key_exp, ""),
            "AI-oordeel": ai_data.get("flag", "Niet Uitgevoerd"),
            "AI-toelichting": ai_data.get("explanation", "")
        })
    edited = pd.DataFrame(rows)

    missing_explanations = edited[
        (edited["Impact"] == "Negatief") & (edited["Toelichting"].astype(str).str.strip() == "")]
    if not missing_explanations.empty:
        st.warning(
            f"{len(missing_explanations)} indicator(en) gemarkeerd als Negatief zonder toelichting. Voeg toelichtingen toe voor deze rijen.")
        with st.expander("Indicatoren zonder toelichting (Negatieve impact)"):
            st.write(missing_explanations["Indicator"].tolist())

    counts = edited["Impact"].value_counts().reindex(["Positief", "Geen impact", "Negatief"]).fillna(0).astype(int)
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("Impactoverzicht")
        st.write(counts.to_frame("aantal"))
        chart_df = pd.DataFrame({"Impact": counts.index, "aantal": counts.values})
        chart = alt.Chart(chart_df).mark_bar().encode(
            x=alt.X("Impact:N", sort=["Positief", "Geen impact", "Negatief"]),
            y="aantal:Q",
            color=alt.Color("Impact:N", scale=alt.Scale(domain=["Positief", "Geen impact", "Negatief"],
                                                        range=["#2ca02c", "#1f77b4", "#d62728"]))
        ).properties(height=240)
        st.altair_chart(chart, use_container_width=True)
    with col2:
        st.subheader("Beleid")
        snippet = (st.session_state.policy_desc[:200] + "...") if len(
            st.session_state.policy_desc) > 200 else st.session_state.policy_desc
        st.write(snippet if snippet else "Geen beschrijving ingevoerd")
        st.write("Indicatoren weergegeven:", len(st.session_state.render_inds))

    st.write("---")
    save_col, hist_col = st.columns([1, 2])
    with save_col:
        if st.button("Momentopname opslaan in sessiegeschiedenis"):
            rec = {
                "policy": st.session_state.policy_desc,
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "table": edited.to_dict(orient="records")
            }
            st.session_state.history.append(rec)
            st.success("Opgeslagen in sessiegeschiedenis (niet persistent).")

        csv_buf = StringIO()
        edited.to_csv(csv_buf, index=False)
        short = sanitize_filename(st.session_state.policy_desc[:30])
        fname = f"{short}_{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}_impacts.csv"
        st.download_button("CSV van tabel downloaden", csv_buf.getvalue(), file_name=fname, mime="text/csv")

    with hist_col:
        st.subheader("Sessiegeschiedenis (recente opslag)")
        history = st.session_state.get("history", [])
        if len(history) == 0:
            st.write(
                "Nog geen momentopnamen opgeslagen. Klik op 'Momentopname opslaan in sessiegeschiedenis' om de huidige tabel te bewaren.")
        else:
            for i, rec in enumerate(reversed(history[-10:])):
                idx = len(history) - 1 - i
                with st.expander(f"Opgeslagen {rec['timestamp']}", expanded=False):
                    hdf = pd.DataFrame(rec["table"])
                    st.write(hdf)
                    buf = StringIO()
                    hdf.to_csv(buf, index=False)
                    st.download_button(f"CSV downloaden (item #{idx})", buf.getvalue(), file_name=f"snapshot_{idx}.csv",
                                       mime="text/csv")

    st.sidebar.write("---")
    st.sidebar.subheader("Huidige Sessiestatus (Debug)")
    st.sidebar.write(st.session_state)
