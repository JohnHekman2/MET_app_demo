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
from openai import OpenAI

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
if "openai_api_key" not in st.session_state:
    st.session_state.openai_api_key = st.secrets.get("OPENAI_API_KEY", "")
if "openai_base_url" not in st.session_state:
    st.session_state.openai_base_url = st.secrets.get("OPENAI_BASE_URL", "")
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
st.sidebar.header("AI-integratie (GPT-5-mini)")
st.sidebar.markdown("Ontvang een snelle beoordeling van de lokale GPT-5 mini server voor de impacts.")

# Check if the API key and base URL are available
api_key_loaded = bool(st.session_state.openai_api_key)
base_url_loaded = bool(st.session_state.openai_base_url)
if api_key_loaded and base_url_loaded:
    st.sidebar.success("OpenAI API-sleutel en basis-URL geladen uit secrets!")
else:
    st.sidebar.warning("OpenAI API-sleutel en/of basis-URL niet gevonden. Voeg ze toe aan uw Streamlit secrets.")

if st.sidebar.button("Ontvang AI-suggesties", disabled=not api_key_loaded or not base_url_loaded):
    if not api_key_loaded or not base_url_loaded:
        st.sidebar.error(
            "Kan geen AI-suggesties ophalen zonder een geldige API-sleutel en basis-URL. Configureer uw secrets.")
    else:
        st.session_state.ai_suggestions = {}
        client = OpenAI(
            api_key=st.session_state.openai_api_key,
            base_url=st.session_state.openai_base_url
        )

        with st.spinner("AI-suggesties aan het genereren..."):
            for ind in st.session_state.render_inds:
                try:
                    # The prompt is in Dutch to instruct the model to respond in the same language.
                    prompt_text = f"Beoordeel of de beleidsoptie '{st.session_state.policy_desc}' een negatieve, positieve, of geen impact heeft op de milieu-indicator '{ind}'. De context zijn beleidsopties in Noordwest-Europa. Antwoord in het Nederlands. Herhaal de beleidsoptie niet. Geef alleen beknopte hints over positieve of negatieve impacts. Het antwoord moet een JSON object zijn met twee sleutels: 'vlag' (met een van de waarden 'Positief', 'Negatief', of 'Geen impact') en 'toelichting' (een korte toelichting van maximaal 3 zinnen)."

                    response = client.chat.completions.create(
                        model="gpt-5-mini",  # Using the model name as specified by the user
                        messages=[
                            {"role": "user", "content": prompt_text}
                        ]
                    )

                    ai_output = response.choices[0].message.content
                    ai_data = json.loads(ai_output)

                    st.session_state.ai_suggestions[ind] = {
                        "vlag": ai_data.get("vlag", "Geen impact"),
                        "toelichting": ai_data.get("toelichting", "Geen toelichting gegeven.")
                    }

                except Exception as e:
                    st.error(f"Fout voor indicator '{ind}': {e}")
                    st.session_state.ai_suggestions[ind] = {
                        "vlag": "Fout",
                        "toelichting": f"API-aanroep mislukt: {e}"
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
            ai_flag = ai_data.get('vlag', 'Niet Uitgevoerd')
            ai_explanation = ai_data.get('toelichting', '')

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
            "AI-oordeel": ai_data.get("vlag", "Niet Uitgevoerd"),
            "AI-toelichting": ai_data.get("toelichting", "")
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
