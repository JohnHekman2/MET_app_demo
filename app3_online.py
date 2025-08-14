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

st.set_page_config(page_title="Policy Environmental Impact Scoping (2-page)", layout="wide")


# --- Helpers -----------------------------------------------------------------
@st.cache_data
def load_indicators(uploaded_file):
    if uploaded_file is None:
        return [
            "Air quality",
            "Greenhouse gas emissions",
            "Water quality",
            "Biodiversity",
            "Land use",
            "Energy consumption",
            "Waste generation",
            "Noise pollution",
            "Soil contamination",
            "Resource depletion"
        ]
    else:
        try:
            df = pd.read_csv(uploaded_file, header=0)
            first_col = df.columns[0]
            items = df[first_col].dropna().astype(str).tolist()
            return items
        except Exception as e:
            st.sidebar.error(f"Could not read uploaded CSV: {e}")
            return []


def sanitize_key(s):
    s2 = s.strip().lower()
    s2 = re.sub(r"\s+", "_", s2)
    s2 = re.sub(r"[^a-z0-9_]", "", s2)
    return s2 or "indicator"


def sanitize_filename(s, maxlen=30):
    s2 = re.sub(r"\s+", "_", s.strip())
    s2 = re.sub(r"[^A-Za-z0-9_\-]", "", s2)
    return (s2[:maxlen] or "policy").rstrip("_")


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
st.sidebar.title("Policy Environmental Impact Scoping")

st.sidebar.markdown("---")
st.sidebar.header("Step 1 — Describe the policy option")
st.sidebar.markdown("Provide a clear explanation of what the policy option entails.")
st.sidebar.text_area(
    "Policy option explanation",
    value=st.session_state.policy_desc,
    placeholder="Describe the policy (what it does, who it affects, timeframe, boundaries, any important assumptions)...",
    height=260,
    key="policy_desc"
)

st.sidebar.markdown("---")
st.sidebar.header("Indicators source")
uploaded = st.sidebar.file_uploader("Upload a CSV containing indicators (first column used)", type=["csv"],
                                    key="indicator_uploader")

if uploaded:
    st.session_state.indicator_list = load_indicators(uploaded)
    st.session_state.selected_inds = st.session_state.indicator_list
    st.session_state.render_inds = st.session_state.indicator_list
st.sidebar.write(f"Loaded indicators: {len(st.session_state.indicator_list)}")

with st.sidebar.expander("Choose indicators to show"):
    st.multiselect(
        "Pick indicators (leave empty to show all)",
        options=st.session_state.indicator_list,
        default=st.session_state.selected_inds,
        key="selected_inds"
    )
    st.session_state.render_inds = st.session_state.selected_inds if st.session_state.selected_inds else st.session_state.indicator_list

impact_options = ["Negative", "No impact", "Positive"]

# Initialize per-indicator session_state defaults (do this before widget creation)
for ind in st.session_state.render_inds:
    sk = sanitize_key(ind)
    key_imp = f"imp__{sk}"
    key_exp = f"exp__{sk}"
    # Initialize keys if they don't already exist
    if key_imp not in st.session_state:
        st.session_state[key_imp] = "No impact"
    if key_exp not in st.session_state:
        st.session_state[key_exp] = ""

# --- Sidebar AI Integration --------------------------------------------------
st.sidebar.markdown("---")
st.sidebar.header("AI Integration (Gemini)")
st.sidebar.markdown("Get a quick assessment from Gemini for the impacts.")

# Check if the API key is available
api_key_loaded = bool(st.session_state.gemini_api_key)
if api_key_loaded:
    st.sidebar.success("Gemini API key loaded from secrets!")
else:
    st.sidebar.warning("Gemini API key not found. Please add it to your Streamlit secrets.")

if st.sidebar.button("Get AI Suggestions", disabled=not api_key_loaded):
    if not api_key_loaded:
        st.sidebar.error("Cannot get AI suggestions without a valid API key. Please configure your secrets.")
    else:
        st.session_state.ai_suggestions = {}
        with st.spinner("Generating AI suggestions..."):
            for ind in st.session_state.render_inds:
                try:
                    # Updated prompt to Gemini with new constraints
                    prompt = f"add a flag and a short (maximum 3 sentence) explanation if the policy option '{st.session_state.policy_desc}' is impacting the environmental indicator '{ind}' negatively, positively, or has no impact. The context are policy options in Northwest Europe. Do not repeat the policy option in your explanation, only provide succinct hints on positive or negative impacts. The flag must be one of these three options: 'Positive', 'Negative', or 'No impact'."

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
                                        "enum": ["Positive", "Negative", "No impact"]
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
                        "flag": ai_data.get("flag", "No impact"),
                        "explanation": ai_data.get("explanation", "No explanation provided.")
                    }

                except Exception as e:
                    st.error(f"Error for indicator '{ind}': {e}")
                    st.session_state.ai_suggestions[ind] = {
                        "flag": "Error",
                        "explanation": f"API call failed: {e}"
                    }
        st.session_state.show_ai_columns = True
        st.success("AI suggestions generated!")
        st.rerun()

# --- Main Page: Assess impacts ----------------------------------------------
st.title("Step 2 — Assess environmental impacts")
if not st.session_state.policy_desc.strip():
    st.info("Please enter a policy description in the sidebar to begin the assessment.")
else:
    st.subheader("Policy explanation")
    st.write(st.session_state.policy_desc)

    st.markdown("---")
    st.markdown(
        "For each indicator below, select the impact (Negative / No impact / Positive) and provide an explanation for your judgement.")

    # Determine the columns to display based on whether AI suggestions have been run
    if st.session_state.show_ai_columns:
        # Show all 5 columns
        cols_config = [3, 2, 2, 2, 3]
        col_names = ["Indicator", "Impact", "Your Explanation", "AI Judgement", "AI Explanation"]
    else:
        # Show only the original 3 columns
        cols_config = [3, 2, 5]
        col_names = ["Indicator", "Impact", "Explanation"]

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
                init_idx = impact_options.index(st.session_state.get(key_imp, "No impact"))
            except ValueError:
                init_idx = 1
            cols[1].selectbox(f"Impact {i + 1}", options=impact_options, index=init_idx, key=key_imp,
                              label_visibility="collapsed")
        with cols[2]:
            cols[2].text_area(f"Explanation {i + 1}", value=st.session_state.get(key_exp, ""), key=key_exp, height=90,
                              label_visibility="collapsed")

        # Conditionally render AI columns
        if st.session_state.show_ai_columns:
            ai_data = st.session_state.ai_suggestions.get(ind, {})
            ai_flag = ai_data.get('flag', 'Not Run')
            ai_explanation = ai_data.get('explanation', '')

            with cols[3]:
                st.write(f"**{ai_flag}**")
            with cols[4]:
                cols[4].text_area(f"AI Suggestion {i + 1}", value=ai_explanation, disabled=True, height=90,
                                  label_visibility="collapsed")

    rows = []
    for ind in st.session_state.render_inds:
        sk = sanitize_key(ind)
        key_imp = f"imp__{sk}"
        key_exp = f"exp__{sk}"
        ai_data = st.session_state.ai_suggestions.get(ind, {})
        rows.append({
            "Indicator": ind,
            "Impact": st.session_state.get(key_imp, "No impact"),
            "Explanation": st.session_state.get(key_exp, ""),
            "AI Judgement": ai_data.get("flag", "Not Run"),
            "AI Explanation": ai_data.get("explanation", "")
        })
    edited = pd.DataFrame(rows)

    missing_explanations = edited[
        (edited["Impact"] == "Negative") & (edited["Explanation"].astype(str).str.strip() == "")]
    if not missing_explanations.empty:
        st.warning(
            f"{len(missing_explanations)} indicator(s) marked Negative without an explanation. Please add explanations for these rows.")
        with st.expander("Indicators missing explanation (Negative impact)"):
            st.write(missing_explanations["Indicator"].tolist())

    counts = edited["Impact"].value_counts().reindex(["Positive", "No impact", "Negative"]).fillna(0).astype(int)
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("Impact summary")
        st.write(counts.to_frame("count"))
        chart_df = pd.DataFrame({"Impact": counts.index, "count": counts.values})
        chart = alt.Chart(chart_df).mark_bar().encode(
            x=alt.X("Impact:N", sort=["Positive", "No impact", "Negative"]),
            y="count:Q",
            color=alt.Color("Impact:N", scale=alt.Scale(domain=["Positive", "No impact", "Negative"],
                                                        range=["#2ca02c", "#1f77b4", "#d62728"]))
        ).properties(height=240)
        st.altair_chart(chart, use_container_width=True)
    with col2:
        st.subheader("Policy")
        snippet = (st.session_state.policy_desc[:200] + "...") if len(
            st.session_state.policy_desc) > 200 else st.session_state.policy_desc
        st.write(snippet if snippet else "No description entered")
        st.write("Indicators shown:", len(st.session_state.render_inds))

    st.write("---")
    save_col, hist_col = st.columns([1, 2])
    with save_col:
        if st.button("Save snapshot to session history"):
            rec = {
                "policy": st.session_state.policy_desc,
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "table": edited.to_dict(orient="records")
            }
            st.session_state.history.append(rec)
            st.success("Saved to session history (not persistent).")

        csv_buf = StringIO()
        edited.to_csv(csv_buf, index=False)
        short = sanitize_filename(st.session_state.policy_desc[:30])
        fname = f"{short}_{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}_impacts.csv"
        st.download_button("Download CSV of table", csv_buf.getvalue(), file_name=fname, mime="text/csv")

    with hist_col:
        st.subheader("Session history (recent saves)")
        history = st.session_state.get("history", [])
        if len(history) == 0:
            st.write("No saved snapshots yet. Click 'Save snapshot to session history' to keep the current table.")
        else:
            for i, rec in enumerate(reversed(history[-10:])):
                idx = len(history) - 1 - i
                with st.expander(f"Saved {rec['timestamp']}", expanded=False):
                    hdf = pd.DataFrame(rec["table"])
                    st.write(hdf)
                    buf = StringIO()
                    hdf.to_csv(buf, index=False)
                    st.download_button(f"Download CSV (entry #{idx})", buf.getvalue(), file_name=f"snapshot_{idx}.csv",
                                       mime="text/csv")

    st.sidebar.write("---")
    st.sidebar.subheader("Current Session State (Debug)")
    st.sidebar.write(st.session_state)
