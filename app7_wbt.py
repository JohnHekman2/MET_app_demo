import streamlit as st
import pandas as pd
import altair as alt
from io import StringIO
from datetime import datetime
import re
import textwrap
from openai import OpenAI
import json

st.set_page_config(page_title="Modulaire Beleidsoptie Milieu-impact Scoping", layout="wide")

# --- 1. Data Model Class -----------------------------------------------------

class PolicyData:
    """
    Klasse om alle beleidsinformatie, diagnostische stappen en resultaten 
    op een gestructureerde manier op te slaan, onafhankelijk van st.session_state.
    
    De app is vereenvoudigd naar 3 stappen: 1. Beschrijving, 2. Doelgroep, 3. Beoordeling.
    """
    def __init__(self):
        # Stap 1: Beschrijving
        self.policy_desc = ""
        
        # Stap 2: Doelgroep
        self.target_group_desc = ""
        self.target_group_ai = ""
        
        # Nieuwe Stap 3 (oude Stap 5): Beoordeling
        self.indicator_list = [] # Alle beschikbare indicatoren
        self.selected_inds = []  # Indicatoren geselecteerd voor weergave
        self.assessment_data = {} # Dict: {indicator_key: {"impact": "...", "explanation": "..."}}
        
        # NIEUW: AI Beoordelingsdata voor Stap 3
        self.ai_assessment_data = {} # Dict: {indicator_key: "AI's impact text"}
        self.ai_assessment_ran = False # Flag om te controleren of AI-beoordeling is uitgevoerd

    def initialize_indicators(self, default_list):
        """Initialiseert de indicatorenlijsten als ze leeg zijn."""
        if not self.indicator_list:
            self.indicator_list = default_list
            self.selected_inds = default_list

    def update_assessment(self, indicator, impact, explanation):
        """Update de impact en toelichting voor een specifieke indicator."""
        key = self._sanitize_key(indicator)
        if key not in self.assessment_data:
            self.assessment_data[key] = {}
        self.assessment_data[key]["impact"] = impact
        self.assessment_data[key]["explanation"] = explanation

    def get_assessment_value(self, indicator, field):
        """Haalt een specifieke waarde op (impact of explanation) voor een indicator."""
        key = self._sanitize_key(indicator)
        return self.assessment_data.get(key, {}).get(field, "Geen impact" if field == "impact" else "")

    def get_ai_assessment_value(self, indicator):
        """Haalt de AI-beoordelingstekst op voor een indicator."""
        key = self._sanitize_key(indicator)
        return self.ai_assessment_data.get(key, "Niet beoordeeld door AI")

    def get_full_diagnostic_context(self):
        """Genereert een samenvattende tekst van Stap 1 en 2, voor context op Stap 3."""
        return (
            f"**Beleidsoptie (Stap 1):** {self.policy_desc}\n\n"
            f"**Doelgroep/Gedrag (Stap 2):** {self.target_group_desc}"
        )

    def _sanitize_key(self, s):
        """Hulpfunctie voor het genereren van veilige sleutels."""
        s2 = s.strip().lower()
        s2 = re.sub(r"\s+", "_", s2)
        s2 = re.sub(r"[^a-z0-9_]", "", s2)
        return s2 or "indicator"

# --- 2. Hulpfuncties en Initialisatie ---------------------------------------

# Constanten voor navigatie. Nu slechts 2 stappen in de flow.
PAGE_INPUT = "input" # Stap 1 & 2
PAGE_ASSESSMENT = "assessment" # Stap 3

@st.cache_data
def load_default_indicators():
    """Laadt standaard indicatoren."""
    return [
        "Luchtkwaliteit", "Broeikasgasuitstoot", "Waterkwaliteit", 
        "Biodiversiteit", "Landgebruik", "Energieverbruik", 
        "Afvalgeneratie", "Geluidsoverlast", "Bodemverontreiniging",
        "Uitputting natuurlijke hulpbronnen"
    ]

@st.cache_data
def load_indicators_from_file(uploaded_file):
    """Laadt indicatoren uit een geüpload CSV-bestand."""
    if uploaded_file is None:
        return []
    try:
        df = pd.read_csv(uploaded_file, header=0)
        first_col = df.columns[0]
        items = df[first_col].dropna().astype(str).tolist()
        return items
    except Exception as e:
        st.sidebar.error(f"Kon geüploade CSV niet lezen: {e}")
        return []

def sanitize_filename(s, maxlen=30):
    """
    Genereert een veilige bestandsnaam uit een string.
    """
    s2 = re.sub(r"\s+", "_", s.strip())
    s2 = re.sub(r"[^A-Za-z0-9_\-]", "", s2)
    return (s2[:maxlen] or "beleid").rstrip("_")

def set_page(page):
    """Stelt de huidige pagina in de sessiestatus in."""
    st.session_state.page = page

@st.cache_resource
def get_openai_client():
    """
    Maakt en cachet een OpenAI client-instantie.
    Haalt de credentials op uit st.session_state, die worden geinitialiseerd
    via st.secrets.
    """
    api_key = st.session_state.openai_api_key
    base_url = st.session_state.openai_base_url
    # Zorg ervoor dat de base_url eindigt op /v1 als dat nodig is voor de API
    if base_url and not base_url.endswith("/v1"):
         # Dit is een heuristiek, de uiteindelijke implementatie hangt af van de AI-proxy
        base_url = base_url.rstrip('/') + '/v1' 
        
    if api_key and base_url:
        return OpenAI(api_key=api_key, base_url=base_url)
    return None

def run_ai_step_analysis(prompt_key, output_attr, title):
    """
    Voert een AI-analyse uit voor de Doelgroep/Gedrag stap (Stap 2).
    """
    client = get_openai_client()
    data = st.session_state.data

    if not client:
        st.session_state.data.__dict__[output_attr] = "Kan geen AI-suggesties ophalen zonder geldige API-sleutel en basis-URL."
        return

    policy_context = data.policy_desc
    
    # Definieer de Nederlandse context die aan de prompt wordt toegevoegd.
    nl_context = "De analyse betreft een beleidsvoorstel van de Nederlandse regering. De invloedssfeer is beperkt tot het Nederlands grondgebied en haar inwoners. Houd hier rekening mee bij het inschatten van de effecten. Specifiek, ga na wat de context is voor de vraagkant van de economie en de aanbodkant. Actoren aan de vraagkant (nederlandse consumenten) kunnen makkelijker worden beinvloedt dan vaak internationale actoren aan de aanbodkant, omdat Nederland een klein land is op een grote internationale markt. Deze context hoeft niet terug te komen in het antwoord specifiek, maar je moet er wel rekening mee houden."

    if prompt_key == "target_group_prompt":
        prompt = f"""{nl_context}\n\nDe beleidsoptie is: '{policy_context}'.\n\nBepaal de meest waarschijnlijke **doelgroepen** en het **gedrag/de activiteit** dat direct door dit beleid wordt beïnvloed. Geef een beknopte, beredeneerde analyse van maximaal 5 zinnen. Antwoord in het Nederlands."""
    else:
        st.session_state.data.__dict__[output_attr] = "Interne fout: Onbekende prompt sleutel."
        return

    with st.spinner(f"AI-suggesties aan het genereren voor {title}..."):
        try:
            # Gebruik het gewenste model
            response = client.chat.completions.create(
                model="gpt-5-mini", 
                messages=[{"role": "user", "content": prompt}],
            )
            # Update het juiste attribuut in de PolicyData instantie
            st.session_state.data.__dict__[output_attr] = response.choices[0].message.content
            st.success("AI-suggesties gegenereerd!")
            st.rerun()
        except Exception as e:
            st.error(f"Fout bij AI-analyse voor {title}: {e}")
            st.session_state.data.__dict__[output_attr] = f"API-aanroep mislukt: {e}"

def run_ai_impact_assessment(data: PolicyData):
    """
    NIEUW: Voert een AI-analyse uit voor elke geselecteerde indicator (Stap 3), 
    waarbij de 3-staps logica wordt afgedwongen.
    """
    client = get_openai_client()
    
    if not client:
        st.error("Kan geen AI-suggesties ophalen zonder geldige API-sleutel en basis-URL.")
        return

    # Gebruik geselecteerde indicatoren, of de volledige lijst als er niets is geselecteerd
    indicators_to_assess = data.selected_inds if data.selected_inds else data.indicator_list
    
    # --- NIEUWE PROMPT LOGICA ---
    nl_context = """Je bent een gespecialiseerde milieu-analist. Je taak is om de verwachte milieu-impact van het beleid op de specifieke indicator te beoordelen. 
Je analyse moet strikt de volgende driedelige causale keten volgen, waarbij elk punt kort en direct wordt toegelicht (maximaal 1-2 zinnen per punt) en gemarkeerd met de vetgedrukte stapnaam:
1. **De Verandering**: Wat is de primaire fysieke of gedragsverandering die het beleid veroorzaakt, gebaseerd op de context?
2. **De Gevolgen**: Wat zijn de directe milieugevolgen (toename/afname van emissies, grondstofverbruik, etc.) die voortvloeien uit De Verandering?
3. **De Impact**: Wat is het uiteindelijke effect op de staat van de leefomgeving (de indicator zelf) als gevolg van De Gevolgen?
Gebruik de verstrekte beleidscontext. Antwoord in het Nederlands."""
    
    # Zorg ervoor dat de diagnostische context goed geformatteerd is voor de AI
    full_context = f"Beleidsoptie: {data.policy_desc}\nDoelgroep en Gedrag: {data.target_group_desc}"
    
    data.ai_assessment_data = {} # Reset vorige resultaten
    
    with st.spinner(f"Beoordeling voor {len(indicators_to_assess)} indicatoren genereren... Dit kan even duren."):
        all_results = {}
        success_count = 0
        
        for i, indicator in enumerate(indicators_to_assess):
            st.info(f"Beoordeelt: {indicator} ({i+1}/{len(indicators_to_assess)})")
            
            prompt = f"""{nl_context}

Context:
{full_context}

Indicator die beoordeeld moet worden:
'{indicator}'

Voer de 3-staps analyse uit voor deze indicator. Begin elke stap met het vetgedrukte label."""
            
            try:
                response = client.chat.completions.create(
                    model="gpt-5-mini", 
                    messages=[{"role": "user", "content": prompt}],
                )
                result_text = response.choices[0].message.content
                key = data._sanitize_key(indicator)
                all_results[key] = result_text
                success_count += 1

            except Exception as e:
                # Log de fout, maar ga door met de volgende indicator
                st.warning(f"Fout bij AI-analyse voor '{indicator}': {e}. Sla deze over.")
                key = data._sanitize_key(indicator)
                all_results[key] = f"Fout bij ophalen AI-analyse: {e}"

        data.ai_assessment_data = all_results
        data.ai_assessment_ran = True
        st.success(f"AI-beoordeling voltooid: {success_count} van {len(indicators_to_assess)} indicatoren beoordeeld.")
        st.rerun()


# --- 3. Initialisatie van Session State --------------------------------------

if "data" not in st.session_state:
    st.session_state.data = PolicyData()
    st.session_state.data.initialize_indicators(load_default_indicators())

if "page" not in st.session_state:
    st.session_state.page = PAGE_INPUT

# Initialiseer AI Config (haalt direct de waarden op uit secrets)
# Geen zichtbare widgets meer voor de API-sleutel en basis-URL
if "openai_api_key" not in st.session_state:
    # Haal de sleutel op uit st.secrets (of een lege string als deze niet bestaat)
    st.session_state.openai_api_key = st.secrets.get("OPENAI_API_KEY", "") 
if "openai_base_url" not in st.session_state:
    # Haal de basis-URL op uit st.secrets (of een lege string als deze niet bestaat)
    st.session_state.openai_base_url = st.secrets.get("OPENAI_BASE_URL", "")
    

# --- 4. Pagina Render Functies -----------------------------------------------

def render_input_page(data: PolicyData):
    """
    Rendert de gecombineerde UI voor Stap 1 (Beleid) en Stap 2 (Doelgroep/Gedrag).
    """
    
    # --- STAP 1: Beleidsoptie omschrijven ---
    st.title("Stap 1: Beleidsoptie Omschrijven")
    st.markdown("Geef een duidelijke toelichting op wat de beleidsoptie inhoudt.")
    
    new_policy_desc = st.text_area(
        "Beleidsoptie toelichting",
        value=data.policy_desc,
        placeholder="Beschrijf het beleid (wat het doet, wie het beïnvloedt, tijdsbestek, grenzen, belangrijke aannames)...",
        height=200,
        key="policy_desc_input",
        label_visibility="collapsed"
    )
    data.policy_desc = new_policy_desc
    
    st.markdown("---")

    # --- STAP 2: Doelgroep en Gedrag ---
    st.title("Stap 2: Doelgroep en Gedrag")
    st.markdown("Identificeer de doelgroep van het beleid en de specifieke activiteiten of gedragingen die door het beleid zullen veranderen.")

    # Gebruikersanalyse Doelgroep
    st.subheader("Uw analyse: Doelgroep en activiteit")
    new_target_desc = st.text_area(
        "Doelgroep en Gedrag",
        value=data.target_group_desc,
        placeholder="Wie is de doelgroep? Welke specifieke acties/gedragingen worden aangepast?",
        height=150,
        key="target_group_desc_widget"
    )
    data.target_group_desc = new_target_desc

    # AI Suggestie Doelgroep
    st.subheader("AI-Suggestie (Ter overweging)")
    st.text_area(
        "AI-Oordeel Doelgroep",
        value=data.target_group_ai, # Lees direct uit PolicyData
        height=150,
        disabled=False, # Selecteren/kopiëren is toegestaan
        label_visibility="collapsed"
    )
    
    # AI Knop logica: Alleen actief als Stap 1 is ingevuld
    ai_button_disabled = not data.policy_desc.strip()
    if st.button("Ontvang AI-suggestie voor Doelgroep", key="run_ai_target", disabled=ai_button_disabled, help="Vul eerst de beleidsoptie in (Stap 1) om de AI-suggestie te activeren."):
        run_ai_step_analysis("target_group_prompt", "target_group_ai", "Doelgroep en Gedrag")

    st.markdown("---")
    
    # Navigatie naar Nieuwe Stap 3 (Beoordeling): Beide velden moeten gevuld zijn
    if data.policy_desc.strip() and data.target_group_desc.strip():
        st.button("Naar Stap 3: Finale Impact Beoordeling →", on_click=set_page, args=[PAGE_ASSESSMENT], type="primary")
    else:
        st.button("Naar Stap 3: Finale Impact Beoordeling →", disabled=True, help="Vul de beleidsbeschrijving én uw doelgroepanalyse in om verder te gaan.")


def generate_markdown_report(data: PolicyData, counts: dict, impact_df: pd.DataFrame):
    """
    Genereert een volledig en gestructureerd Markdown-rapport van de analyse.
    """
    
    timestamp = datetime.now().strftime("%d-%m-%Y om %H:%M:%S")

    report_content = f"""# Milieu-impact Analyse Rapport (Diagnostisch)
Datum van de Analyse: {timestamp}
Beleidsoptie: {data.policy_desc[:50] + '...' if len(data.policy_desc) > 50 else data.policy_desc}

## 1. Beleidsbeschrijving
---
{data.policy_desc}
---

## 2. Diagnostische Analyse (Stap 1 en 2)

### Beleidsoptie (Stap 1)
{data.policy_desc}

### Stap 2: Doelgroep en Gedrag
| Bron | Inhoud |
| :--- | :--- |
| **Uw Analyse** | {data.target_group_desc} |
| **AI-Suggestie** | {data.target_group_ai if data.target_group_ai else '*Geen AI-suggestie uitgevoerd.*'} |

## 3. Impactoverzicht
Dit overzicht toont het aantal indicatoren dat als Positief, Neutraal of Negatief is beoordeeld.

| Impact | Aantal Indicatoren |
| :--- | :--- |
| Positief | {counts.get('Positief', 0)} |
| Geen impact | {counts.get('Geen impact', 0)} |
| Negatief | {counts.get('Negatief', 0)} |

## 4. Gedetailleerde Indicatorbeoordeling

"""
    # Gebruik de DataFrame met de gecombineerde toelichting
    for row in impact_df.to_dict(orient="records"):
        ind = row["Indicator"]
        impact = row["Impact"]
        toelichting = row["Toelichting (Rapport)"]
        ai_insight = row.get("AI Inzicht", None) # Haal AI Inzicht op
        
        report_content += f"""
### Indicator: {ind}
- **Beoordeelde Impact (Uw):** **{impact}**
"""
        # Voeg AI inzicht toe als het beschikbaar is en niet de standaard 'Niet beoordeeld door AI' tekst is.
        if ai_insight and ai_insight != "Niet beoordeeld door AI":
            report_content += f"- **AI Inzicht (Autonoom, 3-staps logica):**\n"
            # Inspringen van de 3 stappen in het rapport voor betere leesbaarheid
            indented_ai = textwrap.indent(ai_insight, '  > ')
            report_content += f"{indented_ai}\n"
        
        report_content += f"- **Toelichting (Handmatig + Diagnostiek):** {textwrap.indent(toelichting, '  - ')}\n"
        report_content += "\n***\n" 

    return report_content

def render_assessment_page(data: PolicyData):
    """
    Rendert de UI voor de Nieuwe Stap 3: Finale Impact Beoordeling.
    """
    
    st.title("Stap 3: Finale Impact Beoordeling")
    st.button("← Terug naar Stap 1 & 2", on_click=set_page, args=[PAGE_INPUT]) # Aangepaste terugknop

    if not data.policy_desc.strip() or not data.target_group_desc.strip():
        st.error("Ga terug om de Beleidsoptie en Doelgroep (Stap 1 & 2) te voltooien.")
        return

    st.subheader("Diagnostische Context (Stap 1 & 2)")
    
    # Toon de context uit Stap 1 en 2 als input voor de beoordeling
    st.text_area("Diagnostische Context", value=data.get_full_diagnostic_context(), height=200, disabled=False)
    
    st.markdown("---")

    # --- NIEUW: AI Inzicht sectie ----------------------------------------------
    st.subheader("AI Milieu-impact Inzicht (3-Staps Logica)")
    
    ai_col1, ai_col2 = st.columns([1, 4])
    ai_button_disabled = not (data.policy_desc.strip() and data.target_group_desc.strip())
    
    with ai_col1:
        if st.button("Genereer AI Indicator Inzichten", type="secondary", disabled=ai_button_disabled, help="Laat de AI de impact per indicator inschatten op basis van de ingevoerde beleidscontext. Dit kan even duren."):
            run_ai_impact_assessment(data)
    
    with ai_col2:
        if data.ai_assessment_ran:
             st.success("AI Inzichten (Verandering → Gevolgen → Impact) zijn beschikbaar in de tabel hieronder (kolom 'AI Inzicht').")
        elif not ai_button_disabled:
            st.info("Druk op de knop om de AI-analyse te starten.")
        else:
            st.warning("Vul Stap 1 en 2 in om de AI-inzichten te genereren.")

    st.markdown("---") # Scheiding tussen AI-sectie en tabel
    # --------------------------------------------------------------------------

    # --- Sidebar: Indicator Source & Selection ----------------------------------
    st.sidebar.header("Indicatorbron")
    uploaded = st.sidebar.file_uploader("Upload een CSV met indicatoren (eerste kolom wordt gebruikt)", type=["csv"], key="indicator_uploader")
    
    if uploaded:
        new_list = load_indicators_from_file(uploaded)
        # Reset AI assessment if the indicator list changes
        data.ai_assessment_ran = False 
        data.ai_assessment_data = {}
        # Update de PolicyData instantie met de nieuwe lijst
        data.indicator_list = new_list
        data.selected_inds = new_list
        # Reset de selectie widget via een sleutel
        st.session_state.selected_inds_widget = new_list
        st.session_state.render_inds = new_list
        st.rerun() # Rerun to update the selection box instantly

    st.sidebar.write(f"Geladen indicatoren: {len(data.indicator_list)}")
    
    # Weergavelijst is ofwel de geselecteerde items of de volledige lijst
    render_inds = data.selected_inds if data.selected_inds else data.indicator_list

    with st.sidebar.expander("Kies indicatoren om te tonen"):
        # Gebruik een widget key en update de PolicyData wanneer de widget verandert
        selected_inds_widget = st.multiselect(
            "Kies indicatoren (laat leeg om alles te tonen)",
            options=data.indicator_list,
            default=data.selected_inds,
            key="selected_inds_widget"
        )
        data.selected_inds = selected_inds_widget
        render_inds = data.selected_inds if data.selected_inds else data.indicator_list

    impact_options = ["Negatief", "Geen impact", "Positief"]

    # --- Main Page: Assess impacts Table ----------------------------------------
    st.markdown("Beoordeel nu de impact voor elke indicator. Gebruik de diagnostische context hierboven als basis voor uw toelichting.")

    # Dynamische kolomconfiguratie
    if data.ai_assessment_ran:
        # Pas de verhoudingen aan om ruimte te maken voor de gedetailleerde AI-analyse
        cols_config = [2, 4, 2, 4] # Indicator, AI Inzicht, Uw Impact, Uw Toelichting
        col_names = ["Indicator", "AI Inzicht", "Uw Impact", "Uw Toelichting"]
    else:
        cols_config = [3, 2, 5] # Indicator, Impact, Toelichting (Oude configuratie)
        col_names = ["Indicator", "Impact", "Toelichting"]

    # Display headers
    cols = st.columns(cols_config)
    for i, name in enumerate(col_names):
        with cols[i]:
            st.write(f"**{name}**")

    # Display rows and update PolicyData on change
    for i, ind in enumerate(render_inds):
        sk = data._sanitize_key(ind)
        
        # Haal de huidige waarden op uit de PolicyData instantie
        current_impact = data.get_assessment_value(ind, "impact")
        current_explanation = data.get_assessment_value(ind, "explanation")
        
        # Haal AI-inzicht op
        ai_insight = data.get_ai_assessment_value(ind)

        cols = st.columns(cols_config)
        
        col_idx = 0
        with cols[col_idx]:
            st.write(f"**{ind}**")
        col_idx += 1
        
        # NIEUW: AI Inzicht kolom
        if data.ai_assessment_ran:
            with cols[col_idx]:
                # Gebruik st.caption/markdown en pas de hoogte aan voor de 3 stappen
                st.markdown(ai_insight, help="Inzicht via de Verandering → Gevolgen → Impact logica")
            col_idx += 1
            
        with cols[col_idx]:
            # Gebruik een unieke sleutel per rij en type
            impact_key = f"impact_{sk}" 
            new_impact = cols[col_idx].selectbox(f"Impact {i+1}", options=impact_options, index=impact_options.index(current_impact), key=impact_key, label_visibility="collapsed")
            
        col_idx += 1
        with cols[col_idx]:
            explanation_key = f"explanation_{sk}"
            new_explanation = cols[col_idx].text_area(f"Toelichting {i+1}", value=current_explanation, key=explanation_key, height=130, label_visibility="collapsed") # Hoogte aangepast

        # Update PolicyData zodra de widget is gerenderd/gewijzigd
        data.update_assessment(ind, new_impact, new_explanation)

    # Re-collect data into DataFrame for reporting and visualization
    rows = []
    # De volledige context van Stap 1 & 2 wordt gebruikt
    full_diagnostic_text = data.get_full_diagnostic_context() 
    
    for ind in render_inds:
        impact = data.get_assessment_value(ind, "impact")
        explanation = data.get_assessment_value(ind, "explanation")
        ai_insight = data.get_ai_assessment_value(ind)
        
        # Combineer de toelichting voor het rapport
        combined_explanation = f"{full_diagnostic_text}\n\n**Uw Toelichting (per indicator)**: {explanation}"
        
        rows.append({
            "Indicator": ind,
            "Impact": impact,
            "AI Inzicht": ai_insight,             # NIEUW: AI Inzicht voor opslag/rapport
            "Toelichting (Rapport)": combined_explanation, # Dit gaat in het Markdown rapport
            "Toelichting (Tabel)": explanation             # Dit gaat in de CSV/DataFrame
        })
        
    edited_df = pd.DataFrame(rows)

    # --- Summary and Chart ---
    counts = edited_df["Impact"].value_counts().reindex(["Positief", "Geen impact", "Negatief"]).fillna(0).astype(int)
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Impactoverzicht")
        st.write(counts.to_frame("Aantal"))
        
        if not counts.empty and counts.sum() > 0:
            chart_df = pd.DataFrame({"Impact": counts.index, "Aantal": counts.values})
            chart = alt.Chart(chart_df).mark_bar().encode(
                x=alt.X("Impact:N", sort=["Positief", "Geen impact", "Negatief"]),
                y="Aantal:Q",
                color=alt.Color("Impact:N", scale=alt.Scale(domain=["Positief", "Geen impact", "Negatief"],
                                                           range=["#2ca02c", "#1f77b4", "#d62728"]))
            ).properties(height=240)
            st.altair_chart(chart, use_container_width=True)
        else:
            st.info("Geen gegevens beschikbaar om een grafiek te tonen.")
            
    with col2:
        st.subheader("Beleid")
        snippet = (data.policy_desc[:200] + "...") if len(data.policy_desc) > 200 else data.policy_desc
        st.write(snippet if snippet else "Geen beschrijving ingevoerd")
        st.write("Indicatoren weergegeven:", len(render_inds))

    st.write("---")
    
    # --- Reporting/Saving Section ---
    
    # Bepaal welke kolommen te behouden voor de CSV/Historie
    cols_for_save = ["Indicator", "Impact", "Toelichting (Tabel)"]
    if data.ai_assessment_ran:
        cols_for_save.insert(2, "AI Inzicht")
        
    edited_for_save = edited_df[cols_for_save].rename(columns={"Toelichting (Tabel)": "Toelichting"})
    
    save_col, download_col, hist_col = st.columns([1, 1, 2])
    
    with save_col:
        # Dit is de enige plek waar we data buiten de PolicyData klasse opslaan (voor geschiedenis)
        if st.button("Momentopname opslaan in sessiegeschiedenis"):
            rec = {
                "policy": data.policy_desc,
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "table": edited_for_save.to_dict(orient="records")
            }
            # Initialiseer de geschiedenislijst als deze nog niet bestaat
            if "history" not in st.session_state:
                st.session_state.history = []
            st.session_state.history.append(rec)
            st.success("Opgeslagen in sessiegeschiedenis (niet persistent).")
            
    with download_col:
        # Download CSV
        csv_buf = StringIO()
        edited_for_save.to_csv(csv_buf, index=False)
        short_csv = sanitize_filename(data.policy_desc[:30])
        fname_csv = f"{short_csv}_{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}_impacts.csv"
        st.download_button("CSV van tabel downloaden", csv_buf.getvalue(), file_name=fname_csv, mime="text/csv")
        
        # Download MARKDOWN Report
        if data.policy_desc.strip():
            markdown_report = generate_markdown_report(
                data, 
                counts.to_dict(), 
                edited_df # Pass the DataFrame with the full combined explanation and AI Inzicht
            )
            short_md = sanitize_filename(data.policy_desc[:30])
            fname_md = f"{short_md}_{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}_rapport.md"
            st.download_button("Markdown Rapport downloaden", markdown_report, file_name=fname_md, mime="text/markdown")
        else:
            st.button("Markdown Rapport downloaden", disabled=True, help="Voer eerst een beleidsbeschrijving in.")

    with hist_col:
        st.subheader("Sessiegeschiedenis (recente opslag)")
        history = st.session_state.get("history", [])
        if len(history) == 0:
            st.write("Nog geen momentopnamen opgeslagen.")
        else:
            for i, rec in enumerate(reversed(history[-5:])): 
                idx = len(history) - 1 - i
                with st.expander(f"Opgeslagen {rec['timestamp']}", expanded=False):
                    # We gebruiken de opgeslagen dictionary, niet de live dataframe
                    hdf = pd.DataFrame(rec["table"])
                    st.dataframe(hdf, use_container_width=True)
                    buf = StringIO()
                    hdf.to_csv(buf, index=False)
                    st.download_button(f"CSV downloaden (item #{idx})", buf.getvalue(), file_name=f"snapshot_{idx}.csv", mime="text/csv")


# --- 5. Main Application Flow --------------------------------------------------

data = st.session_state.data

st.sidebar.title("Beleidsoptie Milieu-impact Scoping")
st.sidebar.markdown("Navigatie tussen de stappen.")
# Nieuwe navigatie
st.sidebar.button("Stap 1 & 2: Beleid & Doelgroep", on_click=set_page, args=[PAGE_INPUT], use_container_width=True) 
st.sidebar.button("Stap 3: Beoordeling", on_click=set_page, args=[PAGE_ASSESSMENT], type="primary", use_container_width=True)

# Render de geselecteerde pagina en geef de PolicyData instantie door
if st.session_state.page == PAGE_INPUT:
    render_input_page(data)
elif st.session_state.page == PAGE_ASSESSMENT:
    render_assessment_page(data)
