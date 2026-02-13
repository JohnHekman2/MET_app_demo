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
    """
    def __init__(self):
        # Stap 1: Beschrijving
        self.policy_desc = ""
        
        # Stap 2: Doelgroep
        self.target_group_desc = ""
        self.target_group_ai = ""
        
        # Stap 3: Emissies
        self.emissions_desc = ""
        self.emissions_ai = ""
        
        # Stap 4: Gevolgen
        self.consequences_desc = ""
        self.consequences_ai = ""

        # Stap 5: Beoordeling
        # Lijsten om indicatoren en selecties op te slaan
        self.indicator_list = [] # Alle beschikbare indicatoren
        self.selected_inds = []  # Indicatoren geselecteerd voor weergave
        self.assessment_data = {} # Dict: {indicator_key: {"impact": "...", "explanation": "..."}}

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

    def get_full_diagnostic_context(self):
        """Genereert een samenvattende tekst van Stap 2 t/m 4."""
        return (
            f"**Doelgroep/Gedrag (Stap 2):** {self.target_group_desc}\n\n"
            f"**Emissies/Veranderingen (Stap 3):** {self.emissions_desc}\n\n"
            f"**Fysieke Gevolgen/Mens (Stap 4):** {self.consequences_desc}"
        )

    def _sanitize_key(self, s):
        """Hulpfunctie voor het genereren van veilige sleutels."""
        s2 = s.strip().lower()
        s2 = re.sub(r"\s+", "_", s2)
        s2 = re.sub(r"[^a-z0-9_]", "", s2)
        return s2 or "indicator"

# --- 2. Hulpfuncties en Initialisatie ---------------------------------------

# Constanten voor navigatie
PAGE_INPUT = "input"
PAGE_TARGET_GROUP = "target_group"
PAGE_EMISSIONS = "emissions"
PAGE_CONSEQUENCES = "consequences"
PAGE_ASSESSMENT = "assessment"

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

    Args:
        s (str): De input string (bv. de beleidsomschrijving).
        maxlen (int): De maximale lengte van de bestandsnaam.
    Returns:
        str: Een opgeschoonde string die geschikt is voor gebruik als bestandsnaam.
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

    De functie gebruikt de API-sleutel en basis-URL uit de Streamlit session_state.
    De `@st.cache_resource` decorator zorgt ervoor dat de client slechts eenmaal
    wordt aangemaakt en hergebruikt wordt bij volgende runs.
    Returns:
        OpenAI | None: Een geïnitialiseerde OpenAI client of None als de credentials ontbreken.
    """
    api_key = st.session_state.openai_api_key
    base_url = st.session_state.openai_base_url
    if api_key and base_url:
        return OpenAI(api_key=api_key, base_url=base_url)
    return None

def run_ai_step_analysis(prompt_key, output_attr, title):
    """
    Voert een AI-analyse uit voor een specifieke diagnostische stap.

    Deze functie stelt een prompt samen op basis van de context uit de `PolicyData`
    instantie, stuurt deze naar een private-cloud OpenAI API, en slaat het resultaat op in
    het gespecificeerde attribuut van de `PolicyData` instantie.
    Vervolgens wordt de Streamlit app herladen met `st.rerun()`.
    Args:
        prompt_key (str): Een sleutel om de juiste prompt-template te selecteren 
                          (bv. "target_group_prompt").
        output_attr (str): De naam van het attribuut in de `PolicyData` klasse waar 
                           het AI-resultaat moet worden opgeslagen (bv. "target_group_ai").
        title (str): Een titel voor de spinner-tekst (bv. "Doelgroep en Gedrag").
    """
    client = get_openai_client()
    data = st.session_state.data

    if not client:
        st.session_state.data.__dict__[output_attr] = "Kan geen AI-suggesties ophalen zonder geldige API-sleutel en basis-URL."
        return

    policy_context = data.policy_desc
    prompt = ""

    # Definieer de Nederlandse context die aan elke prompt wordt toegevoegd.
    nl_context = "De analyse betreft een beleidsvoorstel van de Nederlandse regering. De invloedssfeer is beperkt tot het Nederlands grondgebied en haar inwoners. Houd hier rekening mee bij het inschatten van de effecten. Specifiek, ga na wat de context is voor de vraagkant van de economie en de aanbodkant. Actoren aan de vraagkant (nederlandse consumenten) kunnen makkelijker worden beinvloedt dan vaak internationale actoren aan de aanbodkant, omdat Nederland een klein land is op een grote internationale markt. Deze context hoeft niet terug te komen in het antwoord specifiek, maar je moet er wel rekening mee houden."

    if prompt_key == "target_group_prompt":
        prompt = f"""{nl_context}\n\nDe beleidsoptie is: '{policy_context}'.\n\nBepaal de meest waarschijnlijke **doelgroepen** en het **gedrag/de activiteit** dat direct door dit beleid wordt beïnvloed. Geef een beknopte, beredeneerde analyse van maximaal 5 zinnen. Antwoord in het Nederlands."""
    elif prompt_key == "emissions_prompt":
        target_group_context = data.target_group_desc
        prompt = f"""{nl_context}\n\nDe beleidsoptie is: '{policy_context}'. De gedragsverandering is: '{target_group_context}'.\n\nWelke directe en indirecte **emissies, onttrekkingen, of fysieke veranderingen** (zoals habitatverlies) zullen hierdoor optreden? Geef een beknopte analyse van maximaal 5 zinnen. Antwoord in het Nederlands."""
    elif prompt_key == "consequences_prompt":
        emissions_context = data.emissions_desc
        prompt = f"""{nl_context}\n\nDe fysieke veranderingen/emissies zijn: '{emissions_context}'.\n\nWat zijn de daaruit voortvloeiende **veranderingen in het fysieke leefmilieu** (bv. slechtere luchtkwaliteit, watervervuiling) en wat zijn de uiteindelijke **gevolgen hiervan voor de mens**? Geef een beknopte analyse van maximaal 5 zinnen. Antwoord in het Nederlands."""
    
    if not prompt:
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


# --- 3. Initialisatie van Session State --------------------------------------

if "data" not in st.session_state:
    st.session_state.data = PolicyData()
    st.session_state.data.initialize_indicators(load_default_indicators())

if "page" not in st.session_state:
    st.session_state.page = PAGE_INPUT

# Initialiseer AI Config (kan in sidebar worden aangepast)
if "openai_api_key" not in st.session_state:
    st.session_state.openai_api_key = st.secrets.get("OPENAI_API_KEY", "") 
if "openai_base_url" not in st.session_state:
    st.session_state.openai_base_url = st.secrets.get("OPENAI_BASE_URL", "")
    

# --- 4. Pagina Render Functies -----------------------------------------------

def render_input_page(data: PolicyData):
    """
    Rendert de UI voor Stap 1: Het omschrijven van de beleidsoptie.

    Deze pagina bevat een text_area waarin de gebruiker de beleidsoptie kan
    beschrijven. De ingevoerde tekst wordt opgeslagen in `data.policy_desc`.
    Args:
        data (PolicyData): De data-instantie uit de sessiestatus.
    """
    st.title("Stap 1: Beleidsoptie omschrijven")
    st.markdown("Geef een duidelijke toelichting op wat de beleidsoptie inhoudt.")
    
    new_desc = st.text_area(
        "Beleidsoptie toelichting",
        value=data.policy_desc,
        placeholder="Beschrijf het beleid (wat het doet, wie het beïnvloedt, tijdsbestek, grenzen, belangrijke aannames)...",
        height=300,
        key="policy_desc_input",
        label_visibility="collapsed"
    )
    # Update de PolicyData instantie direct
    data.policy_desc = new_desc
    
    st.markdown("---")
    
    if data.policy_desc.strip():
        st.button("Naar Stap 2: Doelgroep & Gedrag →", on_click=set_page, args=[PAGE_TARGET_GROUP], type="primary")
    else:
        st.button("Naar Stap 2: Doelgroep & Gedrag →", disabled=True, help="Vul een beleidsbeschrijving in om verder te gaan.")


def render_target_group_page(data: PolicyData):
    """
    Rendert de UI voor Stap 2: Doelgroep en Gedrag.

    De gebruiker kan zijn analyse invoeren, die wordt opgeslagen in `data.target_group_desc`.
    Er is ook een knop om een AI-suggestie op te vragen, die wordt weergegeven
    in een apart tekstvak en wordt opgeslagen in `data.target_group_ai`.
    Args:
        data (PolicyData): De data-instantie uit de sessiestatus.
    """
    st.title("Stap 2: Doelgroep en Gedrag")
    st.button("← Terug naar Stap 1", on_click=set_page, args=[PAGE_INPUT])
    st.markdown("Identificeer de doelgroep van het beleid en de specifieke activiteiten of gedragingen die door het beleid zullen veranderen.")

    st.subheader("Uw analyse: Doelgroep en activiteit")
    # Gebruik een tijdelijke widget-sleutel en update de PolicyData bij wijziging
    new_desc = st.text_area(
        "Doelgroep en Gedrag",
        value=data.target_group_desc,
        placeholder="Wie is de doelgroep? Welke specifieke acties/gedragingen worden aangepast?",
        height=150,
        key="target_group_desc_widget"
    )
    data.target_group_desc = new_desc

    st.subheader("AI-Suggestie (Ter overweging)")
    st.text_area(
        "AI-Oordeel Doelgroep",
        value=data.target_group_ai, # Lees direct uit PolicyData
        height=150,
        label_visibility="collapsed"
    )

    if st.button("Ontvang AI-suggestie voor Doelgroep", key="run_ai_target"):
        # De functie update nu de PolicyData instantie direct (attribuut: 'target_group_ai')
        run_ai_step_analysis("target_group_prompt", "target_group_ai", "Doelgroep en Gedrag")

    st.markdown("---")
    
    if data.target_group_desc.strip():
        st.button("Naar Stap 3: Emissies & Veranderingen →", on_click=set_page, args=[PAGE_EMISSIONS], type="primary")
    else:
        st.button("Naar Stap 3: Emissies & Veranderingen →", disabled=True, help="Vul uw analyse in om verder te gaan.")


def render_emissions_page(data: PolicyData):
    """
    Rendert de UI voor Stap 3: Emissies, Onttrekkingen en Fysieke Veranderingen.

    De gebruiker kan zijn analyse van de fysieke veranderingen invoeren, die wordt
    opgeslagen in `data.emissions_desc`. Er is ook een knop om een AI-suggestie
    op te vragen, die wordt opgeslagen in `data.emissions_ai`.
    Args:
        data (PolicyData): De data-instantie uit de sessiestatus.
    """
    st.title("Stap 3: Emissies, Onttrekkingen en Fysieke Veranderingen")
    st.button("← Terug naar Stap 2", on_click=set_page, args=[PAGE_TARGET_GROUP])
    st.markdown("Vertaal het veranderde gedrag (Stap 2) naar directe en indirecte veranderingen in de fysieke omgeving (emissies, onttrekkingen of habitatverlies).")

    st.subheader("Uw analyse: Emissies en Veranderingen")
    new_desc = st.text_area(
        "Emissies, Onttrekkingen, Fysieke Veranderingen",
        value=data.emissions_desc,
        placeholder="Welke emissies/onttrekkingen ontstaan direct? En indirect? Denk aan CO2, stikstof, landgebruik.",
        height=150,
        key="emissions_desc_widget"
    )
    data.emissions_desc = new_desc

    st.subheader("AI-Suggestie (Ter overweging)")
    st.text_area(
        "AI-Oordeel Emissies",
        value=data.emissions_ai,
        height=150,
        label_visibility="collapsed"
    )

    if st.button("Ontvang AI suggestie voor vorm van milieudruk (emissies, hinder, of fysieke veranderingen)", key="run_ai_emissions"):
        run_ai_step_analysis("emissions_prompt", "emissions_ai", "Emissies en Veranderingen")

    st.markdown("---")
    
    if data.emissions_desc.strip():
        st.button("Naar Stap 4: Milieu & Menselijke Gevolgen →", on_click=set_page, args=[PAGE_CONSEQUENCES], type="primary")
    else:
        st.button("Naar Stap 4: Milieu & Menselijke Gevolgen →", disabled=True, help="Vul uw analyse in om verder te gaan.")


def render_consequences_page(data: PolicyData):
    """
    Rendert de UI voor Stap 4: Fysieke Gevolgen en Impact op Mensen.

    De gebruiker kan zijn analyse van de gevolgen voor milieu en mens invoeren,
    die wordt opgeslagen in `data.consequences_desc`. Er is ook een knop om een
    AI-suggestie op te vragen, die wordt opgeslagen in `data.consequences_ai`.
    Args:
        data (PolicyData): De data-instantie uit de sessiestatus.
    """
    st.title("Stap 4: Fysieke Gevolgen en Impact op Mensen")
    st.button("← Terug naar Stap 3", on_click=set_page, args=[PAGE_EMISSIONS])
    st.markdown("Welke veranderingen in het fysieke leefmilieu treden dan op, en wat zijn daarvan de gevolgen voor de mens?")

    st.subheader("Uw analyse: Milieu- en Menselijke Gevolgen")
    new_desc = st.text_area(
        "Gevolgen voor het Milieu en de Mens",
        value=data.consequences_desc,
        placeholder="Leidt dit tot verslechtering/verbetering van lucht/water/bodem? Wat is de impact op de gezondheid of leefbaarheid van mensen?",
        height=150,
        key="consequences_desc_widget"
    )
    data.consequences_desc = new_desc

    st.subheader("AI-Suggestie (Ter overweging)")
    st.text_area(
        "AI-Oordeel Gevolgen",
        value=data.consequences_ai,
        height=150,
        label_visibility="collapsed"
    )

    if st.button("Ontvang AI-suggestie voor Gevolgen", key="run_ai_consequences"):
        run_ai_step_analysis("consequences_prompt", "consequences_ai", "Gevolgen")

    st.markdown("---")
    
    if data.consequences_desc.strip():
        st.button("Naar Stap 5: Finale Impact Beoordeling →", on_click=set_page, args=[PAGE_ASSESSMENT], type="primary")
    else:
        st.button("Naar Stap 5: Finale Impact Beoordeling →", disabled=True, help="Vul uw analyse in om verder te gaan.")


def generate_markdown_report(data: PolicyData, counts: dict, impact_df: pd.DataFrame):
    """
    Genereert een volledig en gestructureerd Markdown-rapport van de analyse.

    Het rapport bevat de beleidsbeschrijving, de analyses van de diagnostische
    stappen (zowel van de gebruiker als van de AI), een samenvatting van de
    impactscores, en een gedetailleerde uitsplitsing per indicator.
    Args:
        data (PolicyData): De data-instantie met alle invoer.
        counts (dict): Een dictionary met het aantal positieve, neutrale en negatieve scores.
        impact_df (pd.DataFrame): DataFrame met de gedetailleerde beoordelingen per indicator.
    Returns:
        str: De volledige inhoud van het rapport als een Markdown-geformatteerde string.
    """
    
    timestamp = datetime.now().strftime("%d-%m-%Y om %H:%M:%S")

    report_content = f"""# Milieu-impact Analyse Rapport (Diagnostisch)
Datum van de Analyse: {timestamp}
Beleidsoptie: {data.policy_desc[:50] + '...' if len(data.policy_desc) > 50 else data.policy_desc}

## 1. Beleidsbeschrijving
---
{data.policy_desc}
---

## 2. Diagnostische Analyse (Stappen 2 t/m 4)

### Stap 2: Doelgroep en Gedrag
| Bron | Inhoud |
| :--- | :--- |
| **Uw Analyse** | {data.target_group_desc} |
| **AI-Suggestie** | {data.target_group_ai if data.target_group_ai else '*Geen AI-suggestie uitgevoerd.*'} |

### Stap 3: Emissies, Onttrekkingen en Fysieke Veranderingen
| Bron | Inhoud |
| :--- | :--- |
| **Uw Analyse** | {data.emissions_desc} |
| **AI-Suggestie** | {data.emissions_ai if data.emissions_ai else '*Geen AI-suggestie uitgevoerd.*'} |

### Stap 4: Fysieke Gevolgen en Impact op Mensen
| Bron | Inhoud |
| :--- | :--- |
| **Uw Analyse** | {data.consequences_desc} |
| **AI-Suggestie** | {data.consequences_ai if data.consequences_ai else '*Geen AI-suggestie uitgevoerd.*'} |

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
        
        report_content += f"""
### Indicator: {ind}
- **Beoordeelde Impact:** **{impact}**
- **Toelichting (Handmatig + Diagnostiek):** {textwrap.indent(toelichting, '  - ')}\n
"""
        report_content += "\n***\n" 

    return report_content

def render_assessment_page(data: PolicyData):
    """
    Rendert de UI voor Stap 5: Finale Impact Beoordeling.

    Deze pagina toont een samenvatting van de diagnostische stappen en biedt
    een interface om de impact per milieu-indicator te beoordelen. Het bevat:
    - Sidebar voor het selecteren/uploaden van indicatoren.
    - Een beoordelingsgrid waar de gebruiker impact (positief/negatief/neutraal)
      en een toelichting kan invoeren.
    - Een samenvattingstabel en een grafiek van de impactscores.
    - Knoppen om de resultaten als CSV of als een volledig Markdown-rapport te downloaden.
    Args:
        data (PolicyData): De data-instantie uit de sessiestatus.
    """
    
    st.title("Stap 5: Finale Impact Beoordeling")
    st.button("← Terug naar Stap 4", on_click=set_page, args=[PAGE_CONSEQUENCES])

    if not data.policy_desc.strip() or not data.consequences_desc.strip():
        st.error("Ga terug om alle diagnostische stappen (1 t/m 4) te voltooien.")
        return

    st.subheader("Samenvatting Diagnostische Stappen (Handmatig)")
    
    st.text_area("Diagnostische Context", value=data.get_full_diagnostic_context(), height=200, disabled=True)
    
    st.markdown("---")

    # --- Sidebar: Indicator Source & Selection ----------------------------------
    st.sidebar.header("Indicatorbron")
    uploaded = st.sidebar.file_uploader("Upload een CSV met indicatoren (eerste kolom wordt gebruikt)", type=["csv"], key="indicator_uploader")
    
    if uploaded:
        new_list = load_indicators_from_file(uploaded)
        # Update de PolicyData instantie met de nieuwe lijst
        data.indicator_list = new_list
        data.selected_inds = new_list
        # Reset de selectie widget via een sleutel
        st.session_state.selected_inds_widget = new_list
        st.session_state.render_inds = new_list

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

    # --- Main Page: Assess impacts ----------------------------------------------
    st.markdown("Beoordeel nu de impact voor elke indicator. Gebruik uw diagnostische analyse hierboven als basis voor uw toelichting.")

    cols_config = [3, 2, 5]
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

        cols = st.columns(cols_config)
        with cols[0]:
            st.write(f"**{ind}**")
        with cols[1]:
            # Gebruik een unieke sleutel per rij en type
            impact_key = f"impact_{sk}" 
            new_impact = cols[1].selectbox(f"Impact {i+1}", options=impact_options, index=impact_options.index(current_impact), key=impact_key, label_visibility="collapsed")
            
        with cols[2]:
            explanation_key = f"explanation_{sk}"
            new_explanation = cols[2].text_area(f"Toelichting {i+1}", value=current_explanation, key=explanation_key, height=90, label_visibility="collapsed")

        # Update PolicyData zodra de widget is gerenderd/gewijzigd
        data.update_assessment(ind, new_impact, new_explanation)

    # Re-collect data into DataFrame for reporting and visualization
    rows = []
    full_diagnostic_text = data.get_full_diagnostic_context()
    
    for ind in render_inds:
        impact = data.get_assessment_value(ind, "impact")
        explanation = data.get_assessment_value(ind, "explanation")
        
        # Combineer de toelichting voor het rapport
        combined_explanation = f"{full_diagnostic_text}\n\n**Uw Toelichting (per indicator)**: {explanation}"
        
        rows.append({
            "Indicator": ind,
            "Impact": impact,
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
    edited_for_save = edited_df.drop(columns=["Toelichting (Rapport)"]).rename(columns={"Toelichting (Tabel)": "Toelichting"})
    
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
                edited_df # Pass the DataFrame with the full combined explanation
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
                    hdf = pd.DataFrame(rec["table"])
                    st.dataframe(hdf, use_container_width=True)
                    buf = StringIO()
                    hdf.to_csv(buf, index=False)
                    st.download_button(f"CSV downloaden (item #{idx})", buf.getvalue(), file_name=f"snapshot_{idx}.csv", mime="text/csv")


# --- 5. Main Application Flow --------------------------------------------------

data = st.session_state.data

st.sidebar.title("Beleidsoptie Milieu-impact Scoping")
st.sidebar.markdown("Navigatie tussen de stappen.")
st.sidebar.button("Stap 1: Beleidsinvoer", on_click=set_page, args=[PAGE_INPUT], use_container_width=True)
st.sidebar.button("Stap 2: Doelgroep", on_click=set_page, args=[PAGE_TARGET_GROUP], use_container_width=True)
st.sidebar.button("Stap 3: Emissies", on_click=set_page, args=[PAGE_EMISSIONS], use_container_width=True)
st.sidebar.button("Stap 4: Gevolgen", on_click=set_page, args=[PAGE_CONSEQUENCES], use_container_width=True)
st.sidebar.button("Stap 5: Beoordeling", on_click=set_page, args=[PAGE_ASSESSMENT], type="primary", use_container_width=True)

# Render de geselecteerde pagina en geef de PolicyData instantie door
if st.session_state.page == PAGE_INPUT:
    render_input_page(data)
elif st.session_state.page == PAGE_TARGET_GROUP:
    render_target_group_page(data)
elif st.session_state.page == PAGE_EMISSIONS:
    render_emissions_page(data)
elif st.session_state.page == PAGE_CONSEQUENCES:
    render_consequences_page(data)
elif st.session_state.page == PAGE_ASSESSMENT:
    render_assessment_page(data)

# Toon de AI/API configuratie in de sidebar (optioneel, voor debug/configuratie)
with st.sidebar.expander("AI API Configuratie"):
    st.text_input("OpenAI API Key", st.session_state.openai_api_key, key="openai_api_key_widget", type="password")
    st.text_input("OpenAI Base URL", st.session_state.openai_base_url, key="openai_base_url_widget")

    # Update de sessiestatus en herlaad de AI client bij wijziging van de widget
    st.session_state.openai_api_key = st.session_state.openai_api_key_widget
    st.session_state.openai_base_url = st.session_state.openai_base_url_widget

    if st.button("AI Client Herladen", key="reload_ai"):
        st.cache_resource.clear()
        st.success("AI Client geherladen.")
