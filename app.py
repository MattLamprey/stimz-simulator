import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import streamlit as st

st.set_page_config(page_title="Stimz Recommender Prototype", layout="wide")

# ----------------------------
# LOAD DATA
# ----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("FilteredScoredData.csv")
    pd.set_option("future.no_silent_downcasting", True)
    df = df.replace(r"^\s*$", np.nan, regex=True)
    df = df.infer_objects(copy=False)
    return df

df = load_data()

# ----------------------------
# DISPLAY LABELS
# ----------------------------
condition_options = {
    "ADHD": "adhd_attention_differences_diagnosis",
    "Autism": "autism_diagnosis",
    "Anxiety": "anxiety_diagnosis",
    "Depression": "depression_diagnosis",
    "OCD": "ocd_diagnosis",
    "Sensory Processing Differences": "sensory_processing_differences_diagnosis",
    "Dyspraxia": "dyspraxia_diagnosis",
    "Dyslexia": "dyslexia_diagnosis",
    "Tourette's / Tics": "tourettes_tics_diagnosis",
}

severity_cols = {
    "ADHD": "adhd_attention_differences_severity",
    "Autism": "autism_severity",
    "Anxiety": "anxiety_severity",
    "Depression": "depression_severity",
    "OCD": "ocd_severity",
    "Sensory Processing Differences": "sensory_processing_differences_severity",
    "Dyspraxia": "dyspraxia_severity",
    "Dyslexia": "dyslexia_severity",
    "Tourette's / Tics": "tourettes_tics_severity",
}

symptom_labels = {
    "Difficultyfocusingorstayingontask": "Difficulty focusing or staying on task",
    "Restlessnessorneedingtomovefidget": "Restlessness or needing to move / fidget",
    "Feelingoverwhelmedsensoryormental": "Feeling overwhelmed (sensory or mental)",
    "Suddenanxietyspikes": "Sudden anxiety spikes",
    "Bigemotionsthatarehardtoregulate": "Big emotions that are hard to regulate",
    "Difficultywindingdownforsleep": "Difficulty winding down for sleep",
    "Skinpickingornailbiting": "Skin picking or nail biting",
    "Chewingormouthingurges": "Chewing or mouthing urges",
    "Needingdeeppressureorstrongsensoryinput": "Needing deep pressure or strong sensory input",
    "Seekingpainfulorveryintensesensation": "Seeking painful or very intense sensation",
    "Feelingmentallydrainedorshutdown": "Feeling mentally drained or shut down",
}

stim_labels = {
    "Clickingbuttonpressing": "Clicking / button pressing",
    "Rolling": "Rolling",
    "Rubbingovertexturetracing": "Rubbing over texture / tracing",
    "Twistingspinning": "Twisting / spinning",
    "Squeezingsquishing": "Squeezing / squishing",
    "Stretchingpulling": "Stretching / pulling",
    "Tappingdrumming": "Tapping / drumming",
    "Intenseinputspikypain": "Intense input (spiky / pain)",
    "Lookingatcolourormovement": "Looking at colour or movement",
    "Chewingmouthing": "Chewing / mouthing",
    "Flipfoldmovingbackandforth": "Flip / fold / moving back-and-forth",
    "Weightedpressure": "Weighted pressure",
}

feature_labels = {
    "Quiet": "Quiet",
    "Discreet": "Discreet",
    "Easytouseonehanded": "Easy to use one-handed",
    "Pocketsizedeasytocarry": "Pocket sized / easy to carry",
    "Wearable": "Wearable",
}

persona_names = {
    0: "Deep Pressure Regulator",
    1: "Emotional Regulator",
    2: "High-Intensity Seeker",
    3: "Anxious Habit Regulator",
    4: "Focus & Fidget Regulator",
}

persona_descriptions = {
    0: "Tends to feel overwhelmed or mentally drained and benefits from grounding sensory input like pressure, resistance, or oral stimulation.",
    1: "Experiences intense emotional fluctuations and may use sensory input or repetitive behaviours to regulate anxiety and stress.",
    2: "Actively seeks strong sensory input such as pressure, resistance, or intense feedback to regulate sensory overload and internal tension.",
    3: "Tends to develop repetitive habits (e.g. picking, fidgeting) in response to anxiety or stress, benefiting from controlled sensory alternatives.",
    4: "Struggles with focus and restlessness, often benefiting from movement-based or tactile stimulation to maintain attention and calm.",
}


symptom_cols = list(symptom_labels.keys())
stim_cols = list(stim_labels.keys())
stim_cols = list(stim_labels.keys())
feature_cols = list(feature_labels.keys())

age_col = "AgeBands"

age_labels = {
    0: "Prefer not to say",
    1: "4–7",
    2: "8–12",
    3: "13–17",
    4: "18–24",
    5: "25–34",
    6: "35–44",
    7: "45–54",
    8: "55+",
}

condition_options = {k: v for k, v in condition_options.items() if v in df.columns}
severity_cols = {k: v for k, v in severity_cols.items() if v in df.columns}

required_cols = [age_col] + list(condition_options.values()) + list(severity_cols.values()) + symptom_cols + stim_cols + feature_cols
missing_cols = [c for c in required_cols if c not in df.columns]

if missing_cols:
    st.error("These columns are missing from the CSV:")
    st.write(missing_cols)
    st.stop()


# ----------------------------
# HELPERS
# ----------------------------
def safe_cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    if np.all(vec1 == 0) or np.all(vec2 == 0):
        return 0.0
    return float(cosine_similarity([vec1], [vec2])[0][0])

def severity_group_from_score(score: float) -> str:
    if score <= 2:
        return "Mild"
    elif score == 3:
        return "Moderate"
    else:
        return "High"


def severity_group_similarity(user_group: str, respondent_group: str) -> float:
    if user_group == respondent_group:
        return 1.0

    adjacent = {
        "Mild": ["Moderate"],
        "Moderate": ["Mild", "High"],
        "High": ["Moderate"],
    }

    if respondent_group in adjacent.get(user_group, []):
        return 0.5

    return 0.0

def normalise_severity_value(value: float) -> float:
    if pd.isna(value):
        return 0.0
    value = float(value)
    value = max(1.0, min(5.0, value))
    return (value - 1.0) / 4.0

def get_row_condition_vector(row: pd.Series) -> np.ndarray:
    return np.array([float(row[col]) for col in condition_options.values()], dtype=float)

def get_row_severity_value(row: pd.Series) -> float:
    found = []
    for condition_name, diag_col in condition_options.items():
        sev_col = severity_cols.get(condition_name)
        if sev_col is None:
            continue
        if float(row[diag_col]) > 0:
            found.append(normalise_severity_value(float(row[sev_col])))

    if found:
        return float(np.mean(found))

    fallback = []
    for sev_col in severity_cols.values():
        fallback.append(normalise_severity_value(float(row[sev_col])))

    return float(np.mean(fallback)) if fallback else 0.0

def compute_similarity(row, user_age_band, user_condition_vec, user_symptom_vec, user_severity, user_severity_group):

    respondent_condition_vec = get_row_condition_vector(row)
    respondent_symptom_vec = row[symptom_cols].values.astype(float)
    respondent_severity = get_row_severity_value(row)
    respondent_age_band = str(row[age_col]).strip()

    respondent_severity_raw = 1 + (respondent_severity * 4)
    respondent_severity_group = severity_group_from_score(round(respondent_severity_raw))

    age_sim = age_similarity(user_age_band, respondent_age_band)
    condition_sim = safe_cosine_similarity(user_condition_vec, respondent_condition_vec)
    symptom_sim = safe_cosine_similarity(user_symptom_vec, respondent_symptom_vec)

    severity_sim = 1.0 - abs(user_severity - respondent_severity)
    severity_sim = max(0.0, min(1.0, severity_sim))

    severity_group_sim = severity_group_similarity(user_severity_group, respondent_severity_group)

    return (
        0.12 * age_sim +
        0.23 * condition_sim +
        0.40 * symptom_sim +
        0.15 * severity_sim +
        0.10 * severity_group_sim
    )

def join_nicely(items):
    if len(items) == 0:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + f", and {items[-1]}"

def get_age_options(df: pd.DataFrame, age_col: str) -> list:
    vals = df[age_col].dropna()

    cleaned = []
    for v in vals:
        try:
            code = int(float(str(v).strip()))
            if code != 0:  # exclude "Prefer not to say"
                cleaned.append(code)
        except:
            pass

    return sorted(set(cleaned))

def age_similarity(user_age, respondent_age) -> float:
    try:
        user_age = int(user_age)
        respondent_age = int(float(str(respondent_age).strip()))
    except:
        return 0.0

    distance = abs(user_age - respondent_age)

    if distance == 0:
        return 1.0
    elif distance == 1:
        return 0.6
    elif distance == 2:
        return 0.3
    else:
        return 0.0

@st.cache_data
def build_personas(df: pd.DataFrame, persona_cols: list[str], n_clusters: int = 5):

    X = df[persona_cols].astype(float).copy()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
    clusters = model.fit_predict(X_scaled)

    df_with_clusters = df.copy()
    df_with_clusters["persona_cluster"] = clusters

    cluster_profiles = (
        df_with_clusters
        .groupby("persona_cluster")[persona_cols]
        .mean()
    )

    # 🔥 normalisation you added
    cluster_profiles = cluster_profiles.div(cluster_profiles.sum(axis=1), axis=0)

    cluster_profiles = cluster_profiles.reset_index()

    # ✅ THIS LINE MUST ALIGN WITH THE REST (same indent level)
    centroids_scaled = model.cluster_centers_

    return df_with_clusters, cluster_profiles, scaler, model, centroids_scaled

# ----------------------------
# CLEAN DATA
# ----------------------------
for col in required_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df[required_cols] = df[required_cols].fillna(0)

# ----------------------------
# BUILD PERSONA
# ----------------------------

persona_cols = symptom_cols + stim_cols + feature_cols

df, cluster_profiles, persona_scaler, persona_model, persona_centroids = build_personas(
    df, persona_cols, n_clusters=5
)

# ----------------------------
# UI
# ----------------------------
st.title("Stimz Recommender Prototype")
st.write("Select condition, severity, and top symptoms to see what types of stim characteristics similar respondents tend to prefer.")

col1, col2 = st.columns(2)

with col1:
    age_options = get_age_options(df, age_col)

    selected_age_band = st.selectbox(
        "Select your age group",
        age_options,
        format_func=lambda x: age_labels.get(int(float(x)), str(x))
    )

    selected_conditions = st.multiselect(
        "Select your condition(s)",
        list(condition_options.keys())
    )

    severity = st.slider(
        "How severe are your symptoms overall?",
        min_value=1,
        max_value=5,
        value=3
    )
user_severity = (float(severity) - 1.0) / 4.0
user_severity_group = severity_group_from_score(severity)

st.caption(f"Severity group: {user_severity_group}")

with col2:
    selected_symptoms_display = st.multiselect(
    "Select your top 3 symptoms",
    list(symptom_labels.values()),
    max_selections=3
)

if len(selected_conditions) == 0 or len(selected_symptoms_display) == 0:
    st.warning("Please select at least one condition and at least one top symptom.")
    st.stop()

reverse_symptom_labels = {v: k for k, v in symptom_labels.items()}
selected_symptoms = [reverse_symptom_labels[x] for x in selected_symptoms_display]

# ----------------------------
# BUILD USER INPUT VECTORS
# ----------------------------
user_condition_vec = np.array(
    [1.0 if condition_name in selected_conditions else 0.0 for condition_name in condition_options.keys()],
    dtype=float,
)

user_symptom_vec = np.array(
    [1.0 if symptom in selected_symptoms else 0.0 for symptom in symptom_cols],
    dtype=float,
)

# ----------------------------
# CALCULATE SIMILARITY
# ----------------------------
working_df = df.copy()

working_df["similarity"] = working_df.apply(
    compute_similarity,
    axis=1,
    user_age_band=selected_age_band,
    user_condition_vec=user_condition_vec,
    user_symptom_vec=user_symptom_vec,
    user_severity=user_severity,
    user_severity_group=user_severity_group,
)

# Sort by strongest similarity first
working_df = working_df.sort_values("similarity", ascending=False).copy()

# Sharpen the difference between strong and weak matches
working_df["similarity"] = working_df["similarity"].astype(float) ** 2

# Keep only reasonably strong matches
top_df = working_df[working_df["similarity"] > (0.45 ** 2)].copy()

# Hard cap so recommendations are driven by nearest matches, not broad averages
top_n = 25
top_df = top_df.head(top_n).copy()

# Fallback if too few matches survive
if len(top_df) < 15:
    top_df = working_df.head(top_n).copy()

# Build base similarity weights
weights = top_df["similarity"].astype(float)

# Reduce dominance from over-represented persona clusters
cluster_counts = top_df["persona_cluster"].value_counts(normalize=True)

top_df["cluster_weight"] = top_df["persona_cluster"].map(
    lambda c: 1 / (cluster_counts[c] + 0.01)
)

# Combine similarity with cluster diversity weighting
top_df["final_weight"] = weights * top_df["cluster_weight"]
weights = top_df["final_weight"]

# Add respondent severity group
top_df["respondent_severity_group"] = top_df.apply(
    lambda row: severity_group_from_score(round(1 + (get_row_severity_value(row) * 4))),
    axis=1
)

# Boost same/near severity groups
severity_boost = top_df["respondent_severity_group"].apply(
    lambda x: 1.10 if x == user_severity_group else (1.03 if severity_group_similarity(user_severity_group, x) == 0.5 else 1.0)
)

# Final weights
weights = weights * severity_boost

# Safety check
if weights.sum() == 0:
    st.error("Similarity scores summed to zero. Try selecting more common conditions or symptoms.")
    st.stop()


# ----------------------------
# AGGREGATE RESULTS
# ----------------------------
predicted_symptoms = (
    top_df[symptom_cols]
    .astype(float)
    .multiply(weights, axis=0)
    .sum() / weights.sum()
).sort_values(ascending=False)

predicted_stims = (
    top_df[stim_cols]
    .astype(float)
    .multiply(weights, axis=0)
    .sum() / weights.sum()
).sort_values(ascending=False)

predicted_features = (
    top_df[feature_cols]
    .astype(float)
    .multiply(weights, axis=0)
    .sum() / weights.sum()
).sort_values(ascending=False)

user_persona_df = pd.DataFrame([{
    **predicted_symptoms[symptom_cols].to_dict(),
    **predicted_stims[stim_cols].to_dict(),
    **predicted_features[feature_cols].to_dict(),
}])

user_persona_vector_scaled = persona_scaler.transform(user_persona_df[persona_cols])
user_cluster = int(persona_model.predict(user_persona_vector_scaled)[0])

# ----------------------------
# TUNE STIM SIGNALS BEFORE PRODUCT MAPPING
# ----------------------------
predicted_stims = predicted_stims.copy()

# Reduce dominance of visual signal
# Context-aware visual dampening
visual_base = predicted_stims["Lookingatcolourormovement"]
intensity_need = predicted_stims["Intenseinputspikypain"] + predicted_stims["Weightedpressure"]

# Strong suppression when intensity dominates
if intensity_need > 0.3:
    predicted_stims["Lookingatcolourormovement"] *= 0.3

# Moderate suppression for high severity users
elif user_severity_group == "High":
    predicted_stims["Lookingatcolourormovement"] *= 0.5

# Light suppression otherwise
else:
    predicted_stims["Lookingatcolourormovement"] *= 0.7

persona_name = persona_names.get(user_cluster, "")

if persona_name == "Anxious Habit Regulator":
    predicted_stims["Intenseinputspikypain"] *= 0.6
    predicted_stims["Weightedpressure"] *= 0.7
    predicted_stims["Squeezingsquishing"] *= 0.85
    predicted_stims["Stretchingpulling"] *= 0.8
elif persona_name in ["Deep Pressure Regulator", "High-Intensity Seeker"]:
    predicted_stims["Intenseinputspikypain"] *= 1.5
    predicted_stims["Weightedpressure"] *= 1.4
    predicted_stims["Squeezingsquishing"] *= 1.3
    predicted_stims["Stretchingpulling"] *= 1.2
else:
    predicted_stims["Intenseinputspikypain"] *= 1.1
    predicted_stims["Weightedpressure"] *= 1.05
    predicted_stims["Squeezingsquishing"] *= 1.05
    predicted_stims["Stretchingpulling"] *= 1.0


# ----------------------------
# PRODUCT TYPE CONCEPTS
# ----------------------------
product_types = pd.DataFrame([
    {
        "Product type": "Chewable stim",
        "Clickingbuttonpressing": 0.0,
        "Rolling": 0.0,
        "Rubbingovertexturetracing": 0.0,
        "Twistingspinning": 0.0,
        "Squeezingsquishing": 0.2,
        "Stretchingpulling": 0.1,
        "Tappingdrumming": 0.0,
        "Intenseinputspikypain": 0.1,
        "Lookingatcolourormovement": 0.0,
        "Chewingmouthing": 1.0,
        "Flipfoldmovingbackandforth": 0.0,
        "Weightedpressure": 0.0,
        "Quiet": 0.7,
        "Discreet": 0.8,
        "Easytouseonehanded": 0.8,
        "Pocketsizedeasytocarry": 0.8,
        "Wearable": 0.1,
    },
    {
    "Product type": "Stretch-based stim",
    "Clickingbuttonpressing": 0.0,
    "Rolling": 0.1,
    "Rubbingovertexturetracing": 0.1,
    "Twistingspinning": 0.2,
    "Squeezingsquishing": 0.4,
    "Stretchingpulling": 1.0,
    "Tappingdrumming": 0.0,
    "Intenseinputspikypain": 0.15,
    "Lookingatcolourormovement": 0.0,
    "Chewingmouthing": 0.0,
    "Flipfoldmovingbackandforth": 0.2,
    "Weightedpressure": 0.1,
    "Quiet": 0.65,
    "Discreet": 0.6,
    "Easytouseonehanded": 0.65,
    "Pocketsizedeasytocarry": 0.5,
    "Wearable": 0.0,
},
    {
        "Product type": "Wearable sensory stim",
        "Clickingbuttonpressing": 0.1,
        "Rolling": 0.1,
        "Rubbingovertexturetracing": 0.4,
        "Twistingspinning": 0.2,
        "Squeezingsquishing": 0.2,
        "Stretchingpulling": 0.3,
        "Tappingdrumming": 0.1,
        "Intenseinputspikypain": 0.0,
        "Lookingatcolourormovement": 0.2,
        "Chewingmouthing": 0.0,
        "Flipfoldmovingbackandforth": 0.2,
        "Weightedpressure": 0.1,
        "Quiet": 0.9,
        "Discreet": 1.0,
        "Easytouseonehanded": 0.8,
        "Pocketsizedeasytocarry": 0.7,
        "Wearable": 1.0,
    },
    {
        "Product type": "Quiet handheld stim",
        "Clickingbuttonpressing": 0.3,
        "Rolling": 0.4,
        "Rubbingovertexturetracing": 0.6,
        "Twistingspinning": 0.4,
        "Squeezingsquishing": 0.5,
        "Stretchingpulling": 0.3,
        "Tappingdrumming": 0.1,
        "Intenseinputspikypain": 0.0,
        "Lookingatcolourormovement": 0.1,
        "Chewingmouthing": 0.0,
        "Flipfoldmovingbackandforth": 0.4,
        "Weightedpressure": 0.1,
        "Quiet": 1.0,
        "Discreet": 0.9,
        "Easytouseonehanded": 0.9,
        "Pocketsizedeasytocarry": 0.8,
        "Wearable": 0.0,
    },
    {
    "Product type": "Visual stim",
    "Clickingbuttonpressing": 0.0,
    "Rolling": 0.0,
    "Rubbingovertexturetracing": 0.0,
    "Twistingspinning": 0.1,
    "Squeezingsquishing": 0.0,
    "Stretchingpulling": 0.0,
    "Tappingdrumming": 0.0,
    "Intenseinputspikypain": 0.0,
    "Lookingatcolourormovement": 0.6,
    "Chewingmouthing": 0.0,
    "Flipfoldmovingbackandforth": 0.1,
    "Weightedpressure": 0.0,
    "Quiet": 0.3,
    "Discreet": 0.2,
    "Easytouseonehanded": 0.2,
    "Pocketsizedeasytocarry": 0.1,
    "Wearable": 0.0,
},
    {
    "Product type": "Deep-pressure stim",
    "Clickingbuttonpressing": 0.0,
    "Rolling": 0.0,
    "Rubbingovertexturetracing": 0.1,
    "Twistingspinning": 0.0,
    "Squeezingsquishing": 0.9,
    "Stretchingpulling": 0.6,
    "Tappingdrumming": 0.0,
    "Intenseinputspikypain": 0.4,
    "Lookingatcolourormovement": 0.0,
    "Chewingmouthing": 0.0,
    "Flipfoldmovingbackandforth": 0.0,
    "Weightedpressure": 1.0,
    "Quiet": 0.8,
    "Discreet": 0.5,
    "Easytouseonehanded": 0.6,
    "Pocketsizedeasytocarry": 0.3,
    "Wearable": 0.1,
},
{
    "Product type": "Tactile texture stim",
    "Clickingbuttonpressing": 0.1,
    "Rolling": 0.2,
    "Rubbingovertexturetracing": 1.0,
    "Twistingspinning": 0.1,
    "Squeezingsquishing": 0.4,
    "Stretchingpulling": 0.2,
    "Tappingdrumming": 0.0,
    "Intenseinputspikypain": 0.0,
    "Lookingatcolourormovement": 0.1,
    "Chewingmouthing": 0.0,
    "Flipfoldmovingbackandforth": 0.3,
    "Weightedpressure": 0.2,
    "Quiet": 0.95,
    "Discreet": 0.85,
    "Easytouseonehanded": 0.95,
    "Pocketsizedeasytocarry": 0.85,
    "Wearable": 0.1,
},
{
    "Product type": "Squeeze / resistance stim",
    "Clickingbuttonpressing": 0.0,
    "Rolling": 0.1,
    "Rubbingovertexturetracing": 0.15,
    "Twistingspinning": 0.1,
    "Squeezingsquishing": 0.9,
    "Stretchingpulling": 0.55,
    "Tappingdrumming": 0.0,
    "Intenseinputspikypain": 0.15,
    "Lookingatcolourormovement": 0.0,
    "Chewingmouthing": 0.0,
    "Flipfoldmovingbackandforth": 0.15,
    "Weightedpressure": 0.6,
    "Quiet": 0.85,
    "Discreet": 0.7,
    "Easytouseonehanded": 0.8,
    "Pocketsizedeasytocarry": 0.7,
    "Wearable": 0.0,
},
{
    "Product type": "Fidget / motor stim",
    "Clickingbuttonpressing": 0.9,
    "Rolling": 0.7,
    "Rubbingovertexturetracing": 0.3,
    "Twistingspinning": 0.8,
    "Squeezingsquishing": 0.4,
    "Stretchingpulling": 0.2,
    "Tappingdrumming": 0.9,
    "Intenseinputspikypain": 0.0,
    "Lookingatcolourormovement": 0.2,
    "Chewingmouthing": 0.0,
    "Flipfoldmovingbackandforth": 0.8,
    "Weightedpressure": 0.0,
    "Quiet": 0.6,
    "Discreet": 0.7,
    "Easytouseonehanded": 0.95,
    "Pocketsizedeasytocarry": 0.9,
    "Wearable": 0.0,
},
{
    "Product type": "Intense sensory stim",
    "Clickingbuttonpressing": 0.0,
    "Rolling": 0.1,
    "Rubbingovertexturetracing": 0.0,
    "Twistingspinning": 0.1,
    "Squeezingsquishing": 1.0,        # ↑ was 0.8
    "Stretchingpulling": 0.7,         # ↑ was 0.6
    "Tappingdrumming": 0.0,
    "Intenseinputspikypain": 1.0,
    "Lookingatcolourormovement": 0.0,
    "Chewingmouthing": 0.2,
    "Flipfoldmovingbackandforth": 0.0,
    "Weightedpressure": 1.0,          # ↑ was 0.9
    "Quiet": 0.3,
    "Discreet": 0.2,
    "Easytouseonehanded": 0.5,
    "Pocketsizedeasytocarry": 0.3,
    "Wearable": 0.0,
},


])

# ----------------------------
# PRODUCT TYPE DESCRIPTIONS
# ----------------------------
product_type_descriptions = {
    "Chewable stim": "Helps regulate through oral sensory input, especially useful for chewing urges or anxiety.",
    
    "Stretch-based stim": "Provides resistance and tension release through pulling or stretching movements.",
    
    "Wearable sensory stim": "Allows ongoing, discreet sensory input throughout the day without needing to hold an object.",
    
    "Quiet handheld stim": "Designed for subtle use in social or work settings where noise and visibility matter.",
    
    "Visual stim": "Engages visual focus and can help with calming or redirecting attention.",
    
    "Deep-pressure stim": "Provides firm sensory input that can help reduce overwhelm and promote grounding.",

"Tactile texture stim": "Focuses on rubbing or tracing textures to provide calming, repetitive sensory feedback.",

"Squeeze / resistance stim": "Provides compression-based input through squeezing or resistance, helping release tension and regulate stress.",

"Fidget / motor stim": "Encourages repetitive movement such as clicking, tapping, or spinning to support focus and restlessness.",

"Intense sensory stim": "Designed for those who seek strong sensory input, including spiky, firm, or high-intensity feedback.",
}

# ----------------------------
# SCORE PRODUCT TYPES
# ----------------------------
stim_weight = 0.80
feature_weight = 0.20

def score_product_type(row):
    stim_score = float(np.dot(row[stim_cols].values, predicted_stims[stim_cols].values))
    feature_score = float(np.dot(row[feature_cols].values, predicted_features[feature_cols].values))

    persona_name = persona_names.get(user_cluster, "")

    # Persona-based tuning
    if persona_name == "Anxious Habit Regulator":
        if row["Product type"] in ["Intense sensory stim", "Stretch-based stim", "Squeeze / resistance stim", "Deep-pressure stim"]:
            stim_score *= 0.5
        if row["Product type"] in ["Tactile texture stim", "Fidget / motor stim", "Quiet handheld stim"]:
            stim_score *= 1.35
        if row["Product type"] == "Visual stim":
            stim_score *= 0.65

    # Intensity mismatch
    intensity_need = predicted_stims["Intenseinputspikypain"] + predicted_stims["Weightedpressure"]
    product_intensity = row["Intenseinputspikypain"] + row["Weightedpressure"]

    mismatch_penalty = 0.0
    if intensity_need > 0.2 and product_intensity < 0.3:
        mismatch_penalty += 0.2

    # Visual penalties (clean version)
    visual_need = predicted_stims["Lookingatcolourormovement"]

    visual_penalty = 0.0
    if row["Product type"] == "Visual stim":
        if visual_need < 0.08:
            visual_penalty += 0.25
        if intensity_need > 0.3:
            visual_penalty += 0.20
        if user_severity_group == "High" and intensity_need > 0.2:
            visual_penalty += 0.20

    return (stim_weight * stim_score) + (feature_weight * feature_score) - mismatch_penalty - visual_penalty

product_types["Score"] = product_types.apply(score_product_type, axis=1)
product_types = product_types.sort_values("Score", ascending=False).reset_index(drop=True)

# Split into primary vs supporting
primary_candidates = []
supporting_candidates = []

supporting_types = []

# Only allow visual as supporting if it actually has signal
if predicted_stims["Lookingatcolourormovement"] > 0.12:
    supporting_types.append("Visual stim")

for p in product_types["Product type"]:
    if p in supporting_types:
        supporting_candidates.append(p)
    else:
        primary_candidates.append(p)

top_primary_product_types = primary_candidates[:2]
top_supporting_product_types = supporting_candidates[:1]

top_3_stims = [stim_labels[col] for col in predicted_stims.head(3).index]
top_3_features = [feature_labels[col] for col in predicted_features.head(3).index]
top_3_symptoms = [symptom_labels[col] for col in predicted_symptoms.head(3).index]

# ----------------------------
# FRIENDLY SUMMARY
# ----------------------------
st.subheader("Your sensory persona")
st.markdown(f"### {persona_names.get(user_cluster, f'Persona {user_cluster}')}")
st.write(persona_descriptions.get(user_cluster, ""))

st.subheader("Summary recommendation")

summary_text = (
    f"People with a similar profile to you tend to benefit from stims that provide **{join_nicely(top_3_stims)}**. "
    f"These are often most helpful when they are **{join_nicely(top_3_features)}**. "
    f"A good place to start would be looking for **{join_nicely(top_primary_product_types + top_supporting_product_types)}**. "
    f"This is likely because the main challenges reported are **{join_nicely(top_3_symptoms)}**."
)

# ----------------------------
# HERO OUTPUT (MAIN RESULT)
# ----------------------------
st.subheader("What should you look for")

# PRIMARY
st.markdown("## Primary recommendations")

for i, item in enumerate(top_primary_product_types, start=1):
    st.markdown(f"### {i}. {item}")
    st.caption(product_type_descriptions.get(item, ""))


# SUPPORTING
if top_supporting_product_types:
    st.markdown("## Supporting options")
    st.caption("Additional tools that may complement your primary regulation style")

    for i, item in enumerate(top_supporting_product_types, start=1):
        st.markdown(f"### {i}. {item}")
        st.caption(product_type_descriptions.get(item, ""))

st.divider()

st.subheader("How this recommendation is built")

rec1, rec2, rec3 = st.columns(3)

with rec1:
    st.markdown("### Stim interaction types")
    for i, item in enumerate(top_3_stims, start=1):
        st.markdown(f"{i}. {item}")

with rec2:
    st.markdown("### Practical features")
    for i, item in enumerate(top_3_features, start=1):
        st.markdown(f"{i}. {item}")

with rec3:
    st.markdown("### Symptom drivers")
    for i, item in enumerate(top_3_symptoms, start=1):
        st.markdown(f"{i}. {item}")

# ----------------------------
# DETAILED TABLES
# ----------------------------
st.subheader("Full matched profile insight")

insight1, insight2, insight3 = st.columns(3)

with insight1:
    symptom_display = pd.DataFrame({
        "Symptom": [symptom_labels[col] for col in predicted_symptoms.index],
        "Score": predicted_symptoms.values
    })
    st.dataframe(symptom_display, width="stretch", hide_index=True)

with insight2:
    stim_display = pd.DataFrame({
        "Stim type": [stim_labels[col] for col in predicted_stims.index],
        "Score": predicted_stims.values
    })
    st.dataframe(stim_display, width="stretch", hide_index=True)

with insight3:
    feature_display = pd.DataFrame({
        "Feature": [feature_labels[col] for col in predicted_features.index],
        "Score": predicted_features.values
    })
    st.dataframe(feature_display, width="stretch", hide_index=True)

# ----------------------------
# MODEL DIAGNOSTICS
# ----------------------------
st.subheader("Model diagnostics")
st.write(f"Number of matched respondents used: {len(top_df)}")
st.write(f"Average similarity of matched respondents: {top_df['similarity'].mean():.3f}")

with st.expander("Show top matched respondents"):
    display_cols = ["similarity"] + list(condition_options.values()) + list(severity_cols.values())
    st.dataframe(top_df[display_cols].head(20), width="stretch", hide_index=True)

with st.expander("🔍 Show persona cluster profiles"):
    st.dataframe(cluster_profiles, width="stretch")