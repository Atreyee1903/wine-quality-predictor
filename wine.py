from pathlib import Path

import pandas as pd
import streamlit as st
from lightgbm import LGBMClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, train_test_split


DATA_FILE = Path(__file__).with_name("wine__12.csv")
SEED = 8
GOOD_QUALITY_THRESHOLD = 7
FEATURES = [
    "fixed acidity",
    "volatile acidity",
    "citric acid",
    "residual sugar",
    "chlorides",
    "free sulfur dioxide",
    "total sulfur dioxide",
    "density",
    "pH",
    "sulphates",
    "alcohol",
]

SAMPLE_PROFILES = {
    "Startup Demo": {
        "fixed acidity": 7.3,
        "volatile acidity": 0.32,
        "citric acid": 0.36,
        "residual sugar": 2.0,
        "chlorides": 0.033,
        "free sulfur dioxide": 28.0,
        "total sulfur dioxide": 87.0,
        "density": 0.9908,
        "pH": 3.28,
        "sulphates": 0.78,
        "alcohol": 12.4,
    },
    "Budget Table Wine": {
        "fixed acidity": 11.2,
        "volatile acidity": 0.76,
        "citric acid": 0.04,
        "residual sugar": 2.3,
        "chlorides": 0.092,
        "free sulfur dioxide": 6.0,
        "total sulfur dioxide": 18.0,
        "density": 0.9991,
        "pH": 3.20,
        "sulphates": 0.45,
        "alcohol": 8.9,
    },
    "Balanced Everyday Wine": {
        "fixed acidity": 7.9,
        "volatile acidity": 0.43,
        "citric acid": 0.21,
        "residual sugar": 1.6,
        "chlorides": 0.106,
        "free sulfur dioxide": 10.0,
        "total sulfur dioxide": 37.0,
        "density": 0.9966,
        "pH": 3.17,
        "sulphates": 0.91,
        "alcohol": 9.5,
    },
}


st.set_page_config(
    page_title="ai_wineQualitypredictor",
    page_icon=":wine_glass:",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data
def load_data():
    return pd.read_csv(DATA_FILE)


def prepare_features(dataframe):
    feature_frame = dataframe[FEATURES].copy()
    feature_frame["total_acidity"] = (
        feature_frame["fixed acidity"] + feature_frame["volatile acidity"] + feature_frame["citric acid"]
    )
    feature_frame["fixed_to_volatile_ratio"] = feature_frame["fixed acidity"] / (
        feature_frame["volatile acidity"] + 0.01
    )
    feature_frame["sulfur_ratio"] = feature_frame["free sulfur dioxide"] / (
        feature_frame["total sulfur dioxide"] + 1.0
    )
    feature_frame["sulfur_load"] = feature_frame["total sulfur dioxide"] - feature_frame["free sulfur dioxide"]
    feature_frame["alcohol_density_ratio"] = feature_frame["alcohol"] / feature_frame["density"]
    feature_frame["sulphates_chlorides_ratio"] = feature_frame["sulphates"] / (
        feature_frame["chlorides"] + 0.001
    )
    feature_frame["residual_sugar_alcohol_ratio"] = feature_frame["residual sugar"] / (
        feature_frame["alcohol"] + 0.01
    )
    feature_frame["ph_sulphates_interaction"] = feature_frame["pH"] * feature_frame["sulphates"]
    return feature_frame


@st.cache_resource
def train_model(dataframe):
    cleaned_dataframe = dataframe.drop_duplicates().reset_index(drop=True)
    features = prepare_features(cleaned_dataframe)
    score_target = cleaned_dataframe["quality"]
    verdict_target = (cleaned_dataframe["quality"] >= GOOD_QUALITY_THRESHOLD).astype(int)
    x_train, x_test, y_score_train, y_score_test, y_verdict_train, y_verdict_test = train_test_split(
        features,
        score_target,
        verdict_target,
        test_size=0.2,
        random_state=SEED,
        stratify=score_target,
    )

    score_model = VotingClassifier(
        estimators=[
            (
                "rf",
                RandomForestClassifier(
                    n_estimators=350,
                    random_state=SEED,
                    class_weight="balanced_subsample",
                    max_depth=14,
                    min_samples_leaf=2,
                    min_samples_split=6,
                ),
            ),
            (
                "lgbm",
                LGBMClassifier(
                    n_estimators=350,
                    learning_rate=0.05,
                    num_leaves=31,
                    max_depth=-1,
                    subsample=0.85,
                    colsample_bytree=0.85,
                    random_state=SEED,
                    class_weight="balanced",
                    verbosity=-1,
                ),
            ),
            (
                "et",
                ExtraTreesClassifier(
                    n_estimators=350,
                    random_state=SEED,
                    class_weight="balanced",
                    max_depth=14,
                    min_samples_leaf=2,
                    min_samples_split=6,
                ),
            ),
        ],
        voting="soft",
        weights=[2, 3, 2],
    )
    verdict_model = VotingClassifier(
        estimators=[
            (
                "rf",
                RandomForestClassifier(
                    n_estimators=400,
                    random_state=SEED,
                    class_weight="balanced",
                    max_depth=12,
                    min_samples_leaf=2,
                    min_samples_split=6,
                ),
            ),
            (
                "et",
                ExtraTreesClassifier(
                    n_estimators=400,
                    random_state=SEED,
                    class_weight="balanced",
                    max_depth=12,
                    min_samples_leaf=2,
                    min_samples_split=6,
                ),
            ),
            (
                "lgbm",
                LGBMClassifier(
                    n_estimators=300,
                    learning_rate=0.05,
                    num_leaves=31,
                    max_depth=-1,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    random_state=SEED,
                    class_weight="balanced",
                    verbosity=-1,
                ),
            ),
        ],
        voting="soft",
        weights=[2, 2, 3],
    )

    score_cv_scores = cross_val_score(score_model, x_train, y_score_train, cv=3, scoring="accuracy")
    verdict_cv_scores = cross_val_score(verdict_model, x_train, y_verdict_train, cv=3, scoring="accuracy")

    score_model.fit(x_train, y_score_train)
    verdict_model.fit(x_train, y_verdict_train)

    score_test_predictions = score_model.predict(x_test)
    verdict_test_predictions = verdict_model.predict(x_test)

    metrics = {
        "score_cv_accuracy": score_cv_scores.mean(),
        "score_test_accuracy": accuracy_score(y_score_test, score_test_predictions),
        "verdict_cv_accuracy": verdict_cv_scores.mean(),
        "verdict_test_accuracy": accuracy_score(y_verdict_test, verdict_test_predictions),
        "training_rows": len(cleaned_dataframe),
        "quality_range": f"{int(score_target.min())} - {int(score_target.max())}",
        "score_model_name": "LightGBM Ensemble",
        "verdict_model_name": "LightGBM Ensemble",
    }
    return score_model, verdict_model, metrics


def quality_status(is_good_quality):
    return "Good Quality" if is_good_quality else "Needs Improvement"


def build_input_form(dataframe, container):
    defaults = SAMPLE_PROFILES[st.session_state.sample_profile]
    input_left, input_right = container.columns(2, gap="medium")
    feature_inputs = {}
    for index, feature in enumerate(FEATURES):
        min_value = float(dataframe[feature].min())
        max_value = float(dataframe[feature].max())
        median_value = float(dataframe[feature].median())
        value = float(defaults.get(feature, median_value))
        column = input_left if index < 6 else input_right
        feature_inputs[feature] = column.number_input(
            feature.title(),
            min_value=min_value,
            max_value=max_value,
            value=value,
            step=(max_value - min_value) / 100 if max_value != min_value else 0.1,
            format="%.4f",
        )
    return feature_inputs


def main():
    dataframe = load_data()
    score_model, verdict_model, metrics = train_model(dataframe)

    st.markdown(
        """
        <style>
        .stApp {
            background:
                radial-gradient(circle at top left, #fff4d6 0%, transparent 30%),
                radial-gradient(circle at top right, #f3c4a2 0%, transparent 25%),
                linear-gradient(180deg, #fffaf3 0%, #f7efe6 100%);
            color: #2f1b12;
        }
        .stApp, .stApp p, .stApp label, .stApp span, .stApp li {
            color: #2f1b12;
        }
        .stApp h1, .stApp h2, .stApp h3 {
            color: #4a1223;
        }
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #fff7ec 0%, #f6e8d8 100%);
            border-right: 1px solid rgba(95, 15, 64, 0.08);
        }
        div[data-testid="stMetric"] {
            background: rgba(255, 255, 255, 0.92);
            border: 1px solid rgba(95, 15, 64, 0.08);
            border-radius: 18px;
            padding: 0.8rem 1rem;
            box-shadow: 0 12px 30px rgba(95, 15, 64, 0.08);
        }
        div[data-testid="stMetricLabel"] > div,
        div[data-testid="stMetricValue"] > div {
            color: #4a1223;
        }
        div[data-baseweb="input"] > div,
        div[data-baseweb="select"] > div {
            background: rgba(255, 255, 255, 0.96);
            color: #2f1b12;
            border-radius: 14px;
        }
        div[data-baseweb="input"] input {
            color: #2f1b12 !important;
        }
        div[data-testid="stDataFrame"],
        div[data-testid="stTable"] {
            background: rgba(255, 255, 255, 0.96);
            border-radius: 18px;
            padding: 0.35rem;
            box-shadow: 0 12px 30px rgba(95, 15, 64, 0.08);
        }
        div[data-testid="stAlert"] {
            border-radius: 18px;
        }
        .hero {
            padding: 1.6rem 1.8rem;
            border-radius: 24px;
            background: linear-gradient(135deg, #5f0f40 0%, #9a031e 55%, #fb8b24 100%);
            color: white;
            box-shadow: 0 18px 50px rgba(95, 15, 64, 0.22);
            margin-bottom: 1rem;
        }
        .hero h1 {
            margin-bottom: 0.3rem;
        }
        .subtle {
            opacity: 0.9;
            font-size: 1rem;
        }
        .info-card {
            background: rgba(255, 255, 255, 0.88);
            border: 1px solid rgba(95, 15, 64, 0.08);
            border-radius: 20px;
            padding: 1rem 1.1rem;
            box-shadow: 0 12px 30px rgba(95, 15, 64, 0.08);
            margin-bottom: 1rem;
        }
        </style>
        <div class="hero">
            <h1>ai_wineQualitypredictor</h1>
            <p class="subtle">A startup-style wine quality intelligence dashboard for quick grading, confidence scoring, and production guidance.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if "sample_profile" not in st.session_state:
        st.session_state.sample_profile = "Startup Demo"

    st.sidebar.header("Input Studio")
    st.sidebar.selectbox(
        "Load a sample profile",
        options=list(SAMPLE_PROFILES.keys()),
        key="sample_profile",
    )
    st.sidebar.caption("Adjust the chemistry values to simulate a new wine batch.")

    left, right = st.columns([1.2, 1.0], gap="large")

    with left:
        st.subheader("Batch Inputs")
        st.markdown(
            """
            <div class="info-card">
                <strong>Batch Inputs</strong><br>
                Enter the chemistry values for the wine batch you want to score.
            </div>
            """,
            unsafe_allow_html=True,
        )
        feature_inputs = build_input_form(dataframe, left)
        predict_clicked = st.button("Predict Wine Quality", type="primary", use_container_width=True)

    with right:
        st.subheader("Model Snapshot")
        metric_a, metric_b, metric_c, metric_d = st.columns(4)
        metric_a.metric("Score CV Accuracy", f"{metrics['score_cv_accuracy']:.2%}")
        metric_b.metric("Score Test Accuracy", f"{metrics['score_test_accuracy']:.2%}")
        metric_c.metric("Training Rows", f"{metrics['training_rows']}")
        metric_d.metric("Quality Range", metrics["quality_range"])
        st.markdown(
            f"""
            <div class="info-card">
                Score model: <strong>{metrics['score_model_name']}</strong><br>
                Verdict model: <strong>{metrics['verdict_model_name']}</strong><br>
                Verdict CV Accuracy: <strong>{metrics['verdict_cv_accuracy']:.2%}</strong><br>
                Verdict Test Accuracy: <strong>{metrics['verdict_test_accuracy']:.2%}</strong>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if predict_clicked:
        input_frame = prepare_features(pd.DataFrame([feature_inputs]))
        predicted_quality = int(score_model.predict(input_frame)[0])
        verdict_probability = float(verdict_model.predict_proba(input_frame)[0][1])
        is_good_quality = verdict_probability >= 0.5
        quality_bucket = quality_status(is_good_quality)
        verdict_confidence = verdict_probability if is_good_quality else 1.0 - verdict_probability

        result_left, result_mid, result_right = st.columns(3)
        result_left.metric("Predicted Score", predicted_quality)
        result_mid.metric("Quality Verdict", quality_bucket)
        result_right.metric("Prediction Confidence", f"{verdict_confidence:.1%}")

        if is_good_quality:
            st.success(
                f"This batch is predicted to be {quality_bucket.lower()} with {verdict_confidence:.1%} confidence."
            )
        else:
            st.warning(
                f"This batch is predicted to {quality_bucket.lower()} with {verdict_confidence:.1%} confidence."
            )

if __name__ == "__main__":
    main()
