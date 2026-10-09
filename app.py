import os
import zipfile
from html import escape
from pathlib import Path

import keras_hub
import streamlit as st
import tensorflow as tf
from tensorflow.keras import layers

import utilities.utils as utilities


BASE_DIR = Path(__file__).resolve().parent
st.set_page_config(page_title="Structura", layout="centered", initial_sidebar_state="collapsed")

SAMPLE_ABSTRACTS = [
    ("Orthopedics", "Chronic low back pain remains a major cause of disability worldwide, with limited long-term efficacy from standard physical therapies. We aimed to evaluate whether an 8-week mindfulness-based stress reduction (MBSR) program reduces pain intensity compared to usual care. A randomized, parallel-group trial was conducted with 240 adults aged 18 to 65 years diagnosed with persistent non-specific low back pain. Participants were randomly assigned 1:1 to either the MBSR group or a waitlist control receiving standard medical care. The primary outcome was the change in back pain intensity measured on a 10-point visual analog scale at 24 weeks. A total of 218 participants completed the final 24-week follow-up assessment. The MBSR group demonstrated a significant reduction in pain scores compared to the control group (mean difference -1.4 points; 95% CI, -1.9 to -0.9; P < 0.001). No serious adverse events associated with the intervention were reported during the trial period. An 8-week MBSR program provides clinically meaningful, sustained improvement in pain and functional limitations for patients with chronic low back pain."),
    ("Endocrinology", "The optimal macronutrient distribution for managing metabolic markers in recently diagnosed type 2 diabetes patients is still heavily debated. This study sought to compare the effects of a low-carbohydrate Mediterranean diet versus a standard low-fat diet on glycemic control. We designed a single-center, open-label, randomized controlled trial involving 180 treatment-naive adults. Over a 12-month period, the intervention group followed a low-carbohydrate Mediterranean diet, while the comparator group adhered to a low-fat diet. Hemoglobin A1c (HbA1c) levels and lipid profiles were evaluated at baseline, 6 months, and 12 months. At the 12-month mark, the Mediterranean diet group achieved a significantly greater reduction in HbA1c than the low-fat group (-1.2% vs -0.7%, P = 0.02). Additionally, high-density lipoprotein cholesterol levels increased more prominently in the low-carbohydrate cohort. Adherence to a low-carbohydrate Mediterranean diet leads to superior glycemic control and better lipid profiles than a conventional low-fat diet."),
    ("Pulmonology", "Pediatric asthma management heavily relies on consistent medication adherence, which is traditionally poor among school-aged children. We investigated whether an interactive mobile health application could improve inhaler adherence and reduce emergency department visits. A multi-center randomized controlled trial enrolled 350 children aged 7 to 12 with moderate-to-severe persistent asthma. Families were randomized to either use the gamified smartphone app linked to an electronic inhaler sensor or receive standard asthma education. Adherence data were automatically logged via electronic sensors, and emergency visits were tracked over 6 months. The intervention group achieved an average daily inhaler adherence rate of 78%, compared to 54% in the control group (P < 0.001). Emergency department visits dropped by 45% in the app group relative to the standard care group over the 6-month tracking window. Integrating mobile health applications with electronic trackers significantly boosts pediatric asthma adherence and lowers acute care utilization."),
    ("Robotics", "Autonomous vehicle navigation relies heavily on real-time object detection, but current deep learning models often struggle under heavy rain and foggy conditions. This paper introduces an adaptive feature-fusion network designed to maintain high object-detection accuracy across diverse adverse weather scenarios. We trained and evaluated our convolutional neural network architecture on a newly curated synthetic dataset containing 50,000 degraded driving images. The proposed model dynamically adjusts convolutional filter weights based on estimated ambient lighting and visibility metrics extracted from the input frame. Performance was benchmarked against three industry-standard architectures using mean Average Precision (mAP) calculations. The adaptive network achieved an mAP of 84.3% in heavy rain simulations, outperforming the baseline model by 11.5 percentage points. Processing speeds remained stable at 45 frames per second on standard edge-computing hardware. Dynamic feature-fusion networks offer a reliable and computationally efficient solution for enhancing autonomous vehicle safety in unpredictable environments."),
]


def inject_frontend_css() -> None:
    st.markdown(
        """<style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@200;300;400;500&display=swap');
        .stApp { background: #0d0d0f; color: #e5e5ea; font-family: Inter, sans-serif; }
        [data-testid="stAppViewContainer"] > .main { padding: 0; }
        [data-testid="stHeader"], [data-testid="stToolbar"] { background: transparent; }
        [data-testid="stTextArea"] label { display: none; }
        .structura-wordmark { text-align: center; font-size: 3.2rem; font-weight: 300;
            letter-spacing: 0.25em; color: #8da3ff; margin: 1rem 0 0.1rem; }
        .structura-tagline { text-align: center; font-size: 1.05rem; font-weight: 300;
            letter-spacing: 0.05em; color: #8e8e93; margin-bottom: 3.5rem; }
        [data-testid="stTextArea"] textarea { background: rgba(25,25,25,.4) !important;
            border-radius: 16px !important; border: 1px solid rgba(255,255,255,.1) !important;
            color: #fff !important; font-size: 1.05rem !important; line-height: 1.6 !important;
            padding: 1.2rem !important; box-shadow: 0 8px 32px rgba(0,0,0,.3) !important; }
        .stButton > button { background: transparent !important; border: 1px solid rgba(255,255,255,.1) !important;
            border-radius: 8px !important; color: #8e8e93 !important; font-weight: 400 !important;
            transition: all .3s ease !important; }
        .stButton > button:hover { background: rgba(255,255,255,.05) !important;
            color: #d1d1d6 !important; border-color: rgba(255,255,255,.2) !important; }
        .stButton > button[kind="primary"] { background: #8da3ff !important; color: #0d0d0f !important; }
        .glass-card { background: rgba(20,20,22,.5); border-radius: 18px; padding: 2rem;
            box-shadow: 0 12px 40px rgba(0,0,0,.4); margin: 1rem 0 2rem; }
        .abstract-paragraph { font-size: 1.1rem; line-height: 1.75; margin-bottom: 1.5rem;
            color: #e5e5ea; font-weight: 300; }
        .category-label { font-weight: 500; font-size: .95rem; letter-spacing: .08em;
            color: #8da3ff; text-transform: uppercase; position: relative; display: inline-block;
            margin-right: .5rem; }
        .tooltip-text { visibility: hidden; position: absolute; bottom: 140%; left: 50%;
            transform: translateX(-50%); background: #1e1e20; color: #fff; border-radius: 8px;
            padding: 6px 12px; font-size: .8rem; white-space: nowrap; z-index: 2; }
        .category-label:hover .tooltip-text { visibility: visible; }
        .result-heading { text-align: center; font-size: 1.4rem; font-weight: 400;
            letter-spacing: .1em; color: #e5e5ea; margin: 3rem 0 1rem; }
        .samples-section { margin-top: 4rem; }
        .sample-card { background: rgba(20,20,22,.5); border: 1px solid rgba(255,255,255,.08);
            border-radius: 16px; padding: 1.25rem; margin-bottom: 1rem; }
        .sample-topic { color: #8da3ff; font-weight: 500; letter-spacing: .08em;
            text-transform: uppercase; margin-bottom: .5rem; }
        .sample-preview { color: #8e8e93; font-size: .9rem; line-height: 1.5; margin-bottom: 1rem; }
        .limitations-section { margin-top: 3rem; padding: 1.5rem; color: #8e8e93; }
        .limitations-section h2 { color: #e5e5ea; font-size: 1.15rem; }
        </style>""",
        unsafe_allow_html=True,
    )


@st.cache_resource
def load_structura_model():
    keras_file_path = BASE_DIR / "model" / "structura.keras"
    extract_dir = BASE_DIR / "model_extract"
    os.makedirs(extract_dir, exist_ok=True)
    with zipfile.ZipFile(keras_file_path, "r") as archive:
        bert_vocab_path = archive.extract(
            "assets/layers/bert_text_embedder_preprocessor/tokenizer/vocabulary.txt",
            extract_dir,
        )
        char_vocab_path = archive.extract(
            "assets/layers/text_vectorization/vocabulary.txt", extract_dir
        )
        weights_path = archive.extract("model.weights.h5", extract_dir)

    token_inputs = layers.Input(shape=(), dtype=tf.string, name="token_inputs")
    tokenizer = keras_hub.tokenizers.BertTokenizer(
        vocabulary=bert_vocab_path,
        lowercase=True,
        strip_accents=False,
        split=True,
        suffix_indicator="##",
        oov_token="[UNK]",
    )
    preprocessor = keras_hub.models.BertTextEmbedderPreprocessor(
        tokenizer=tokenizer, sequence_length=256, truncate="round_robin"
    )
    backbone = keras_hub.models.BertBackbone.from_preset("all_minilm_l6_v2_en")
    embedder = keras_hub.models.BertTextEmbedder(
        backbone=backbone,
        preprocessor=preprocessor,
        pooling_mode="mean",
        normalize=True,
        name="token_embedder",
    )
    embedder.trainable = False
    token_embeddings = embedder(embedder.preprocessor(token_inputs))
    token_model = tf.keras.Model(
        token_inputs,
        layers.Dense(128, activation="relu", name="dense")(token_embeddings),
        name="token_model",
    )

    char_inputs = layers.Input(shape=(1,), dtype=tf.string, name="char_inputs")
    with open(char_vocab_path, encoding="utf-8") as file:
        char_vocab = [line.strip() for line in file if line.strip() and line.strip() != "[UNK]"]
    char_vectorizer = layers.TextVectorization(name="text_vectorization", output_mode="int")
    char_vectorizer.set_vocabulary(char_vocab)
    char_vectors = char_vectorizer(char_inputs)
    char_embeddings = layers.Embedding(input_dim=70, output_dim=25, name="embedding")(char_vectors)
    char_output = layers.Bidirectional(layers.LSTM(32), name="bidirectional")(char_embeddings)
    char_model = tf.keras.Model(char_inputs, char_output, name="char_model")

    line_inputs = layers.Input(shape=(15,), dtype=tf.int32, name="line_number_input")
    line_model = tf.keras.Model(line_inputs, layers.Dense(32, activation="relu", name="dense_1")(line_inputs))
    total_inputs = layers.Input(shape=(20,), dtype=tf.int32, name="total_lines_input")
    total_model = tf.keras.Model(total_inputs, layers.Dense(32, activation="relu", name="dense_2")(total_inputs))

    combined = layers.Concatenate(name="tokne_char_hybrid_embedding")(
        [token_model.output, char_model.output]
    )
    combined = layers.Dropout(0.5, name="dropout")(
        layers.Dense(256, activation="relu", name="dense_3")(combined)
    )
    combined = layers.Concatenate(name="token_char_positional_embedding")(
        [line_model.output, total_model.output, combined]
    )
    output_layer = layers.Dense(5, activation="softmax", name="output_layer")(combined)
    model = tf.keras.Model(
        [line_model.input, total_model.input, token_model.input, char_model.input],
        output_layer,
        name="model_3",
    )
    model.load_weights(weights_path, skip_mismatch=True)
    return model


def set_text(text: str) -> None:
    st.session_state.abstract_input = text
    st.session_state.analysis = None


def clear_text() -> None:
    st.session_state.abstract_input = ""
    st.session_state.analysis = None


def analyze() -> None:
    text = st.session_state.get("abstract_input", "").strip()
    if text:
        st.session_state.analysis = text


def render_grouped_output(formatted_output: list[dict]) -> None:
    groups = []
    for line in formatted_output:
        label = line.get("label", "OTHER").upper()
        if groups and groups[-1][0] == label:
            groups[-1][1].append(line)
        else:
            groups.append((label, [line]))
    html = '<div class="glass-card">'
    for label, lines in groups:
        text = " ".join(line["text"] for line in lines)
        confidence = sum(line.get("confidence probability", 0) for line in lines) / len(lines)
        html += (
            '<div class="abstract-paragraph">'
            f'<span class="category-label">{escape(label)}'
            f'<span class="tooltip-text">Mean confidence: {confidence:.2%}</span></span>'
            f'{escape(text)}</div>'
        )
    st.markdown(html + "</div>", unsafe_allow_html=True)


def render_raw_output(predictions, probabilities, lines) -> None:
    with st.popover("Raw output"):
        rows = []
        for index, (line, scores) in enumerate(zip(lines, probabilities), start=1):
            rows.append(
                {
                    "#": index,
                    "Assigned label": predictions[index - 1]["label"],
                    "Line": line,
                    "BACKGROUND": float(scores[0]),
                    "CONCLUSIONS": float(scores[1]),
                    "METHODS": float(scores[2]),
                    "OBJECTIVE": float(scores[3]),
                    "RESULTS": float(scores[4]),
                }
            )
        st.dataframe(rows, use_container_width=True, hide_index=True)


def main() -> None:
    inject_frontend_css()
    st.markdown(
        '<div class="container"><main><section class="hero">'
        '<div class="structura-wordmark">STRUCTURA</div>'
        '<div class="structura-tagline">See what the research is about.</div></section>',
        unsafe_allow_html=True,
    )

    st.text_area(
        "Paste scientific abstract",
        placeholder="Paste a scientific abstract here...",
        height=220,
        key="abstract_input",
        label_visibility="collapsed",
    )
    st.markdown('<p class="input-hint">Works best with scientific or medical research abstracts.</p>', unsafe_allow_html=True)
    analyze_col, clear_col = st.columns([3, 1])
    with analyze_col:
        st.button("Analyze abstract", on_click=analyze, type="primary", use_container_width=True)
    with clear_col:
        st.button("Clear", on_click=clear_text, use_container_width=True)

    analysis_text = st.session_state.get("analysis")
    if analysis_text:
        with st.spinner("Analyzing your abstract..."):
            model = load_structura_model()
            line_numbers, total_lines, lines, chars = utilities.preprocess_text(analysis_text)
            probabilities = model.predict(
                (line_numbers, total_lines, tf.constant(lines), tf.expand_dims(tf.constant(chars), axis=-1)),
                verbose=0,
            )
        primary = utilities.output_formatting(probabilities, lines)
        alternate = utilities.output_formatting(probabilities, lines, k=2)
        st.markdown('<section class="results-section"><div class="result-heading">Structured abstract</div>', unsafe_allow_html=True)
        render_raw_output(primary, probabilities, lines)
        render_grouped_output(primary)
        with st.expander("See alternative predictions"):
            render_grouped_output(alternate)
        st.markdown("</section>", unsafe_allow_html=True)

    st.markdown('<section class="samples-section"><div class="result-heading">Try a sample abstract</div>', unsafe_allow_html=True)
    for index, (topic, text) in enumerate(SAMPLE_ABSTRACTS):
        st.markdown(
            f'<div class="sample-card"><div class="sample-topic">{escape(topic)}</div>'
            f'<div class="sample-preview">{escape(text)}</div>',
            unsafe_allow_html=True,
        )
        st.button("Try this", key=f"sample_{index}", on_click=set_text, args=(text,))
        st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</section>", unsafe_allow_html=True)

    st.markdown(
        '<section class="limitations-section"><h2>Why might Structura get a line wrong?</h2>'
        '<p>Scientific language can be highly nuanced and sentence classification is not perfect. You might notice occasional misclassifications because:</p>'
        '<ul class="limitations-list"><li>Some sentences naturally contain information belonging to multiple categories.</li>'
        '<li>Not all scientific abstracts follow a rigid, predictable structure.</li>'
        '<li>The model was trained primarily on medical and life-science datasets.</li></ul></section>'
        '<footer><p class="footer-copy">&copy; 2026 Structura. Interface Prototype.</p></footer></main></div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
