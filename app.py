import streamlit as st

import os
import zipfile
from pathlib import Path

import tensorflow as tf
from tensorflow.keras import layers
import keras_hub

import utilities.utils as utilities

# --------------------------------------------------
# UI / UX STYLING (CSS)
# --------------------------------------------------
st.set_page_config(page_title="Structura", layout="centered")

st.markdown(
    """
    <style>
    /* Base Theme & Typography */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@200;300;400;500&display=swap');

    .stApp {
        background-color: #0d0d0f;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        color: #e5e5ea;
    }

    /* STRUCTURA Wordmark */
    .structura-wordmark {
        text-align: center;
        font-size: 3.2rem;
        font-weight: 300;
        letter-spacing: 0.25em;
        color: #8da3ff; /* Elegant accent color */
        margin-bottom: 0.1rem;
        margin-top: 1rem;
    }
    
    .structura-tagline {
        text-align: center;
        font-size: 1.05rem;
        font-weight: 300;
        letter-spacing: 0.05em;
        color: #8e8e93;
        margin-bottom: 3.5rem;
    }

    /* Glassmorphism Input Area */
    .stTextArea textarea {
        background: rgba(25, 25, 25, 0.4) !important;
        backdrop-filter: blur(12px) !important;
        -webkit-backdrop-filter: blur(12px) !important;
        border-radius: 16px !important;
        border-top: 1px solid rgba(255, 255, 255, 0.2) !important;
        border-left: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.03) !important;
        border-bottom: 1px solid rgba(255, 255, 255, 0.03) !important;
        color: #ffffff !important;
        font-size: 1.05rem !important;
        line-height: 1.6 !important;
        padding: 1.2rem !important;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3) !important;
        transition: all 0.4s cubic-bezier(0.165, 0.84, 0.44, 1) !important;
    }
    .stTextArea textarea:focus {
        border-top: 1px solid rgba(141, 163, 255, 0.5) !important;
        border-left: 1px solid rgba(141, 163, 255, 0.3) !important;
        background: rgba(35, 35, 35, 0.5) !important;
        box-shadow: 0 8px 40px rgba(141, 163, 255, 0.1) !important;
    }

    /* Secondary Clear Button */
    .stButton>button {
        background: transparent !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 8px !important;
        color: #8e8e93 !important;
        font-weight: 400 !important;
        padding: 0.3rem 1rem !important;
        transition: all 0.3s ease !important;
        float: right;
    }
    .stButton>button:hover {
        background: rgba(255, 255, 255, 0.05) !important;
        color: #d1d1d6 !important;
        border-color: rgba(255, 255, 255, 0.2) !important;
    }

    /* Glassmorphism Output Cards */
    .glass-card {
        background: rgba(20, 20, 22, 0.5);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border-radius: 18px;
        /* Fading white edge: brightest top/left, receding to right/bottom */
        border-top: 1px solid rgba(255, 255, 255, 0.25);
        border-left: 1px solid rgba(255, 255, 255, 0.15);
        border-right: 1px solid rgba(255, 255, 255, 0.02);
        border-bottom: 1px solid rgba(255, 255, 255, 0.02);
        padding: 2rem;
        box-shadow: 0 12px 40px rgba(0, 0, 0, 0.4);
        margin-bottom: 2rem;
        margin-top: 1rem;
    }

    /* Continuous Prose Structure */
    .abstract-paragraph {
        font-size: 1.1rem;
        line-height: 1.75;
        margin-bottom: 1.5rem;
        color: #e5e5ea;
        font-weight: 300;
    }
    .abstract-paragraph:last-child {
        margin-bottom: 0;
    }

    /* Category Labels & Tooltips */
    .category-label {
        font-weight: 500;
        font-size: 0.95rem;
        letter-spacing: 0.08em;
        color: #8da3ff;
        text-transform: uppercase;
        position: relative;
        display: inline-block;
        cursor: pointer;
        margin-right: 0.5rem;
    }
    
    .category-label .tooltip-text {
        visibility: hidden;
        background: rgba(30, 30, 32, 0.85);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        color: #ffffff;
        text-align: center;
        border-radius: 8px;
        padding: 6px 12px;
        position: absolute;
        z-index: 100;
        bottom: 140%;
        left: 50%;
        transform: translateX(-50%) scale(0.9) translateY(8px);
        opacity: 0;
        transition: opacity 0.3s cubic-bezier(0.175, 0.885, 0.32, 1.275), transform 0.3s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        border-top: 1px solid rgba(255,255,255,0.2);
        border-left: 1px solid rgba(255,255,255,0.1);
        font-size: 0.8rem;
        letter-spacing: normal;
        font-weight: 400;
        white-space: nowrap;
        box-shadow: 0 8px 24px rgba(0,0,0,0.5);
    }
    
    .category-label:hover .tooltip-text {
        visibility: visible;
        opacity: 1;
        transform: translateX(-50%) scale(1) translateY(0);
    }

    /* Raw Output Popover Styling */
    div[data-testid="stPopoverBody"] {
        background: rgba(15, 15, 18, 0.45) !important;
        backdrop-filter: blur(20px) !important; /* Strong backdrop blur */
        -webkit-backdrop-filter: blur(20px) !important;
        border-top: 1px solid rgba(255, 255, 255, 0.25) !important;
        border-left: 1px solid rgba(255, 255, 255, 0.15) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.05) !important;
        border-bottom: 1px solid rgba(255, 255, 255, 0.05) !important;
        border-radius: 16px !important;
        box-shadow: 0 16px 48px rgba(0,0,0,0.6) !important;
        color: #fff !important;
    }
    
    /* Expander Styling */
    .stExpander {
        border: none !important;
        background: transparent !important;
    }
    .stExpander > summary {
        background: rgba(255, 255, 255, 0.03) !important;
        border-radius: 12px !important;
        border-top: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-left: 1px solid rgba(255, 255, 255, 0.05) !important;
        color: #8e8e93 !important;
        transition: background 0.3s ease !important;
    }
    .stExpander > summary:hover {
        background: rgba(255, 255, 255, 0.06) !important;
        color: #e5e5ea !important;
    }

    /* Header adjustments */
    .result-heading {
        text-align: center;
        font-size: 1.4rem;
        font-weight: 400;
        letter-spacing: 0.1em;
        color: #e5e5ea;
        margin-top: 3rem;
        margin-bottom: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# --------------------------------------------------
# CACHED MODEL LOADING & RECONSTRUCTION
# --------------------------------------------------
@st.cache_resource
def load_structura_model():
    """
    Extracts assets from the .keras archive, reconstructs the model architecture,
    and loads the trained weights directly from the internal .h5 file.
    """
    CURR_DIR = Path(__file__).resolve().parent
    keras_file_path = CURR_DIR / "model" / "structura.keras"
    extract_dir = CURR_DIR / "model_extract"
    
    os.makedirs(extract_dir, exist_ok=True)

    # 1. EXTRACT VOCABULARIES & WEIGHTS FROM ZIP
    with zipfile.ZipFile(keras_file_path, "r") as z:
        bert_vocab_path = z.extract(
            "assets/layers/bert_text_embedder_preprocessor/tokenizer/vocabulary.txt",
            extract_dir
        )
        char_vocab_path = z.extract(
            "assets/layers/text_vectorization/vocabulary.txt",
            extract_dir
        )
        weights_path = z.extract("model.weights.h5", extract_dir)

    # 2. RECONSTRUCT TOKEN BRANCH (MINILM)
    token_inputs = layers.Input(shape=(), dtype=tf.string, name='token_inputs')

    tokenizer = keras_hub.tokenizers.BertTokenizer(
        vocabulary=bert_vocab_path,
        lowercase=True,
        strip_accents=False,
        split=True,
        suffix_indicator="##",
        oov_token="[UNK]"
    )

    preprocessor = keras_hub.models.BertTextEmbedderPreprocessor(
        tokenizer=tokenizer,
        sequence_length=256,
        truncate="round_robin"
    )

    backbone = keras_hub.models.BertBackbone.from_preset("all_minilm_l6_v2_en")

    embedder = keras_hub.models.BertTextEmbedder(
        backbone=backbone,
        preprocessor=preprocessor,
        pooling_mode="mean",
        normalize=True,
        name="token_embedder"
    )
    embedder.trainable = False

    preprocessed_tokens = embedder.preprocessor(token_inputs)
    token_embeddings = embedder(preprocessed_tokens)
    token_outputs = layers.Dense(128, activation='relu', name='dense')(token_embeddings)
    token_model = tf.keras.Model(inputs=token_inputs, outputs=token_outputs, name='token_model')

    # 3. RECONSTRUCT CHARACTER BRANCH
    char_inputs = layers.Input(shape=(1,), dtype=tf.string, name='char_inputs')

    with open(char_vocab_path, "r", encoding="utf-8") as f:
        char_vocab = [line.strip() for line in f if line.strip() and line.strip() != "[UNK]"]

    char_vectorizer = layers.TextVectorization(
        name="text_vectorization",
        output_mode="int"
    )
    char_vectorizer.set_vocabulary(char_vocab)

    char_vectors = char_vectorizer(char_inputs)

    char_embed = layers.Embedding(
        input_dim=70, 
        output_dim=25,
        name="embedding"
    )
    char_embeddings = char_embed(char_vectors)

    char_bi_lstm = layers.Bidirectional(
        layers.LSTM(32),
        name="bidirectional"
    )(char_embeddings)

    char_model = tf.keras.Model(inputs=char_inputs, outputs=char_bi_lstm, name='char_model')

    # 4. RECONSTRUCT POSITIONAL BRANCHES
    line_number_inputs = layers.Input(shape=(15,), dtype=tf.int32, name='line_number_input')
    x = layers.Dense(32, activation='relu', name='dense_1')(line_number_inputs)
    line_number_model = tf.keras.Model(inputs=line_number_inputs, outputs=x, name='line_number_model')

    total_lines_inputs = layers.Input(shape=(20,), dtype=tf.int32, name='total_lines_input')
    y = layers.Dense(32, activation='relu', name='dense_2')(total_lines_inputs)
    total_line_model = tf.keras.Model(inputs=total_lines_inputs, outputs=y, name='total_line_model')

    # 5. COMBINE BRANCHES & CLASSIFICATION HEAD
    combined_embeddings = layers.Concatenate(
        name='tokne_char_hybrid_embedding'
    )([token_model.output, char_model.output])

    z = layers.Dense(256, activation='relu', name='dense_3')(combined_embeddings)
    z = layers.Dropout(0.5, name='dropout')(z)

    z = layers.Concatenate(
        name='token_char_positional_embedding'
    )([line_number_model.output, total_line_model.output, z])

    output_layer = layers.Dense(5, activation='softmax', name='output_layer')(z)

    # 6. REASSEMBLE FULL MODEL
    model_3_reconstructed = tf.keras.Model(
        inputs=[
            line_number_model.input,
            total_line_model.input,
            token_model.input,
            char_model.input
        ],
        outputs=output_layer,
        name='model_3'
    )

    # 7. RESTORE TRAINED WEIGHTS
    model_3_reconstructed.load_weights(weights_path, skip_mismatch=True)
    
    return model_3_reconstructed

# --------------------------------------------------
# MODEL INITIALIZATION
# --------------------------------------------------
left_col, middle_col, right_col = st.columns([1, 2, 1])

with middle_col:
    try:
        model = load_structura_model()
    except Exception as exc:
        st.error(f"Failed to load model: {exc}")
        st.stop()       

# -----------------( TITLE & HEADING )-----------------
st.markdown("<div class='structura-wordmark'>STRUCTURA</div>", unsafe_allow_html=True)
st.markdown("<div class='structura-tagline'>See what the research is about.</div>", unsafe_allow_html=True)

# -----------------( INPUT )-----------------
def clear_text():
    """Clears the text area input by setting its value to an empty string."""
    st.session_state['abstract_input'] = ''

st.text_area(
    label="Paste Abstract Here",
    placeholder="Paste abstract of a research paper here...",
    height=250,
    key="abstract_input",
    label_visibility="collapsed" 
)

st.button("Clear", on_click=clear_text)

# -----------------( OUTPUT PIPELINE )-----------------
if not st.session_state.get('abstract_input', '').strip():
    st.markdown("<div style='text-align: center; color: #555; margin-top: 3rem; font-weight: 300;'>Paste an abstract to begin structuring.</div>", unsafe_allow_html=True)
    st.stop()

# Preprocess input using utilities
abstract_line_numbers_one_hot, abstract_total_lines_one_hot, abstract_lines, abstract_chars = utilities.preprocess_text(st.session_state['abstract_input'])

if not abstract_lines:
    st.stop()

left_col, middle_col, right_col = st.columns([1, 2, 1])
with middle_col:
    with st.spinner('Structuring...'):
        with tf.device('/CPU:0'):
            # Pass inputs to the model for prediction
            model_pred_probs = model.predict(x=(
                abstract_line_numbers_one_hot,
                abstract_total_lines_one_hot,
                tf.constant(abstract_lines),
                tf.expand_dims(tf.constant(abstract_chars), axis=-1)
            ))

# Formatting output
output = utilities.output_formatting(model_pred_probs, abstract_lines)

# Display Primary Output
st.markdown("<div class='result-heading'>Structured Abstract</div>", unsafe_allow_html=True)
utilities.render_structured_abstract(output)

# Display Alternate Output
with st.expander('Alternate Structure'):
    second_best_output = utilities.output_formatting(model_pred_probs, abstract_lines, k=2)
    utilities.render_structured_abstract(second_best_output)