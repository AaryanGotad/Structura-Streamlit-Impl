import streamlit as st

import os
import zipfile
from pathlib import Path

import tensorflow as tf
from tensorflow.keras import layers
import keras_hub

import utilities.utils as utilities

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
        char_vocab = [line.strip() for line in f if line.strip()]

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
    # Initialize model
    try:
        model = load_structura_model()
    except Exception as exc:
        st.error(f"Failed to load model: {exc}")
        st.stop()       

# -----------------( TITLE & HEADING )-----------------
st.markdown(
    """
        <h1 
            style='text-align: center;
                   font-size: 4rem;
                   letter-spacing: 0.03rem;
                   font-family: "Copperplate";
                   text-decoration: underline;
                   margin-bottom: 0.05em;'>
            Structura
        </h1>
    """,
    unsafe_allow_html=True)

st.markdown(
    """
        <h2 
            style='text-align: center;
                    font-size: 1.95rem;
                    letter-spacing: 0.15rem;
                    font-family: "Roboto";'>
            Understand what the Research is About
        </h2>
    """,
    unsafe_allow_html=True)

# -----------------( INPUT )-----------------
def clear_text():
    """
    Clears the text area input by setting its value to an empty string.
    """
    st.session_state['abstract_input'] = ''

st.text_area(
    label="Paste Abstract Here",
    placeholder="Paste abstract of a research paper here...",
    height=250,
    key="abstract_input"
)

st.button("Clear", on_click=clear_text)

# -----------------( OUTPUT PIPELINE )-----------------
if not st.session_state.get('abstract_input', '').strip():
    st.info('Paste an abstract to structure it.')
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

output = utilities.output_formatting(model_pred_probs, abstract_lines)

st.divider()

st.markdown(
    """
        <h3 
            style='text-align: center;
                    font-size: 1.5rem;
                    letter-spacing: 0.15rem;
                    font-family: "Roboto";'>
            We Think The Research is About
        </h3>
    """,
    unsafe_allow_html=True)

st.divider()

utilities.pretty_output(output)

with st.expander('Alternate Output'):
    second_best_output = utilities.output_formatting(model_pred_probs, abstract_lines, k=2)
    utilities.pretty_output(second_best_output)
