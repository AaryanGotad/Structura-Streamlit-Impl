from html import escape
import tensorflow as tf
from pathlib import Path
from spacy.lang.en import English
import re
import streamlit as st

class_names_path = Path(__file__).with_name('class_names.txt')

with class_names_path.open('r', encoding='utf=8') as file:
    class_names = file.read().splitlines()

def split_chars(text):
    """
    Splits the input text into individual characters and returns them as a space-separated string.
    """
    return " ".join(list(text))

def preprocess_text(text):
    """
    Preprocesses the input text by performing the following steps:
    1. Sentence tokenization using spaCy's sentencizer.
    2. Creating a list of dictionaries containing line number and total lines for each sentence.
    3. One-hot encoding the line numbers and total lines.
    4. Splitting each sentence into individual characters.

    Args:
    - text: Input text to be preprocessed.

    Returns:
    - abstract_line_numbers_one_hot: One-hot encoded line numbers.
    - abstract_total_lines_one_hot: One-hot encoded total lines.
    - abstract_lines: List of sentences.
    - abstract_chars: List of sentences split into individual characters.
    """
    # -----------( SENTENCIZER )-----------
    nlp = English() # setup English sentence parser
    sentencizer = nlp.add_pipe('sentencizer') # create sentencizer splitting pipeline object

    # creating a 'doc' of parsed sequences
    doc = nlp(text)
    abstract_lines = [str(sent) for sent in list(doc.sents)]

    # -----------( LINE NUMBER & TOTAL NO. OF LINES )-----------
    total_lines = len(abstract_lines)

    # going through each line in abstract and creating a list dictionaries
    # containing features for each line
    lines = []
    for i, line in enumerate(abstract_lines):
        line_dict = {}
        line_dict['text'] = str(line)
        line_dict['line_number'] = i
        line_dict['total_lines'] = total_lines - 1
        lines.append(line_dict)

    # -----------( ONE-HOT ENCODING L.N.s & T.L.s )-----------
    # getting all line_number values from text abstract and one-hot endoding them
    line_numbers = [line['line_number'] for line in lines]
    abstract_line_numbers_one_hot = tf.one_hot(line_numbers, depth=15)

    # getting all total_lines values from text abstract and one-hot encoding them
    total_lines_values = [line['total_lines'] for line in lines]
    abstract_total_lines_one_hot = tf.one_hot(total_lines_values, depth=20)

    # -----------( CHARACTER SPLITTING )-----------
    # splitting each line into individual characters
    abstract_chars = [split_chars(line) for line in abstract_lines]

    return abstract_line_numbers_one_hot, abstract_total_lines_one_hot, abstract_lines, abstract_chars

def output_formatting(model_pred_probs, abstract_lines, class_names=class_names, k=1):
    """
    Formats the model prediction probabilities into a structured output.
    Depending on the value of k, it either returns the top prediction or the kth best prediction
    for each line in the abstract.
    """
    if k == 1:
        # turning prediction probabilities into prediction classes
        abstract_preds = tf.argmax(model_pred_probs, axis=1)
    
        # turning prediction class integers into string class names
        abstract_pred_classes = [class_names[i] for i in abstract_preds]
    
        formatted_lines = []
        for i, line in enumerate(abstract_lines):
            formatted_line = {}
            formatted_line['label'] = abstract_pred_classes[i]
            formatted_line['text'] = line
            formatted_line['confidence probability'] = float(model_pred_probs[i][abstract_preds[i]])
            formatted_lines.append(formatted_line)
    
        return formatted_lines

    else:
        # getting the top k prediction probabilities and their corresponding class indices
        top_k_probs, top_k_indices = tf.math.top_k(model_pred_probs, k=k)

        kth_best_class_names = [class_names[i] for i in top_k_indices[:, k-1]]
        kth_best_probs = [float(top_k_probs[i][k-1]) for i in range(len(top_k_probs))]

        formatted_lines = []
        for i, line in enumerate(abstract_lines):        
            formatted_line = {}
            formatted_line['label'] = kth_best_class_names[i]
            formatted_line['text'] = line
            formatted_line['confidence probability'] = float(kth_best_probs[i])
            formatted_lines.append(formatted_line)

        return formatted_lines

def pretty_output(output):
    """
    Displays the formatted output in a user-friendly manner.
    """
    # Format the output for display
    pretty_output = {'BACKGROUND': [], 'OBJECTIVE': [], 'METHODS': [], 'RESULTS': [], 'CONCLUSIONS': [], 'OTHER': []}
    for line in output:
        pretty_output[line['label']].append(line)

    pretty_output = {k: v for k, v in pretty_output.items() if v}

    left_col, middle_col, right_col = st.columns([1, 2, 1])

    with right_col:
        # displaying raw model probability values
        with st.popover('Raw Model Outputs'):                    
            st.dataframe(output)

    for label, lines in pretty_output.items():
        text = re.sub(r"\s+", " ", " ".join(line['text'] for line in lines)).strip()
        mean_confidence = sum(line['confidence probability'] for line in lines) / len(lines)
        confidence_text = f"{mean_confidence:.2%}"

        st.markdown(
            f"""
                <h4 
                    style='text-align: left;
                            font-size: 1.25rem;
                            letter-spacing: 0.15rem;'>
                    <span title='Mean confidence: {confidence_text}'>
                        {escape(label)}
                    </span>: {escape(text)}
                </h4>
            """,
            unsafe_allow_html=True)
