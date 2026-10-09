# Structura

Structura is a Streamlit application that classifies the sentences in a
scientific abstract as background, objective, methods, results, or
conclusions. The interface includes the liquid-glass frontend design and
four ready-to-run research abstract samples.

The standalone frontend prototype source files are kept in
[frontend/](/workspaces/Structura/frontend): `index.html`, `styles.css`, and
`script.js`.

### Run locally

1. Create and set up the virtual environment.

   ```bash
   python3 -m venv .venv
   .venv/bin/python -m pip install --upgrade pip
   .venv/bin/python -m pip install -r requirements.txt
   ```

2. Start the app.

   ```bash
   .venv/bin/streamlit run app.py --server.address=0.0.0.0 --server.port=8501
   ```

Paste an abstract and select **Analyze abstract**, or choose **Try this** on
one of the sample cards to populate the editor. The raw model probabilities
and second-best sentence predictions are available below the primary result;
sample cards remain below the results so the analysis stays immediately
visible.
