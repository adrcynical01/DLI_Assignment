# streamlit_app.py
import numpy as np
import pandas as pd
import streamlit as st
from tensorflow.keras.models import load_model

st.set_page_config(page_title="Phishing Website Detector", page_icon="🔒", layout="wide")
st.title("🔒 Phishing Website Detector")
st.caption("Kaggle: Phishing Website Detector — MLP model")

# --- Feature schema (matches your notebook) ---
FEATURES = [
    'UsingIP', 'LongURL', 'ShortURL', 'Symbol@', 'Redirecting//', 'PrefixSuffix-', 'SubDomains',
    'HTTPS', 'DomainRegLen', 'Favicon', 'NonStdPort', 'HTTPSDomainURL', 'RequestURL', 'AnchorURL',
    'LinksInScriptTags', 'ServerFormHandler', 'InfoEmail', 'AbnormalURL', 'WebsiteForwarding',
    'StatusBarCust', 'DisableRightClick', 'UsingPopupWindow', 'IframeRedirection', 'AgeofDomain',
    'DNSRecording', 'WebsiteTraffic', 'PageRank', 'GoogleIndex', 'LinksPointingToPage', 'StatsReport'
]

@st.cache_resource
def load_keras_model(path: str):
    return load_model(path)

st.sidebar.header("⚙️ Model")
model_file = st.sidebar.file_uploader("Upload trained Keras model (.keras)", type=["keras"])
default_path = st.sidebar.text_input("…or local path", value="phishing_model.keras")

model = None
if model_file is not None:
    # Save uploaded model to a temporary file
    import tempfile, os
    tmpdir = tempfile.mkdtemp()
    mdl_path = os.path.join(tmpdir, "phishing_model.keras")
    with open(mdl_path, "wb") as f:
        f.write(model_file.read())
    model = load_keras_model(mdl_path)
else:
    # try default path
    try:
        model = load_keras_model(default_path)
    except Exception as e:
        st.warning("No model loaded yet. Upload a `.keras` file or fix the path in the sidebar.")
        st.stop()

st.success("Model loaded ✅")

st.divider()
col1, col2 = st.columns([1,1])

with col1:
    st.subheader("🧪 Single Prediction")
    st.caption("All features in this dataset are typically encoded as integers (−1, 0, 1). -1 = legitimate 0 = Suspicious/Unsure 1 = phishy")
    inputs = {}
    for feat in FEATURES:
        inputs[feat] = st.select_slider(
            feat, options=[-1, 0, 1], value=0,
            help="Encoded feature value (−1, 0, 1)"
        )

    if st.button("Predict", type="primary"):
        x = np.array([[inputs[f] for f in FEATURES]], dtype=np.float32)
        prob = float(model.predict(x, verbose=0)[0][0])
        label01 = 1 if prob >= 0.5 else 0       # your notebook maps final to {0,1}
        label_txt = "PHISHING" if label01 == 1 else "LEGITIMATE"

        st.metric("Probability (phishing)", f"{prob:.3f}")
        st.metric("Predicted label", label_txt)

        st.info("Threshold = 0.50. You can adjust thresholding in code if needed.")

with col2:
    st.subheader("📦 Batch Prediction (CSV)")
    st.caption("Upload CSV with **exactly these 30 columns** (in any order):")
    st.code(", ".join(FEATURES), language="text")

    up = st.file_uploader("Upload feature CSV", type=["csv"], key="csv")
    if up is not None:
        try:
            df = pd.read_csv(up)
            missing = [c for c in FEATURES if c not in df.columns]
            extra = [c for c in df.columns if c not in FEATURES]
            if missing:
                st.error(f"Missing columns: {missing}")
            else:
                X = df[FEATURES].astype(float).values
                probs = model.predict(X, verbose=0).ravel()
                labels = (probs >= 0.5).astype(int)
                out = df.copy()
                out["prob_phishing"] = probs
                out["pred_label"] = np.where(labels == 1, "PHISHING", "LEGITIMATE")
                st.dataframe(out.head(20))
                st.download_button("⬇️ Download predictions", out.to_csv(index=False), "predictions.csv", "text/csv")
                if extra:
                    st.warning(f"Ignored extra columns: {extra}")
        except Exception as e:
            st.exception(e)

st.divider()
