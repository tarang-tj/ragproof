"""Streamlit dashboard for retrieval-query drift.

Paste a baseline window of queries (what you tested on) and a current window (recent
production traffic). The app embeds both and runs ragproof's drift detection so you
can see, at a glance, when your real queries have wandered away from your eval set --
the moment your retrieval metrics stop meaning anything.

    pip install ragproof[dashboard,benchmark]
    streamlit run ragproof/dashboard.py
"""
import numpy as np
import streamlit as st

from ragproof.drift import detect_drift

BASELINE = "When is the final exam?\nHow is the class graded?\nWhat textbook is required?\nWhen are office hours?"
CURRENT = "Can I use a calculator on the test?\nWhat is the attendance policy?\nWhen is the group project due?\nHow many quizzes are there?"


@st.cache_resource
def _embedder():
    from ragproof.benchmarks.beir import model2vec_embedder
    return model2vec_embedder()


def _lines(text):
    return [ln.strip() for ln in text.splitlines() if ln.strip()]


def main():
    st.set_page_config(page_title="ragproof drift", page_icon="📊")
    st.title("ragproof — retrieval query drift")
    st.caption("Detect when production queries drift away from your evaluation set.")

    col_a, col_b = st.columns(2)
    baseline_text = col_a.text_area("Baseline queries (your eval set)", BASELINE, height=180)
    current_text = col_b.text_area("Current queries (production)", CURRENT, height=180)

    if st.button("Check drift", type="primary"):
        baseline, current = _lines(baseline_text), _lines(current_text)
        if len(baseline) < 2 or len(current) < 2:
            st.warning("Give at least two queries in each window.")
            return
        embed = _embedder()
        report = detect_drift(np.asarray(embed(baseline)), np.asarray(embed(current)))

        m1, m2 = st.columns(2)
        m1.metric("PSI", f"{report['psi']:.3f}", help=f"alert threshold {report['psi_threshold']}")
        m2.metric("Centroid cosine distance", f"{report['centroid_cosine_distance']:.3f}")
        if report["drift_detected"]:
            st.error("Drift detected — your production traffic no longer matches your eval set. Re-evaluate.")
        else:
            st.success("No significant drift — your eval set still represents production.")


if __name__ == "__main__":
    main()
