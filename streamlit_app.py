"""Streamlit UI — primary application for predictions + persona explanations."""

from __future__ import annotations

import asyncio
import json
from datetime import date, datetime

import pandas as pd
import streamlit as st

from app.config import get_settings
from app.genai.service import explain_personas, generate_synthetic_json, summarize_template
from app.genai.text_repair import repair_text_to_english
from app.ml.inference_inputs import claim_row_dataframe, dataframe_row_to_claim
from app.ml.mlops_sim import (
    compare_retrain_holdout,
    raw_claims_from_eval_drift_csv,
    simulated_export_payload,
)
from app.ml.predict import load_metadata, predict_batch, top_feature_names
from app.schemas import ClaimInput, Persona

SETTINGS = get_settings()

_MLOPS_RETRAIN_REQUIRED_KEYS: frozenset[str] = frozenset(
    {
        "n_rows_base",
        "n_rows_drift",
        "n_rows_combined",
        "metrics_before",
        "metrics_after",
        "baseline_before",
        "baseline_after",
        "data_before_tooltip",
        "data_after_tooltip",
    }
)

# Exactly the same ordering as ``app.ml.features._DAMAGE_FLOAT`` (``other`` last).
_SYMPTOM_BINARY_FIELDS: tuple[str, ...] = (
    "turnOnOff",
    "touchScreen",
    "smashed",
    "frontCamera",
    "backCamera",
    "frontOrBackCamera",
    "audio",
    "mic",
    "buttons",
    "connection",
    "charging",
)
_OTHER_SYMPTOM_FIELD: str = "other"
_SYMPTOM_RADIO_OPTIONS: tuple[str, ...] = ("Yes", "No", "Unknown")


_MLOPS_RETRAIN_STATE_KEY = "mlops_retrain_result"


def _sanitize_mlops_retrain_session() -> None:
    """Drop retrain results from older runs / wrong shapes (avoids KeyError on every rerun)."""
    # Legacy session key collided with ``st.button(key="mlops_retrain")`` (widget keys own ``session_state``).
    st.session_state.pop("mlops_retrain", None)
    blob = st.session_state.get(_MLOPS_RETRAIN_STATE_KEY)
    if blob is None:
        return
    if not isinstance(blob, dict) or not _MLOPS_RETRAIN_REQUIRED_KEYS.issubset(blob.keys()):
        st.session_state.pop(_MLOPS_RETRAIN_STATE_KEY, None)


def _symptom_seed_radio_index(value: object) -> int:
    """Map stored float/None to index in ``_SYMPTOM_RADIO_OPTIONS`` (Yes=0, No=1, Unknown=2)."""
    if value is None:
        return 2
    xf = pd.to_numeric(value, errors="coerce")
    if pd.isna(xf):
        return 2
    return 0 if float(xf) >= 0.5 else 1


def _symptom_radio_to_model_value(choice: str) -> float | None:
    if choice == "Yes":
        return 1.0
    if choice == "No":
        return 0.0
    return None


@st.cache_data(show_spinner=True)
def load_claim_sheet(path_str: str) -> pd.DataFrame:
    df = pd.read_excel(path_str, engine="openpyxl")
    return df


def coerce_date_any(v):
    if v is None:
        return datetime.now().date()
    if isinstance(v, datetime):
        return v.date()
    if isinstance(v, date):
        return v
    try:
        return pd.Timestamp(v).normalize().date()
    except Exception:
        return datetime.now().date()


def parse_maybe_float(txt: str) -> float | None:
    s = txt.strip().lower()
    if s == "" or s == "nan":
        return None
    try:
        return float(txt)
    except ValueError:
        return None


def badge_for_band(band: str) -> str:
    return {
        "likely_approve": "🟢",
        "lean_approve": "🟡",
        "borderline": "🟠",
        "lean_decline": "🔴",
        "likely_decline": "⛔",
    }.get(band, "⚪")


def band_from_p(p: float) -> str:
    if p >= 0.75:
        return "likely_approve"
    if p >= 0.55:
        return "lean_approve"
    if p >= 0.45:
        return "borderline"
    if p >= 0.25:
        return "lean_decline"
    return "likely_decline"


def _select_or_text_input(
    label: str,
    seed_text: str | None,
    series: pd.Series | None,
    max_uniques: int,
    *,
    widget_key: str,
) -> str:
    """Prefer ``st.selectbox`` when the training sheet has few enough distinct strings (categorical UX)."""
    cur = (seed_text or "").strip()
    if series is not None and not series.empty:
        raw = series.dropna().astype(str).str.strip()
        raw = raw[raw.ne("")]
        uniques = sorted(raw.unique().tolist())
        if 0 < len(uniques) <= max_uniques:
            options = [""] + uniques
            try:
                idx = options.index(cur)
            except ValueError:
                idx = 0
            picked = st.selectbox(label, options=options, index=idx, key=widget_key)
            return str(picked).strip()
    return str(st.text_input(label, value=cur, key=widget_key)).strip()


def render_predict_form(seed: ClaimInput, df_all: pd.DataFrame, *, seed_row_index: int) -> ClaimInput:
    c1, c2, c3 = st.columns(3)
    excess = c1.number_input("excessFee", value=float(seed.excessFee or 0.0))
    rrp = c2.number_input("rrp", value=float(seed.rrp or 0.0))
    balance_rrp = c3.number_input("balanceRRP", value=float(seed.balanceRRP or 0.0))
    c4, c5 = st.columns(2)
    old_bal = c4.number_input("oldBalanceRRP", value=float(seed.oldBalanceRRP or 0.0))
    device_cost = c5.number_input("deviceCost", value=float(seed.deviceCost or 0.0))

    def _col(name: str) -> pd.Series | None:
        return df_all[name] if name in df_all.columns else None

    pname = _select_or_text_input(
        "productName",
        seed.productName,
        _col("productName"),
        90,
        widget_key="form_productName",
    )
    pdesc = st.text_area("productDesc", value=str(seed.productDesc or ""), key="form_productDesc")
    cov = _select_or_text_input(
        "coverage",
        seed.coverage,
        _col("coverage"),
        60,
        widget_key="form_coverage",
    )
    pcode = _select_or_text_input(
        "productCode",
        seed.productCode,
        _col("productCode"),
        90,
        widget_key="form_productCode",
    )

    d1, d2, d3 = st.columns(3)
    ps_start = d1.date_input("policyStartDate", value=coerce_date_any(seed.policyStartDate))
    ps_end = d2.date_input("policyEndDate", value=coerce_date_any(seed.policyEndDate))
    purch = d3.date_input("purchaseDate", value=coerce_date_any(seed.purchaseDate))

    e1, e2, e3 = st.columns(3)
    pol_stat = _select_or_text_input(
        "policyStatus",
        seed.policyStatus,
        _col("policyStatus"),
        40,
        widget_key="form_policyStatus",
    )
    retailer = _select_or_text_input(
        "retailerName",
        seed.retailerName,
        _col("retailerName"),
        120,
        widget_key="form_retailerName",
    )
    dtype = _select_or_text_input(
        "deviceType",
        seed.deviceType,
        _col("deviceType"),
        40,
        widget_key="form_deviceType",
    )
    model_txt = _select_or_text_input(
        "model",
        seed.model,
        _col("model"),
        160,
        widget_key="form_model",
    )

    ch1, ch2 = st.columns(2)
    channel_txt = _select_or_text_input(
        "channel",
        seed.channel,
        _col("channel"),
        30,
        widget_key="form_channel",
    )
    ctype_txt = _select_or_text_input(
        "claimType",
        seed.claimType,
        _col("claimType"),
        35,
        widget_key="form_claimType",
    )
    country_txt = _select_or_text_input(
        "country",
        seed.country,
        _col("country"),
        45,
        widget_key="form_country",
    )

    st.markdown(
        "Symptom flags — **Yes / No / Unknown** (model gets **1 / 0 / NaN**). "
        "**other** stays a free numeric field (dataset sometimes uses mixed encodings)."
    )
    dmg_vals: dict[str, float | None] = {}
    dmg_cols = st.columns(4)
    for idx, fname in enumerate(_SYMPTOM_BINARY_FIELDS):
        ridx = _symptom_seed_radio_index(getattr(seed, fname, None))
        with dmg_cols[idx % len(dmg_cols)]:
            pick = st.radio(
                fname,
                _SYMPTOM_RADIO_OPTIONS,
                index=ridx,
                horizontal=True,
                key=f"sym_r{seed_row_index}_{fname}",
            )
        dmg_vals[fname] = _symptom_radio_to_model_value(str(pick))

    fv_other = getattr(seed, _OTHER_SYMPTOM_FIELD, None)
    starter_other = ""
    if fv_other is not None:
        xf_o = pd.to_numeric(fv_other, errors="coerce")
        if pd.notna(xf_o):
            xv_o = float(xf_o)
            starter_other = str(int(xv_o)) if xv_o.is_integer() else str(xv_o)
    with dmg_cols[len(_SYMPTOM_BINARY_FIELDS) % len(dmg_cols)]:
        raw_other = st.text_input(
            _OTHER_SYMPTOM_FIELD,
            value=starter_other,
            key=f"sym_r{seed_row_index}_other",
        )
    dmg_vals[_OTHER_SYMPTOM_FIELD] = parse_maybe_float(raw_other)

    issue = st.text_area("issueDesc", value=str(seed.issueDesc or ""), height=100)

    return ClaimInput(
        excessFee=excess,
        rrp=rrp,
        balanceRRP=balance_rrp,
        oldBalanceRRP=old_bal,
        productName=pname or None,
        productDesc=pdesc or None,
        coverage=cov or None,
        productCode=pcode or None,
        policyStartDate=datetime.combine(ps_start, datetime.min.time()),
        policyEndDate=datetime.combine(ps_end, datetime.min.time()),
        purchaseDate=datetime.combine(purch, datetime.min.time()),
        policyStatus=pol_stat or None,
        retailerName=retailer or None,
        deviceType=dtype or None,
        make=str(seed.make or "") or None,
        model=model_txt or None,
        deviceCost=device_cost,
        relationship=str(seed.relationship or "") or None,
        channel=channel_txt or None,
        claimType=ctype_txt or None,
        country=country_txt or None,
        issueDesc=issue or None,
        **dmg_vals,
    )


st.set_page_config(page_title="Claim Approval Agent", layout="wide")
st.title("Claim approval agent")

st.caption(
    "Dataset `claim_use_case_dataset.xlsx` — **`status`** label: Completed ⇒ approved pathway, "
    "Declined ⇒ denied."
)

sidebar = st.sidebar
with sidebar:
    sidebar.markdown("### Training data path")
    data_path_input = sidebar.text_input("Excel (.xlsx)", value=str(SETTINGS.claim_data_xlsx.resolve()))

    try:
        df_all = load_claim_sheet(str(data_path_input))
        sidebar.success(f"Loaded {len(df_all)} rows")
    except Exception as exc:  # noqa: BLE001
        sidebar.error(str(exc))
        st.stop()

    seed_idx = sidebar.slider(
        "Seed defaults from Excel row index",
        0,
        max(0, len(df_all) - 1),
        min(20, len(df_all) - 1),
        help="Pre-fills the form from a persisted claim observation.",
    )

    meta_md = load_metadata()
    with sidebar.expander("Artifact metadata"):
        if meta_md:
            sidebar.json(meta_md)
        else:
            sidebar.write("`python -m app.ml.train --no-tune`")

    show_seed_json = sidebar.checkbox("Show JSON seed snapshot")

seed_claim = dataframe_row_to_claim(df_all, int(seed_idx))
if show_seed_json:
    st.sidebar.code(json.dumps(seed_claim.model_dump(mode="json"), indent=2, default=str), language="json")

rebuilt_claim = render_predict_form(seed_claim, df_all, seed_row_index=int(seed_idx))

tabs = st.tabs(["Predict + drivers", "Multi-persona explain", "Synthetic stress", "DB & MLOps Pipeline"])

_sanitize_mlops_retrain_session()

with tabs[0]:
    if st.button("Run prediction", type="primary", use_container_width=True, key="run_pred_btn"):
        frame = claim_row_dataframe(rebuilt_claim)
        preds, probs = predict_batch(frame)
        ap_prob = float(probs[0])
        is_pos = bool(int(preds[0]) == 1)
        feats = top_feature_names(14)
        band_txt = band_from_p(ap_prob)

        cols = st.columns(3)
        cols[0].metric("p(approve)", f"{ap_prob:.4f}")
        cols[1].metric("band", f"{badge_for_band(band_txt)} {band_txt}")
        cols[2].metric("predicted outcome", "Completed style" if is_pos else "Declined")

        tmpl = summarize_template(rebuilt_claim, ap_prob, is_pos, feats)
        with st.expander("Template synopsis"):
            st.write(tmpl)
        with st.expander("Top forest importances"):
            for nm, wt in feats:
                st.markdown(f"- **{nm}** `{wt:.4f}`")

with tabs[1]:
    if st.button("Generate personas", use_container_width=True, key="gen_exp_btn"):
        frame = claim_row_dataframe(rebuilt_claim)
        preds, probs = predict_batch(frame)
        ap_prob = float(probs[0])
        is_pos = bool(int(preds[0]) == 1)
        feats = top_feature_names(14)
        payload = {
            "approved": is_pos,
            "probability_approved": float(ap_prob),
            "risk_band": band_from_p(ap_prob),
        }
        with st.spinner("LLM personas (offline stubs unless API key configured) …"):
            outs = asyncio.run(
                explain_personas(
                    claim=rebuilt_claim,
                    personas=[Persona.customer, Persona.claims_adjuster],
                    model_payload=payload,
                    top_features=feats,
                )
            )
        for role, blob in outs.items():
            with st.container():
                st.subheader(role.replace("_", " "))
                st.write(blob)

with tabs[2]:
    st.caption(
        "**Synthetic stress** fabricates claim-shaped JSON + short narratives to probe denial patterns and "
        "borderline cases—useful for QA and storytelling, not for ground truth. With ``GEMINI_API_KEY`` the LLM "
        "fills this tab; otherwise you get deterministic stubs, or an automatic fallback if the model returns "
        "unparseable JSON."
    )
    sc_count = int(st.number_input("How many scenarios", min_value=1, max_value=15, value=4))
    focus = st.radio("Focus", ["denial_patterns", "borderline"], horizontal=True)

    if st.button("Produce synthetic vignettes", key="synth_go"):
        with st.spinner("Synthetic generator…"):
            raw, ration = asyncio.run(
                generate_synthetic_json(
                    n_scenarios=sc_count,
                    focus_value=str(focus),
                    deny_rate_hint=0.55,
                    narrative_words_max=220,
                )
            )
        st.info(ration)
        for i, block in enumerate(raw):
            st.subheader(f"Scenario {i + 1}")
            st.json(block)
            payload = block.get("structured_claim") if isinstance(block, dict) else None
            if isinstance(payload, dict):
                try:
                    syn_claim = ClaimInput.model_validate(payload)
                    frame = claim_row_dataframe(syn_claim)
                    _pred, prob = predict_batch(frame)
                    st.caption(f"Stress score with the currently loaded bundle: p(approve) ≈ {float(prob[0]):.4f}")
                except Exception as exc:  # noqa: BLE001
                    st.caption(f"Optional scoring skipped for this payload ({exc}).")


with tabs[3]:
    st.markdown("### Simulated DB fetch · in-memory retrain · S3 export")
    st.caption(
        "Drift rows are read from ``data/eval_drift_logs.csv`` using **only** ``feature_*`` columns plus labels "
        "from ``ground_truth_approved``. Base rows use ``data/claim_use_case_dataset_enriched.csv`` with enrichment "
        "columns stripped—matching the live Excel schema. **Retrain does not** overwrite ``artifacts/`` or persist "
        "the concatenated frame."
    )
    data_enriched = SETTINGS.project_root / "data" / "claim_use_case_dataset_enriched.csv"
    data_drift = SETTINGS.project_root / "data" / "eval_drift_logs.csv"

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### 1. Fetch (simulated DB → drift table)")
        if st.button("Fetch drift rows", use_container_width=True, key="mlops_fetch"):
            try:
                drift_df = raw_claims_from_eval_drift_csv(data_drift)
                st.session_state["mlops_fetched_n"] = len(drift_df)
                st.success(
                    f"Loaded **{len(drift_df)}** raw claim rows from `{data_drift.name}` "
                    f"(explanations / metadata columns dropped)."
                )
            except Exception as exc:  # noqa: BLE001
                st.error(f"Could not read drift CSV: {exc}")
        if st.session_state.get("mlops_fetched_n"):
            st.metric(
                "Last fetch row count",
                int(st.session_state["mlops_fetched_n"]),
                help=f"Source file: `{data_drift}` — values are ``feature_*`` cells only.",
            )

    with col2:
        st.markdown("#### 2. Retrain (memory-only metrics)")
        st.caption(
            f"Base: `{data_enriched.name}` · Drift: `{data_drift.name}`. "
            "Each side uses a fresh stratified 80/20 holdout with the same `--no-tune` random forest settings as "
            "`python -m app.ml.train --no-tune`."
        )
        if st.button("Run in-memory retrain comparison", use_container_width=True, type="primary", key="mlops_retrain_run_btn"):
            try:
                with st.spinner("Fitting two holdout evaluations…"):
                    st.session_state[_MLOPS_RETRAIN_STATE_KEY] = compare_retrain_holdout(
                        base_data_path=data_enriched,
                        drift_csv_path=data_drift,
                        tune=False,
                    )
                st.success("Compared base vs base+drift (nothing written to disk).")
            except Exception as exc:  # noqa: BLE001
                st.session_state.pop(_MLOPS_RETRAIN_STATE_KEY, None)
                st.error(f"Retrain simulation failed: {exc}")

    if _MLOPS_RETRAIN_STATE_KEY in st.session_state:
        r = st.session_state[_MLOPS_RETRAIN_STATE_KEY]
        prev_n = int(r["n_rows_base"])
        new_n = int(r["n_rows_combined"])
        drift_n = int(r["n_rows_drift"])
        st.markdown(
            f"**Training rows:** previous **{prev_n}** (base only) → **{new_n}** "
            f"after adding **{drift_n}** drift rows."
        )
        m_b, m_a = r["metrics_before"], r["metrics_after"]
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric(
                "ROC-AUC (before)",
                f"{m_b['roc_auc']:.4f}",
                help=r["data_before_tooltip"],
            )
        with c2:
            st.metric(
                "Accuracy (before)",
                f"{m_b['accuracy']:.4f}",
                help=r["data_before_tooltip"],
            )
        with c3:
            st.metric(
                "F1 (before)",
                f"{m_b['f1']:.4f}",
                help=r["data_before_tooltip"],
            )
        d1, d2, d3 = st.columns(3)
        with d1:
            st.metric(
                "ROC-AUC (after)",
                f"{m_a['roc_auc']:.4f}",
                help=r["data_after_tooltip"],
            )
        with d2:
            st.metric(
                "Accuracy (after)",
                f"{m_a['accuracy']:.4f}",
                help=r["data_after_tooltip"],
            )
        with d3:
            st.metric(
                "F1 (after)",
                f"{m_a['f1']:.4f}",
                help=r["data_after_tooltip"],
            )
        with st.expander("Baseline vs model (holdout)"):
            st.caption("Dummy classifier on the same split—for context only.")
            st.json(
                {
                    "baseline_before": r["baseline_before"],
                    "baseline_after": r["baseline_after"],
                }
            )

    st.markdown("#### 3. Export bundle (simulated S3 PUT)")
    if st.button("Simulate pushing artifacts to S3", use_container_width=True, key="mlops_s3"):
        summary = simulated_export_payload()
        st.json(summary)
        st.caption(
            "No AWS calls are made. Sizes reflect your local ``artifacts/approval_model*.`` files if present—train "
            "with `python -m app.ml.train --no-tune` first if you need non-empty uploads."
        )

    st.markdown("#### 4. Single-row save (still mock)")
    if st.button("Save current form as unlabeled DB row (mock)", use_container_width=True, key="mlops_save_row"):
        st.success("Recorded the edited ``ClaimInput`` JSON as a pending row—no real database in this demo.")

with st.expander("Text repair (encoding cleanup + optional Gemini → English)", expanded=False):
    st.caption(
        "Separate from prediction: fixes mojibake with `text_clean`/`ftfy`, then asks Gemini for JSON English "
        "(when `GEMINI_API_KEY` is set). See docs/FEATURE_ENGINEERING.md."
    )
    tr_issue = st.text_area("issueDesc (raw)", value=str(rebuilt_claim.issueDesc or ""), height=120, key="trepi")
    tr_pn = st.text_input("productName", value=str(rebuilt_claim.productName or ""), key="treppn")
    tr_pd = st.text_area("productDesc", value=str(rebuilt_claim.productDesc or ""), height=80, key="treppd")
    if st.button("Run /v1/repair-text flow", key="text_repair_go"):
        fixed, rat, backend = asyncio.run(
            repair_text_to_english(
                issue_desc=tr_issue or None,
                product_name=tr_pn or None,
                product_desc=tr_pd or None,
            )
        )
        st.write("**backend:**", backend)
        st.info(rat)
        st.json(
            {
                "issueDesc": fixed["issueDesc"],
                "productName": fixed["productName"],
                "productDesc": fixed["productDesc"],
            }
        )


st.divider()
st.markdown("Run `python -m app.ml.train [--no-tune]` after updating the workbook to refresh coefficients.")
