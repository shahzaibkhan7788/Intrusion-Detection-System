from __future__ import annotations

import ast
import json
import re
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_REPORTS_DIR = BASE_DIR / "reports"
RUNS_DIR = BASE_DIR / "attack_pruning_search" / "runs"
DEFAULT_RUN_NAME = "run_09"
CHART_TEMPLATE = "plotly_white"
PRIMARY = "#0f766e"
ACCENT = "#f97316"
HIGHLIGHT = "#14b8a6"


def style_page() -> None:
    st.markdown(
        """
        <style>
            .stApp {
                background:
                    radial-gradient(circle at top left, rgba(20,184,166,0.12), transparent 28%),
                    linear-gradient(180deg, #f7fffd 0%, #eefbf8 100%);
            }
            .block-container {
                padding-top: 1.5rem;
                padding-bottom: 2rem;
            }
            div[data-testid="stMetric"] {
                background: white;
                border: 1px solid rgba(15, 118, 110, 0.12);
                border-radius: 16px;
                padding: 0.9rem;
                box-shadow: 0 8px 24px rgba(15, 23, 42, 0.06);
            }
            .dashboard-banner {
                padding: 1.35rem 1.5rem;
                border-radius: 20px;
                background: linear-gradient(135deg, #0f766e 0%, #14b8a6 60%, #99f6e4 100%);
                color: white;
                box-shadow: 0 16px 40px rgba(15, 118, 110, 0.18);
                margin-bottom: 1rem;
            }
            .dashboard-banner h1 {
                margin: 0;
                font-size: 2rem;
            }
            .dashboard-banner p {
                margin: 0.35rem 0 0;
                font-size: 0.98rem;
                opacity: 0.92;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header() -> None:
    st.markdown(
        """
        <div class="dashboard-banner">
            <h1>TabNet Save History Analytics</h1>
            <p>Complete visualization for all generated Save_history artifacts: metrics,
            confusion matrix, class metrics, top confusions, histories, feature importance,
            data-preparation report, raw tables, pies, histograms, and derived analysis.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def build_file_map(reports_dir: Path, run_dir: Path | None = None, weights_dir: Path | None = None) -> dict[str, Path]:
    file_map = {
        "metrics": reports_dir / "All_metrics.json",
        "metrics_summary": reports_dir / "All_metrics_summary.csv",
        "confusion_matrix": reports_dir / "All_confusion_matrix.csv",
        "class_metrics": reports_dir / "All_class_metrics.csv",
        "top_confusions": reports_dir / "All_top_confusions.csv",
        "feature_importance": reports_dir / "All_feature_importance.csv",
        "finetune_summary": reports_dir / "All_finetune_history.csv",
        "finetune_live": reports_dir / "All_finetune_history_live.csv",
        "finetune_raw": reports_dir / "All_finetune_history_raw.json",
        "pretrain_summary": reports_dir / "All_pretrain_history.csv",
        "pretrain_live": reports_dir / "All_pretrain_history_live.csv",
        "pretrain_raw": reports_dir / "All_pretrain_history_raw.json",
        "data_preparation_report": reports_dir / "data_preparation_report.json",
    }
    if run_dir is not None:
        file_map["run_configuration"] = run_dir / "run_configuration.json"
    if weights_dir is not None:
        file_map["pretrain_weight"] = weights_dir / "All_pretrain.zip"
        file_map["finetune_weight"] = weights_dir / "All_finetune.zip"
    return file_map


def discover_run_directories() -> list[Path]:
    if not RUNS_DIR.exists():
        return []
    return sorted([path for path in RUNS_DIR.iterdir() if path.is_dir() and path.name.startswith("run_")])


def resolve_selected_sources() -> tuple[Path, Path | None, Path | None]:
    run_dirs = discover_run_directories()
    options = ["Default reports"] + [run_dir.name for run_dir in run_dirs] + ["Custom run path"]
    preferred_run = next((run_dir for run_dir in run_dirs if run_dir.name == DEFAULT_RUN_NAME), None)
    if preferred_run is not None:
        default_index = options.index(preferred_run.name)
    elif run_dirs:
        default_index = options.index(run_dirs[-1].name)
    else:
        default_index = 0
    selected_option = st.sidebar.selectbox("Dashboard Source", options, index=default_index)

    if selected_option == "Default reports":
        return DEFAULT_REPORTS_DIR, None, None

    if selected_option == "Custom run path":
        custom_default = str(preferred_run) if preferred_run else (str(run_dirs[-1]) if run_dirs else str(RUNS_DIR / DEFAULT_RUN_NAME))
        custom_input = st.sidebar.text_input("Run folder path", value=custom_default)
        custom_run_dir = Path(custom_input).expanduser()
        return custom_run_dir / "reports", custom_run_dir, custom_run_dir / "weights"

    selected_run_dir = next((run_dir for run_dir in run_dirs if run_dir.name == selected_option), None)
    if selected_run_dir is None:
        return DEFAULT_REPORTS_DIR, None, None
    return selected_run_dir / "reports", selected_run_dir, selected_run_dir / "weights"


@st.cache_data(show_spinner=False)
def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def raw_history_to_df(payload: dict) -> pd.DataFrame:
    if not payload:
        return pd.DataFrame()
    frame = pd.DataFrame(payload)
    if not frame.empty and "epoch" not in frame.columns:
        frame.insert(0, "epoch", range(len(frame)))
    return frame


def parse_trainer_config(raw_text: str) -> dict:
    if not raw_text or raw_text == "nan":
        return {}
    cleaned = re.sub(r"<class '([^']+)'>", r"'\1'", raw_text)
    try:
        parsed = ast.literal_eval(cleaned)
    except Exception:
        return {"raw_config": raw_text}
    return parsed if isinstance(parsed, dict) else {"raw_config": raw_text}


def trainer_table(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["parameter", "value"])
    config = parse_trainer_config(str(df.loc[0, "trainer"])) if "trainer" in df.columns else {}
    rows = [{"parameter": key, "value": stringify_value(value)} for key, value in config.items()]
    if "verbose" in df.columns:
        rows.append({"parameter": "verbose", "value": stringify_value(df.loc[0, "verbose"])})
    return pd.DataFrame(rows)


def stringify_value(value) -> str:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value)
    return str(value)


@st.cache_data(show_spinner=False)
def load_dashboard_data(reports_dir: str, run_dir: str | None, weights_dir: str | None) -> dict:
    reports_path = Path(reports_dir)
    run_path = Path(run_dir) if run_dir else None
    weights_path = Path(weights_dir) if weights_dir else None
    file_map = build_file_map(reports_path, run_path, weights_path)

    data: dict[str, object] = {
        "files": {},
        "reports_dir": reports_path,
        "run_dir": run_path,
        "weights_dir": weights_path,
    }
    for key, path in file_map.items():
        entry = {
            "path": path,
            "exists": path.exists(),
            "size_kb": round(path.stat().st_size / 1024, 2) if path.exists() else 0,
        }
        if path.exists():
            if path.suffix == ".csv":
                entry["data"] = load_csv(path)
            elif path.suffix == ".json":
                entry["data"] = load_json(path)
        data["files"][key] = entry

    metrics = data["files"]["metrics"].get("data", {}) if data["files"]["metrics"]["exists"] else {}
    feature_df = (
        data["files"]["feature_importance"].get("data", pd.DataFrame())
        if data["files"]["feature_importance"]["exists"]
        else pd.DataFrame()
    )
    metrics_summary = (
        data["files"]["metrics_summary"].get("data", pd.DataFrame())
        if data["files"]["metrics_summary"]["exists"]
        else pd.DataFrame()
    )
    confusion_matrix_df = (
        data["files"]["confusion_matrix"].get("data", pd.DataFrame())
        if data["files"]["confusion_matrix"]["exists"]
        else pd.DataFrame()
    )
    class_metrics_df = (
        data["files"]["class_metrics"].get("data", pd.DataFrame())
        if data["files"]["class_metrics"]["exists"]
        else pd.DataFrame()
    )
    top_confusions_df = (
        data["files"]["top_confusions"].get("data", pd.DataFrame())
        if data["files"]["top_confusions"]["exists"]
        else pd.DataFrame()
    )
    finetune_summary = (
        data["files"]["finetune_summary"].get("data", pd.DataFrame())
        if data["files"]["finetune_summary"]["exists"]
        else pd.DataFrame()
    )
    finetune_live = (
        data["files"]["finetune_live"].get("data", pd.DataFrame())
        if data["files"]["finetune_live"]["exists"]
        else pd.DataFrame()
    )
    finetune_raw = (
        raw_history_to_df(data["files"]["finetune_raw"].get("data", {}))
        if data["files"]["finetune_raw"]["exists"]
        else pd.DataFrame()
    )
    pretrain_summary = (
        data["files"]["pretrain_summary"].get("data", pd.DataFrame())
        if data["files"]["pretrain_summary"]["exists"]
        else pd.DataFrame()
    )
    pretrain_live = (
        data["files"]["pretrain_live"].get("data", pd.DataFrame())
        if data["files"]["pretrain_live"]["exists"]
        else pd.DataFrame()
    )
    pretrain_raw = (
        raw_history_to_df(data["files"]["pretrain_raw"].get("data", {}))
        if data["files"]["pretrain_raw"]["exists"]
        else pd.DataFrame()
    )
    data_preparation_report = (
        data["files"]["data_preparation_report"].get("data", {})
        if data["files"]["data_preparation_report"]["exists"]
        else {}
    )
    run_configuration = (
        data["files"]["run_configuration"].get("data", {})
        if "run_configuration" in data["files"] and data["files"]["run_configuration"]["exists"]
        else {}
    )

    data.update(
        {
            "metrics": metrics,
            "metrics_summary_df": metrics_summary,
            "confusion_matrix_df": prepare_confusion_csv(confusion_matrix_df),
            "class_metrics_df": class_metrics_df,
            "top_confusions_df": top_confusions_df,
            "feature_df": prepare_feature_importance(feature_df),
            "finetune_summary_df": finetune_summary,
            "finetune_config_df": trainer_table(finetune_summary),
            "finetune_live_df": finetune_live,
            "finetune_raw_df": finetune_raw,
            "pretrain_summary_df": pretrain_summary,
            "pretrain_config_df": trainer_table(pretrain_summary),
            "pretrain_live_df": pretrain_live,
            "pretrain_raw_df": pretrain_raw,
            "data_preparation_report": data_preparation_report,
            "run_configuration": run_configuration,
        }
    )
    return data


def prepare_confusion_csv(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    prepared = df.copy()
    if "true_label" in prepared.columns:
        prepared = prepared.set_index("true_label")
    return prepared


def prepare_feature_importance(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    prepared = df.copy().sort_values("importance", ascending=False).reset_index(drop=True)
    total = prepared["importance"].sum()
    prepared["rank"] = range(1, len(prepared) + 1)
    prepared["share_pct"] = prepared["importance"] / total * 100 if total > 0 else 0.0
    prepared["cumulative_pct"] = prepared["share_pct"].cumsum()
    if "feature_name" not in prepared.columns:
        prepared["feature_name"] = prepared["feature_idx"].apply(lambda value: f"Feature {value}")
    return prepared


def scalar_metrics_table(metrics: dict) -> pd.DataFrame:
    rows = []
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            rows.append({"metric": key, "value": float(value)})
    report = metrics.get("classification_report", {})
    for label in ["macro avg", "weighted avg"]:
        if isinstance(report.get(label), dict):
            for metric_name, metric_value in report[label].items():
                rows.append({"metric": f"{label}_{metric_name}", "value": metric_value})
    return pd.DataFrame(rows)


def metrics_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    prepared = df.copy()
    if "metric" in prepared.columns and "value" in prepared.columns:
        return prepared
    return pd.DataFrame(columns=["metric", "value"])


def class_counts_df(metrics: dict) -> pd.DataFrame:
    counts = metrics.get("class_counts_test", {})
    if not counts:
        return pd.DataFrame(columns=["Class", "Count"])
    return (
        pd.DataFrame({"Class": list(counts.keys()), "Count": list(counts.values())})
        .sort_values("Count", ascending=False)
        .reset_index(drop=True)
    )


def classification_tables(metrics: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    report = metrics.get("classification_report", {})
    if not report:
        empty = pd.DataFrame(columns=["label", "precision", "recall", "f1-score", "support"])
        return empty, empty

    rows = []
    for label, values in report.items():
        if isinstance(values, dict):
            row = {"label": label}
            row.update(values)
        else:
            row = {"label": label, "accuracy": values}
        rows.append(row)

    report_df = pd.DataFrame(rows)
    per_class = report_df[
        ~report_df["label"].isin(["accuracy", "macro avg", "weighted avg"])
    ].copy()
    summary = report_df[report_df["label"].isin(["accuracy", "macro avg", "weighted avg"])].copy()
    return per_class, summary


def confusion_df(metrics: dict) -> pd.DataFrame:
    matrix = metrics.get("confusion_matrix", [])
    labels = metrics.get("classes", [f"Class {index}" for index in range(len(matrix))])
    if not matrix:
        return pd.DataFrame()
    return pd.DataFrame(matrix, index=labels, columns=labels)


def normalized_confusion(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    row_totals = df.sum(axis=1).replace(0, float("nan"))
    return df.div(row_totals, axis=0).fillna(0)


def misclassification_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["True Class", "Predicted Class", "Count"])
    rows = []
    for true_label in df.index:
        for pred_label in df.columns:
            count = int(df.loc[true_label, pred_label])
            if true_label != pred_label and count > 0:
                rows.append(
                    {
                        "True Class": true_label,
                        "Predicted Class": pred_label,
                        "Count": count,
                    }
                )
    if not rows:
        return pd.DataFrame(columns=["True Class", "Predicted Class", "Count"])
    return pd.DataFrame(rows).sort_values("Count", ascending=False).reset_index(drop=True)


def numeric_columns(df: pd.DataFrame, skip: set[str] | None = None) -> list[str]:
    skip = skip or set()
    return [
        column
        for column in df.columns
        if column not in skip and pd.api.types.is_numeric_dtype(df[column])
    ]


def plot_heatmap(
    df: pd.DataFrame,
    title: str,
    color_scale: str = "Tealgrn",
    text_format: str = ".0f",
) -> go.Figure:
    figure = go.Figure(
        data=[
            go.Heatmap(
                z=df.values,
                x=df.columns.tolist(),
                y=df.index.tolist(),
                colorscale=color_scale,
                text=df.values,
                texttemplate=f"%{{text:{text_format}}}",
                hovertemplate="True: %{y}<br>Predicted: %{x}<br>Value: %{z}<extra></extra>",
            )
        ]
    )
    figure.update_layout(
        title=title,
        xaxis_title="Predicted label",
        yaxis_title="True label",
        template=CHART_TEMPLATE,
        height=520,
    )
    return figure


def plot_history_lines(
    df: pd.DataFrame,
    title: str,
    columns: list[str],
    x_col: str = "epoch",
) -> go.Figure:
    melted = df[[x_col] + columns].melt(id_vars=x_col, var_name="Metric", value_name="Value")
    figure = px.line(
        melted,
        x=x_col,
        y="Value",
        color="Metric",
        markers=True,
        template=CHART_TEMPLATE,
        title=title,
    )
    figure.update_layout(legend_title="Metric", hovermode="x unified")
    return figure


def plot_history_histogram(df: pd.DataFrame, column: str, title: str) -> go.Figure:
    figure = px.histogram(
        df,
        x=column,
        nbins=20,
        marginal="box",
        template=CHART_TEMPLATE,
        color_discrete_sequence=[ACCENT],
        title=title,
    )
    return figure


def render_sidebar(data: dict) -> None:
    return


def render_overview(data: dict) -> None:
    metrics = data["metrics"]
    counts_df = class_counts_df(metrics)
    feature_df = data["feature_df"]
    finetune_live = data["finetune_live_df"]
    pretrain_live = data["pretrain_live_df"]
    metrics_summary_df = metrics_summary_table(data["metrics_summary_df"])
    prep_report = data["data_preparation_report"]
    run_config = data["run_configuration"]

    total_classes = len(metrics.get("classes", []))
    total_samples = int(counts_df["Count"].sum()) if not counts_df.empty else 0

    col1, col2, col3 = st.columns(3)
    col1.metric("Macro F1", f"{metrics.get('test_f1_macro', 0):.4f}")
    col2.metric("Classes", total_classes)
    col3.metric("Test Samples", f"{total_samples:,}")

    col4, col5 = st.columns(2)
    col4.metric("Finetune Epochs", len(finetune_live))
    col5.metric("Pretrain Epochs", len(pretrain_live))

    if run_config:
        st.subheader("Run Configuration")
        c1, c2, c3 = st.columns(3)
        c1.metric("Run Index", run_config.get("run_index", "N/A"))
        c2.metric("Removed Attacks", len(run_config.get("removed_attacks", [])))
        c3.metric("Kept Attacks", len(run_config.get("kept_attacks", metrics.get("kept_attacks", []))))
        st.write(f"Strategy: `{run_config.get('strategy_used', 'N/A')}`")
        if run_config.get("removed_attacks"):
            st.write("Removed attacks:", run_config["removed_attacks"])

    if not counts_df.empty:
        left, right = st.columns(2)
        pie = px.pie(
            counts_df,
            names="Class",
            values="Count",
            hole=0.42,
            title="Test Class Distribution",
            template=CHART_TEMPLATE,
        )
        pie.update_traces(textposition="inside", textinfo="percent+label")
        left.plotly_chart(pie, use_container_width=True)

        bar = px.bar(
            counts_df,
            x="Class",
            y="Count",
            color="Count",
            color_continuous_scale="Tealgrn",
            title="Class Counts",
            template=CHART_TEMPLATE,
        )
        right.plotly_chart(bar, use_container_width=True)

        hist = px.histogram(
            counts_df,
            x="Count",
            nbins=min(10, max(len(counts_df), 1)),
            title="Histogram of Class Supports",
            template=CHART_TEMPLATE,
            color_discrete_sequence=[PRIMARY],
        )
        st.plotly_chart(hist, use_container_width=True)

    scalar_df = metrics_summary_df if not metrics_summary_df.empty else scalar_metrics_table(metrics)
    if not scalar_df.empty:
        st.subheader("Scalar Metrics Table")
        st.dataframe(
            scalar_df.reset_index(drop=True),
            use_container_width=True,
            hide_index=True,
        )

    if not feature_df.empty:
        st.subheader("Feature Importance Snapshot")
        st.dataframe(
            feature_df[["rank", "feature_name", "importance", "importance_pct"]].head(15),
            use_container_width=True,
            hide_index=True,
        )


def render_metrics_tab(data: dict) -> None:
    metrics = data["metrics"]
    counts_df = class_counts_df(metrics)
    per_class_df, summary_df = classification_tables(metrics)
    class_metrics_df = data["class_metrics_df"]
    conf_df = data["confusion_matrix_df"] if not data["confusion_matrix_df"].empty else confusion_df(metrics)
    mis_df = data["top_confusions_df"] if not data["top_confusions_df"].empty else misclassification_df(conf_df)

    if summary_df.empty and per_class_df.empty and conf_df.empty:
        st.warning("`All_metrics.json` was not found or is empty.")
        return

    if not summary_df.empty:
        st.subheader("Classification Summary")
        st.dataframe(summary_df, use_container_width=True, hide_index=True)

    if not per_class_df.empty:
        st.subheader("Per-Class Classification Report")
        st.dataframe(per_class_df, use_container_width=True, hide_index=True)

        score_cols = [column for column in ["precision", "recall", "f1-score"] if column in per_class_df.columns]
        if score_cols:
            melted = per_class_df[["label"] + score_cols].melt(
                id_vars="label", var_name="Metric", value_name="Score"
            )
            score_fig = px.bar(
                melted,
                x="label",
                y="Score",
                color="Metric",
                barmode="group",
                template=CHART_TEMPLATE,
                title="Per-Class Precision / Recall / F1",
            )
            st.plotly_chart(score_fig, use_container_width=True)

            heatmap_df = per_class_df.set_index("label")[score_cols]
            st.plotly_chart(
                plot_heatmap(heatmap_df, "Per-Class Score Heatmap", "YlGnBu", ".3f"),
                use_container_width=True,
            )

        if "support" in per_class_df.columns:
            support_fig = px.pie(
                per_class_df,
                names="label",
                values="support",
                title="Support Share by Class",
                template=CHART_TEMPLATE,
            )
            st.plotly_chart(support_fig, use_container_width=True)

    if not conf_df.empty:
        st.subheader("Confusion Matrix")
        left, right = st.columns(2)
        left.plotly_chart(
            plot_heatmap(conf_df, "Confusion Matrix", "Tealgrn", ".0f"),
            use_container_width=True,
        )
        right.plotly_chart(
            plot_heatmap(normalized_confusion(conf_df), "Normalized Confusion Matrix", "Blues", ".2%"),
            use_container_width=True,
        )
        st.dataframe(conf_df, use_container_width=True)

    if not class_metrics_df.empty:
        st.subheader("Attack Performance Diagnostics")
        left, right = st.columns(2)
        left.dataframe(class_metrics_df, use_container_width=True, hide_index=True)
        radar_source = class_metrics_df.head(15).copy()
        diag_fig = px.bar(
            radar_source,
            x="class_name",
            y=["precision", "recall", "f1_score"],
            barmode="group",
            template=CHART_TEMPLATE,
            title="Class Metrics from `All_class_metrics.csv`",
        )
        right.plotly_chart(diag_fig, use_container_width=True)

    if not mis_df.empty:
        st.subheader("Top Misclassifications")
        left, right = st.columns([1.1, 1.4])
        left.dataframe(mis_df.head(15), use_container_width=True, hide_index=True)
        mis_fig = px.bar(
            mis_df.head(15),
            x="count" if "count" in mis_df.columns else "Count",
            y="true_label" if "true_label" in mis_df.columns else "True Class",
            color="predicted_label" if "predicted_label" in mis_df.columns else "Predicted Class",
            orientation="h",
            template=CHART_TEMPLATE,
            title="Largest Off-Diagonal Errors",
        )
        right.plotly_chart(mis_fig, use_container_width=True)

    if not counts_df.empty:
        st.subheader("Class Count Table")
        st.dataframe(counts_df, use_container_width=True, hide_index=True)


def best_epoch_card(df: pd.DataFrame, priority: list[str], mode: str) -> tuple[str, str]:
    if df.empty:
        return "Best Epoch", "N/A"
    for column in priority:
        if column in df.columns:
            index = int(df[column].idxmax() if mode == "max" else df[column].idxmin())
            epoch = int(df.loc[index, "epoch"]) if "epoch" in df.columns else index
            value = df.loc[index, column]
            return f"Best {column}", f"epoch {epoch} ({value:.4f})"
    return "Best Epoch", "N/A"


def render_history_section(
    title: str,
    summary_df: pd.DataFrame,
    config_df: pd.DataFrame,
    live_df: pd.DataFrame,
    raw_df: pd.DataFrame,
    focus: str,
) -> None:
    st.subheader(title)
    metrics = data["metrics"] if False else None

    card1, card2, card3, card4 = st.columns(4)
    card1.metric("Live Rows", len(live_df))
    card2.metric("Raw Rows", len(raw_df))
    last_lr = live_df["lr"].iloc[-1] if not live_df.empty and "lr" in live_df.columns else 0
    card3.metric("Last Learning Rate", f"{last_lr:.6f}")
    best_label, best_value = best_epoch_card(
        live_df,
        ["valid_accuracy", "valid_f1_macro", "train_accuracy", "val_0_unsup_loss_numpy"],
        "max" if focus == "finetune" else "min",
    )
    card4.metric(best_label, best_value)

    left, right = st.columns(2)
    left.markdown("**Saved History CSV**")
    left.dataframe(summary_df, use_container_width=True, hide_index=True)
    right.markdown("**Parsed Trainer Configuration**")
    if config_df.empty:
        right.info("No trainer configuration is stored in the current history CSV format.")
    else:
        right.dataframe(config_df, use_container_width=True, hide_index=True)

    if not live_df.empty:
        st.markdown("**Live CSV Analytics**")
        x_col = "epoch" if "epoch" in live_df.columns else live_df.columns[0]
        numeric_live = numeric_columns(live_df, skip={x_col})
        loss_cols = [column for column in numeric_live if "loss" in column.lower()]
        score_cols = [
            column
            for column in numeric_live
            if any(token in column.lower() for token in ["acc", "auc", "f1", "precision", "recall", "average_precision"])
        ]

        if loss_cols:
            st.plotly_chart(
                plot_history_lines(live_df, f"{title} Loss Curves", loss_cols, x_col=x_col),
                use_container_width=True,
            )
        if score_cols:
            st.plotly_chart(
                plot_history_lines(live_df, f"{title} Score Curves", score_cols, x_col=x_col),
                use_container_width=True,
            )

        selectable = numeric_live[:]
        default_columns = selectable[: min(6, len(selectable))]
        selected = st.multiselect(
            f"{title} metrics to compare",
            options=selectable,
            default=default_columns,
            key=f"{focus}_metric_selector",
        )
        if selected:
            st.plotly_chart(
                plot_history_lines(live_df, f"{title} Selected Metrics", selected, x_col=x_col),
                use_container_width=True,
            )

        dist_col = st.selectbox(
            f"{title} histogram metric",
            options=selectable,
            index=0 if selectable else None,
            key=f"{focus}_histogram_selector",
        )
        if dist_col:
            hist_left, hist_right = st.columns(2)
            hist_left.plotly_chart(
                plot_history_histogram(live_df, dist_col, f"{title} Histogram: {dist_col}"),
                use_container_width=True,
            )
            box = px.box(
                live_df,
                y=dist_col,
                points="all",
                template=CHART_TEMPLATE,
                title=f"{title} Box Plot: {dist_col}",
                color_discrete_sequence=[HIGHLIGHT],
            )
            hist_right.plotly_chart(box, use_container_width=True)

        corr_cols = numeric_live[: min(12, len(numeric_live))]
        if corr_cols:
            corr = live_df[corr_cols].corr(numeric_only=True)
            st.plotly_chart(
                plot_heatmap(corr, f"{title} Correlation Heatmap", "RdBu", ".2f"),
                use_container_width=True,
            )

        st.dataframe(live_df, use_container_width=True)

    if not raw_df.empty:
        st.markdown("**Raw JSON Analytics**")
        x_col = "epoch" if "epoch" in raw_df.columns else raw_df.columns[0]
        numeric_raw = numeric_columns(raw_df, skip={x_col})
        if numeric_raw:
            st.plotly_chart(
                plot_history_lines(raw_df, f"{title} Raw JSON Trends", numeric_raw, x_col=x_col),
                use_container_width=True,
            )
            raw_pick = st.selectbox(
                f"{title} raw metric",
                options=numeric_raw,
                index=0,
                key=f"{focus}_raw_metric_selector",
            )
            st.plotly_chart(
                plot_history_histogram(raw_df, raw_pick, f"{title} Raw Histogram: {raw_pick}"),
                use_container_width=True,
            )
        st.dataframe(raw_df, use_container_width=True)


def render_feature_tab(data: dict) -> None:
    feature_df = data["feature_df"]
    if feature_df.empty:
        st.warning("`All_feature_importance.csv` was not found or is empty.")
        return

    top_n = st.slider("Top features to highlight", min_value=5, max_value=min(40, len(feature_df)), value=min(20, len(feature_df)))
    top_df = feature_df.head(top_n).copy()

    st.subheader("Feature Importance Table")
    st.dataframe(feature_df, use_container_width=True, hide_index=True)

    left, right = st.columns(2)
    bar = px.bar(
        top_df.sort_values("importance"),
        x="importance",
        y="feature_name",
        orientation="h",
        color="importance",
        color_continuous_scale="Tealgrn",
        template=CHART_TEMPLATE,
        title=f"Top {top_n} Feature Importance",
    )
    left.plotly_chart(bar, use_container_width=True)

    pie = px.pie(
        top_df,
        names="feature_name",
        values="share_pct",
        hole=0.45,
        template=CHART_TEMPLATE,
        title=f"Top {top_n} Feature Share",
    )
    right.plotly_chart(pie, use_container_width=True)

    hist_left, hist_right = st.columns(2)
    hist = px.histogram(
        feature_df,
        x="importance",
        nbins=20,
        marginal="rug",
        template=CHART_TEMPLATE,
        color_discrete_sequence=[PRIMARY],
        title="Histogram of Feature Importance Values",
    )
    hist_left.plotly_chart(hist, use_container_width=True)

    cumulative = px.area(
        feature_df,
        x="rank",
        y="cumulative_pct",
        template=CHART_TEMPLATE,
        title="Cumulative Importance Coverage",
    )
    cumulative.update_yaxes(title="Cumulative %")
    hist_right.plotly_chart(cumulative, use_container_width=True)

    scatter = px.scatter(
        feature_df,
        x="rank",
        y="importance",
        size="share_pct",
        hover_name="feature_name",
        template=CHART_TEMPLATE,
        title="Feature Rank vs Importance",
        color="importance",
        color_continuous_scale="Viridis",
    )
    st.plotly_chart(scatter, use_container_width=True)


def render_data_prep_tab(data: dict) -> None:
    report = data["data_preparation_report"]
    if not report:
        st.warning("`data_preparation_report.json` was not found or is empty.")
        return

    stats = report.get("global_statistics", {})
    columns = report.get("column_standardization", {})
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Files Processed", report.get("summary", {}).get("files_processed", 0))
    c2.metric("Total Rows", f"{stats.get('total_rows_after_filtering', 0):,}")
    c3.metric("Attack Samples", f"{stats.get('attack_samples', {}).get('count', 0):,}")
    c4.metric("Common Features", columns.get("common_feature_column_count", 0))

    label_dist = stats.get("label_distribution_after", {})
    if label_dist:
        label_df = (
            pd.DataFrame({"Class": list(label_dist.keys()), "Count": list(label_dist.values())})
            .sort_values("Count", ascending=False)
            .reset_index(drop=True)
        )
        left, right = st.columns(2)
        left.dataframe(label_df, use_container_width=True, hide_index=True)
        prep_fig = px.bar(
            label_df,
            x="Class",
            y="Count",
            color="Count",
            color_continuous_scale="Tealgrn",
            title="Global Label Distribution After Preparation",
            template=CHART_TEMPLATE,
        )
        right.plotly_chart(prep_fig, use_container_width=True)

    file_details = report.get("file_details", [])
    if file_details:
        detail_rows = [
            {
                "file_name": entry.get("file_name"),
                "rows": entry.get("rows"),
                "columns": entry.get("columns"),
                "rows_dropped_missing": entry.get("metadata", {}).get("rows_dropped_missing", 0),
                "non_numeric_dropped": len(entry.get("metadata", {}).get("non_numeric_dropped", [])),
                "constant_dropped": len(entry.get("metadata", {}).get("constant_or_zero_dropped", [])),
            }
            for entry in file_details
        ]
        st.subheader("Per-File Preparation Summary")
        st.dataframe(pd.DataFrame(detail_rows), use_container_width=True, hide_index=True)

    common_cols = columns.get("common_feature_columns", [])
    if common_cols:
        with st.expander("Common Standardized Feature Columns", expanded=False):
            st.write(common_cols)


def render_raw_files_tab(data: dict) -> None:
    st.subheader("All Raw File Previews")
    for key, entry in data["files"].items():
        path: Path = entry["path"]
        with st.expander(f"{path.name} ({key})", expanded=False):
            st.caption(f"Path: `{path}`")
            st.caption(f"Size: {entry['size_kb']} KB")
            if not entry["exists"]:
                st.warning("File not found.")
                continue

            payload = entry.get("data")
            if isinstance(payload, pd.DataFrame):
                st.dataframe(payload, use_container_width=True)
                st.download_button(
                    label=f"Download {path.name}",
                    data=payload.to_csv(index=False).encode("utf-8"),
                    file_name=path.name,
                    mime="text/csv",
                    key=f"download_{key}",
                )
            elif isinstance(payload, (dict, list)):
                st.json(payload)
                st.download_button(
                    label=f"Download {path.name}",
                    data=json.dumps(payload, indent=2).encode("utf-8"),
                    file_name=path.name,
                    mime="application/json",
                    key=f"download_{key}",
                )
            else:
                st.info("Preview is not available for this file type.")


def main() -> None:
    st.set_page_config(page_title="TabNet Save History Dashboard", layout="wide")
    style_page()
    render_header()

    selected_reports_dir, selected_run_dir, selected_weights_dir = resolve_selected_sources()
    if not selected_reports_dir.exists():
        st.error(f"Reports folder not found: `{selected_reports_dir}`")
        return

    data = load_dashboard_data(
        str(selected_reports_dir),
        str(selected_run_dir) if selected_run_dir else None,
        str(selected_weights_dir) if selected_weights_dir else None,
    )
    render_sidebar(data)

    tabs = st.tabs(
        [
            "Overview",
            "Metrics & Confusion",
            "Finetune History",
            "Pretrain History",
            "Feature Importance",
        ]
    )

    with tabs[0]:
        render_overview(data)
    with tabs[1]:
        render_metrics_tab(data)
    with tabs[2]:
        render_history_section(
            "Finetune History",
            data["finetune_summary_df"],
            data["finetune_config_df"],
            data["finetune_live_df"],
            data["finetune_raw_df"],
            focus="finetune",
        )
    with tabs[3]:
        render_history_section(
            "Pretrain History",
            data["pretrain_summary_df"],
            data["pretrain_config_df"],
            data["pretrain_live_df"],
            data["pretrain_raw_df"],
            focus="pretrain",
        )
    with tabs[4]:
        render_feature_tab(data)


if __name__ == "__main__":
    main()
