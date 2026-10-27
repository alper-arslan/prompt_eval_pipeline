"""Streamlit UI for the prompt eval workflow.

Run with: uv run streamlit run app.py
"""
import glob
import json
import os
from statistics import mean

import pandas as pd
import streamlit as st
from anthropic import AuthenticationError, PermissionDeniedError

from client import model
from dataset_generator import DEFAULT_DATASET_PROMPT, generate_dataset
from main import DEFAULT_PROMPT_TEMPLATE, RUNS_DIR, run_eval, save_run

FORMATS = ["json", "python", "regex"]

st.set_page_config(page_title="Prompt Eval", page_icon="🧪", layout="wide")


def load_dataset(path):
    """Return the test cases in a dataset file, or None if it can't be read."""
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def load_runs():
    """Return all saved runs, newest first."""
    runs = []
    for path in sorted(glob.glob(os.path.join(RUNS_DIR, "*.json")), reverse=True):
        try:
            with open(path) as f:
                run = json.load(f)
            run["name"] = os.path.basename(path)
            runs.append(run)
        except (OSError, json.JSONDecodeError):
            continue
    return runs


def format_averages(results):
    """Return {format: average score} for the formats present in the results."""
    return {
        fmt: mean(r["score"] for r in results if r["test_case"]["format"] == fmt)
        for fmt in FORMATS
        if any(r["test_case"]["format"] == fmt for r in results)
    }


def results_table(results):
    """Return a DataFrame with one row per scored case."""
    return pd.DataFrame(
        {
            "Format": r["test_case"]["format"],
            "Task": r["test_case"]["task"],
            "Score": r["score"],
            "Model": r.get("model_score"),
            "Syntax": r.get("syntax_score"),
        }
        for r in results
    )


def show_table(results):
    """Render the results table with score bars."""
    st.dataframe(
        results_table(results),
        hide_index=True,
        width="stretch",
        column_config={
            "Score": st.column_config.ProgressColumn("Score", min_value=0, max_value=10, format="%.1f"),
            "Model": st.column_config.NumberColumn("Model", format="%.1f", help="Grader score, 1-10"),
            "Syntax": st.column_config.NumberColumn("Syntax", format="%d", help="10 if valid, else 0"),
            "Task": st.column_config.TextColumn("Task", width="large"),
        },
    )


def show_summary(results, total=None):
    """Render the headline metrics for a set of results."""
    averages = format_averages(results)
    cols = st.columns(2 + len(averages))
    cols[0].metric("Average score", f"{mean(r['score'] for r in results):.2f}")
    cols[1].metric("Cases scored", f"{len(results)}/{total or len(results)}")
    for col, (fmt, avg) in zip(cols[2:], averages.items()):
        col.metric(fmt.capitalize(), f"{avg:.2f}")


def show_details(results):
    """Render one expander per case, lowest scores first."""
    st.subheader("Case details")
    st.caption("Lowest scores first")
    for r in sorted(results, key=lambda r: r["score"]):
        case = r["test_case"]
        with st.expander(f"{r['score']:.1f} · {case['format']} · {case['task'][:90]}"):
            st.markdown(f"**Task:** {case['task']}")
            st.markdown(f"**Criteria:** {case['solution_criteria']}")
            st.markdown("**Output**")
            st.code(r["output"].strip(), language=case["format"] if case["format"] != "regex" else "text")
            st.markdown(f"**Grader reasoning:** {r['reasoning']}")
            good, bad = st.columns(2)
            good.markdown("**Strengths**\n" + "\n".join(f"- {s}" for s in r.get("strengths", [])))
            bad.markdown("**Weaknesses**\n" + "\n".join(f"- {w}" for w in r.get("weaknesses", [])))


def reset_template():
    """Restore the default prompt template in the editor."""
    st.session_state["template"] = DEFAULT_PROMPT_TEMPLATE


def reset_dataset_prompt():
    """Restore the default dataset-generation prompt in the editor."""
    st.session_state["dataset_prompt"] = DEFAULT_DATASET_PROMPT


def set_draft(cases):
    """Replace the dataset draft shown in the editor, discarding unsaved edits."""
    st.session_state["draft"] = pd.DataFrame(cases, columns=["task", "format", "solution_criteria"])
    st.session_state.pop("draft_editor", None)


def clean_cases(draft):
    """Return the draft as a list of case dicts, dropping blank rows, or raise ValueError."""
    cases = [row for row in draft.fillna("").to_dict("records") if row["task"].strip()]
    for i, case in enumerate(cases, start=1):
        if case["format"] not in FORMATS:
            raise ValueError(f"Case {i} needs a format of json, python or regex.")
        if not case["solution_criteria"].strip():
            raise ValueError(f"Case {i} needs solution criteria.")
    if not cases:
        raise ValueError("There are no cases to save.")
    return cases


st.session_state.setdefault("template", DEFAULT_PROMPT_TEMPLATE)
st.session_state.setdefault("dataset_prompt", DEFAULT_DATASET_PROMPT)
if "draft" not in st.session_state:
    set_draft(load_dataset("dataset.json") or [])
has_key = bool(os.environ.get("ANTHROPIC_API_KEY"))

with st.sidebar:
    st.title("🧪 Prompt Eval")
    st.caption("Test a prompt against a dataset and see how it scores.")
    st.divider()
    dataset_path = st.text_input("Dataset", "dataset.json")
    dataset = load_dataset(dataset_path)
    if dataset is None and not os.path.exists(dataset_path):
        st.info("No file there yet. Create one in the Dataset tab.")
    elif dataset is None:
        st.error("Could not read that dataset file.")
    else:
        mix = {fmt: sum(1 for c in dataset if c["format"] == fmt) for fmt in FORMATS}
        st.write(f"**{len(dataset)} cases** · " + " · ".join(f"{n} {fmt}" for fmt, n in mix.items()))
    workers = st.slider("Parallel requests", 1, 8, 4, help="Higher is faster but may hit rate limits.")
    st.divider()
    st.caption(f"Model: `{model}`")
    if has_key:
        st.caption("✅ API key found")
    else:
        st.error("ANTHROPIC_API_KEY is not set. Export it or add it to a .env file, then restart.")

run_tab, dataset_tab, history_tab = st.tabs(["Run", "Dataset", "History"])

with run_tab:
    editor, output = st.columns([2, 3], gap="large")

    with editor:
        st.subheader("Prompt")
        st.text_area(
            "Prompt template",
            key="template",
            height=320,
            label_visibility="collapsed",
            help="Use {task} and {format}; they are filled in for each test case.",
        )
        st.caption("Use `{task}` and `{format}` as placeholders.")
        missing = "{task}" not in st.session_state["template"]
        if missing:
            st.warning("The template needs a `{task}` placeholder.")
        run_col, reset_col = st.columns([3, 2])
        run_clicked = run_col.button(
            "Run eval",
            type="primary",
            width="stretch",
            disabled=not has_key or dataset is None or missing,
        )
        reset_col.button("Reset prompt", on_click=reset_template, width="stretch")

    with output:
        st.subheader("Results")
        if run_clicked:
            bar = st.progress(0.0, text=f"Running 0/{len(dataset)}...")

            def on_progress(done, total, _result):
                bar.progress(done / total, text=f"Running {done}/{total}...")

            try:
                results = run_eval(dataset, st.session_state["template"], on_progress, workers)
            except (AuthenticationError, PermissionDeniedError) as e:
                bar.empty()
                st.error(f"Authentication failed: {e}")
            else:
                bar.empty()
                if results:
                    path = save_run(results, st.session_state["template"])
                    st.session_state["last_run"] = {"results": results, "total": len(dataset), "path": path}
                else:
                    st.error("No test cases were scored. Check the terminal for errors.")

        last = st.session_state.get("last_run")
        if last:
            show_summary(last["results"], last["total"])
            show_table(last["results"])
            st.caption(f"Saved to `{last['path']}`")
        else:
            st.info("Edit the prompt on the left and press **Run eval**.")

    if st.session_state.get("last_run"):
        show_details(st.session_state["last_run"]["results"])

with dataset_tab:
    prompt_col, cases_col = st.columns([2, 3], gap="large")

    with prompt_col:
        st.subheader("Dataset prompt")
        st.text_area(
            "Dataset prompt",
            key="dataset_prompt",
            height=380,
            label_visibility="collapsed",
            help="Use {count} for the number of cases. The reply must be a JSON array.",
        )
        st.caption("Use `{count}` as a placeholder. Each case needs `task`, `format` and `solution_criteria`.")
        count = st.number_input("Number of cases", 1, 30, 15, help="Capped at 30 to fit in one reply.")
        gen_col, reset_col = st.columns([3, 2])
        generate_clicked = gen_col.button("Generate", type="primary", width="stretch", disabled=not has_key)
        reset_col.button("Reset prompt", on_click=reset_dataset_prompt, width="stretch", key="reset_dataset")

    with cases_col:
        st.subheader("Cases")
        if generate_clicked:
            with st.spinner("Generating..."):
                try:
                    set_draft(generate_dataset(st.session_state["dataset_prompt"], count))
                except (AuthenticationError, PermissionDeniedError) as e:
                    st.error(f"Authentication failed: {e}")
                except (json.JSONDecodeError, TypeError, KeyError):
                    st.error("The model's reply was not a valid list of cases. Try generating again.")
        st.caption("Edit cells, add rows at the bottom, or select a row and press Delete. Nothing is saved until you press Save.")
        edited = st.data_editor(
            st.session_state["draft"],
            key="draft_editor",
            num_rows="dynamic",
            hide_index=True,
            width="stretch",
            column_config={
                "task": st.column_config.TextColumn("Task", width="large", required=True),
                "format": st.column_config.SelectboxColumn("Format", options=FORMATS, required=True),
                "solution_criteria": st.column_config.TextColumn("Solution criteria", width="large", required=True),
            },
        )
        st.caption(f"Saving overwrites `{dataset_path}`.")
        if st.button("Save dataset", disabled=not dataset_path):
            try:
                cases = clean_cases(edited)
            except ValueError as e:
                st.error(str(e))
            else:
                with open(dataset_path, "w") as f:
                    json.dump(cases, f, indent=2)
                st.success(f"Saved {len(cases)} cases to `{dataset_path}`.")

with history_tab:
    runs = load_runs()
    if not runs:
        st.info("No saved runs yet. Every eval run is saved to the `runs/` folder.")
    else:
        st.subheader("Saved runs")
        st.dataframe(
            pd.DataFrame(
                {
                    "Run": r["name"].removesuffix(".json"),
                    "Average": r["average"],
                    "Cases": len(r["results"]),
                    "Prompt": r["template"].strip().replace("\n", " ")[:80],
                }
                for r in runs
            ),
            hide_index=True,
            width="stretch",
            column_config={"Average": st.column_config.NumberColumn("Average", format="%.2f")},
        )

        st.subheader("Compare two runs")
        names = [r["name"] for r in runs]
        left_col, right_col = st.columns(2)
        base_name = left_col.selectbox("Baseline", names, index=min(1, len(names) - 1))
        new_name = right_col.selectbox("New", names, index=0)
        base = next(r for r in runs if r["name"] == base_name)
        new = next(r for r in runs if r["name"] == new_name)

        st.metric("Average score", f"{new['average']:.2f}", delta=f"{new['average'] - base['average']:+.2f}")
        base_avgs, new_avgs = format_averages(base["results"]), format_averages(new["results"])
        compare = pd.DataFrame(
            {
                "Baseline": [base_avgs.get(f) for f in FORMATS],
                "New": [new_avgs.get(f) for f in FORMATS],
            },
            index=FORMATS,
        )
        st.bar_chart(compare, horizontal=False)

        with st.expander("Prompt templates"):
            a, b = st.columns(2)
            a.markdown("**Baseline**")
            a.code(base["template"], language="text")
            b.markdown("**New**")
            b.code(new["template"], language="text")
