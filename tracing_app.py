import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

RUNS_PATH = Path("memory/runs.jsonl")
EPISODIC_PATH = Path("memory/episodic.jsonl")

st.set_page_config(page_title="Harness Traces", page_icon="🔍", layout="wide")


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


@st.cache_data(ttl=5)
def load_runs() -> pd.DataFrame:
    df = pd.DataFrame(read_jsonl(RUNS_PATH))
    if df.empty:
        return df
    df["started_at"] = pd.to_datetime(df["started_at"], utc=True)
    df["finished_at"] = pd.to_datetime(df["finished_at"], utc=True)
    df["duration_s"] = (df["finished_at"] - df["started_at"]).dt.total_seconds()
    df["tools"] = df["tool_calls"].apply(lambda t: ", ".join(t) if t else "—")
    df["prompt"] = df["user_message"].str.slice(0, 80)
    return df.sort_values("started_at", ascending=False).reset_index(drop=True)


@st.cache_data(ttl=5)
def load_episodes() -> dict[str, dict]:
    return {e["run_id"]: e for e in read_jsonl(EPISODIC_PATH) if "run_id" in e}


df = load_runs()
episodes = load_episodes()

st.title("Harness traces")
st.caption(f"Reading `{RUNS_PATH}`")

if df.empty:
    st.info("No runs logged yet. Run the harness and come back.")
    st.stop()

# ---- filters ----
with st.sidebar:
    st.header("Filters")
    models = st.multiselect("Model", sorted(df["model"].dropna().unique()))
    complexity = st.multiselect("Complexity", sorted(df["complexity"].dropna().unique()))
    all_tools = sorted({t for ts in df["tool_calls"] for t in ts})
    tools = st.multiselect("Used tool", all_tools)
    search = st.text_input("Search prompt")
    if st.button("Refresh"):
        st.cache_data.clear()
        st.rerun()

view = df
if models:
    view = view[view["model"].isin(models)]
if complexity:
    view = view[view["complexity"].isin(complexity)]
if tools:
    view = view[view["tool_calls"].apply(lambda ts: any(t in ts for t in tools))]
if search:
    view = view[view["user_message"].str.contains(search, case=False, na=False)]

if view.empty:
    st.warning("No runs match these filters.")
    st.stop()

# ---- KPIs ----
k1, k2, k3, k4 = st.columns(4)
k1.metric("Runs", len(view))
k2.metric("Total tokens", f"{int(view['total_tokens'].sum()):,}")
k3.metric("Avg tokens / run", f"{view['total_tokens'].mean():,.0f}")
k4.metric("Avg latency", f"{view['duration_s'].mean():.1f}s")


# ---- runs table ----
st.subheader("Runs")
table = view[
    ["started_at", "prompt", "model", "complexity", "tools", "total_tokens", "duration_s"]
]
event = st.dataframe(
    table,
    width="stretch",
    hide_index=True,
    on_select="rerun",
    selection_mode="single-row",
    column_config={
        "started_at": st.column_config.DatetimeColumn("Started", format="MMM D, HH:mm:ss"),
        "total_tokens": st.column_config.NumberColumn("Tokens"),
        "duration_s": st.column_config.NumberColumn("Duration (s)", format="%.2f"),
    },
)

# ---- run detail ----
rows = event.selection.rows
run = view.iloc[rows[0]] if rows else view.iloc[0]
st.subheader("Run detail")
st.caption(f"`{run['run_id']}`" + ("" if rows else " · latest run (select a row above)"))

st.markdown(f"**Prompt:** {run['user_message']}")
d1, d2, d3, d4, d5, d6 = st.columns(6)
d1.metric("Model", run["model"])
d2.metric("Input tokens", f"{int(run['input_tokens']):,}")
d3.metric("Output tokens", f"{int(run['output_tokens']):,}")
d4.metric("Groq time", f"{run['groq_time']:.2f}s")
d5.metric("LLM wall time", f"{run['wall_time']:.2f}s")
d6.metric("Run duration", f"{run['duration_s']:.2f}s")

spans = run.get("spans") if "spans" in run.index else None
if isinstance(spans, list) and spans:
    sp = pd.DataFrame(spans)
    sp["start"] = sp["start_offset_s"]
    sp["end"] = sp["start_offset_s"] + sp["duration_s"]
    fig = px.timeline(
        sp.assign(
            start=pd.to_datetime(sp["start"], unit="s"),
            end=pd.to_datetime(sp["end"], unit="s"),
        ),
        x_start="start",
        x_end="end",
        y="name",
        color="type",
    )
    fig.update_yaxes(autorange="reversed")
    st.plotly_chart(fig, width="stretch")
    st.dataframe(sp.drop(columns=["start", "end"]), width="stretch", hide_index=True)
else:
    st.caption("No per-step spans logged for this run yet, so only run totals are shown.")

if run["tool_calls"]:
    st.markdown("**Tools used:** " + " ".join(f"`{t}`" for t in run["tool_calls"]))

ep = episodes.get(run["run_id"])
if ep:
    st.markdown("**Memory distilled from this run**")
    st.write(ep.get("episode", ""))
    if ep.get("knowledge"):
        st.dataframe(pd.DataFrame(ep["knowledge"]), hide_index=True, width="stretch")
    rej = ep.get("guardrail_rejections", {})
    if any(rej.values()):
        st.warning(f"Guardrail rejections: {rej}")

with st.expander("Raw record"):
    st.json(json.loads(run.to_json(date_format="iso")))
