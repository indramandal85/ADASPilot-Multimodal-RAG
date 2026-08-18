"""Minimal HTTP-only UI for ADASPilot.

Run the API first with ``make run-api``, then start this app with ``make run-ui``.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from typing import Any
from urllib.parse import quote

import httpx
import streamlit as st

API_URL = os.getenv("ADASPILOT_API_URL", "http://127.0.0.1:8000").rstrip("/")
TIMEOUT = httpx.Timeout(180.0, connect=10.0)


def get_json(path: str) -> dict[str, Any] | None:
    try:
        response = httpx.get(f"{API_URL}{path}", timeout=TIMEOUT)
        response.raise_for_status()
        return response.json()
    except httpx.HTTPError as exc:
        st.warning(f"API unavailable: {exc}")
        return None


@st.cache_data(ttl=300, show_spinner=False)
def get_citation(document_id: str, reference_number: int) -> dict[str, Any] | None:
    encoded = quote(document_id, safe="")
    try:
        response = httpx.get(
            f"{API_URL}/corpus/documents/{encoded}/citation",
            params={"ref": reference_number},
            timeout=TIMEOUT,
        )
        response.raise_for_status()
        return response.json()
    except httpx.HTTPError:
        return None


def stream_answer(
    question: str,
    mode: str,
    run_state: dict[str, Any],
) -> Iterator[str]:
    with httpx.stream(
        "POST",
        f"{API_URL}/query/stream",
        json={"question": question, "mode": mode},
        timeout=TIMEOUT,
    ) as response:
        response.raise_for_status()
        event = ""
        for line in response.iter_lines():
            if line.startswith("event: "):
                event = line.removeprefix("event: ")
            elif line.startswith("data: "):
                payload = json.loads(line.removeprefix("data: "))
                if event == "answer":
                    yield str(payload["text"])
                elif event == "timeline":
                    run_state.setdefault("timeline", []).append(payload)
                else:
                    run_state[event] = payload


def _document_label(document: dict[str, Any]) -> str:
    title = " ".join(str(document["title"]).split())
    display_title = title if len(title) <= 34 else f"{title[:31]}..."
    pages = document.get("page_count")
    return f"{display_title}  ·  {pages if pages is not None else '?'} pages"


@st.fragment(run_every=2.0)
def show_ingestion_progress() -> None:
    job = get_json("/corpus/ingest-status")
    if not job or job["status"] == "idle":
        return

    status = str(job["status"])
    previous = st.session_state.get("last_ingestion_status")
    st.session_state["last_ingestion_status"] = status
    if status == "running":
        st.info(job["message"], icon="⏳")
        return
    if status == "failed":
        st.error(job["message"])
    else:
        st.success(job["message"])
        if job.get("ingested"):
            st.caption(f"Added: {', '.join(job['ingested'])}")
        if job.get("reactivated"):
            st.caption(f"Reactivated: {', '.join(job['reactivated'])}")
    for error in job.get("errors", []):
        st.warning(error)

    if status == "succeeded" and previous == "running":
        st.session_state.pop("staged_upload_token", None)
        st.cache_data.clear()
        st.rerun()


def show_corpus_sidebar() -> None:
    with st.sidebar:
        st.header("Your Documents")
        status = get_json("/corpus/status")
        if status:
            st.caption(status["corpus_version"])
            active_documents = [
                item for item in status["documents"] if item["status"] != "inactive"
            ]
            st.caption(f"{len(active_documents)} documents · {status['page_count']} indexed pages")
            document_ids = [document["document_id"] for document in active_documents]
            selected_key = "selected_corpus_document"
            if st.session_state.get(selected_key) not in [None, *document_ids]:
                st.session_state[selected_key] = None
            by_id = {document["document_id"]: document for document in active_documents}
            with st.container(height=280, border=True):
                selected_id = st.radio(
                    "Corpus documents",
                    document_ids,
                    index=None,
                    format_func=lambda document_id: _document_label(by_id[document_id]),
                    key=selected_key,
                    label_visibility="collapsed",
                )
            if selected_id:
                selected = by_id[selected_id]
                st.caption(f"Selected: {selected['title']}")
                if st.button(
                    "Remove selected document",
                    key=f"del_{selected_id}",
                    use_container_width=True,
                    help="Deactivate this document without deleting its PDF from disk.",
                ):
                    try:
                        response = httpx.delete(
                            f"{API_URL}/corpus/documents/{quote(selected_id, safe='')}",
                            timeout=TIMEOUT,
                        )
                        response.raise_for_status()
                        st.cache_data.clear()
                        st.rerun()
                    except httpx.HTTPError as exc:
                        st.error(f"Remove failed: {exc}")
            if status["staged_upload_count"]:
                st.info(f"{status['staged_upload_count']} PDF(s) ready to process")

        st.divider()
        upload = st.file_uploader("Add a PDF", type=["pdf"])
        if upload is not None:
            upload_token = f"{upload.name}:{upload.size}"
            previous_token = st.session_state.get("staged_upload_token")
            if upload_token != previous_token:
                try:
                    response = httpx.post(
                        f"{API_URL}/corpus/upload",
                        files={"file": (upload.name, upload.getvalue(), "application/pdf")},
                        timeout=TIMEOUT,
                    )
                    response.raise_for_status()
                    result = response.json()
                    st.session_state["staged_upload_token"] = upload_token
                    if result["status"] == "already_staged":
                        st.info("This PDF is already staged.")
                    else:
                        st.success(f"Staged {upload.name}")
                except httpx.HTTPError as exc:
                    st.error(f"Upload failed: {exc}")

        if st.button("Process uploaded files", use_container_width=True):
            try:
                response = httpx.post(
                    f"{API_URL}/corpus/ingest-staged",
                    timeout=TIMEOUT,
                )
                response.raise_for_status()
                result = response.json()
                st.session_state["last_ingestion_status"] = result["status"]
                st.info("Processing started in the background. Progress appears below.")
            except httpx.HTTPError as exc:
                st.error(f"Processing failed: {exc}")

        show_ingestion_progress()


def show_evidence(payload: dict[str, Any]) -> None:
    evidence = payload.get("evidence", [])
    citations = payload.get("citations", [])
    if not evidence and not citations:
        return
    with st.expander("Evidence and citations", expanded=True):
        by_record = {item["record_id"]: item for item in evidence}
        for reference_number, citation in enumerate(citations, start=1):
            item = by_record.get(citation["record_id"])
            page_number = citation["page_number"]
            page_label = "Web source" if page_number == 0 else f"Page {page_number}"
            st.markdown(f"**[{reference_number}] {page_label}**")
            if item:
                passage = " ".join(item["text"].split())
                preview = passage if len(passage) <= 300 else f"{passage[:300]}..."
                st.caption(preview)
                if len(passage) > 300:
                    with st.expander("Show full passage"):
                        st.write(passage)
                if item.get("image_path"):
                    image_url = f"{API_URL}/assets/{quote(item['image_path'])}"
                    st.image(
                        image_url,
                        caption=item.get("asset_id") or "Retrieved page/figure",
                    )

            document_id = citation["document_id"]
            if document_id.startswith("web:"):
                url = document_id.removeprefix("web:")
                st.caption(f"Web literature result: {url}")
            else:
                formatted = get_citation(document_id, reference_number)
                if formatted:
                    st.caption(formatted["citation"])
                    if formatted.get("doi_url"):
                        st.link_button("Open DOI", formatted["doi_url"])
                else:
                    st.caption(f"[{reference_number}] {document_id}, page {page_number}.")


def show_timeline(items: list[dict[str, Any]]) -> None:
    if not items:
        return
    with st.expander("Plan / task timeline"):
        for item in items:
            duration = (
                f" · {item['duration_ms']:.0f} ms" if item.get("duration_ms") is not None else ""
            )
            st.markdown(f"- **{item['label']}** — {item['status']}{duration}")
            if item.get("detail"):
                st.caption(item["detail"])


def show_evaluation() -> None:
    report = get_json("/evaluation/latest")
    if report is None:
        return
    st.subheader("Latest evaluation")
    st.caption(
        f"{report['benchmark_split']} split · {report['execution_policy']} · "
        f"router accuracy {report['router']['accuracy']:.1%}"
    )
    rows = []
    for mode, metrics in report["modes"].items():
        rows.append(
            {
                "mode": mode,
                "examples": metrics["example_count"],
                "Recall@5": metrics["text_recall_at_5"],
                "MRR": metrics["mrr"],
                "citation recall": metrics["citation_recall"],
                "abstention F1": metrics["abstention_f1"],
                "goal accuracy": metrics["agent_goal_accuracy"],
                "p95 ms": metrics["p95_latency_ms"],
                "cost USD": metrics["estimated_cost_usd"],
            }
        )
    st.dataframe(rows, hide_index=True, use_container_width=True)
    statuses = {mode: metrics["ragas"]["status"] for mode, metrics in report["modes"].items()}
    st.caption(f"RAGAS judge status: {statuses}")


st.set_page_config(page_title="ADASPilot", page_icon="🚗", layout="wide")
show_corpus_sidebar()
st.title("ADASPilot")
st.caption("Evidence-grounded ADAS research over the verified local corpus")

mode = st.selectbox(
    "Execution mode",
    ["auto", "direct", "react", "plan"],
    help=(
        "Auto uses the explainable router; direct is cheapest, ReAct uses tools, "
        "and plan handles comparisons."
    ),
)
question = st.chat_input("Ask about the ADAS corpus")
if question:
    run_state: dict[str, Any] = {}
    with st.chat_message("user"):
        st.write(question)
    with st.chat_message("assistant"):
        try:
            answer = st.write_stream(stream_answer(question, mode, run_state), cursor="▌")
            if not answer:
                st.warning("The workflow returned no answer text.")
            execution = run_state.get("execution", {})
            if execution:
                st.caption(f"Mode: `{execution['mode']}` · {execution['latency_ms']:.0f} ms")
            show_timeline(run_state.get("timeline", []))
            show_evidence(run_state.get("evidence", {}))
        except httpx.HTTPError as exc:
            st.error(f"Query failed: {exc}")

st.divider()
show_evaluation()
