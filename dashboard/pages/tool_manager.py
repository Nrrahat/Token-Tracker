"""Inspect allowlisted tools and current process execution metrics."""

import json

import streamlit as st

from dashboard.components import page_header
from dashboard.document_storage import DocumentUploadError, store_document
from dashboard.runtime import AppServices


def render(services: AppServices) -> None:
    page_header("Tool manager", "Review registered tools, schemas, and in-process execution signals.")
    st.subheader("Reader documents")
    uploads = st.file_uploader(
        "Add documents for the safe reader",
        type=["txt", "md", "csv", "json"],
        accept_multiple_files=True,
        help="UTF-8 documents up to 1 MB are stored in the reader's confined directory.",
    )
    if uploads:
        for upload in uploads:
            try:
                stored_path = store_document(services.uploaded_root, upload.name, upload.getvalue())
                st.success(f"Added {stored_path.name}")
            except DocumentUploadError as error:
                st.error(f"{upload.name}: {error}")
    stored_documents = sorted(
        path for path in services.uploaded_root.iterdir() if path.is_file()
    )
    if stored_documents:
        st.dataframe(
            [{"File": path.name, "Size bytes": path.stat().st_size} for path in stored_documents],
            use_container_width=True,
            hide_index=True,
        )
    metrics = services.tool_manager.metrics()
    for tool in services.tool_registry.list_tools():
        with st.container(border=True):
            st.markdown(f"**{tool.name}**")
            st.caption(tool.description)
            columns = st.columns(5)
            current = metrics.get(tool.name)
            columns[0].metric("Timeout", f"{tool.timeout_seconds:g}s")
            columns[1].metric("Calls", current.call_count if current else 0)
            columns[2].metric("Failures", current.failure_count if current else 0)
            columns[3].metric("Duplicates", current.duplicate_call_count if current else 0)
            columns[4].metric("Output tokens est.", current.estimated_output_tokens if current else 0)
            with st.expander("Input schema"):
                st.code(json.dumps(tool.provider_definition().parameters, indent=2), language="json")
    st.caption("Tool metrics are in memory for this Streamlit process; durable tool-call analytics are a later phase.")