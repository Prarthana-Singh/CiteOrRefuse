"""Minimal Streamlit frontend for CiteOrRefuse -- a thin client with no
business logic of its own: a file uploader that calls `POST /ingest`, and
a query box that calls `POST /answer` scoped to whatever was just
uploaded, displaying exactly what the API already returns (answer or
refusal, citations, confidence).

This is a manual-verification-only component (see README limitations) --
there is no automated test suite for it, since a thin client has no
business logic of its own to unit test; the API it calls is fully tested
in tests/api.

Run: streamlit run streamlit_app.py
Requires the FastAPI service (libs/api/app.py) already running separately,
e.g.: uvicorn libs.api.app:app
"""
import requests
import streamlit as st

st.set_page_config(page_title="CiteOrRefuse", page_icon="\U0001F4C4")

st.title("CiteOrRefuse")
st.caption("Upload a 10-K, then ask a question. Every answer is cited or refused -- never guessed.")

api_url = st.sidebar.text_input("API base URL", value="http://127.0.0.1:8000")

if "filing_id" not in st.session_state:
    st.session_state.filing_id = None
    st.session_state.company = None

st.header("1. Upload a 10-K")
uploaded_file = st.file_uploader("SEC 10-K filing (.htm/.html only)", type=["htm", "html"])
company_name = st.text_input("Company name")

if st.button("Ingest", disabled=uploaded_file is None or not company_name):
    with st.spinner("Ingesting -- parsing, chunking, and indexing the filing..."):
        response = requests.post(
            f"{api_url}/ingest",
            files={"file": (uploaded_file.name, uploaded_file.getvalue(), "text/html")},
            data={"company": company_name},
        )
    if response.ok:
        body = response.json()
        st.session_state.filing_id = body["filing_id"]
        st.session_state.company = body["company"]
        st.success(
            f"Indexed {body['company']}: {body['sections_detected']} sections, "
            f"{body['chunks_indexed']} chunks. filing_id = {body['filing_id']}"
        )
    else:
        st.session_state.filing_id = None
        st.error(f"Ingest failed ({response.status_code}): {response.json().get('detail', response.text)}")

st.header("2. Ask a question")
if st.session_state.filing_id is None:
    st.info("Upload and ingest a filing above before asking a question.")
else:
    st.caption(f"Scoped to: {st.session_state.company} (filing_id = {st.session_state.filing_id})")
    query = st.text_input("Question")
    if st.button("Ask", disabled=not query):
        with st.spinner("Retrieving, generating, and checking groundedness..."):
            response = requests.post(
                f"{api_url}/answer",
                json={"query": query, "filing_id": st.session_state.filing_id},
            )
        if not response.ok:
            st.error(f"Request failed ({response.status_code}): {response.text}")
        else:
            result = response.json()
            if result["answered"]:
                st.success(result["answer"])
                confidence = result["groundedness"]["overall_confidence"]
                st.metric("Groundedness confidence", f"{confidence:.2f}")
                st.subheader("Citations")
                for citation in result["citations"]:
                    with st.expander(f"{citation['chunk_id']} -- {citation['section_title']}"):
                        st.write(citation["text"])
            else:
                st.warning(result["refusal_reason"])
