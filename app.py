import streamlit as st
from config import *
from ingestion import load
from retrieval import get_retriever, get_ans, characters

st.set_page_config(page_title="Harry Potter RAG", page_icon="🪄", layout="centered")

st.title("🪄 Harry Potter RAG")
st.caption("Ask a question, pick a character, get an answer in their voice.")

# Cache the retriever so it's built once per session, not on every question
@st.cache_resource(show_spinner="Loading Harry Potter knowledge base...")
def load_retriever():
    docs, parent_splitter, child_splitter = load(PDF_PATH)
    retriever = get_retriever(docs, parent_splitter, child_splitter)
    return retriever

retriever = load_retriever()

# Sidebar - character selector
st.sidebar.header("Choose your character")
character = st.sidebar.selectbox(
    "Who should answer?",
    options=list(characters.keys()),
    index=None,
    placeholder="Select a character..."
)

# Main chat interface
question = st.text_input("Ask something about the Harry Potter world:")

if st.button("Ask"):
    if character is None:
        st.warning("Please select a character from the sidebar first.")
    elif not question.strip():
        st.warning("Please type a question.")
    else:
        with st.spinner(f"{character} is thinking..."):
            answer = get_ans(retriever, question, character)
        st.markdown(f"### {character} says:")
        st.write(answer)