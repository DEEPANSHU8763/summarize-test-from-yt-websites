import streamlit as st
import validators
import ssl
import certifi
import re

from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.document_loaders import UnstructuredURLLoader


ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=certifi.where())


# -----------------------------
# Streamlit UI
# -----------------------------

st.set_page_config(
    page_title="Summarize Website",
    page_icon="🦜"
)

st.title("🦜 AI Website Summarizer")
st.write("Paste any article URL and get a summary.")


# -----------------------------
# Sidebar
# -----------------------------

with st.sidebar:
    groq_api_key = st.text_input("Enter GROQ API Key", type="password")


# URL input
url = st.text_input("Enter website URL")


# -----------------------------
# LLM
# -----------------------------

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    groq_api_key=groq_api_key,
    temperature=0,
    max_tokens=600
)


# -----------------------------
# Prompt
# -----------------------------

prompt = PromptTemplate.from_template(
"""
Summarize the following content in about 300 words.

{text}
"""
)


# -----------------------------
# Summarization function
# -----------------------------

def summarize_large_text(docs):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=200
    )

    split_docs = splitter.split_documents(docs)

    summaries = []

    for doc in split_docs:

        formatted_prompt = prompt.format(text=doc.page_content)

        response = llm.invoke(formatted_prompt)

        summaries.append(response.content)

    combined_text = "\n".join(summaries)

    final_prompt = f"""
Combine the following partial summaries into one final summary of about 300 words.

{combined_text}
"""

    final_summary = llm.invoke(final_prompt)

    return final_summary.content


# -----------------------------
# Button
# -----------------------------

if st.button("Summarize"):

    if not groq_api_key:
        st.error("Please enter GROQ API key")

    elif not validators.url(url):
        st.error("Please enter a valid URL")

    else:

        try:

            with st.spinner("Loading website..."):

                loader = UnstructuredURLLoader(
                    urls=[url],
                    headers={"User-Agent": "Mozilla/5.0"}
                )

                docs = loader.load()

                summary = summarize_large_text(docs)

                st.success(summary)

        except Exception as e:
            st.exception(e)
