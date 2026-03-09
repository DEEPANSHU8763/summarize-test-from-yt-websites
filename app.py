import streamlit as st
import validators
import ssl
import certifi

from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_community.document_loaders import UnstructuredURLLoader


ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=certifi.where())


# ------------------------
# Streamlit UI
# ------------------------

st.set_page_config(page_title="AI Website Summarizer", page_icon="🦜")

st.title("🦜 AI Website Summarizer")


with st.sidebar:
    groq_api_key = st.text_input("Enter GROQ API Key", type="password")


url = st.text_input("Enter Website URL")


# ------------------------
# LLM
# ------------------------

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    groq_api_key=groq_api_key,
    temperature=0,
    max_tokens=600
)


prompt = PromptTemplate.from_template(
"""
Summarize the following content in about 300 words.

{text}
"""
)


# ------------------------
# Simple Text Splitter
# ------------------------

def split_text(text, chunk_size=2000, overlap=200):

    chunks = []
    start = 0

    while start < len(text):

        end = start + chunk_size
        chunks.append(text[start:end])

        start += chunk_size - overlap

    return chunks


# ------------------------
# Summarization Pipeline
# ------------------------

def summarize_docs(docs):

    full_text = "\n".join([doc.page_content for doc in docs])

    chunks = split_text(full_text)

    summaries = []

    for chunk in chunks:

        formatted = prompt.format(text=chunk)

        response = llm.invoke(formatted)

        summaries.append(response.content)

    combined = "\n".join(summaries)

    final_prompt = f"""
Combine the following summaries into one final summary of about 300 words.

{combined}
"""

    final = llm.invoke(final_prompt)

    return final.content


# ------------------------
# Run Button
# ------------------------

if st.button("Summarize"):

    if not groq_api_key:
        st.error("Enter GROQ API key")

    elif not validators.url(url):
        st.error("Enter valid URL")

    else:

        try:

            with st.spinner("Loading website..."):

                loader = UnstructuredURLLoader(
                    urls=[url],
                    headers={"User-Agent": "Mozilla/5.0"}
                )

                docs = loader.load()

                summary = summarize_docs(docs)

                st.success(summary)

        except Exception as e:
            st.exception(e)
