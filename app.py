import streamlit as st
import validators
import ssl
import certifi
import requests
from bs4 import BeautifulSoup

from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq

ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=certifi.where())


# -------------------------
# Streamlit UI
# -------------------------

st.set_page_config(page_title="AI Website Summarizer", page_icon="🦜")

st.title("🦜 AI Website Summarizer")

with st.sidebar:
    groq_api_key = st.text_input("Enter GROQ API Key", type="password")

url = st.text_input("Enter Website URL")


# -------------------------
# LLM
# -------------------------

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


# -------------------------
# Fetch Website Content
# -------------------------

def fetch_text_from_url(url):

    headers = {"User-Agent": "Mozilla/5.0"}

    response = requests.get(url, headers=headers, timeout=20)

    soup = BeautifulSoup(response.text, "html.parser")

    paragraphs = soup.find_all("p")

    text = " ".join([p.get_text() for p in paragraphs])

    return text


# -------------------------
# Text Splitter
# -------------------------

def split_text(text, chunk_size=2000, overlap=200):

    chunks = []
    start = 0

    while start < len(text):

        end = start + chunk_size
        chunks.append(text[start:end])

        start += chunk_size - overlap

    return chunks


# -------------------------
# Summarization
# -------------------------

def summarize_text(text):

    chunks = split_text(text)

    summaries = []

    for chunk in chunks:

        formatted_prompt = prompt.format(text=chunk)

        response = llm.invoke(formatted_prompt)

        summaries.append(response.content)

    combined = "\n".join(summaries)

    final_prompt = f"""
Combine the following summaries into a single final summary of about 300 words.

{combined}
"""

    final = llm.invoke(final_prompt)

    return final.content


# -------------------------
# Button
# -------------------------

if st.button("Summarize"):

    if not groq_api_key:
        st.error("Please enter GROQ API key")

    elif not validators.url(url):
        st.error("Please enter a valid URL")

    else:

        try:

            with st.spinner("Fetching article..."):

                text = fetch_text_from_url(url)

                summary = summarize_text(text)

                st.success(summary)

        except Exception as e:
            st.exception(e)
