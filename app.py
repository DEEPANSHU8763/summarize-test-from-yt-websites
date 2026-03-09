import validators
import streamlit as st
import ssl
import certifi
import re

from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

from youtube_transcript_api import YouTubeTranscriptApi


# Fix SSL issues (macOS)
ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=certifi.where())


# -----------------------------
# Streamlit UI
# -----------------------------

st.set_page_config(page_title="LangChain: Summarize Text From YT or Website", page_icon="🦜")

st.title("🦜 LangChain: Summarize Text From YT or Website")
st.subheader("Summarize URL")


# -----------------------------
# Sidebar
# -----------------------------

with st.sidebar:
    groq_api_key = st.text_input("Enter GROQ API Key", type="password")


# URL input
generic_url = st.text_input("Enter YouTube or Website URL", label_visibility="collapsed")


# -----------------------------
# Prompt
# -----------------------------

prompt_template = """
Provide a concise summary of the following content in about 300 words.

Content:
{text}
"""

prompt = PromptTemplate(
    template=prompt_template,
    input_variables=["text"]
)


# -----------------------------
# Extract YouTube Video ID
# -----------------------------

def get_video_id(url):
    pattern = r"(?:v=|\/)([0-9A-Za-z_-]{11})"
    match = re.search(pattern, url)
    return match.group(1) if match else None


# -----------------------------
# Button
# -----------------------------

if st.button("Summarize the Content from YT or Website"):

    if not groq_api_key.strip():
        st.error("Please enter your GROQ API key")

    elif not generic_url.strip():
        st.error("Please enter a URL")

    elif not validators.url(generic_url):
        st.error("Please enter a valid URL")

    else:
        try:
            with st.spinner("Loading content..."):

                # -----------------------------
                # Load Data
                # -----------------------------

                if "youtube.com" in generic_url or "youtu.be" in generic_url:

                    video_id = get_video_id(generic_url)

                    try:
                        transcript = YouTubeTranscriptApi.get_transcript(
                            video_id,
                            languages=["en", "hi"]
                        )

                        text = " ".join([t["text"] for t in transcript])

                        docs = [Document(page_content=text)]

                    except Exception:
                        st.error("Could not fetch transcript from this YouTube video.")
                        st.stop()

                else:
                    loader = UnstructuredURLLoader(
                        urls=[generic_url],
                        ssl_verify=True,
                        headers={"User-Agent": "Mozilla/5.0"}
                    )

                    docs = loader.load()

                    if not docs:
                        st.error("Could not extract text from the webpage.")
                        st.stop()


                # -----------------------------
                # Split documents to avoid
                # Groq token overflow
                # -----------------------------

                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=5000,
                    chunk_overlap=200
                )

                split_docs = text_splitter.split_documents(docs)


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
                # Summarization Chain
                # -----------------------------

                chain = load_summarize_chain(
                    llm,
                    chain_type="map_reduce",
                    map_prompt=prompt,
                    combine_prompt=prompt
                )


                # -----------------------------
                # Generate Summary
                # -----------------------------

                summary = chain.run(split_docs)

                st.success(summary)


        except Exception as e:

            st.exception(e)
