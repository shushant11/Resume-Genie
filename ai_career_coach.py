import hashlib
import io
import os

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from pypdf import PdfReader

st.set_page_config(page_title="Resume Career Coach", page_icon="🧭", layout="wide")

# Pane heights (px). The chat box is shorter because the chat input sits below it.
PANE_HEIGHT = 620
CHAT_HEIGHT = 545

SYSTEM_TEMPLATE = """
You are a professional career coach and resume mentor.

You help with:
- career guidance
- resume improvements
- interview preparation
- job search strategy
- skill gap analysis

Base your advice on the candidate's resume below. Do not invent experience,
skills, or achievements that are not in it.

Candidate Resume:
{context}
"""

GREETING = (
    "Hi! I've read your resume. I can help with career guidance, resume "
    "improvements, interview preparation, job search strategy, and skill gap "
    "analysis. What would you like to work on?"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def extract_resume_text(data: bytes) -> str:
    reader = PdfReader(io.BytesIO(data))
    pages = [(page.extract_text() or "") for page in reader.pages]
    return "\n\n".join(pages).strip()


@st.cache_data(show_spinner=False)
def render_pdf_pages(data: bytes) -> list:
    """Render each PDF page to a PNG so the resume can be shown on the left."""
    import fitz  # PyMuPDF

    doc = fitz.open(stream=data, filetype="pdf")
    try:
        return [page.get_pixmap(dpi=120).tobytes("png") for page in doc]
    finally:
        doc.close()


def get_chat_model(api_key: str) -> ChatGoogleGenerativeAI:
    os.environ["GOOGLE_API_KEY"] = api_key
    return ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite")


def chunk_to_text(content) -> str:
    """Gemini may return a chunk's content as a string or as a list of parts."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and item.get("type", "text") == "text":
                parts.append(item.get("text", ""))
        return "".join(parts)
    return ""


def stream_reply(chat, system_message, history):
    """Yield the coach's reply chunk by chunk, given the conversation so far."""
    messages = [system_message]
    for m in history:
        if m["role"] == "user":
            messages.append(HumanMessage(content=m["content"]))
        else:
            messages.append(AIMessage(content=m["content"]))

    for chunk in chat.stream(messages):
        text = chunk_to_text(chunk.content)
        if text:
            yield text


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
st.session_state.setdefault("messages", [])
st.session_state.setdefault("resume_id", None)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Settings")
    api_key = st.text_input(
        "Gemini API Key",
        type="password",
        value=os.environ.get("GOOGLE_API_KEY", ""),
        help="Your key is only used for this session and is never stored.",
    )
    uploaded = st.file_uploader("Upload your resume (PDF)", type=["pdf"])
    if st.button("Clear chat"):
        st.session_state["messages"] = []

# ---------------------------------------------------------------------------
# Main UI
# ---------------------------------------------------------------------------
st.title("🧭 Resume Career Coach")

if not uploaded:
    st.info("Upload your resume (PDF) in the sidebar to start chatting with your career coach.")
    st.stop()

data = uploaded.getvalue()

# A different resume means a fresh conversation
resume_id = hashlib.sha1(data).hexdigest()
if st.session_state["resume_id"] != resume_id:
    st.session_state["resume_id"] = resume_id
    st.session_state["messages"] = []

resume_text = extract_resume_text(data)

left, right = st.columns(2, gap="medium")

# ---- Left half: the resume ------------------------------------------------
with left:
    st.subheader("Your resume")
    with st.container(height=PANE_HEIGHT):
        try:
            for page_png in render_pdf_pages(data):
                st.image(page_png)
        except ImportError:
            st.error("PyMuPDF is needed to show the resume. Run: pip install pymupdf")
        except Exception as e:
            st.error(f"Couldn't display the PDF: {e}")

# ---- Right half: the chatbot ----------------------------------------------
with right:
    st.subheader("Career coach")
    chat_box = st.container(height=CHAT_HEIGHT)

    if not resume_text:
        st.warning(
            "I couldn't read any text from this PDF (it may be a scanned image). "
            "Try uploading a text-based PDF."
        )

    if not api_key:
        placeholder = "Enter your Gemini API key in the sidebar (press Enter) to start"
    else:
        placeholder = "Ask about your resume, interviews, skill gaps..."

    prompt = st.chat_input(placeholder, disabled=not (api_key and resume_text))

    with chat_box:
        with st.chat_message("assistant"):
            st.markdown(GREETING)

        for m in st.session_state["messages"]:
            with st.chat_message(m["role"]):
                st.markdown(m["content"])

        if prompt:
            st.session_state["messages"].append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            try:
                chat = get_chat_model(api_key)
                system_message = SystemMessage(
                    content=SYSTEM_TEMPLATE.format(context=resume_text)
                )
                with st.chat_message("assistant"):
                    reply = st.write_stream(
                        stream_reply(chat, system_message, st.session_state["messages"])
                    )
                st.session_state["messages"].append(
                    {"role": "assistant", "content": reply}
                )
            except Exception as e:
                st.session_state["messages"].pop()  # drop the unanswered question
                st.error(f"Something went wrong: {e}")