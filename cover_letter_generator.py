import os
import tempfile

import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import PromptTemplate

st.set_page_config(page_title="Cover Letter Generator", page_icon="✉️", layout="wide")

# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------
TEMPLATE = """
You are an expert career coach and professional cover letter writer.

Your task is to write a compelling, personalized cover letter based on the
candidate's resume and the target job description provided below. Use ONLY
the experience, skills, and achievements that actually appear in the resume
— do not invent anything not present there.

Use the job description to determine which skills and experiences from the
resume are most relevant, and to mirror the role's key requirements and
terminology naturally in the letter, so it reads as tailored to this
specific job rather than generic.

WRITING GUIDELINES:
- Keep the tone professional, confident, and genuine — avoid generic or
  overly formal phrasing.
- Open with a strong hook that names the role being applied for and why the
  candidate is a strong fit for it specifically.
- Identify 2-3 requirements from the job description that the candidate's
  resume actually supports, and build the body paragraphs around those —
  connecting specific resume evidence (skills, projects, experience) to
  what the role needs.
- Do not claim a skill or requirement from the job description that the
  resume does not support.
- Show genuine enthusiasm for the role without sounding exaggerated.
- Close with a clear, confident call to action (e.g. requesting an
  interview).
- Keep the letter to 3-4 concise paragraphs (roughly 250-350 words).
- Do not repeat the resume verbatim — synthesize and reframe it in the
  context of this job.
- Do not fabricate a company name, hiring manager name, or any job detail
  not present in the job description; use a neutral placeholder like
  "[Company Name]" or "[Hiring Manager]" if none was given.

JOB DESCRIPTION:
{job_description}

CANDIDATE'S RESUME:
{context}

Return ONLY the final cover letter text, formatted as a standard business
letter (greeting, body paragraphs, sign-off) — no extra commentary,
headers, or explanations before or after it.
"""

PROMPT = PromptTemplate(
    input_variables=["context", "job_description"],
    template=TEMPLATE,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_resume_text(uploaded_file) -> str:
    """Save the uploaded PDF to a temp path and extract its text via PyPDFLoader."""
    tmp_path = os.path.join(st.session_state["_tmp_dir"], uploaded_file.name)
    with open(tmp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    loader = PyPDFLoader(tmp_path)
    documents = loader.load()
    return "\n\n".join(doc.page_content for doc in documents)


def get_chat_model(api_key: str) -> ChatGoogleGenerativeAI:
    os.environ["GOOGLE_API_KEY"] = api_key
    return ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite")


def chunk_to_text(content) -> str:
    """Gemini/langchain streaming chunks can return .content as a plain
    string OR as a list of content-part dicts/strings. Normalize either
    shape into plain text so it can be safely concatenated."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                parts.append(part.get("text", ""))
        return "".join(parts)
    return str(content)


# ---------------------------------------------------------------------------
# Session state setup
# ---------------------------------------------------------------------------
if "_tmp_dir" not in st.session_state:
    st.session_state["_tmp_dir"] = tempfile.mkdtemp()

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Settings")
    api_key = st.text_input(
        "Gemini API Key",
        type="password",
        help="Your key is only used for this session and is never stored.",
    )

# ---------------------------------------------------------------------------
# Main UI
# ---------------------------------------------------------------------------
st.title("✉️ Cover Letter Generator")
st.write(
    "Upload your resume (PDF) and paste the job description below to get a "
    "tailored, ready-to-send cover letter."
)

col1, col2 = st.columns(2)
with col1:
    uploaded_file = st.file_uploader("Upload your resume (PDF)", type=["pdf"])
with col2:
    job_description = st.text_area(
        "Job description",
        height=220,
        placeholder="Paste the full job description here...",
    )

submit = st.button("Generate Cover Letter", type="primary")

output_placeholder = st.empty()

if submit:
    if not api_key:
        st.error("Please enter your Gemini API key in the sidebar.")
    elif not uploaded_file:
        st.error("Please upload a resume PDF first.")
    elif not job_description.strip():
        st.error("Please paste the job description.")
    else:
        try:
            with st.spinner("Reading resume..."):
                context = load_resume_text(uploaded_file)

            chat = get_chat_model(api_key)
            formatted_prompt = PROMPT.format(
                context=context, job_description=job_description
            )

            with output_placeholder.container():
                st.subheader("Your Cover Letter")
                response_area = st.empty()
                full_response = ""
                with st.spinner("Writing your cover letter..."):
                    for chunk in chat.stream(formatted_prompt):
                        full_response += chunk_to_text(chunk.content)
                        response_area.markdown(full_response)

            st.success("Done.")
            st.download_button(
                "Download as .txt",
                data=full_response,
                file_name="cover_letter.txt",
                mime="text/plain",
            )
        except Exception as e:
            st.error(f"Something went wrong: {e}")