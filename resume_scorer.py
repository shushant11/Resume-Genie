import math
import os
import re
import tempfile

import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import PromptTemplate

st.set_page_config(page_title="Resume Scorer", page_icon="🎯", layout="wide")

# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------
TEMPLATE = """
You are an expert Technical Recruiter and ATS (Applicant Tracking System)
analyst. Score the candidate's resume against the job description below.

Base your evaluation ONLY on what is actually written in the resume. Do not
assume or invent skills, experience, or achievements. Do not give credit for
information that is not present.

JOB DESCRIPTION:
{job_description}

CANDIDATE'S RESUME:
{context}

SCORING DEFINITIONS:

1. SCORE (out of 100) - weighted sum of these internal rubrics:
   - Skills & Tools Match: 30 points
   - Experience Relevance: 25 points
   - Projects & Achievements (impact, measurable results): 15 points
   - Education & Certifications: 10 points
   - Keyword Alignment with the job description: 10 points
   - Resume Quality (clarity, grammar, structure): 10 points

2. OVERALL MATCH (out of 100%) - how closely the candidate fits the role as
   a whole (skills, experience level, domain, and responsibilities),
   independent of how well the resume is written.

3. RELIABILITY SCORE (out of 100) - how credible and verifiable the resume
   appears: consistent dates and timelines, claims backed by specific
   projects or metrics, no vague buzzwords or inflated titles, skills
   supported by evidence in experience/projects, and presence of
   verifiable links (GitHub, LinkedIn, portfolio).

4. ATS COMPATIBILITY SCORE (out of 100) - standard section headings,
   machine-readable layout (no tables/graphics/columns that break parsing),
   keyword usage, consistent formatting, and use of the job description's
   terminology.

RULES:
- Scores must be justified by evidence from the resume and job description.
- A keyword counts as "matched" only if it (or a clear synonym) appears in
  the resume.
- "Missing keywords" must come from the job description, and must not
  already appear in the resume.
- Only treat a skill as a gap if the job description requires or prefers it.
- Keep every section concise and practical.

Return the output in EXACTLY the following format:

RESUME SCORE REPORT

Score: XX/100
Overall Match: XX%
Reliability Score: XX/100
ATS Compatibility Score: XX/100

Score Breakdown:
- Skills & Tools Match: X/30
- Experience Relevance: X/25
- Projects & Achievements: X/15
- Education & Certifications: X/10
- Keyword Alignment: X/10
- Resume Quality: X/10

Keywords Matched:
- ...

Missing Keywords:
- ...

Two-Line Analysis:
[Exactly two lines: line 1 on the candidate's biggest strength for this role,
line 2 on the biggest weakness or risk.]

Skill Gap Analysis:
For each gap:
- Skill: ...
  Importance for the role: [High / Medium / Low]
  Why it matters: ...

Overall Improvement Suggestions:
Provide 5-7 specific, actionable changes, ordered by impact.

Industry-Specific Feedback:
[Feedback tailored to the industry/domain of the job description: what
hiring managers in this field look for, what the resume does well or
poorly against those expectations, and how to better position the
candidate for this field.]
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
    # temperature=0 keeps scores steadier between runs
    return ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite", temperature=0)


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


def extract_scores(report: str) -> dict:
    """Pull the four headline numbers out of the model's report.
    Returns only the ones it can find, so a format slip never crashes the UI."""
    patterns = {
        "Score": r"^\W*Score\s*:\s*\**\s*(\d{1,3})\s*/\s*100",
        "Overall Match": r"^\W*Overall Match\s*:\s*\**\s*(\d{1,3})\s*%",
        "Reliability Score": r"^\W*Reliability Score\s*:\s*\**\s*(\d{1,3})\s*/\s*100",
        "ATS Compatibility": r"^\W*ATS Compatibility Score\s*:\s*\**\s*(\d{1,3})\s*/\s*100",
    }
    found = {}
    for label, pattern in patterns.items():
        m = re.search(pattern, report, flags=re.IGNORECASE | re.MULTILINE)
        if m:
            found[label] = int(m.group(1))
    return found


def render_metrics(scores: dict, container) -> None:
    if not scores:
        return
    with container.container():
        cols = st.columns(len(scores))
        for col, (label, value) in zip(cols, scores.items()):
            suffix = "%" if label == "Overall Match" else " / 100"
            col.metric(label, f"{value}{suffix}")
            col.progress(min(max(value, 0), 100) / 100)


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
st.title("🎯 Resume Scorer")
st.write(
    "Paste a job description and upload your resume (PDF) to see how well "
    "they match, with ATS and reliability scores, keyword gaps, and "
    "improvement suggestions."
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

submit = st.button("Score My Resume", type="primary")

metrics_placeholder = st.empty()
report_placeholder = st.empty()

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

            with report_placeholder.container():
                st.subheader("Detailed Report")
                response_area = st.empty()
                full_response = ""
                with st.spinner("Scoring your resume..."):
                    for chunk in chat.stream(formatted_prompt):
                        full_response += chunk_to_text(chunk.content)
                        response_area.markdown(full_response)

            # Headline numbers appear above the report once it's complete
            render_metrics(extract_scores(full_response), metrics_placeholder)

            st.success("Done.")
            st.download_button(
                "Download report as .txt",
                data=full_response,
                file_name="resume_score_report.txt",
                mime="text/plain",
            )
        except Exception as e:
            st.error(f"Something went wrong: {e}")