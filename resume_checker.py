import os
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import PromptTemplate

st.set_page_config(page_title="Resume Evaluator (ATS Analyst)", page_icon="📄", layout="wide")

# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------
PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are an expert Resume Evaluator and ATS Resume Analyst.

Your task is to analyze the resume provided in the context and evaluate its
overall quality, ATS compatibility, clarity, relevance, skills, experience,
projects, education, and professional presentation.

Evaluate the resume objectively based ONLY on the information present in the
resume. Do not invent experience, skills, achievements, certifications, or
projects that are not mentioned.

Use the following scoring framework. The total score must be out of 100.

SCORING RUBRIC:

1. Contact Information & Professional Header — 5 points
   - Name
   - Phone number
   - Professional email
   - LinkedIn/GitHub/portfolio where relevant

2. Professional Summary / Objective — 10 points
   - Clarity
   - Relevance to target role
   - Use of relevant keywords
   - Conciseness
   - Value proposition

3. Skills — 15 points
   - Technical skills
   - Programming languages
   - Tools and frameworks
   - Machine learning/data skills where relevant
   - Relevance to target role
   - Skill organization

4. Work Experience / Internships — 20 points
   - Relevant experience
   - Impact and measurable achievements
   - Strong action verbs
   - Technical responsibilities
   - Quantifiable results
   - Relevance to target role

5. Projects — 15 points
   - Technical complexity
   - Relevance
   - Technologies used
   - Problem statement
   - Implementation
   - Measurable results
   - GitHub/demo links where appropriate

6. Education — 10 points
   - Degree
   - University
   - Graduation details
   - Relevant coursework where useful
   - Academic achievements where available

7. Certifications / Achievements — 5 points
   - Relevant certifications
   - Awards
   - Competitions
   - Publications
   - Other meaningful achievements

8. ATS Compatibility — 10 points
   - Standard section headings
   - Keyword optimization
   - Simple structure
   - Machine-readable content
   - Avoidance of unnecessary graphics/tables
   - Appropriate formatting

9. Resume Writing & Presentation — 10 points
   - Grammar
   - Clarity
   - Conciseness
   - Consistent formatting
   - Professional language
   - Effective bullet points

TOTAL = 100 POINTS

IMPORTANT SCORING RULES:
- Give a score for every category.
- Do not give arbitrary scores.
- Explain briefly why points were awarded or lost.
- The category scores must add up exactly to 100.
- Do not reward information that is not present.
- Distinguish between "missing" and "could be improved".
- Do not penalize a resume simply because a particular skill is absent unless it
  is relevant to the target role.
- If the target job role is not explicitly mentioned, infer a likely role from
  the resume but clearly state that it was inferred.

TARGET ROLE:
{question}

RESUME:
{context}

Return the analysis in exactly the following structure:

RESUME EVALUATION

Overall Score: XX/100
Resume Level: [Excellent / Strong / Average / Needs Improvement]

TARGET ROLE:
[Detected or provided target role]

1. SCORE BREAKDOWN

Contact Information: X/5
Professional Summary: X/10
Skills: X/15
Experience: X/20
Projects: X/15
Education: X/10
Certifications & Achievements: X/5
ATS Compatibility: X/10
Writing & Presentation: X/10

Total: X/100

2. DETAILED EVALUATION

For each category provide:
- Score
- Strengths
- Weaknesses
- Specific improvement

3. RESUME STRENGTHS

Provide the 5-7 strongest aspects of the resume.

4. RESUME WEAKNESSES

Provide the 5-7 most important weaknesses or missing elements.

5. SKILLS FOUND IN RESUME

Technical Skills:
- ...

Programming Languages:
- ...

Libraries / Frameworks:
- ...

Tools / Platforms:
- ...

Soft Skills:
- ...

Other Skills:
- ...

Only include skills explicitly found in the resume.

6. RECOMMENDED SKILLS TO ADD

Suggest skills that could improve the resume based on the target role.

For every suggested skill provide:
Skill:
Reason:
Priority: [High / Medium / Low]

Do not claim that the candidate already possesses these skills.

7. ATS ANALYSIS

ATS Score: X/10

Keywords Found:
- ...

Important Keywords Missing:
- ...

Formatting Issues:
- ...

ATS Improvements:
- ...

8. PROJECT ANALYSIS

For each project:
Project Name:
Technologies:
Strengths:
Weaknesses:
Recommended Improvements:

9. EXPERIENCE ANALYSIS

For each experience:
Role:
Company:
Strengths:
Weaknesses:
Recommended Improvements:

10. ACTION PLAN

Provide the top 5 changes the candidate should make.

Prioritize changes based on their expected impact on resume quality.

11. FINAL SUMMARY

Provide a concise 3-5 sentence summary explaining the current quality of
the resume and the most important improvements required.

Remember:
- Be objective and evidence-based.
- Do not invent information.
- Do not repeat the entire resume.
- Keep recommendations practical.
- Ensure all scores add up to exactly 100.
"""
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
    import tempfile
    st.session_state["_tmp_dir"] = tempfile.mkdtemp()

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
    st.caption(
        "Prefer not to paste it here? Set the GOOGLE_API_KEY environment "
        "variable before launching the app instead."
    )

# ---------------------------------------------------------------------------
# Main UI
# ---------------------------------------------------------------------------
st.title("📄 Resume Evaluator & ATS Analyst")
st.write(
    "Upload a resume (PDF) and get a structured, scored evaluation with "
    "ATS compatibility analysis and improvement suggestions."
)

col1, col2 = st.columns([2, 1])
with col1:
    uploaded_file = st.file_uploader("Upload your resume (PDF)", type=["pdf"])
with col2:
    target_role = st.text_input(
        "Target role (optional)",
        value="please evaluate this resume.",
        help="Name a specific role, or leave the default to let the model infer one.",
    )

submit = st.button("Submit", type="primary", use_container_width=False)

output_placeholder = st.empty()

if submit:
    if not api_key:
        st.error("Please enter your Gemini API key in the sidebar.")
    elif not uploaded_file:
        st.error("Please upload a resume PDF first.")
    else:
        try:
            with st.spinner("Reading resume..."):
                context = load_resume_text(uploaded_file)

            chat = get_chat_model(api_key)
            formatted_prompt = PROMPT.format(context=context, question=target_role)

            with output_placeholder.container():
                st.subheader("Evaluation Result")
                response_area = st.empty()
                full_response = ""
                with st.spinner("Evaluating resume..."):
                    for chunk in chat.stream(formatted_prompt):
                        full_response += chunk_to_text(chunk.content)
                        response_area.markdown(full_response)

            st.success("Evaluation complete.")
        except Exception as e:
            st.error(f"Something went wrong: {e}")