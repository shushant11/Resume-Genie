"""Resume Genie - your AI resume and career toolkit.

One Streamlit app with four AI resume / career tools:
  Resume Evaluator, Resume Scorer, Cover Letter Generator and Career Coach.

Run:  streamlit run app.py
(Optional, recommended: put .streamlit/config.toml next to this file for the
matching theme.)
"""

import base64
import hashlib
import html
import io
import math
import os
import re
from functools import lru_cache
from types import SimpleNamespace

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from pypdf import PdfReader

APP_TITLE = "Resume Genie"
BRAND = "Resume Genie"
TAGLINE = "Make a wish. Get a stronger resume."

MODEL_NAME = "gemini-3.5-flash-lite"

# Sidebar logo.  True  -> your logo exactly as supplied (icon + wordmark).
#                False -> only the lamp/document icon next to the text "Resume Genie".
SHOW_FULL_LOGO = True

# Career-coach page: pane heights in px (the chat box is shorter because the
# chat input sits below it).
PANE_HEIGHT = 620
CHAT_HEIGHT = 520

# ===========================================================================
# Prompts (unchanged from the working version)
# ===========================================================================
EVALUATOR_TEMPLATE = """
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

SCORER_TEMPLATE = """
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

COVER_LETTER_TEMPLATE = """
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

EVALUATOR_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=EVALUATOR_TEMPLATE,
)
SCORER_PROMPT = PromptTemplate(
    input_variables=["context", "job_description"],
    template=SCORER_TEMPLATE,
)
COVER_LETTER_PROMPT = PromptTemplate(
    input_variables=["context", "job_description"],
    template=COVER_LETTER_TEMPLATE,
)

CHAT_STARTERS = [
    ("🔍 Review my resume", "Review my resume and tell me the three most important things to fix."),
    ("🎤 Interview prep", "Based on my resume, which interview questions should I prepare for, and how should I answer them?"),
    ("📈 Skill gaps", "Which skill gaps should I close to reach the next level in my career?"),
    ("🧭 Job search plan", "Help me build a job search strategy based on my resume."),
]


# ===========================================================================
# Shared helpers: PDF, model, streaming
# ===========================================================================
@st.cache_data(show_spinner=False)
def extract_resume_text(data: bytes) -> str:
    reader = PdfReader(io.BytesIO(data))
    pages = [(page.extract_text() or "") for page in reader.pages]
    return "\n\n".join(pages).strip()


@st.cache_data(show_spinner=False)
def render_pdf_pages(data: bytes) -> list:
    """Render each PDF page to a PNG so the resume can be shown on screen."""
    try:
        import pymupdf as fitz  # PyMuPDF (current name)
    except ImportError:
        import fitz  # older PyMuPDF

    doc = fitz.open(stream=data, filetype="pdf")
    try:
        return [page.get_pixmap(dpi=120).tobytes("png") for page in doc]
    finally:
        doc.close()


def get_chat_model(api_key: str, temperature=None) -> ChatGoogleGenerativeAI:
    os.environ["GOOGLE_API_KEY"] = api_key
    kwargs = {"model": MODEL_NAME}
    if temperature is not None:
        kwargs["temperature"] = temperature
    return ChatGoogleGenerativeAI(**kwargs)


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


def explain_error(e: Exception):
    """Turn an API exception into (headline, detail) for the user."""
    msg = str(e)
    low = msg.lower()
    if "api key" in low or "api_key" in low or "401" in low or "permission_denied" in low:
        return "Gemini didn't accept the API key", "Check the key in the sidebar and try again."
    if "429" in low or "quota" in low or "rate limit" in low or "resource_exhausted" in low:
        return "Gemini rate limit reached", "Wait a minute and try again, or check your quota."
    if "404" in low and "model" in low:
        return "The Gemini model wasn't found", f"Check MODEL_NAME ({MODEL_NAME}) at the top of app.py."
    return "Something went wrong", msg[:400]


# ===========================================================================
# Text helpers (the LLM's output is untrusted: always escape before HTML)
# ===========================================================================
_BULLET = re.compile(r"^\s*(?:[-*•–·]|\d{1,2}[.)])\s+")
_FIELD = re.compile(r"^([A-Za-z][A-Za-z /&()'\-]{1,48}?)\s*:\s*(.*)$")


def esc(text) -> str:
    return html.escape("" if text is None else str(text), quote=True)


def esc_md(text: str) -> str:
    """Stop Streamlit's markdown from treating $...$ as LaTeX."""
    return text.replace("$", "\\$")


def plain(text: str) -> str:
    return text.replace("**", "").replace("__", "").strip()


def inline(text: str) -> str:
    """Escape, then allow **bold**, *italic* and `code` only."""
    t = esc(text)
    t = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", t)
    t = re.sub(r"(?<![\w*])\*(?!\s)([^*\n]+?)(?<!\s)\*(?![\w*])", r"<em>\1</em>", t)
    t = re.sub(r"`([^`\n]+)`", r"<code>\1</code>", t)
    return t.replace("**", "")


def norm_heading(line: str) -> str:
    s = re.sub(r"^\s*#{1,6}\s*", "", line.strip())
    s = plain(s)
    s = re.sub(r"^\d{1,2}[.)]\s*", "", s)
    s = s.strip(" *_:-—\t").lower()
    return re.sub(r"\s+", " ", s)


def parse_sections(text: str, headings: dict) -> dict:
    """Split a report into {key: text}. Anything before the first heading is '_pre'."""
    lookup = {alias: key for key, aliases in headings.items() for alias in aliases}
    sections = {"_pre": []}
    current = "_pre"
    for line in text.splitlines():
        key, content = lookup.get(norm_heading(line)), None
        if key is None and ":" in line and not _BULLET.match(line):
            head, _, rest = line.partition(":")
            k2 = lookup.get(norm_heading(head))
            if k2 and rest.strip():
                key, content = k2, rest.strip()
        if key is not None:
            current = key
            sections.setdefault(current, [])
            if content:
                sections[current].append(content)
            continue
        sections[current].append(line)
    return {k: "\n".join(v).strip() for k, v in sections.items()}


def bullets(text: str, split_commas: bool = False) -> list:
    items, seen = [], set()
    for raw in text.splitlines():
        line = plain(_BULLET.sub("", raw.strip()))
        if not line or line in ("...", "…") or line.lower() in ("none", "n/a", "none found"):
            continue
        parts = [line]
        if split_commas and "," in line and "(" not in line:
            parts = [p.strip(" .") for p in line.split(",")]
        for p in parts:
            if p and p.lower() not in seen:
                seen.add(p.lower())
                items.append(p)
    return items


def list_items(text: str) -> list:
    """Group lines into items: a bullet or number starts a new item,
    other lines continue it, and a blank line ends it."""
    items, cur = [], ""
    for raw in text.splitlines():
        s = raw.strip()
        if not s:
            if cur:
                items.append(cur)
                cur = ""
            continue
        if _BULLET.match(s):
            if cur:
                items.append(cur)
            cur = _BULLET.sub("", s)
        else:
            cur = f"{cur} {s}" if cur else s
    if cur:
        items.append(cur)
    return [i.strip() for i in items if i.strip()]


def split_title(item: str):
    """'**Title**: body' or 'Title: body' -> (title, body)."""
    m = re.match(r"^\*\*(.+?)\*\*\s*[:\-–—.]?\s*(.*)$", item, re.S)
    if m:
        return plain(m.group(1)).rstrip(":"), m.group(2).strip()
    m = re.match(r"^([^:]{3,70}?):\s+(.+)$", item, re.S)
    if m and len(m.group(1).split()) <= 9:
        return plain(m.group(1)), m.group(2).strip()
    return "", item


def parse_records(text: str, start_key: str, known) -> list:
    """Parse repeated 'Label: value' blocks. A record starts at `start_key`.
    Returns a list of {label: [lines]} dicts (labels lower-cased)."""
    known = set(known) | {start_key}
    records, cur, field = [], None, None
    for raw in text.splitlines():
        s = plain(_BULLET.sub("", raw.strip()))
        if not s:
            continue
        m = _FIELD.match(s)
        label = m.group(1).strip().lower() if m else None
        if label in known:
            if label == start_key:
                cur = {}
                records.append(cur)
            if cur is None:
                continue
            field = label
            cur.setdefault(field, [])
            if m.group(2).strip():
                cur[field].append(m.group(2).strip())
        elif cur is not None and field is not None:
            cur[field].append(s)
    return records


def parse_fields(text: str, known) -> tuple:
    """Like parse_records but for ONE block: (preamble_lines, {label: [lines]})."""
    known = set(known)
    pre, fields, field = [], {}, None
    for raw in text.splitlines():
        s = plain(_BULLET.sub("", raw.strip()))
        if not s:
            continue
        m = _FIELD.match(s)
        label = m.group(1).strip().lower() if m else None
        if label in known:
            field = label
            fields.setdefault(field, [])
            if m.group(2).strip():
                fields[field].append(m.group(2).strip())
        elif field is not None:
            fields[field].append(s)
        else:
            pre.append(s)
    return pre, fields


SKILL_GROUP_NAMES = (
    "technical skills", "programming languages", "libraries / frameworks",
    "libraries/frameworks", "libraries", "frameworks", "tools / platforms",
    "tools/platforms", "tools", "platforms", "soft skills", "other skills",
    "languages", "databases", "cloud",
)


def parse_groups(text: str) -> list:
    """'Technical Skills:' + bullets -> [(group, [skills])]."""
    groups, cur = [], None
    for raw in text.splitlines():
        s = raw.strip()
        if not s:
            continue
        is_bullet = bool(_BULLET.match(s))
        body = plain(_BULLET.sub("", s))
        m = _FIELD.match(body)
        if m and (m.group(1).strip().lower() in SKILL_GROUP_NAMES
                  or (not is_bullet and not m.group(2).strip())):
            cur = (m.group(1).strip(), [])
            groups.append(cur)
            value = m.group(2).strip()
            if value:
                cur[1].extend(bullets(value, True))
        elif cur is not None:
            cur[1].extend(bullets(body, True))
    return [(g, items) for g, items in groups if items]


def priority_of(text: str) -> str:
    m = re.search(r"\b(high|medium|low)\b", text or "", re.I)
    return m.group(1).capitalize() if m else ""


def join_lines(lines) -> str:
    return " ".join(lines).strip()


# ---- numbers -----------------------------------------------------------------
def _num(pattern: str, text: str):
    m = re.search(pattern, text, flags=re.IGNORECASE | re.MULTILINE)
    return int(float(m.group(1))) if m else None


def extract_scores(report: str) -> dict:
    """Headline numbers of the scorer report (only the ones that can be found)."""
    patterns = {
        "Score": r"^\W*Score\s*:\s*\**\s*(\d{1,3}(?:\.\d+)?)\s*/\s*100",
        "Overall Match": r"^\W*Overall Match\s*:\s*\**\s*(\d{1,3}(?:\.\d+)?)\s*%",
        "Reliability": r"^\W*Reliability Score\s*:\s*\**\s*(\d{1,3}(?:\.\d+)?)\s*/\s*100",
        "ATS Compatibility": r"^\W*ATS Compatibility Score\s*:\s*\**\s*(\d{1,3}(?:\.\d+)?)\s*/\s*100",
    }
    found = {}
    for label, pattern in patterns.items():
        v = _num(pattern, report)
        if v is not None:
            found[label] = max(0, min(100, v))
    return found


def extract_eval_headline(report: str) -> dict:
    out = {}
    v = _num(r"^\W*Overall Score\s*:\s*\**\s*(\d{1,3}(?:\.\d+)?)\s*/\s*100", report)
    if v is not None:
        out["score"] = max(0, min(100, v))
    m = re.search(r"^\W*Resume Level\s*:\s*(.+)$", report, re.I | re.M)
    if m:
        level = plain(m.group(1)).strip(" []*_.")
        if level:
            out["level"] = level
    v = _num(r"^\W*ATS Score\s*:\s*\**\s*(\d{1,2}(?:\.\d+)?)\s*/\s*10\b", report)
    if v is not None:
        out["ats"] = max(0, min(10, v))
    return out


def parse_breakdown(text: str) -> list:
    """'Skills: 12/15' lines -> [(label, got, max)]; the Total line is skipped."""
    rows = []
    for raw in text.splitlines():
        s = plain(_BULLET.sub("", raw.strip()))
        m = re.match(
            r"^\W*([A-Za-z][A-Za-z &/,'\-]*?)\s*[:\-–—]\s*(\d+(?:\.\d+)?)\s*/\s*(\d+)", s
        )
        if not m or m.group(1).strip().lower().startswith("total"):
            continue
        got, mx = float(m.group(2)), int(m.group(3))
        if mx > 0:
            rows.append((m.group(1).strip(), min(got, mx), mx))
    return rows


def fmt_num(x) -> str:
    return str(int(x)) if float(x) == int(x) else f"{x:.1f}"


def band(score: int):
    """(label, css class) for a 0-100 score."""
    if score >= 85:
        return "Excellent", "ok"
    if score >= 70:
        return "Strong", "ok"
    if score >= 50:
        return "Average", "warn"
    return "Needs work", "bad"


# ===========================================================================
# Logo
# ===========================================================================
def _pil_logo():
    from PIL import Image

    return Image.open(io.BytesIO(base64.b64decode(LOGO_B64))).convert("RGBA")


@lru_cache(maxsize=1)
def icon_image():
    """The lamp + document icon, cropped from the left of the logo."""
    im = _pil_logo()
    w, h = im.size
    icon = im.crop((0, 0, int(w * 0.44), h))
    box = icon.split()[3].getbbox()
    return icon.crop(box) if box else icon


@lru_cache(maxsize=1)
def icon_uri() -> str:
    buf = io.BytesIO()
    icon_image().save(buf, "PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def logo_uri() -> str:
    return "data:image/webp;base64," + LOGO_B64


# ===========================================================================
# Styling
# ===========================================================================
CSS = r"""
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Plus+Jakarta+Sans:wght@600;700;800&display=swap');
:root{
  --blue:#0F37A9; --violet:#7D3ADF; --indigo:#5B3FD4; --gold:#DCA54A; --gold-soft:#FFD98A;
  --ink:#171B3A; --ink2:#4B5072; --muted:#6B7194; --line:#E6E1F7; --tint:#F1EDFD; --bg:#F7F5FF; --card:#FFFFFF;
  --ok:#0B6B43; --ok-bg:#E3F5EC; --bad:#A3281D; --bad-bg:#FDECEA; --warn:#8A4B00; --warn-bg:#FFF1D6; --info:#2F4A8C; --info-bg:#E6ECFB;
  --grad:linear-gradient(135deg,#0F37A9 0%,#5B3FD4 55%,#7D3ADF 100%);
  --shadow:0 1px 2px rgba(23,27,58,.04),0 8px 24px rgba(91,63,212,.08);
  --r:16px;
}
html,body,.stApp,.stApp button,.stApp input,.stApp textarea{font-family:'Inter',system-ui,-apple-system,'Segoe UI',Roboto,sans-serif;}
.stApp{background:radial-gradient(1200px 500px at 85% -10%,#ECE5FF 0%,rgba(236,229,255,0) 60%),var(--bg);color:var(--ink);}
.stApp p,.stApp li,.stApp label,.stApp span,.stApp h1,.stApp h2,.stApp h3,.stApp h4{color:inherit;}
.block-container{padding-top:1.4rem;padding-bottom:4rem;max-width:1180px;}
#MainMenu,footer{visibility:hidden;}
header[data-testid="stHeader"]{background:transparent;}
@keyframes rise{from{opacity:0;transform:translateY(10px)}to{opacity:1;transform:none}}
@keyframes arc{from{stroke-dashoffset:var(--circ)}}
@keyframes fill{from{width:0}}
@media (prefers-reduced-motion:reduce){*{animation:none!important;transition:none!important}}

/* ---------- sidebar ---------- */
[data-testid="stSidebar"]{background:#fff;border-right:1px solid var(--line);}
[data-testid="stSidebar"] .block-container,[data-testid="stSidebarUserContent"]{padding-top:1rem;}
.side-brand{display:flex;align-items:center;gap:10px;padding:2px 2px 14px;}
.side-brand img.full{width:100%;max-width:230px;height:auto;display:block;margin:0 auto;}
.side-brand img.ico{height:44px;width:auto;}
.side-brand b{font-size:1.05rem;letter-spacing:-.01em;color:var(--ink);}
.side-brand small{display:block;color:var(--muted);font-size:.72rem;font-weight:500;}
.side-label{font-size:.68rem;font-weight:700;letter-spacing:.09em;text-transform:uppercase;color:var(--muted);margin:14px 0 6px;}
.pill{display:inline-flex;align-items:center;gap:6px;padding:4px 10px;border-radius:999px;font-size:.76rem;font-weight:600;margin:2px 4px 2px 0;}
.pill.ok{background:var(--ok-bg);color:var(--ok);} .pill.ok::before{content:"✓";font-weight:800;}
.pill.todo{background:var(--tint);color:var(--ink2);} .pill.todo::before{content:"○";}
.side-foot{margin-top:18px;font-size:.72rem;color:var(--muted);line-height:1.5;}

/* ---------- buttons ---------- */
.stApp [data-testid="stBaseButton-primary"],.stApp button[kind="primary"]{background:var(--grad);color:#fff;border:0;border-radius:12px;font-weight:700;padding:.6rem 1.2rem;box-shadow:0 6px 16px rgba(91,63,212,.28);transition:transform .15s ease,box-shadow .15s ease,filter .15s ease;}
.stApp [data-testid="stBaseButton-primary"]:hover,.stApp button[kind="primary"]:hover{transform:translateY(-1px);box-shadow:0 10px 22px rgba(91,63,212,.34);filter:brightness(1.06);color:#fff;}
.stApp [data-testid="stBaseButton-primary"] p,.stApp button[kind="primary"] p{color:#fff;}
.stApp [data-testid="stBaseButton-primary"]:disabled,.stApp button[kind="primary"]:disabled{background:#CFC9E8;box-shadow:none;color:#fff;transform:none;}
.stApp [data-testid="stBaseButton-secondary"],.stApp button[kind="secondary"],.stApp [data-testid="stDownloadButton"] button{border:1px solid var(--line);background:#fff;color:var(--ink);border-radius:12px;font-weight:600;}
.stApp [data-testid="stBaseButton-secondary"]:hover,.stApp button[kind="secondary"]:hover,.stApp [data-testid="stDownloadButton"] button:hover{border-color:var(--violet);color:var(--indigo);background:var(--tint);}

/* sidebar navigation buttons: flat when idle, gradient when active */
[data-testid="stSidebar"] button{justify-content:flex-start;text-align:left;}
[data-testid="stSidebar"] button p{text-align:left;}
.stApp [data-testid="stSidebar"] [data-testid="stBaseButton-secondary"],.stApp [data-testid="stSidebar"] button[kind="secondary"]{background:transparent;border-color:transparent;box-shadow:none;padding:.55rem .8rem;}
.stApp [data-testid="stSidebar"] [data-testid="stBaseButton-secondary"]:hover,.stApp [data-testid="stSidebar"] button[kind="secondary"]:hover{background:var(--tint);border-color:var(--line);color:var(--indigo);}

/* ---------- inputs ---------- */
[data-baseweb="input"],[data-baseweb="textarea"],[data-baseweb="base-input"]{border-radius:12px!important;background:#fff!important;}
[data-baseweb="input"],[data-baseweb="textarea"]{border:1px solid var(--line)!important;}
[data-baseweb="input"]:focus-within,[data-baseweb="textarea"]:focus-within{border-color:var(--violet)!important;box-shadow:0 0 0 3px rgba(125,58,223,.15);}
.stApp textarea,.stApp input{color:var(--ink)!important;}
.stApp label p{font-weight:600;color:var(--ink2);font-size:.86rem;}
[data-testid="stFileUploader"] section{border:1.5px dashed #C9BEF2;background:#FBFAFF;border-radius:14px;}
[data-testid="stFileUploader"] section:hover{border-color:var(--violet);background:var(--tint);}
[data-testid="stVerticalBlockBorderWrapper"]{border-radius:var(--r);}

/* ---------- tabs ---------- */
.stTabs [data-baseweb="tab-list"]{gap:4px;border-bottom:1px solid var(--line);}
.stTabs [data-baseweb="tab"]{height:44px;padding:0 14px;border-radius:10px 10px 0 0;font-weight:600;color:var(--ink2);}
.stTabs [data-baseweb="tab"]:hover{background:var(--tint);color:var(--indigo);}
.stTabs [aria-selected="true"]{color:var(--indigo)!important;}
.stTabs [data-baseweb="tab-highlight"]{background:var(--violet);height:3px;border-radius:3px;}
.stTabs [data-baseweb="tab-panel"]{padding-top:1.1rem;}

/* ---------- status / chat ---------- */
[data-testid="stStatus"],[data-testid="stExpander"]{border-radius:14px;border:1px solid var(--line);background:#fff;}
[data-testid="stChatMessage"]{background:#fff;border:1px solid var(--line);border-radius:16px;padding:.8rem 1rem;box-shadow:0 1px 2px rgba(23,27,58,.03);animation:rise .25s ease both;}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]){background:var(--tint);border-color:#DCD2F7;}
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]){background:var(--tint);border-color:#DCD2F7;}
[data-testid="stChatInput"]{border-radius:14px;border:1px solid var(--line);background:#fff;}
[data-testid="stChatInput"]:focus-within{border-color:var(--violet);box-shadow:0 0 0 3px rgba(125,58,223,.15);}

/* ---------- hero ---------- */
.hero{position:relative;overflow:hidden;border-radius:24px;padding:38px 40px;color:#fff;background:linear-gradient(135deg,#0B2A87 0%,#3A2FC9 52%,#7D3ADF 100%);box-shadow:0 18px 50px rgba(58,47,201,.28);display:grid;grid-template-columns:1.35fr 1fr;gap:28px;align-items:center;animation:rise .45s ease both;}
.hero::before{content:"";position:absolute;right:-90px;top:-110px;width:380px;height:380px;border-radius:50%;background:radial-gradient(circle,rgba(255,217,138,.30),rgba(255,217,138,0) 65%);}
.hero::after{content:"";position:absolute;left:-60px;bottom:-120px;width:320px;height:320px;border-radius:50%;background:radial-gradient(circle,rgba(255,255,255,.14),rgba(255,255,255,0) 65%);}
.hero>*{position:relative;z-index:1;}
.eyebrow{display:inline-flex;align-items:center;gap:8px;padding:6px 12px;border-radius:999px;background:rgba(255,255,255,.14);border:1px solid rgba(255,255,255,.28);font-size:.78rem;font-weight:600;color:#fff;}
.hero h1{margin:16px 0 12px;font-size:2.55rem;line-height:1.12;font-weight:800;letter-spacing:-.025em;color:#fff;padding:0;}
.hero h1 .tag{display:block;margin-top:6px;color:var(--gold-soft);font-size:.84em;}
.hero p.lead{font-size:1.05rem;line-height:1.6;color:#E8E4FF;max-width:560px;margin:0 0 18px;}
.hero .trust{display:flex;flex-wrap:wrap;gap:8px;}
.hero .trust span{font-size:.78rem;font-weight:600;padding:5px 11px;border-radius:999px;background:rgba(255,255,255,.12);color:#fff;border:1px solid rgba(255,255,255,.2);}
.logo-card{background:#fff;border-radius:22px;padding:22px 20px;box-shadow:0 14px 40px rgba(11,42,135,.35);display:flex;align-items:center;justify-content:center;}
.logo-card img{width:100%;max-width:330px;height:auto;display:block;}
@media (max-width:900px){.hero{grid-template-columns:1fr;padding:28px 22px}.hero h1{font-size:1.9rem}}

/* ---------- cards & headings ---------- */
.h2{font-size:1.18rem;font-weight:700;letter-spacing:-.01em;color:var(--ink);margin:26px 0 10px;}
.h2 small{font-weight:500;color:var(--muted);font-size:.85rem;margin-left:6px;}
.card{background:#fff;border:1px solid var(--line);border-radius:var(--r);padding:20px 22px;box-shadow:var(--shadow);animation:rise .35s ease both;}
.feat{height:100%;}
.feat .ico{width:46px;height:46px;border-radius:14px;display:flex;align-items:center;justify-content:center;font-size:1.45rem;background:var(--tint);margin-bottom:12px;}
.feat h3{margin:0 0 6px;font-size:1.06rem;font-weight:700;color:var(--ink);padding:0;}
.feat p{margin:0;color:var(--ink2);font-size:.92rem;line-height:1.55;}
.steps3{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;}
@media (max-width:800px){.steps3{grid-template-columns:1fr}}
.stepcard{display:flex;gap:12px;align-items:flex-start;background:#fff;border:1px solid var(--line);border-radius:14px;padding:14px 16px;}
.stepcard .n{flex:none;width:30px;height:30px;border-radius:50%;background:var(--tint);color:var(--indigo);font-weight:800;display:flex;align-items:center;justify-content:center;font-size:.9rem;}
.stepcard.done .n{background:var(--ok-bg);color:var(--ok);}
.stepcard b{display:block;color:var(--ink);font-size:.93rem;}
.stepcard span{color:var(--ink2);font-size:.83rem;}
.pagehead{display:flex;align-items:center;gap:14px;margin:2px 0 18px;animation:rise .3s ease both;}
.pagehead .ico{width:52px;height:52px;border-radius:16px;background:var(--grad);display:flex;align-items:center;justify-content:center;font-size:1.5rem;box-shadow:0 8px 20px rgba(91,63,212,.28);}
.pagehead h1{margin:0;padding:0;font-size:1.65rem;font-weight:800;letter-spacing:-.02em;color:var(--ink);line-height:1.2;}
.pagehead p{margin:3px 0 0;color:var(--ink2);font-size:.95rem;}
.empty{text-align:center;background:#fff;border:1.5px dashed #C9BEF2;border-radius:20px;padding:44px 24px;animation:rise .3s ease both;}
.empty .big{font-size:2.4rem;margin-bottom:6px;}
.empty h3{margin:0 0 6px;font-size:1.2rem;color:var(--ink);padding:0;}
.empty p{margin:0 auto;max-width:460px;color:var(--ink2);font-size:.95rem;line-height:1.6;}

/* ---------- callouts ---------- */
.callout{display:flex;gap:12px;align-items:flex-start;border-radius:14px;padding:14px 16px;margin:10px 0;border:1px solid transparent;line-height:1.55;font-size:.95rem;}
.callout .ci{flex:none;width:26px;height:26px;border-radius:50%;display:flex;align-items:center;justify-content:center;font-weight:800;font-size:.85rem;background:rgba(255,255,255,.7);}
.callout b.t{display:block;margin-bottom:2px;}
.callout.ok{background:var(--ok-bg);color:#0A4F33;border-color:#BFE6D1;}
.callout.warn{background:var(--warn-bg);color:#6B3A00;border-color:#F3D9A0;}
.callout.bad{background:var(--bad-bg);color:#7F1F16;border-color:#F4C4BE;}
.callout.info{background:var(--info-bg);color:#24396E;border-color:#C9D6F6;}
.callout.plain{background:#fff;color:var(--ink);border-color:var(--line);box-shadow:var(--shadow);}
.callout p{margin:0 0 6px;} .callout p:last-child{margin:0;}

/* ---------- score hero ---------- */
.scorehero{display:grid;grid-template-columns:260px 1fr;gap:22px;align-items:center;background:#fff;border:1px solid var(--line);border-radius:20px;padding:22px 26px;box-shadow:var(--shadow);animation:rise .4s ease both;}
@media (max-width:760px){.scorehero{grid-template-columns:1fr}}
.donutbox{text-align:center;}
.donutbox svg{display:block;margin:0 auto;}
.donutbox .arc{animation:arc 1.1s cubic-bezier(.2,.7,.2,1) both;}
.band{display:inline-flex;align-items:center;gap:6px;margin-top:6px;padding:4px 12px;border-radius:999px;font-weight:700;font-size:.82rem;}
.band.ok{background:var(--ok-bg);color:var(--ok);} .band.ok::before{content:"✓";}
.band.warn{background:var(--warn-bg);color:var(--warn);} .band.warn::before{content:"●";font-size:.6rem;}
.band.bad{background:var(--bad-bg);color:var(--bad);} .band.bad::before{content:"!";font-weight:800;}
.subline{margin-top:8px;color:var(--ink2);font-size:.88rem;}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:14px;}
.tile{background:var(--bg);border:1px solid var(--line);border-radius:14px;padding:14px 16px;}
.tile .lbl{font-size:.78rem;font-weight:600;color:var(--ink2);}
.tile .val{font-size:1.85rem;font-weight:800;letter-spacing:-.02em;color:var(--ink);margin:2px 0 8px;line-height:1.15;}
.tile .val small{font-size:.85rem;font-weight:600;color:var(--muted);margin-left:2px;}
.track{height:8px;border-radius:99px;background:#E4DDF8;overflow:hidden;}
.track>i{display:block;height:100%;border-radius:99px;background:linear-gradient(90deg,#2F55D4,#7D3ADF);animation:fill .9s ease both;}

/* ---------- bars ---------- */
.bars{background:#fff;border:1px solid var(--line);border-radius:var(--r);padding:8px 22px;box-shadow:var(--shadow);}
.bar-row{display:grid;grid-template-columns:minmax(150px,230px) 1fr 64px;gap:14px;align-items:center;padding:11px 0;border-bottom:1px solid #F0ECFB;}
.bar-row:last-child{border-bottom:0;}
.bar-row .name{font-weight:600;color:var(--ink);font-size:.92rem;}
.bar-row .num{text-align:right;font-weight:700;color:var(--ink);font-size:.92rem;font-variant-numeric:tabular-nums;}
.bar-row .num small{color:var(--muted);font-weight:500;}
@media (max-width:640px){.bar-row{grid-template-columns:1fr 54px}.bar-row .track{grid-column:1/-1;order:3}}

/* ---------- chips & badges ---------- */
.chips{display:flex;flex-wrap:wrap;gap:8px;}
.chip{display:inline-flex;align-items:center;gap:6px;padding:6px 12px;border-radius:999px;font-size:.84rem;font-weight:600;line-height:1.2;border:1px solid transparent;}
.chip.ok{background:var(--ok-bg);color:var(--ok);} .chip.ok::before{content:"✓";font-weight:800;}
.chip.miss{background:var(--bad-bg);color:var(--bad);} .chip.miss::before{content:"✕";font-weight:800;font-size:.75rem;}
.chip.tech{background:var(--tint);color:var(--indigo);border-color:#DDD3F8;}
.chip.info{background:var(--info-bg);color:var(--info);}
.badge{display:inline-flex;align-items:center;gap:5px;padding:3px 10px;border-radius:999px;font-size:.76rem;font-weight:700;white-space:nowrap;}
.badge.high{background:var(--bad-bg);color:var(--bad);} .badge.high::before{content:"▲";font-size:.6rem;}
.badge.medium{background:var(--warn-bg);color:var(--warn);} .badge.medium::before{content:"●";font-size:.55rem;}
.badge.low{background:var(--info-bg);color:var(--info);} .badge.low::before{content:"▼";font-size:.6rem;}
.legend{display:flex;gap:16px;flex-wrap:wrap;color:var(--ink2);font-size:.85rem;margin:6px 0 12px;}

/* ---------- list cards, records, steps ---------- */
.cols2{display:grid;grid-template-columns:1fr 1fr;gap:16px;}
@media (max-width:760px){.cols2{grid-template-columns:1fr}}
.listcard{background:#fff;border:1px solid var(--line);border-radius:var(--r);padding:16px 20px;box-shadow:var(--shadow);}
.listcard h4{margin:0 0 10px;font-size:.98rem;font-weight:700;padding:0;display:flex;align-items:center;gap:8px;}
.listcard.ok h4{color:var(--ok);} .listcard.warn h4{color:var(--warn);} .listcard.info h4{color:var(--info);}
.listcard ul{list-style:none;margin:0;padding:0;}
.listcard li{position:relative;padding:7px 0 7px 26px;border-top:1px solid #F3F0FC;font-size:.93rem;line-height:1.5;color:var(--ink);}
.listcard li:first-child{border-top:0;}
.listcard li::before{position:absolute;left:0;top:7px;width:18px;height:18px;border-radius:50%;font-size:.7rem;font-weight:800;display:flex;align-items:center;justify-content:center;}
.listcard.ok li::before{content:"✓";background:var(--ok-bg);color:var(--ok);}
.listcard.warn li::before{content:"!";background:var(--warn-bg);color:var(--warn);}
.listcard.info li::before{content:"→";background:var(--info-bg);color:var(--info);}
.rec{background:#fff;border:1px solid var(--line);border-radius:var(--r);padding:16px 20px;margin:0 0 12px;box-shadow:var(--shadow);animation:rise .3s ease both;}
.rec .top{display:flex;align-items:center;justify-content:space-between;gap:12px;flex-wrap:wrap;}
.rec .title{font-weight:700;font-size:1.02rem;color:var(--ink);}
.rec .sub{color:var(--ink2);font-size:.9rem;margin-top:2px;}
.rec .body{margin-top:8px;color:var(--ink2);font-size:.93rem;line-height:1.6;}
.rec .chips{margin-top:10px;}
.dl{margin-top:12px;}
.dl .k{display:inline-block;font-size:.72rem;font-weight:700;letter-spacing:.06em;text-transform:uppercase;padding:2px 9px;border-radius:6px;margin-bottom:5px;background:var(--tint);color:var(--indigo);}
.dl .k.ok{background:var(--ok-bg);color:var(--ok);} .dl .k.warn{background:var(--warn-bg);color:var(--warn);} .dl .k.info{background:var(--info-bg);color:var(--info);}
.dl ul{margin:0;padding-left:18px;color:var(--ink);font-size:.93rem;line-height:1.55;} .dl p{margin:0;color:var(--ink);font-size:.93rem;line-height:1.6;}
.stepsl{counter-reset:s;display:flex;flex-direction:column;gap:12px;}
.step{display:flex;gap:14px;background:#fff;border:1px solid var(--line);border-radius:var(--r);padding:16px 20px;box-shadow:var(--shadow);animation:rise .3s ease both;}
.step .n{flex:none;width:34px;height:34px;border-radius:50%;background:var(--grad);color:#fff;font-weight:800;display:flex;align-items:center;justify-content:center;font-size:.95rem;}
.step .t{font-weight:700;color:var(--ink);margin-bottom:2px;}
.step .d{color:var(--ink2);font-size:.93rem;line-height:1.6;}
.prose{background:#fff;border:1px solid var(--line);border-radius:var(--r);padding:18px 22px;box-shadow:var(--shadow);color:var(--ink);font-size:.97rem;line-height:1.7;}
.prose p{margin:0 0 10px;} .prose p:last-child{margin:0;} .prose ul{margin:0 0 10px;padding-left:20px;}
code{background:var(--tint);color:var(--indigo);padding:1px 6px;border-radius:6px;font-size:.88em;}

/* ---------- cover letter ---------- */
.paper{max-width:760px;margin:6px auto 0;background:#fff;border:1px solid var(--line);border-radius:6px;padding:46px 56px;box-shadow:0 2px 4px rgba(23,27,58,.05),0 18px 44px rgba(91,63,212,.12);font-family:Georgia,'Times New Roman',serif;font-size:1.03rem;line-height:1.75;color:#20233F;animation:rise .4s ease both;}
.paper p{margin:0 0 1.05em;color:#20233F;}
.paper mark.ph{background:#FFF1D6;color:#6B3A00;border-radius:4px;padding:0 2px;}
@media (max-width:700px){.paper{padding:26px 22px}}
.meta{display:flex;gap:8px;flex-wrap:wrap;justify-content:center;margin:0 0 12px;}

/* ---------- coach ---------- */
.panehead{display:flex;align-items:center;justify-content:space-between;font-weight:700;color:var(--ink);margin:0 0 8px;font-size:1rem;}
.panehead span{font-weight:600;color:var(--muted);font-size:.78rem;background:var(--tint);padding:3px 10px;border-radius:999px;}
.live{font-size:.85rem;color:var(--ink2);line-height:1.5;}
.live .tail{margin-top:8px;padding:10px 12px;border-radius:10px;background:var(--bg);color:var(--muted);max-height:110px;overflow:hidden;font-size:.8rem;}
"""

CSS += r"""
/* =================== v3 colourful / interactive layer =================== */
@property --num{syntax:'<integer>';initial-value:0;inherits:false}
@keyframes cnt{from{--num:0}}
@keyframes floaty{0%,100%{transform:translateY(0) rotate(-2deg)}50%{transform:translateY(-14px) rotate(2deg)}}
@keyframes twinkle{0%,100%{opacity:.25;transform:scale(.7) rotate(0)}50%{opacity:1;transform:scale(1.25) rotate(25deg)}}
@keyframes drift{0%{background-position:0% 50%}50%{background-position:100% 50%}100%{background-position:0% 50%}}
@keyframes shine{0%{background-position:200% 0}100%{background-position:-200% 0}}
@keyframes pulse{0%,100%{box-shadow:0 0 0 0 rgba(125,58,223,.35)}70%{box-shadow:0 0 0 14px rgba(125,58,223,0)}}
@keyframes bob{0%,100%{transform:translateY(0)}50%{transform:translateY(-4px)}}
@keyframes slide{0%{transform:translateX(-100%)}100%{transform:translateX(250%)}}

.count{counter-reset:n var(--num);animation:cnt 1.2s cubic-bezier(.2,.7,.2,1) both;}
.count::after{content:counter(n);}

.stApp{background:
  radial-gradient(700px 420px at 4% 0%,rgba(232,71,139,.16),transparent 62%),
  radial-gradient(800px 480px at 98% 2%,rgba(47,85,212,.18),transparent 62%),
  radial-gradient(700px 520px at 90% 100%,rgba(220,165,74,.20),transparent 62%),
  radial-gradient(600px 420px at 0% 95%,rgba(14,165,164,.12),transparent 62%),var(--bg);background-attachment:fixed;}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#FFFFFF 0%,#F7F1FF 55%,#E9F0FF 100%);}

/* tool colours */
.t-evaluator{--c1:#2F55D4;--c2:#7D3ADF}
.t-scorer{--c1:#E8478B;--c2:#F59E0B}
.t-cover_letter{--c1:#0EA5A4;--c2:#22C55E}
.t-coach{--c1:#F59E0B;--c2:#EF6C3A}

/* hero */
.hero{background:linear-gradient(120deg,#0B2A87,#4A2FD0,#A23AC8,#3A2FC9,#0B2A87);background-size:300% 300%;animation:rise .45s ease both,drift 14s ease infinite;}
.hero h1{font-size:3.4rem;letter-spacing:-.03em;margin:14px 0 4px;color:#fff;background:linear-gradient(100deg,#fff 0%,#fff 38%,#FFD98A 50%,#fff 62%,#fff 100%);background-size:250% 100%;-webkit-background-clip:text;background-clip:text;-webkit-text-fill-color:transparent;animation:shine 6s linear infinite;}
.hero .tagline{font-size:1.35rem;font-weight:700;color:var(--gold-soft);margin:0 0 14px;}
.sp{position:absolute;z-index:0;color:var(--gold-soft);font-style:normal;animation:twinkle 3.2s ease-in-out infinite;pointer-events:none;}
.sp.s1{left:46%;top:14%;font-size:1.3rem}.sp.s2{left:6%;top:70%;font-size:.9rem;animation-delay:.8s}.sp.s3{left:52%;top:78%;font-size:1.6rem;animation-delay:1.4s}.sp.s4{right:6%;top:10%;font-size:1.1rem;animation-delay:.4s}.sp.s5{left:30%;top:8%;font-size:.8rem;animation-delay:2s}
.hero-art{position:relative;display:flex;align-items:center;justify-content:center;min-height:250px;}
.hero-art .orb{position:absolute;width:250px;height:250px;border-radius:50%;background:radial-gradient(circle at 35% 30%,#fff,rgba(255,255,255,.55) 40%,rgba(255,217,138,.30) 65%,rgba(255,217,138,0) 71%);box-shadow:0 0 70px rgba(255,217,138,.55);animation:pulse 3s ease-out infinite;}
.hero-art img{position:relative;height:215px;width:auto;animation:floaty 6s ease-in-out infinite;filter:drop-shadow(0 14px 18px rgba(11,42,135,.35));}
.hero .trust span{transition:transform .2s,background .2s;}.hero .trust span:hover{transform:translateY(-2px);background:rgba(255,255,255,.24);}
@media (max-width:900px){.hero h1{font-size:2.4rem}.hero-art{min-height:190px}.hero-art .orb{width:190px;height:190px}.hero-art img{height:160px}}

/* stats strip */
.stats{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin:18px 0 4px;}
@media (max-width:800px){.stats{grid-template-columns:repeat(2,1fr)}}
.stat{position:relative;overflow:hidden;background:#fff;border:1px solid var(--line);border-radius:18px;padding:16px 18px;box-shadow:var(--shadow);transition:transform .25s,box-shadow .25s;animation:rise .5s ease both;}
.stat::before{content:"";position:absolute;inset:0 0 auto 0;height:5px;background:linear-gradient(90deg,var(--c1),var(--c2));}
.stat:hover{transform:translateY(-5px);box-shadow:0 14px 30px rgba(91,63,212,.18);}
.stat .si{display:block;font-size:1.3rem;color:inherit}
.stat b{display:block;font-size:1.9rem;font-weight:800;letter-spacing:-.02em;color:var(--ink);line-height:1.2;}
.stat span:last-child{color:var(--ink2);font-size:.85rem;font-weight:500;}

/* section headings get a colour dot */
.h2{display:flex;align-items:center;gap:10px;}
.h2::before{content:"";width:10px;height:26px;border-radius:6px;background:linear-gradient(180deg,#2F55D4,#7D3ADF 50%,#E8478B);}

/* steps */
.stepcard{transition:transform .2s,box-shadow .2s;}.stepcard:hover{transform:translateY(-3px);box-shadow:var(--shadow);}
.stepcard.s1 .n{background:#E6ECFB;color:#2F4A8C}.stepcard.s2 .n{background:#FDE8F1;color:#A3185F}.stepcard.s3 .n{background:#DDF5F3;color:#0B6B68}
.stepcard.done .n{background:var(--ok-bg);color:var(--ok)}

/* feature cards */
.feat{position:relative;overflow:hidden;transition:transform .28s cubic-bezier(.2,.8,.2,1),box-shadow .28s;}
.feat::before{content:"";position:absolute;inset:0 0 auto 0;height:6px;background:linear-gradient(90deg,var(--c1),var(--c2));}
.feat::after{content:"";position:absolute;right:-50px;bottom:-60px;width:170px;height:170px;border-radius:50%;background:radial-gradient(circle,var(--c1),transparent 70%);opacity:.12;transition:opacity .3s,transform .4s;}
.feat:hover{transform:translateY(-8px);box-shadow:0 20px 40px rgba(91,63,212,.2);}
.feat:hover::after{opacity:.28;transform:scale(1.25);}
.feat .ico{background:linear-gradient(135deg,var(--c1),var(--c2));box-shadow:0 8px 18px rgba(0,0,0,.14);transition:transform .35s;}
.feat:hover .ico{transform:rotate(-8deg) scale(1.12);}
.feat h3{color:var(--ink)}

/* page header */
.pagehead .ico{background:linear-gradient(135deg,var(--c1,#2F55D4),var(--c2,#7D3ADF));animation:bob 3.4s ease-in-out infinite;}
.pagehead h1{background:linear-gradient(90deg,var(--ink),var(--c1,#2F55D4));-webkit-background-clip:text;background-clip:text;-webkit-text-fill-color:transparent;}

/* buttons: animated gradient + press feedback */
.stApp [data-testid="stBaseButton-primary"],.stApp button[kind="primary"]{background:linear-gradient(110deg,#2F55D4,#7D3ADF,#E8478B,#7D3ADF,#2F55D4);background-size:250% 100%;animation:drift 8s ease infinite;}
.stApp [data-testid="stBaseButton-primary"]:hover,.stApp button[kind="primary"]:hover{transform:translateY(-2px) scale(1.02);}
.stApp [data-testid="stBaseButton-primary"]:active,.stApp button[kind="primary"]:active,.stApp button:active{transform:scale(.97);}
.stApp [data-testid="stBaseButton-secondary"],.stApp button[kind="secondary"]{transition:transform .18s,border-color .18s,background .18s,box-shadow .18s;}
.stApp [data-testid="stBaseButton-secondary"]:hover,.stApp button[kind="secondary"]:hover{transform:translateY(-2px);box-shadow:0 8px 18px rgba(91,63,212,.16);}
.stApp [data-testid="stSidebar"] [data-testid="stBaseButton-secondary"]:hover,.stApp [data-testid="stSidebar"] button[kind="secondary"]:hover{transform:translateX(5px);box-shadow:none;}
.stApp [data-testid="stSidebar"] [data-testid="stBaseButton-primary"],.stApp [data-testid="stSidebar"] button[kind="primary"]{animation:none;background:linear-gradient(110deg,#2F55D4,#7D3ADF 60%,#B03CC8);}

/* tabs as pills */
.stTabs [data-baseweb="tab-list"]{background:#fff;border:1px solid var(--line);border-radius:16px;padding:5px;gap:4px;box-shadow:var(--shadow);overflow-x:auto;}
.stTabs [data-baseweb="tab"]{height:42px;border-radius:12px;padding:0 16px;transition:background .2s,color .2s,transform .2s;}
.stTabs [data-baseweb="tab"]:hover{transform:translateY(-1px)}
.stTabs [aria-selected="true"]{background:linear-gradient(110deg,#2F55D4,#7D3ADF 70%,#B03CC8)!important;color:#fff!important;box-shadow:0 6px 14px rgba(91,63,212,.3);}
.stTabs [aria-selected="true"] p{color:#fff!important}
.stTabs [data-baseweb="tab-highlight"],.stTabs [data-baseweb="tab-border"]{display:none}
.stTabs [data-baseweb="tab-panel"]{padding-top:1.3rem}

/* inputs */
[data-testid="stSidebar"] [data-testid="stFileUploader"] section{background:linear-gradient(135deg,#FBF8FF,#F0F4FF);border-color:#B9A8F0;}
[data-testid="stFileUploader"] section:hover{transform:scale(1.01);transition:transform .2s;box-shadow:0 8px 20px rgba(125,58,223,.15);}

/* results */
.scorehero{position:relative;overflow:hidden;}
.scorehero::before{content:"";position:absolute;inset:0 0 auto 0;height:6px;background:linear-gradient(90deg,#2F55D4,#7D3ADF,#E8478B,#F59E0B);}
.dwrap{position:relative;width:210px;height:210px;margin:0 auto;}
.dnum{position:absolute;inset:0;display:flex;flex-direction:column;align-items:center;justify-content:center;pointer-events:none;}
.dnum .count{font-size:3.6rem;font-weight:800;letter-spacing:-.03em;color:var(--ink);line-height:1;}
.dnum small{margin-top:6px;font-size:.8rem;color:var(--muted);font-weight:500;}
.tile{position:relative;overflow:hidden;transition:transform .22s,box-shadow .22s;}
.tile::before{content:"";position:absolute;inset:0 0 auto 0;height:4px;background:linear-gradient(90deg,var(--c1,#2F55D4),var(--c2,#7D3ADF));}
.tile:hover{transform:translateY(-4px);box-shadow:0 12px 24px rgba(91,63,212,.16);}
.tile .val .count{font-size:inherit}
.track>i{background:linear-gradient(90deg,var(--c1,#2F55D4),var(--c2,#7D3ADF));}
.bar-row .name{display:flex;align-items:center;gap:9px}
.bar-row .name .dot{flex:none;width:10px;height:10px;border-radius:50%;display:inline-block}
.bar-row{transition:background .15s;border-radius:10px}.bar-row:hover{background:#FAF8FF}
.chip{transition:transform .15s,box-shadow .15s;cursor:default;}.chip:hover{transform:translateY(-2px) scale(1.05);box-shadow:0 6px 14px rgba(23,27,58,.12);}
.chip.tc0{background:#E6ECFB;color:#2F4A8C}.chip.tc1{background:#F1EDFD;color:#5B3FD4}.chip.tc2{background:#FDE8F1;color:#A3185F}
.chip.tc3{background:#FFF1D6;color:#8A4B00}.chip.tc4{background:#DDF5F3;color:#0B6B68}.chip.tc5{background:#E3F5EC;color:#0B6B43}
.rec,.step,.listcard,.prose,.callout{transition:transform .2s,box-shadow .2s;}
.rec:hover,.step:hover,.listcard:hover{transform:translateY(-3px);box-shadow:0 14px 28px rgba(91,63,212,.16);}
.rec{border-left:5px solid #7D3ADF;}
.step:nth-child(6n+1) .n{background:linear-gradient(135deg,#2F55D4,#6A5CF0)}.step:nth-child(6n+2) .n{background:linear-gradient(135deg,#E8478B,#F59E0B)}
.step:nth-child(6n+3) .n{background:linear-gradient(135deg,#0EA5A4,#22C55E)}.step:nth-child(6n+4) .n{background:linear-gradient(135deg,#7D3ADF,#E8478B)}
.step:nth-child(6n+5) .n{background:linear-gradient(135deg,#F59E0B,#EF6C3A)}.step:nth-child(6n+6) .n{background:linear-gradient(135deg,#0F37A9,#0EA5E9)}
.listcard.ok{border-top:5px solid #22C55E}.listcard.warn{border-top:5px solid #F59E0B}.listcard.info{border-top:5px solid #2F55D4}
.paper{background:linear-gradient(180deg,#fff,#FFFDF8);}

/* progress + empty + chat */
.live .lamp{display:inline-block;animation:bob 1.2s ease-in-out infinite;font-size:1.2rem;}
.live .shimmer{position:relative;overflow:hidden;height:6px;border-radius:99px;background:#E4DDF8;margin:10px 0 0;}
.live .shimmer::after{content:"";position:absolute;inset:0;width:40%;border-radius:99px;background:linear-gradient(90deg,#2F55D4,#E8478B,#F59E0B);animation:slide 1.3s ease-in-out infinite;}
.empty{background:linear-gradient(135deg,#fff,#F6F1FF);}
.empty .big{display:inline-block;font-size:3rem;animation:floaty 4s ease-in-out infinite;}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]),[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]){background:linear-gradient(135deg,#F1EDFD,#FDE8F1);border-color:#E3D5F7;}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]),[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]){border-left:5px solid #7D3ADF;}
.panehead{border-bottom:3px solid transparent;border-image:linear-gradient(90deg,#2F55D4,#7D3ADF,#E8478B) 1;padding-bottom:6px;}
"""

CSS += r"""
/* =================== v4: always-readable text + aurora background =================== */
:root{color-scheme:light}
@keyframes aurora{0%{transform:translate3d(0,0,0) scale(1)}50%{transform:translate3d(-2%,2%,0) scale(1.06)}100%{transform:translate3d(2%,-2%,0) scale(1.1)}}
@keyframes marquee{from{transform:translateX(0)}to{transform:translateX(-50%)}}
@keyframes grow{from{width:0}}

/* --- readable in light AND dark Streamlit themes: set colours explicitly --- */
.stApp.stApp{color:var(--ink);background:linear-gradient(135deg,#EEF3FF 0%,#F6EEFF 38%,#FFF0F6 70%,#FFF7EA 100%);isolation:isolate;}
.stApp [data-testid="stAppViewContainer"],.stApp [data-testid="stMain"],.stApp [data-testid="stMainBlockContainer"],.stApp [data-testid="stHeader"],.stApp [data-testid="stBottom"],.stApp [data-testid="stBottom"]>div,.stApp [data-testid="stBottomBlockContainer"]{background:transparent!important;}
.stApp [data-testid="stSidebar"],.stApp [data-testid="stSidebarContent"],.stApp [data-testid="stMarkdownContainer"],.stApp [data-testid="stCaptionContainer"],.stApp [data-testid="stWidgetLabel"],.stApp [data-testid="stChatMessage"],.stApp [data-testid="stChatMessageContent"],.stApp [data-testid="stExpander"],.stApp [data-testid="stStatus"],.stApp [data-testid="stText"]{color:var(--ink);}
.stApp [data-testid="stMarkdownContainer"] p,.stApp [data-testid="stMarkdownContainer"] li,.stApp [data-testid="stMarkdownContainer"] strong,.stApp [data-testid="stMarkdownContainer"] em,.stApp [data-testid="stMarkdownContainer"] h1,.stApp [data-testid="stMarkdownContainer"] h2,.stApp [data-testid="stMarkdownContainer"] h3,.stApp [data-testid="stMarkdownContainer"] h4{color:inherit;}
.stApp [data-testid="stCaptionContainer"],.stApp [data-testid="stCaptionContainer"] *{color:var(--muted);}
.stApp [data-testid="stChatMessage"] *{color:inherit;}
.stApp [data-testid="stChatMessage"] :is(p,li,span,strong,em,h1,h2,h3,h4,td,th,blockquote,ol,ul){color:var(--ink)!important;}
.stApp [data-testid="stChatMessage"] code,.stApp code{background:var(--tint)!important;color:var(--indigo)!important;}
.stApp a{color:var(--indigo);}
.stApp [data-testid="stHeader"] *,.stApp [data-testid="stToolbar"] *{color:var(--ink2);}
.stApp [data-testid="stExpander"] details,.stApp [data-testid="stExpander"] summary,.stApp [data-testid="stStatus"] details,.stApp [data-testid="stStatus"] summary{background:#fff;color:var(--ink);}
.stApp [data-testid="stExpander"] summary:hover{background:var(--tint);}
.stApp [data-testid="stCode"] pre,.stApp [data-testid="stCode"] code{background:#F6F3FF!important;color:var(--ink)!important;}
.stApp [data-testid="stFileUploader"] section,.stApp [data-testid="stFileUploader"] section *{color:var(--ink2);}
.stApp [data-testid="stFileUploaderFile"],.stApp [data-testid="stFileUploaderFile"] *{color:var(--ink);}
.stApp [data-testid="stFileUploader"] section small{color:var(--muted);}

/* the text boxes: colour AND background set on the element itself */
.stApp input,.stApp textarea,.stApp [data-baseweb="input"] input,.stApp [data-baseweb="textarea"] textarea,.stApp [data-testid="stChatInputTextArea"]{background:#fff!important;color:var(--ink)!important;-webkit-text-fill-color:var(--ink)!important;caret-color:var(--violet);font-size:.98rem;}
.stApp input::placeholder,.stApp textarea::placeholder{color:#8A90B0!important;-webkit-text-fill-color:#8A90B0!important;opacity:1;}
.stApp [data-baseweb="input"],.stApp [data-baseweb="base-input"],.stApp [data-baseweb="textarea"],.stApp [data-testid="stTextInputRootElement"],.stApp [data-testid="stTextAreaRootElement"],.stApp [data-testid="stChatInput"],.stApp [data-testid="stChatInput"]>div{background:#fff!important;}
.stApp [data-baseweb="input"] button,.stApp [data-baseweb="input"] svg{color:var(--ink2)!important;fill:var(--ink2);}
.stApp [data-testid="stChatInput"] button svg{fill:var(--indigo);}
.stApp [data-testid="stTextArea"] textarea{line-height:1.6;}
.stApp ::selection{background:#D9C8FF;color:var(--ink);}

/* buttons keep their own colours in any theme */
.stApp [data-testid="stBaseButton-secondary"],.stApp button[kind="secondary"],.stApp [data-testid="stDownloadButton"] button{background:#fff;color:var(--ink);}
.stApp [data-testid="stBaseButton-secondary"] *,.stApp [data-testid="stDownloadButton"] button *{color:inherit;}
.stApp [data-testid="stBaseButton-primary"] *,.stApp button[kind="primary"] *{color:#fff!important;}
.stApp [data-testid="stSidebar"] [data-testid="stBaseButton-secondary"]{background:transparent;}
.stApp [data-testid="stVerticalBlockBorderWrapper"]{border-color:rgba(125,58,223,.18);}

/* --- aurora background --- */
.stApp::before{content:"";position:fixed;inset:-12%;z-index:-1;pointer-events:none;
  background:
   radial-gradient(38vw 38vw at 8% 10%,rgba(255,150,196,.46),transparent 62%),
   radial-gradient(42vw 42vw at 94% 6%,rgba(120,160,255,.44),transparent 62%),
   radial-gradient(40vw 40vw at 88% 94%,rgba(255,196,120,.44),transparent 62%),
   radial-gradient(36vw 36vw at 8% 92%,rgba(110,222,196,.40),transparent 62%),
   radial-gradient(34vw 34vw at 52% 48%,rgba(190,160,255,.30),transparent 64%);
  animation:aurora 22s ease-in-out infinite alternate;}
.stApp::after{content:"";position:fixed;inset:0;z-index:-1;pointer-events:none;
  background-image:radial-gradient(rgba(91,63,212,.20) 1px,transparent 1.3px);background-size:26px 26px;
  -webkit-mask-image:linear-gradient(180deg,rgba(0,0,0,.7),rgba(0,0,0,.05) 80%);mask-image:linear-gradient(180deg,rgba(0,0,0,.7),rgba(0,0,0,.05) 80%);}
[data-testid="stSidebar"]{background:linear-gradient(180deg,rgba(255,255,255,.86),rgba(246,240,255,.82) 55%,rgba(232,240,255,.86))!important;backdrop-filter:blur(18px);-webkit-backdrop-filter:blur(18px);border-right:1px solid rgba(125,58,223,.16);}

/* --- glass cards --- */
.card,.stat,.stepcard,.rec,.step,.listcard,.prose,.bars,.scorehero,.tile,.callout.plain,.empty{background:rgba(255,255,255,.88);backdrop-filter:blur(10px);-webkit-backdrop-filter:blur(10px);border-color:rgba(125,58,223,.14);}
.tile{background:rgba(250,248,255,.9)}
.hero .trust span{backdrop-filter:blur(6px)}

/* --- typography --- */
.hero h1,.h2,.pagehead h1,.feat h3,.stat b,.dnum .count,.empty h3,.rec .title,.step .t{font-family:'Plus Jakarta Sans','Inter',system-ui,sans-serif;}
.h2{font-size:1.28rem;}

/* --- ribbon (scrolling keywords) --- */
.ribbon{overflow:hidden;margin:18px 0 2px;border-radius:999px;background:rgba(255,255,255,.7);border:1px solid rgba(125,58,223,.16);backdrop-filter:blur(8px);-webkit-mask-image:linear-gradient(90deg,transparent,#000 8%,#000 92%,transparent);mask-image:linear-gradient(90deg,transparent,#000 8%,#000 92%,transparent);}
.rtrack{display:flex;gap:34px;width:max-content;padding:11px 0;animation:marquee 34s linear infinite;}
.ribbon:hover .rtrack{animation-play-state:paused;}
.rtrack span{white-space:nowrap;font-weight:600;font-size:.88rem;color:var(--ink2);}
.rtrack span:nth-child(5n+1){color:#2F4A8C}.rtrack span:nth-child(5n+2){color:#A3185F}.rtrack span:nth-child(5n+3){color:#0B6B68}.rtrack span:nth-child(5n+4){color:#8A4B00}.rtrack span:nth-child(5n+5){color:#5B3FD4}

/* --- sidebar setup meter --- */
.meter{height:7px;border-radius:99px;background:#E4DDF8;overflow:hidden;margin:10px 2px 4px;}
.meter i{display:block;height:100%;border-radius:99px;background:linear-gradient(90deg,#2F55D4,#7D3ADF,#E8478B);animation:grow .8s ease both;transition:width .4s;}
.meter-note{font-size:.76rem;color:var(--ink2);font-weight:600;margin:0 2px 2px;}
.meter-note.done{color:var(--ok);}

/* --- polish --- */
*{scrollbar-width:thin;scrollbar-color:#C9BBF1 transparent;}
*::-webkit-scrollbar{width:10px;height:10px}*::-webkit-scrollbar-thumb{background:#C9BBF1;border-radius:10px;border:2px solid transparent;background-clip:content-box}
.stat:nth-child(2){animation-delay:.06s}.stat:nth-child(3){animation-delay:.12s}.stat:nth-child(4){animation-delay:.18s}
.stApp [data-testid="stChatMessage"]{box-shadow:0 6px 18px rgba(91,63,212,.08);}
.stApp [data-testid="stChatInput"]{border:1.5px solid rgba(125,58,223,.28);box-shadow:0 6px 18px rgba(91,63,212,.10);}
.hero{box-shadow:0 24px 60px rgba(58,47,201,.32);}
"""


def inject_css() -> None:
    st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)


def render_html(markup: str) -> None:
    """Render trusted-by-construction HTML (every dynamic piece goes through esc())."""
    markup = re.sub(r"\s*\n\s*", " ", markup).strip()
    if hasattr(st, "html"):
        st.html(markup)
    else:  # older Streamlit versions
        st.markdown(markup, unsafe_allow_html=True)


# ===========================================================================
# HTML components
# ===========================================================================
# (from, to) gradient pairs used for bars and tiles (decorative; values are always printed)
TONES = [
    ("#2F55D4", "#6A5CF0"),  # blue
    ("#E8478B", "#F59E0B"),  # pink -> amber
    ("#0EA5A4", "#22C55E"),  # teal -> green
    ("#7D3ADF", "#E8478B"),  # violet -> pink
    ("#F59E0B", "#EF6C3A"),  # amber -> orange
    ("#0F37A9", "#0EA5E9"),  # navy -> sky
]


def h2(text: str, sub: str = "") -> str:
    small = f"<small>{esc(sub)}</small>" if sub else ""
    return f'<div class="h2">{esc(text)}{small}</div>'


def chip(text: str, kind: str = "tech", idx: int = 0) -> str:
    extra = f" tc{idx % 6}" if kind == "tech" else ""
    return f'<span class="chip {kind}{extra}">{esc(text)}</span>'


def chips(items, kind: str = "tech") -> str:
    return f'<div class="chips">{"".join(chip(i, kind, n) for n, i in enumerate(items))}</div>'


def badge(priority: str, label: str = "") -> str:
    p = (priority or "").lower()
    if p not in ("high", "medium", "low"):
        return ""
    return f'<span class="badge {p}">{esc(label or priority.capitalize())}</span>'


def callout(kind: str, title: str, body_html: str, icon: str = "") -> str:
    icon = icon or {"ok": "✓", "warn": "!", "bad": "✕", "info": "i", "plain": "★"}.get(kind, "i")
    t = f'<b class="t">{esc(title)}</b>' if title else ""
    return (
        f'<div class="callout {kind}"><div class="ci">{esc(icon)}</div>'
        f"<div>{t}{body_html}</div></div>"
    )


def rich_text(text: str) -> str:
    """Escape and turn plain/markdown-ish text into <p> and <ul>."""
    out, para, items = [], [], []

    def flush_para():
        if para:
            out.append(f"<p>{inline(' '.join(para))}</p>")
            para.clear()

    def flush_items():
        if items:
            out.append("<ul>" + "".join(f"<li>{inline(i)}</li>" for i in items) + "</ul>")
            items.clear()

    for raw in text.splitlines():
        s = raw.strip()
        if not s:
            flush_para()
            flush_items()
        elif _BULLET.match(s):
            flush_para()
            items.append(_BULLET.sub("", s))
        else:
            flush_items()
            para.append(s)
    flush_para()
    flush_items()
    return "".join(out)


def pagehead(icon: str, title: str, sub: str, key: str = "") -> None:
    render_html(
        f'<div class="pagehead t-{esc(key)}"><div class="ico">{esc(icon)}</div>'
        f"<div><h1>{esc(title)}</h1><p>{esc(sub)}</p></div></div>"
    )


def donut(score: int, uid: str, caption: str = "out of 100") -> str:
    """Donut for a 0-100 score: gradient arc on a light track. The number in the
    middle is HTML so it can count up (CSS only)."""
    score = max(0, min(100, int(score)))
    r = 80
    circ = 2 * math.pi * r
    offset = circ * (1 - score / 100)
    arc = ""
    if score > 0:
        arc = (
            f'<circle class="arc" cx="100" cy="100" r="{r}" fill="none" '
            f'style="stroke:url(#g{uid}) #5B3FD4;--circ:{circ:.2f}" stroke-width="16" '
            f'stroke-linecap="round" stroke-dasharray="{circ:.2f}" '
            f'stroke-dashoffset="{offset:.2f}" transform="rotate(-90 100 100)">'
            f"<title>Score: {score} out of 100</title></circle>"
        )
    return (
        f'<div class="dwrap"><svg viewBox="0 0 200 200" width="210" height="210" role="img" '
        f'aria-label="Score: {score} out of 100">'
        f'<defs><linearGradient id="g{uid}" x1="0" y1="0" x2="1" y2="1">'
        f'<stop offset="0" stop-color="#2F55D4"/><stop offset=".55" stop-color="#7D3ADF"/>'
        f'<stop offset="1" stop-color="#E8478B"/></linearGradient></defs>'
        f'<circle cx="100" cy="100" r="{r}" fill="none" stroke="#E4DDF8" stroke-width="16"/>'
        f"{arc}</svg>"
        f'<div class="dnum"><span class="count" style="--num:{score}"></span>'
        f"<small>{esc(caption)}</small></div></div>"
    )


def tile(label: str, value: str, unit: str, pct: float, tone: int = 0) -> str:
    pct = max(0, min(100, pct))
    c1, c2 = TONES[tone % len(TONES)]
    if re.fullmatch(r"\d{1,3}", str(value)):
        shown = f'<span class="count" style="--num:{int(value)}"></span>'
    else:
        shown = esc(value)
    return (
        f'<div class="tile" style="--c1:{c1};--c2:{c2}"><div class="lbl">{esc(label)}</div>'
        f'<div class="val">{shown}<small>{esc(unit)}</small></div>'
        f'<div class="track"><i style="width:{pct:.0f}%"></i></div></div>'
    )


def score_hero(score: int, uid: str, tiles_html: str, sub: str = "", level: str = "", css: str = "") -> str:
    if not level:
        level, css = band(score)
    sub_html = f'<div class="subline">{esc(sub)}</div>' if sub else ""
    return (
        f'<div class="scorehero"><div class="donutbox">{donut(score, uid)}'
        f'<span class="band {css}">{esc(level)}</span>{sub_html}</div>'
        f'<div class="tiles">{tiles_html}</div></div>'
    )


def bars_html(rows) -> str:
    out = []
    for n, (label, got, mx) in enumerate(rows):
        pct = 100 * got / mx
        c1, c2 = TONES[n % len(TONES)]
        out.append(
            f'<div class="bar-row"><div class="name"><i class="dot" style="background:{c1}"></i>{esc(label)}</div>'
            f'<div class="track" role="img" aria-label="{esc(label)}: {fmt_num(got)} of {mx}">'
            f'<i style="width:{pct:.0f}%;background:linear-gradient(90deg,{c1},{c2});animation-delay:{n * 0.08:.2f}s"></i></div>'
            f'<div class="num">{fmt_num(got)}<small> / {mx}</small></div></div>'
        )
    return f'<div class="bars">{"".join(out)}</div>'


def list_card(title: str, items, kind: str) -> str:
    if not items:
        return ""
    lis = "".join(f"<li>{inline(i)}</li>" for i in items)
    return f'<div class="listcard {kind}"><h4>{esc(title)}</h4><ul>{lis}</ul></div>'


def steps_html(items) -> str:
    out = []
    for n, item in enumerate(items, 1):
        title, body = split_title(item)
        t = f'<div class="t">{inline(title)}</div>' if title else ""
        out.append(
            f'<div class="step"><div class="n">{n}</div><div>{t}'
            f'<div class="d">{inline(body)}</div></div></div>'
        )
    return f'<div class="stepsl">{"".join(out)}</div>'


def dl_html(label: str, lines, kind: str = "") -> str:
    if not lines:
        return ""
    if len(lines) == 1:
        body = f"<p>{inline(lines[0])}</p>"
    else:
        body = "<ul>" + "".join(f"<li>{inline(l)}</li>" for l in lines) + "</ul>"
    return f'<div class="dl"><span class="k {kind}">{esc(label)}</span>{body}</div>'


FIELD_KIND = {
    "strengths": "ok", "strength": "ok",
    "weaknesses": "warn", "weakness": "warn",
    "recommended improvements": "info", "specific improvement": "info",
    "improvement": "info", "specific improvements": "info",
}


def fields_html(fields: dict, skip=()) -> str:
    return "".join(
        dl_html(label.title(), lines, FIELD_KIND.get(label, ""))
        for label, lines in fields.items()
        if label not in skip
    )


def empty_note(text: str = "Nothing was returned for this section.") -> str:
    return callout("info", "", f"<p>{esc(text)}</p>")


def fallback_markdown(text: str) -> None:
    """Last resort for a section we couldn't parse: let Streamlit render it."""
    st.markdown(esc_md(text))


# ===========================================================================
# Rich renderers
# ===========================================================================
SCORE_HEADINGS = {
    "breakdown": ("score breakdown",),
    "matched": ("keywords matched", "matched keywords"),
    "missing": ("missing keywords", "keywords missing"),
    "two_line": ("two-line analysis", "two line analysis", "2-line analysis"),
    "gaps": ("skill gap analysis", "skill gaps"),
    "improve": ("overall improvement suggestions", "improvement suggestions"),
    "industry": ("industry-specific feedback", "industry specific feedback"),
}

EVAL_HEADINGS = {
    "role": ("target role",),
    "breakdown": ("score breakdown",),
    "detail": ("detailed evaluation",),
    "strengths": ("resume strengths",),
    "weaknesses": ("resume weaknesses",),
    "skills": ("skills found in resume", "skills found"),
    "recommended": ("recommended skills to add", "recommended skills"),
    "ats": ("ats analysis",),
    "projects": ("project analysis",),
    "experience": ("experience analysis",),
    "plan": ("action plan",),
    "summary": ("final summary",),
}

ATS_HEADINGS = {
    "found": ("keywords found",),
    "missing": ("important keywords missing", "keywords missing", "missing keywords"),
    "format": ("formatting issues",),
    "improve": ("ats improvements",),
}

GAP_FIELDS = ("skill", "importance for the role", "importance", "why it matters", "reason", "priority")
EVAL_CATEGORIES = (
    "contact information", "professional summary", "skills", "experience",
    "projects", "education", "certifications", "ats compatibility", "writing",
)


def render_gap_cards(records, title_key="skill") -> str:
    out = []
    for r in records:
        name = join_lines(r.get(title_key, []))
        prio = priority_of(join_lines(r.get("importance for the role", r.get("importance", r.get("priority", [])))))
        why = join_lines(r.get("why it matters", r.get("reason", [])))
        body = f'<div class="body">{inline(why)}</div>' if why else ""
        out.append(
            f'<div class="rec"><div class="top"><div class="title">{inline(name)}</div>'
            f"{badge(prio, prio + ' priority' if prio else '')}</div>{body}</div>"
        )
    return "".join(out)


def render_scorer(report: str) -> None:
    scores = extract_scores(report)
    sec = parse_sections(report, SCORE_HEADINGS)
    tabs = st.tabs(["Overview", "Keywords", "Skill gaps", "Improvements", "Full report"])

    with tabs[0]:
        if "Score" in scores:
            tiles = ""
            if "Overall Match" in scores:
                tiles += tile("Overall match", str(scores["Overall Match"]), "%", scores["Overall Match"], 0)
            if "Reliability" in scores:
                tiles += tile("Reliability", str(scores["Reliability"]), "/ 100", scores["Reliability"], 2)
            if "ATS Compatibility" in scores:
                tiles += tile("ATS compatibility", str(scores["ATS Compatibility"]), "/ 100", scores["ATS Compatibility"], 1)
            render_html(score_hero(scores["Score"], "sc", tiles, "Overall resume score"))
        else:
            render_html(callout(
                "warn", "Couldn't read the scores",
                "<p>The model didn't follow the expected format. The full text is in the "
                "<b>Full report</b> tab.</p>"))

        lines = bullets(sec.get("two_line", ""))
        lines = [re.sub(r"^(strengths?|weakness(?:es)?|risk)\s*:\s*", "", l, flags=re.I) for l in lines][:2]
        if lines:
            render_html(h2("At a glance"))
            html_ = callout("ok", "Biggest strength", f"<p>{inline(lines[0])}</p>")
            if len(lines) > 1:
                html_ += callout("warn", "Biggest risk", f"<p>{inline(lines[1])}</p>")
            render_html(html_)

        rows = parse_breakdown(sec.get("breakdown", ""))
        if rows:
            render_html(h2("Score breakdown", "weighted rubric"))
            render_html(bars_html(rows))

    with tabs[1]:
        matched = bullets(sec.get("matched", ""), True)
        missing = bullets(sec.get("missing", ""), True)
        if not (matched or missing):
            render_html(empty_note("No keyword lists were returned."))
        else:
            render_html(h2("Keywords matched", f"{len(matched)} found in your resume"))
            render_html(chips(matched, "ok") if matched else empty_note("None found."))
            render_html(h2("Missing keywords", f"{len(missing)} from the job description"))
            render_html(chips(missing, "miss") if missing else empty_note("None — great coverage."))

    with tabs[2]:
        text = sec.get("gaps", "")
        records = parse_records(text, "skill", GAP_FIELDS)
        if records:
            counts = {p: sum(1 for r in records if priority_of(join_lines(
                r.get("importance for the role", r.get("importance", [])))) == p)
                for p in ("High", "Medium", "Low")}
            render_html(
                '<div class="legend">'
                + "".join(badge(p, f"{n} {p.lower()}") for p, n in counts.items() if n)
                + "</div>"
            )
            render_html(render_gap_cards(records))
        elif text:
            fallback_markdown(text)
        else:
            render_html(empty_note("No skill gaps were reported."))

    with tabs[3]:
        items = list_items(sec.get("improve", ""))
        if items:
            render_html(h2("Overall improvement suggestions", "ordered by impact"))
            render_html(steps_html(items))
        if sec.get("industry"):
            render_html(h2("Industry-specific feedback"))
            render_html(f'<div class="prose">{rich_text(sec["industry"])}</div>')
        if not items and not sec.get("industry"):
            render_html(empty_note())

    with tabs[4]:
        st.markdown(esc_md(report))


def detail_blocks(text: str) -> list:
    """Split 'Detailed evaluation' into [(title, score_text, body)] by category."""
    blocks, cur = [], None
    for raw in text.splitlines():
        s = raw.strip()
        n = norm_heading(s)
        is_head = (
            s and not _BULLET.match(s) and len(s) < 90
            and any(n.startswith(c) for c in EVAL_CATEGORIES)
            and (re.search(r"\d+(?:\.\d+)?\s*/\s*\d+", s) or s.endswith(":") or s.startswith(("#", "**")))
        )
        if is_head:
            m = re.search(r"(\d+(?:\.\d+)?)\s*/\s*(\d+)", s)
            title = re.sub(r"\(?\s*\d+(?:\.\d+)?\s*/\s*\d+\s*(?:points?)?\s*\)?", "", plain(s))
            title = re.sub(r"^#{1,6}\s*|^\d{1,2}[.)]\s*", "", title).strip(" :-—*")
            cur = [title, f"{fmt_num(float(m.group(1)))}/{m.group(2)}" if m else "", []]
            blocks.append(cur)
        elif cur is not None:
            cur[2].append(raw)
    return [(t, sc, "\n".join(b).strip()) for t, sc, b in blocks]


def render_evaluator(report: str) -> None:
    head = extract_eval_headline(report)
    sec = parse_sections(report, EVAL_HEADINGS)
    tabs = st.tabs(["Overview", "Scorecard", "Skills", "ATS", "Projects & experience", "Action plan", "Full report"])

    strengths = bullets(sec.get("strengths", ""))
    weaknesses = bullets(sec.get("weaknesses", ""))

    with tabs[0]:
        if "score" in head:
            level_raw = head.get("level", "")
            low = level_raw.lower()
            if "excellent" in low or "strong" in low:
                css = "ok"
            elif "average" in low:
                css = "warn"
            elif level_raw:
                css = "bad"
            else:
                css = ""
            tiles = ""
            if "ats" in head:
                tiles += tile("ATS score", fmt_num(head["ats"]), "/ 10", head["ats"] * 10, 1)
            tiles += tile("Strengths found", str(len(strengths)), "", min(100, len(strengths) * 100 / 7), 2)
            tiles += tile("Areas to improve", str(len(weaknesses)), "", min(100, len(weaknesses) * 100 / 7), 4)
            role = join_lines(sec.get("role", "").splitlines()).strip("[] ")
            sub = f"Target role: {role}" if role else "Overall resume score"
            if level_raw:
                render_html(score_hero(int(head["score"]), "ev", tiles, sub, level_raw, css))
            else:
                render_html(score_hero(int(head["score"]), "ev", tiles, sub))
        else:
            render_html(callout(
                "warn", "Couldn't read the overall score",
                "<p>The full text is in the <b>Full report</b> tab.</p>"))

        if strengths or weaknesses:
            render_html(h2("Strengths & weaknesses"))
            render_html(
                f'<div class="cols2">{list_card("What’s working", strengths, "ok")}'
                f'{list_card("What to fix", weaknesses, "warn")}</div>'
            )
        if sec.get("summary"):
            render_html(h2("Final summary"))
            render_html(callout("plain", "", rich_text(sec["summary"]), "★"))

    with tabs[1]:
        rows = parse_breakdown(sec.get("breakdown", ""))
        if rows:
            render_html(h2("Score breakdown", "by category"))
            render_html(bars_html(rows))
        elif sec.get("breakdown"):
            fallback_markdown(sec["breakdown"])
        text = sec.get("detail", "")
        if text:
            render_html(h2("Detailed evaluation", "open a category for details"))
            blocks = detail_blocks(text)
            if len(blocks) >= 3:
                for title, sc, body in blocks:
                    label = f"{title} · {sc}" if sc else title
                    with st.expander(label):
                        pre, fields = parse_fields(
                            body, ("score", "strengths", "weaknesses", "specific improvement",
                                   "specific improvements", "improvement"))
                        if pre or fields:
                            render_html((f'<div class="prose">{rich_text(chr(10).join(pre))}</div>' if pre else "")
                                        + fields_html(fields, skip=("score",)))
                        else:
                            st.write("—")
            else:
                fallback_markdown(text)

    with tabs[2]:
        groups = parse_groups(sec.get("skills", ""))
        if groups:
            render_html(h2("Skills found in your resume"))
            for g, items in groups:
                render_html(f'<div class="rec"><div class="title">{esc(g)}</div>{chips(items, "tech")}</div>')
        elif sec.get("skills"):
            fallback_markdown(sec["skills"])
        recs = parse_records(sec.get("recommended", ""), "skill", GAP_FIELDS)
        if recs:
            render_html(h2("Recommended skills to add", "based on your target role"))
            render_html(render_gap_cards(recs))
        elif sec.get("recommended"):
            render_html(h2("Recommended skills to add"))
            fallback_markdown(sec["recommended"])
        if not (groups or recs or sec.get("skills") or sec.get("recommended")):
            render_html(empty_note())

    with tabs[3]:
        ats = parse_sections(sec.get("ats", ""), ATS_HEADINGS)
        if "ats" in head:
            render_html(score_hero(
                int(round(head["ats"] * 10)), "ats",
                tile("ATS score", fmt_num(head["ats"]), "/ 10", head["ats"] * 10, 1),
                "How well automated screeners can read your resume"))
        found, missing = bullets(ats.get("found", ""), True), bullets(ats.get("missing", ""), True)
        if found:
            render_html(h2("Keywords found"))
            render_html(chips(found, "ok"))
        if missing:
            render_html(h2("Important keywords missing"))
            render_html(chips(missing, "miss"))
        fmt, imp = bullets(ats.get("format", "")), bullets(ats.get("improve", ""))
        if fmt or imp:
            render_html(h2("Fixes"))
            render_html(
                f'<div class="cols2">{list_card("Formatting issues", fmt, "warn")}'
                f'{list_card("ATS improvements", imp, "info")}</div>'
            )
        if not (found or missing or fmt or imp):
            if sec.get("ats"):
                fallback_markdown(sec["ats"])
            else:
                render_html(empty_note())

    with tabs[4]:
        shown = False
        proj = parse_records(
            sec.get("projects", ""), "project name",
            ("technologies", "strengths", "weaknesses", "recommended improvements"))
        if proj:
            shown = True
            render_html(h2("Project analysis"))
            for r in proj:
                tech = bullets(join_lines(r.get("technologies", [])), True)
                render_html(
                    f'<div class="rec"><div class="title">{inline(join_lines(r.get("project name", [])))}</div>'
                    f'{chips(tech, "tech") if tech else ""}'
                    f'{fields_html(r, skip=("project name", "technologies"))}</div>')
        elif sec.get("projects"):
            shown = True
            render_html(h2("Project analysis"))
            fallback_markdown(sec["projects"])
        exp = parse_records(
            sec.get("experience", ""), "role",
            ("company", "strengths", "weaknesses", "recommended improvements"))
        if exp:
            shown = True
            render_html(h2("Experience analysis"))
            for r in exp:
                company = join_lines(r.get("company", []))
                sub = f'<div class="sub">{inline(company)}</div>' if company else ""
                render_html(
                    f'<div class="rec"><div class="title">{inline(join_lines(r.get("role", [])))}</div>{sub}'
                    f'{fields_html(r, skip=("role", "company"))}</div>')
        elif sec.get("experience"):
            shown = True
            render_html(h2("Experience analysis"))
            fallback_markdown(sec["experience"])
        if not shown:
            render_html(empty_note())

    with tabs[5]:
        items = list_items(sec.get("plan", ""))
        if items:
            render_html(h2("Your action plan", "highest impact first"))
            render_html(steps_html(items))
        else:
            render_html(empty_note())

    with tabs[6]:
        st.markdown(esc_md(report))


def render_letter(text: str) -> None:
    clean = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", text.strip())
    paras = [p.strip() for p in re.split(r"\n\s*\n", clean) if p.strip()]
    body = "".join(f"<p>{esc(p).replace(chr(10), '<br>')}</p>" for p in paras)
    body = re.sub(r"\[[^\]\n]{1,60}\]", lambda m: f'<mark class="ph">{m.group(0)}</mark>', body)
    has_ph = '<mark class="ph">' in body
    words = len(clean.split())
    tabs = st.tabs(["Letter", "Plain text"])
    with tabs[0]:
        render_html(
            f'<div class="meta">{chip(f"{words} words", "info")}{chip("Tailored to the job description", "ok")}</div>'
        )
        if has_ph:
            render_html(callout(
                "warn", "Fill in the highlighted placeholders",
                "<p>The job description didn't include everything (e.g. company or hiring "
                "manager), so those parts are left in [brackets].</p>"))
        render_html(f'<div class="paper">{body}</div>')
    with tabs[1]:
        st.caption("Use the copy icon at the top-right of the box.")
        try:
            st.code(clean, language=None, wrap_lines=True)
        except TypeError:  # older Streamlit versions
            st.code(clean, language=None)


# ===========================================================================
# Generate-and-show flow shared by the three one-shot tools
# ===========================================================================
def generate(prompt_text: str, *, api_key: str, temperature, label: str, working: str):
    """Stream the model's answer inside a status box. Returns the text or None."""
    with st.status(label, expanded=True) as status:
        live = st.empty()
        full = ""
        try:
            chat = get_chat_model(api_key, temperature)
            for chunk in chat.stream(prompt_text):
                full += chunk_to_text(chunk.content)
                tail = esc(re.sub(r"\s+", " ", full[-260:]))
                live.markdown(
                    f'<div class="live"><span class="lamp">🪔</span> {esc(working)} · '
                    f"<b>{len(full.split())}</b> words so far"
                    f'<div class="shimmer"></div><div class="tail">…{tail}</div></div>',
                    unsafe_allow_html=True,
                )
            if not full.strip():
                raise ValueError("The model returned an empty response.")
        except Exception as e:
            title, detail = explain_error(e)
            status.update(label=title, state="error", expanded=True)
            live.empty()
            render_html(callout("bad", title, f"<p>{esc(detail)}</p>"))
            return None
        status.update(label="✨ Your wish is granted — results are ready", state="complete", expanded=False)
    if hasattr(st, "toast"):
        st.toast("Your wish is granted!", icon="✨")
    return full


def run_and_show(*, state_key, label, working, prompt_text, ctx, clicked, download_name,
                 renderer, temperature=None):
    """Generate on click, keep the result in session state (so it survives
    reruns such as a download click) and show it with `renderer`."""
    out = st.container()
    if clicked:
        with out:
            text = generate(prompt_text, api_key=ctx.api_key, temperature=temperature,
                            label=label, working=working)
        if text:
            st.session_state[state_key] = text
            top = extract_scores(text).get("Score", extract_eval_headline(text).get("score"))
            if top is not None and top >= 85 and hasattr(st, "balloons"):
                st.balloons()

    report = st.session_state.get(state_key)
    if not report:
        return
    with out:
        st.download_button(
            "⬇ Download as .txt", data=report, file_name=download_name,
            mime="text/plain", key=f"download_{state_key}",
        )
        renderer(report)


# ===========================================================================
# Pages
# ===========================================================================
def resume_ready(ctx) -> bool:
    """Show a friendly empty state and return False if there's no usable resume yet."""
    if ctx.data is None:
        render_html(
            '<div class="empty"><div class="big">🪔</div><h3>Upload your resume to make your first wish</h3>'
            "<p>Add your Gemini API key and drop a PDF into the sidebar. "
            "One upload powers all four tools.</p></div>"
        )
        return False
    if not ctx.text:
        render_html(callout(
            "warn", "No text found in this PDF",
            "<p>It may be a scanned image. Please upload a text-based PDF "
            "(export it from Word or Google Docs).</p>"))
        return False
    return True


def key_hint(ctx) -> None:
    if not ctx.api_key:
        render_html(callout("info", "", "<p>Add your <b>Gemini API key</b> in the sidebar to enable this tool.</p>"))


ROLE_SUGGESTIONS = ["Data Scientist", "Software Engineer", "Product Manager", "Marketing Manager", "Data Analyst"]

SAMPLE_JD = """Data Scientist

We are looking for a Data Scientist to build and ship machine learning models that
improve our product. You will clean and analyse large datasets, design experiments,
and work with engineers to deploy models to production.

Requirements:
- Strong Python and SQL skills
- Experience with pandas, NumPy and scikit-learn
- Solid understanding of machine learning and statistics
- Experience with Docker and cloud platforms (AWS or GCP)
- Familiarity with A/B testing and data visualisation
- Excellent communication skills

Nice to have: Airflow, MLOps, NLP experience."""


def set_role(role: str) -> None:
    st.session_state["target_role_box"] = role


def set_sample_jd() -> None:
    st.session_state["job_description_box"] = SAMPLE_JD


def go_to(page: str) -> None:
    st.session_state["page"] = page


def queue_prompt(text: str) -> None:
    st.session_state["pending_prompt"] = text


def clear_chat() -> None:
    st.session_state["messages"] = []


def page_home(ctx):
    render_html(
        '<div class="hero">'
        '<i class="sp s1">✦</i><i class="sp s2">✦</i><i class="sp s3">✧</i><i class="sp s4">✦</i><i class="sp s5">✧</i>'
        '<div class="hero-text">'
        '<div class="eyebrow">🪔 Your AI resume genie</div>'
        f"<h1>{esc(BRAND)}</h1>"
        f'<div class="tagline">{esc(TAGLINE)}</div>'
        '<p class="lead">Evaluate your resume, score it against a real job description, '
        "generate a tailored cover letter, or talk it through with a career coach — "
        "all in one place.</p>"
        '<div class="trust"><span>⚡ Instant feedback</span><span>🎯 ATS-aware</span>'
        "<span>🔒 Key stays in your session</span></div></div>"
        f'<div class="hero-art"><div class="orb"></div><img src="{icon_uri()}" alt="Resume Genie logo"></div>'
        "</div>"
    )

    words = ["ATS checks", "Keyword gaps", "Score out of 100", "Tailored cover letters", "Interview prep",
             "Skill gap analysis", "Industry feedback", "Resume coaching", "Reliability score", "Action plans"]
    track = "".join(f"<span>✦ {esc(w)}</span>" for w in words * 2)
    render_html(f'<div class="ribbon"><div class="rtrack">{track}</div></div>')

    stats = [("🧰", "4", "tools in one place", 0), ("📊", "100", "point scoring rubric", 3),
             ("🤖", "ATS", "ready checks", 2), ("📄", "PDF", "in, insights out", 4)]
    render_html('<div class="stats">' + "".join(
        f'<div class="stat" style="--c1:{TONES[t][0]};--c2:{TONES[t][1]}"><span class="si">{i}</span>'
        f"<b>{esc(v)}</b><span>{esc(l)}</span></div>"
        for i, v, l, t in stats
    ) + "</div>")

    render_html(h2("Get started in three steps"))
    key_done = bool(ctx.api_key)
    resume_done = ctx.data is not None and bool(ctx.text)
    steps = [
        ("1", "Add your Gemini API key", "Paste it in the sidebar", key_done),
        ("2", "Upload your resume", "A text-based PDF works best", resume_done),
        ("3", "Pick a tool", "Choose from the sidebar or below", False),
    ]
    render_html('<div class="steps3">' + "".join(
        f'<div class="stepcard s{n} {"done" if done else ""}"><div class="n">{"✓" if done else n}</div>'
        f"<div><b>{esc(t)}</b><span>{esc(s)}</span></div></div>"
        for n, t, s, done in steps
    ) + "</div>")

    render_html(h2("Make a wish", "pick a tool"))
    for row in (("evaluator", "scorer"), ("cover_letter", "coach")):
        cols = st.columns(2, gap="medium")
        for col, key in zip(cols, row):
            icon, name, desc = TOOL_INFO[key]
            with col:
                render_html(
                    f'<div class="card feat t-{key}"><div class="ico">{icon}</div>'
                    f"<h3>{esc(name)}</h3><p>{esc(desc)}</p></div>"
                )
                st.button(f"Open {name} →", key=f"open_{key}", on_click=go_to, args=(key,))
        st.write("")


def page_evaluator(ctx):
    icon, name, _ = TOOL_INFO["evaluator"]
    pagehead(icon, "Resume Evaluator & ATS Analyst",
             "A structured, scored review of your resume with ATS analysis and a clear action plan.", "evaluator")
    if not resume_ready(ctx):
        return
    key_hint(ctx)

    if "target_role_box" not in st.session_state:
        st.session_state["target_role_box"] = st.session_state.get("target_role", "")
    with st.container(border=True):
        target_role = st.text_input(
            "Target role (optional)",
            placeholder="e.g. Data Scientist, Backend Engineer — leave blank and the AI will infer one",
            key="target_role_box",
        )
        st.session_state["target_role"] = target_role
        st.caption("Quick picks")
        cols = st.columns(len(ROLE_SUGGESTIONS))
        for col, role in zip(cols, ROLE_SUGGESTIONS):
            with col:
                st.button(role, key=f"role_{role}", on_click=set_role, args=(role,))
        clicked = st.button("✨ Evaluate my resume", type="primary", disabled=not ctx.api_key)

    question = target_role.strip() or "please evaluate this resume."
    run_and_show(
        state_key="evaluator_report",
        label="🪔 Rubbing the lamp… evaluating your resume",
        working="Reading every section",
        prompt_text=EVALUATOR_PROMPT.format(context=ctx.text, question=question),
        ctx=ctx,
        clicked=clicked,
        download_name="resume_evaluation.txt",
        renderer=render_evaluator,
    )


def jd_input(ctx, button_label: str):
    """Job-description box shared by the scorer and the cover-letter page."""
    if "job_description_box" not in st.session_state:
        st.session_state["job_description_box"] = st.session_state.get("job_description", "")
    with st.container(border=True):
        jd = st.text_area(
            "Job description",
            key="job_description_box",
            height=200,
            placeholder="Paste the full job description here…",
        )
        st.session_state["job_description"] = jd  # shared by the scorer and cover-letter pages
        words = len(jd.split())
        st.caption(f"{words} words" if words else "Tip: include responsibilities and requirements for the best match.")
        left, right = st.columns([1, 1])
        with left:
            clicked = st.button(button_label, type="primary", disabled=not ctx.api_key)
        with right:
            st.button("🧪 Try a sample job description", key="sample_jd", on_click=set_sample_jd)
    if clicked and not jd.strip():
        render_html(callout("warn", "", "<p>Please paste the job description first.</p>"))
        clicked = False
    return jd, clicked


def page_scorer(ctx):
    icon, _, _ = TOOL_INFO["scorer"]
    pagehead(icon, "Resume Scorer",
             "See how well your resume matches a job description — with ATS, reliability and keyword gaps.", "scorer")
    if not resume_ready(ctx):
        return
    key_hint(ctx)
    jd, clicked = jd_input(ctx, "🎯 Score my resume")
    run_and_show(
        state_key="scorer_report",
        label="🪔 Rubbing the lamp… scoring your resume",
        working="Comparing against the job description",
        prompt_text=SCORER_PROMPT.format(context=ctx.text, job_description=jd),
        ctx=ctx,
        clicked=clicked,
        download_name="resume_score_report.txt",
        renderer=render_scorer,
        # (no temperature: gemini-3.5-flash-lite ignores it and warns)
    )


def page_cover_letter(ctx):
    icon, _, _ = TOOL_INFO["cover_letter"]
    pagehead(icon, "Cover Letter Generator",
             "A tailored, ready-to-send letter built from your resume and the job description.", "cover_letter")
    if not resume_ready(ctx):
        return
    key_hint(ctx)
    jd, clicked = jd_input(ctx, "✉️ Generate cover letter")
    run_and_show(
        state_key="cover_letter",
        label="🪔 Rubbing the lamp… writing your cover letter",
        working="Matching your experience to the role",
        prompt_text=COVER_LETTER_PROMPT.format(context=ctx.text, job_description=jd),
        ctx=ctx,
        clicked=clicked,
        download_name="cover_letter.txt",
        renderer=render_letter,
    )


def stream_reply(chat, system_message, history, sink):
    """Yield the coach's reply chunk by chunk. Raw text goes to `sink`;
    the yielded text has $ escaped so it isn't shown as LaTeX."""
    messages = [system_message]
    for m in history:
        if m["role"] == "user":
            messages.append(HumanMessage(content=m["content"]))
        else:
            messages.append(AIMessage(content=m["content"]))

    for chunk in chat.stream(messages):
        text = chunk_to_text(chunk.content)
        if text:
            sink.append(text)
            yield esc_md(text)


def page_coach(ctx):
    icon, _, _ = TOOL_INFO["coach"]
    pagehead(icon, "Career Coach", "Chat about your resume, interviews and next career move — your resume stays beside the conversation.", "coach")
    if not resume_ready(ctx):
        return

    left, right = st.columns(2, gap="medium")

    # ---- Left half: the resume --------------------------------------------
    with left:
        try:
            pages = render_pdf_pages(ctx.data)
        except Exception:
            pages = None
        count = f"{len(pages)} page{'s' if len(pages) != 1 else ''}" if pages else ""
        render_html(f'<div class="panehead">Your resume<span>{esc(count)}</span></div>' if count
                    else '<div class="panehead">Your resume</div>')
        with st.container(height=PANE_HEIGHT, border=True):
            if pages is None:
                st.error("Couldn't display the PDF. Make sure PyMuPDF is installed: pip install pymupdf")
            else:
                for page_png in pages:
                    st.image(page_png)

    # ---- Right half: the chatbot ------------------------------------------
    with right:
        render_html('<div class="panehead">Career coach<span>Powered by Gemini</span></div>')
        chat_box = st.container(height=CHAT_HEIGHT, border=True)

        placeholder = (
            "Ask about your resume, interviews, skill gaps…" if ctx.api_key
            else "Add your Gemini API key in the sidebar to start"
        )
        typed = st.chat_input(placeholder, disabled=not ctx.api_key)
        prompt = typed or st.session_state.pop("pending_prompt", None)
        st.button("🗑 Clear chat", on_click=clear_chat, key="clear_chat")

        with chat_box:
            with st.chat_message("assistant", avatar="🧞"):
                st.markdown(GREETING)

            for m in st.session_state["messages"]:
                with st.chat_message(m["role"], avatar="🧞" if m["role"] == "assistant" else "🙂"):
                    st.markdown(esc_md(m["content"]))

            if not st.session_state["messages"] and not prompt and ctx.api_key:
                st.caption("Try one of these:")
                for i, (label, text) in enumerate(CHAT_STARTERS):
                    st.button(label, key=f"starter_{i}", on_click=queue_prompt, args=(text,))

            if prompt:
                st.session_state["messages"].append({"role": "user", "content": prompt})
                with st.chat_message("user", avatar="🙂"):
                    st.markdown(esc_md(prompt))
                try:
                    chat = get_chat_model(ctx.api_key)
                    system_message = SystemMessage(content=SYSTEM_TEMPLATE.format(context=ctx.text))
                    sink = []
                    with st.chat_message("assistant", avatar="🧞"):
                        st.write_stream(
                            stream_reply(chat, system_message, st.session_state["messages"], sink)
                        )
                    st.session_state["messages"].append({"role": "assistant", "content": "".join(sink)})
                except Exception as e:
                    st.session_state["messages"].pop()  # drop the unanswered question
                    title, detail = explain_error(e)
                    render_html(callout("bad", title, f"<p>{esc(detail)}</p>"))


# tool key -> (icon, name, description)
TOOL_INFO = {
    "evaluator": ("📄", "Resume Evaluator",
                  "A full, scored review: ATS compatibility, strengths and weaknesses, skills found, "
                  "project and experience analysis, and a prioritised action plan."),
    "scorer": ("🎯", "Resume Scorer",
               "Paste a job description and see your overall score, match %, reliability and ATS "
               "scores, matched and missing keywords, and skill gaps."),
    "cover_letter": ("✉️", "Cover Letter Generator",
                     "Turn your resume and a job description into a tailored, ready-to-send cover letter."),
    "coach": ("🧭", "Career Coach",
              "Chat about your resume, interviews, job search strategy and skill gaps — "
              "with your resume right beside the conversation."),
}

# page key -> page function
PAGES = {
    "evaluator": page_evaluator,
    "scorer": page_scorer,
    "cover_letter": page_cover_letter,
    "coach": page_coach,
}


# ===========================================================================
# Sidebar + app shell
# ===========================================================================
def nav_button(label: str, page: str) -> None:
    kwargs = dict(
        key=f"nav_{page}",
        type="primary" if st.session_state["page"] == page else "secondary",
        on_click=go_to,
        args=(page,),
    )
    try:
        st.button(label, width="stretch", **kwargs)
    except TypeError:  # older Streamlit versions
        st.button(label, use_container_width=True, **kwargs)


def sidebar():
    with st.sidebar:
        if SHOW_FULL_LOGO:
            render_html(f'<div class="side-brand"><img class="full" src="{logo_uri()}" alt="Resume Genie logo"></div>')
        else:
            render_html(
                f'<div class="side-brand"><img class="ico" src="{icon_uri()}" alt="logo">'
                f"<div><b>{esc(BRAND)}</b><small>{esc(TAGLINE)}</small></div></div>"
            )

        render_html('<div class="side-label">Setup</div>')
        api_key = st.text_input(
            "Gemini API key",
            type="password",
            value=os.environ.get("GOOGLE_API_KEY", ""),
            help="Used only for this session and never stored.",
            placeholder="Paste your key, press Enter",
        )
        uploaded = st.file_uploader("Your resume (PDF)", type=["pdf"])

        resume_bytes, resume_text = None, ""
        if uploaded is not None:
            resume_bytes = uploaded.getvalue()
            try:
                resume_text = extract_resume_text(resume_bytes)
            except Exception as e:
                st.error(f"Couldn't read this PDF: {e}")

        done = int(bool(api_key)) + int(bool(resume_text))
        note = "All set — make a wish ✨" if done == 2 else f"Setup {done} of 2 complete"
        render_html(
            f'<span class="pill {"ok" if api_key else "todo"}">API key</span>'
            f'<span class="pill {"ok" if resume_text else "todo"}">Resume</span>'
            f'<div class="meter"><i style="width:{done * 50}%"></i></div>'
            f'<div class="meter-note {"done" if done == 2 else ""}">{note}</div>'
        )

        render_html('<div class="side-label">Tools</div>')
        nav_button("🏠  Home", "home")
        for key in PAGES:
            icon, tool_name, _ = TOOL_INFO[key]
            nav_button(f"{icon}  {tool_name}", key)

        render_html(
            '<div class="side-foot">Built with Streamlit, LangChain and Gemini.<br>'
            "Your resume and key are only used in this session.</div>"
        )
    return api_key, resume_bytes, resume_text


def main() -> None:
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon=icon_image(),
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_css()

    st.session_state.setdefault("page", "home")
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("resume_id", None)

    api_key, resume_bytes, resume_text = sidebar()

    # A different resume means fresh results and a fresh conversation
    resume_id = hashlib.sha1(resume_bytes).hexdigest() if resume_bytes else None
    if st.session_state["resume_id"] != resume_id:
        st.session_state["resume_id"] = resume_id
        for key in ("evaluator_report", "scorer_report", "cover_letter"):
            st.session_state.pop(key, None)
        st.session_state["messages"] = []

    ctx = SimpleNamespace(api_key=api_key, data=resume_bytes, text=resume_text)
    PAGES.get(st.session_state["page"], page_home)(ctx)


# ===========================================================================
# Logo (embedded so the app is a single file)
# ===========================================================================
LOGO_B64 = (
    "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"
    "wAAAAAAAAAAAAAAAAAAA"
)


main()