# 🪔 Resume Genie

> **Make a wish. Get a stronger resume.**
> An AI-powered resume and career toolkit built with Streamlit, LangChain and Google Gemini.

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.40%2B-FF4B4B?logo=streamlit&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-Gemini-1C3C3C?logo=langchain&logoColor=white)
![Gemini](https://img.shields.io/badge/Google-Gemini-4285F4?logo=google&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-7D3ADF)

![Resume Genie home page](docs/home.png)

---

## ✨ About

**Resume Genie** turns one PDF resume into four AI tools. Add your Gemini API key, upload your resume once, and pick a tool from the sidebar.

Results are shown as rich, interactive views rather than plain text: score donuts, progress bars, keyword chips, skill-gap cards, action plans and a letter-style cover letter.

## 🧰 Features

| Tool | What it does |
|------|--------------|
| 📄 **Resume Evaluator** | 100-point scored review across 9 categories, with strengths and weaknesses, skills found, recommended skills, ATS analysis, project and experience analysis, and an action plan. |
| 🎯 **Resume Scorer** | Paste a job description to get an overall score, match %, reliability and ATS scores, a score breakdown, matched and missing keywords, skill gaps and improvement tips. |
| ✉️ **Cover Letter Generator** | A tailored, ready-to-send cover letter built from your resume and a job description. Placeholders like `[Company Name]` are highlighted. |
| 🧭 **Career Coach** | A streaming chatbot that knows your resume. Your PDF sits on the left and the chat on the right. |

**Also included:** animated score donut, tabbed results, quick-pick roles, a sample job description button, light and dark theme support, clear error messages, and `.txt` downloads.

## 📸 Screenshots

| Resume Evaluator | Resume Scorer |
|:---:|:---:|
| ![Resume Evaluator](docs/evaluator.png) | ![Resume Scorer](docs/scorer.png) |

| Cover Letter Generator | Career Coach |
|:---:|:---:|
| ![Cover Letter Generator](docs/cover-letter.png) | ![Career Coach](docs/career-coach.png) |

## 🚀 Getting Started

**Requirements:** Python 3.10+ and a [Google Gemini API key](https://aistudio.google.com/apikey).

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/ResumeGenie.git
cd ResumeGenie

# 2. Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
source venv/bin/activate       # macOS / Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
streamlit run app.py
```

Open **http://localhost:8501**, paste your Gemini API key in the sidebar, upload a PDF resume, and choose a tool.

### 🔑 API key

The key is used only for the current session and is never stored. To pre-fill it, set an environment variable before starting:

```bash
$env:GOOGLE_API_KEY="your-key-here"      # Windows PowerShell
export GOOGLE_API_KEY="your-key-here"    # macOS / Linux
```

> ⚠️ Never commit your API key to GitHub.

## ⚙️ Configuration

| Setting | Where | Default |
|---------|-------|---------|
| Gemini model | `MODEL_NAME` in `app.py` | `gemini-3.5-flash-lite` |
| Sidebar logo style | `SHOW_FULL_LOGO` in `app.py` | `True` |
| Theme and server | `.streamlit/config.toml` | Light theme |

## 🏗️ How It Works

1. **Upload:** text is extracted with `pypdf`, and pages are rendered with `PyMuPDF` for the coach view.
2. **Prompt:** LangChain prompt templates combine your resume with the job description or target role.
3. **Generate:** Gemini
