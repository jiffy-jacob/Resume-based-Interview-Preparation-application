# 🎤 AI Interview Coach

A resume-based mock interview app. Upload your resume, choose a role and round (Technical or HR), and answer by voice. Get feedback and an improved answer after each question, plus an overall score out of 5.

**Live app:** *(https://resume-based-interview-preparation-application-j59wvvzlu7dwmqu.streamlit.app/)*

## Features

- Questions grounded in your resume through semantic search, different every run
- Technical and HR rounds, unlimited questions until you press **Exit**
- Voice answers with editable text
- Feedback and improved answer after each question
- Overall score out of 5

## Tech Stack

Streamlit · openai/gpt-oss-20b · Groq Whisper · Sentence Transformers (`all-MiniLM-L6-v2`) · FAISS · pypdf

## How retrieval works

1. The resume PDF is cleaned (contact details removed) and split into section-aware chunks.
2. Each chunk is embedded with MiniLM and stored in a FAISS index.
3. Question generation and answer evaluation retrieve the most relevant resume chunks by semantic similarity and use them in the prompt.

## Setup

Requires Python 3.10+ and a free [Groq API key](https://console.groq.com/keys).

```bash
git clone <your-repo-url>
cd <your-repo-folder>
pip install -r requirements.txt
```

Create `.streamlit/secrets.toml`:

```toml
GROQ_API_KEY = "your_key_here"
```

Run:

```bash
streamlit run app.py
```

## Deploy

Push the repo to GitHub, create an app on [Streamlit Community Cloud](https://share.streamlit.io) with `app.py` as the main file, and add `GROQ_API_KEY` under **Settings → Secrets**.

## Author

**Jiffy Jacob**
