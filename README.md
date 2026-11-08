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

Pushed the repo to GitHub, created an app on [Streamlit Community Cloud](https://share.streamlit.io) with `app.py` as the main file, and added `GROQ_API_KEY` under **Settings → Secrets**.

## Conclusion on Model Evaluation

The evaluation of the Personalized AI Interview Coach demonstrates that the system performs well in generating relevant interview feedback and assessing candidate answers. It achieved an overall model-quality rating of **4.00 out of 5**, with particularly good results in answer grounding (4.67), feedback relevance (4.67), and evaluation usefulness (4.33). However, context relevance scored 2.33, indicating that the retrieval component requires further improvement to consistently retrieve the most relevant information from the resume and knowledge base. Overall, the evaluation shows that the system provides a useful foundation for personalized interview preparation. Future improvements will focus on optimizing information retrieval and enhancing the consistency and accuracy of the evaluation process.

## Author

**Jiffy Jacob**
