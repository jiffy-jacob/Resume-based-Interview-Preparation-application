# 🎤 AI Interview Coach

A resume-based mock interview app that runs locally. Upload your resume, choose a role and round (Technical or HR), and answer by voice. Get feedback and an improved answer after each question, plus an overall score out of 5.

**Live app:** *(https://mockinterviewapp-zxfauwpcjqcxerydbrfvbx.streamlit.app/)*


## Features

- Questions based on your resume and chosen role, different every run
- Technical and HR rounds, unlimited questions until you press **Exit**
- Voice answers with editable text
- Feedback and improved answer after each question
- Overall score out of 5

## Tech Stack

Streamlit · Qwen 2.5 7B (Ollama) · MiniLM + FAISS · faster-whisper · pypdf

## Setup

Requires Python 3.10+ and [Ollama](https://ollama.com).

```bash
git clone <your-repo-url>
cd <your-repo-folder>

conda create -n interviewcoach python=3.11
conda activate interviewcoach
pip install -r requirements.txt

ollama pull qwen2.5:7b
streamlit run app.py
```

Optional: add interview-prep PDFs to a `knowledge_base/` folder next to `app.py`.

## Usage

1. Upload your resume (PDF) and select a role.
2. Choose Technical or HR.
3. Record your answer, edit the text if needed, and submit.
4. Read the feedback and continue, or click **Exit** to see your score.

## Author

**Jiffy Jacob**
