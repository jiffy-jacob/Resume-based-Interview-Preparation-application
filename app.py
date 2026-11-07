"""
AI Interview Coach  -  CLOUD version with embeddings + FAISS
------------------------------------------------------------
  * Interface        -> Streamlit
  * LLM              -> Llama 3.1 8B through the Groq API (falls back automatically
                        to another Groq model if Llama 3.1 8B is not available)
  * Speech-to-text   -> Groq Whisper
  * Resume processing-> PDF text extraction, cleaning, contact-detail removal and
                        section-aware chunking
  * Embeddings       -> Sentence Transformers all-MiniLM-L6-v2
  * Vector search    -> FAISS (semantic similarity over the candidate's resume chunks)
  * No knowledge-base PDFs are used: questions and evaluation are grounded in the
    resume passages retrieved from the vector index.

Secrets needed (Streamlit Cloud -> App settings -> Secrets):
  GROQ_API_KEY = "gsk_..."
Optional:
  APP_PASSWORD = "something"          # asks visitors for a password
  GROQ_CHAT_MODEL = "..."   # normally NOT needed: the app picks an available model itself
  GROQ_STT_MODEL  = "..."
"""

import io
import json
import os
import random
import re
import time
import unicodedata
from pathlib import Path

import requests
import streamlit as st
from pypdf import PdfReader

st.set_page_config(page_title="AI Interview Coach", page_icon="🎤", layout="centered")

# =====================================================================
# CONFIG
# =====================================================================
GROQ_CHAT_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_STT_URL = "https://api.groq.com/openai/v1/audio/transcriptions"
GROQ_MODELS_URL = "https://api.groq.com/openai/v1/models"


def get_secret(name, default=""):
    try:
        if name in st.secrets:
            return str(st.secrets[name])
    except Exception:
        pass
    return os.environ.get(name, default)


GROQ_API_KEY = get_secret("GROQ_API_KEY")
# Groq retires models often, so the app asks Groq which models exist and picks
# the first available one from these lists (best first). You can force a model
# with the optional GROQ_CHAT_MODEL / GROQ_STT_MODEL secrets.
CHAT_PREFERRED = [
    "llama-3.1-8b-instant",       # Llama 3.1 8B (first choice)
    "llama-3.3-70b-versatile",    # fallbacks if Groq has retired the model above
    "openai/gpt-oss-20b",
    "openai/gpt-oss-120b",
]
STT_PREFERRED = ["whisper-large-v3-turbo", "whisper-large-v3"]
CHAT_OVERRIDE = get_secret("GROQ_CHAT_MODEL")
STT_OVERRIDE = get_secret("GROQ_STT_MODEL")
APP_PASSWORD = get_secret("APP_PASSWORD")

ROLE_OPTIONS = [
    "AI/ML Intern",
    "Data Scientist",
    "AI/ML Engineer",
    "Data Analyst",
    "NLP Engineer",
    "Generative AI Engineer",
    "Python Developer",
    "Other (type your own)",
]

TECH_FOCUS_AREAS = [
    "Python programming",
    "machine learning fundamentals",
    "model evaluation metrics",
    "overfitting and regularization",
    "feature engineering and data preprocessing",
    "deep learning and neural networks",
    "NLP and text processing",
    "statistics and probability",
    "SQL and data handling",
    "ensemble methods",
    "model deployment and real-world issues",
]

HR_FOCUS_AREAS = [
    "motivation for the role",
    "strengths",
    "weaknesses and how the candidate improves them",
    "teamwork and collaboration",
    "handling a challenge or failure",
    "adaptability and learning new things",
    "career goals",
    "handling pressure and deadlines",
    "communication and conflict",
]



# =====================================================================
# RESUME PROCESSING + EMBEDDINGS + FAISS VECTOR SEARCH
# =====================================================================
def extract_pdf_text(source) -> str:
    """`source` can be a file path or a file-like object."""
    reader = PdfReader(source)
    text = "\n".join(page.extract_text() or "" for page in reader.pages)
    return text.strip()


# ---- 1. cleaning ------------------------------------------------------
HEADING_WORDS = (
    "summary", "profile", "objective", "skills", "project", "experience", "internship",
    "education", "certification", "achievement", "award", "publication", "training",
    "interest", "language",
)
TITLE_CASE_HEADINGS = {
    "summary", "profile", "objective", "skills", "projects", "project", "experience",
    "internships", "education", "certifications", "certification", "achievements",
    "awards", "publications", "training", "interests", "languages",
}
BULLETS = ("\u2022", "-", "\u2013", "*", "\u00b7")

EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
URL_RE = re.compile(r"(?:https?://|www\.)\S+|\b(?:linkedin|github)\.com/\S*", re.I)
PHONE_RE = re.compile(
    r"(?<![\w.])(?:\+?\d{1,3}[\s-]?)?\d{10}(?![\w])"
    r"|(?<![\w.])\+\d{1,3}[\s-]\d{3,5}[\s-]\d{3,5}(?![\w])"
)


def is_heading(line: str) -> bool:
    text = line.strip().rstrip(":")
    words = text.split()
    if not text or len(text) > 45 or len(words) > 5:
        return False
    low = text.lower()
    if text.isupper():
        return any(w in low for w in HEADING_WORDS)
    return text.istitle() and len(words) <= 3 and words[-1].lower() in TITLE_CASE_HEADINGS


def preprocess_resume(raw: str) -> str:
    """Normalise the PDF text and remove contact details (phone, email, links)."""
    text = unicodedata.normalize("NFKC", raw).replace("\u00a0", " ")
    # Many PDFs store bullet points as odd symbols; turn them all into a plain bullet
    text = re.sub(r"[\x7f\uf0b7\uf0a7\uf0d8\u25aa\u25cf\u25e6\u2023\u2043\u2219]", "\u2022", text)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)

    out, seen_heading = [], False
    for line in text.splitlines():
        line = re.sub(r"[ \t]+", " ", line).strip()
        if not line:
            out.append("")
            continue
        if is_heading(line):
            seen_heading = True
        if EMAIL_RE.search(line) or URL_RE.search(line) or PHONE_RE.search(line):
            if not seen_heading:
                continue                      # drop the whole contact line in the header block
            line = PHONE_RE.sub("", URL_RE.sub("", EMAIL_RE.sub("", line))).strip()
        if line and not re.fullmatch(r"[\s|,;:\-\u2022]+", line):
            out.append(line)
    return re.sub(r"\n{3,}", "\n\n", "\n".join(out)).strip()


# ---- 2. section-aware chunking ---------------------------------------
def _is_title_line(line: str, in_bullets: bool) -> bool:
    return (
        in_bullets
        and not line.startswith(BULLETS)
        and len(line) <= 110
        and not line.endswith((".", ","))
        and line[:1].isupper()
    )


def chunk_resume(text: str, max_chars: int = 450):
    """Split the resume by section, then pack lines into chunks of about max_chars.
    A new chunk starts at each project/title line; one line of overlap is kept when a
    chunk is cut because it got too long. Every chunk is labelled with its section."""
    sections, label, lines = [], "HEADER", []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if is_heading(line):
            if lines:
                sections.append((label, lines))
            label, lines = line.rstrip(":").upper(), []
        else:
            lines.append(line)
    if lines:
        sections.append((label, lines))

    chunks = []
    for label, lines in sections:
        cur, size, in_bullets = [], 0, False

        def flush():
            if cur:
                chunks.append({"section": label, "text": f"[{label}] " + "\n".join(cur)})

        for line in lines:
            new_title = bool(cur) and size > 120 and _is_title_line(line, in_bullets)
            too_long = bool(cur) and size + len(line) + 1 > max_chars
            if new_title or too_long:
                flush()
                cur = [] if new_title else cur[-1:]
                size = sum(len(x) + 1 for x in cur)
            cur.append(line)
            size += len(line) + 1
            if line.startswith(BULLETS):
                in_bullets = True
            elif _is_title_line(line, True) and len(cur) == 1:
                in_bullets = False
        flush()
    return chunks


# ---- 3. embeddings + FAISS -------------------------------------------
@st.cache_resource(show_spinner="Loading the embedding model (first time only)...")
def load_embedder():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


def build_resume_index(resume_text: str):
    """Chunk the resume, embed every chunk with MiniLM and store the vectors in FAISS."""
    import faiss

    chunks = chunk_resume(resume_text)
    if not chunks:
        raise ValueError("No usable text was found in the resume.")
    embeddings = load_embedder().encode(
        [c["text"] for c in chunks], convert_to_numpy=True, normalize_embeddings=True
    ).astype("float32")
    index = faiss.IndexFlatIP(embeddings.shape[1])   # inner product on unit vectors = cosine similarity
    index.add(embeddings)
    return {"chunks": chunks, "index": index}


def search_resume(rindex, query: str, top_k: int = 3):
    """Semantic similarity search over the resume chunks."""
    q = load_embedder().encode([query], convert_to_numpy=True, normalize_embeddings=True).astype("float32")
    k = min(top_k, len(rindex["chunks"]))
    scores, ids = rindex["index"].search(q, k)
    return [
        {"id": int(i), "score": float(sc), **rindex["chunks"][int(i)]}
        for sc, i in zip(scores[0], ids[0])
        if i != -1
    ]


def gather_context(rindex, queries, per_query=2, max_chunks=6, spotlight=False):
    """Retrieve resume passages for several queries and merge them without duplicates.
    `spotlight` adds one random project chunk so different projects come up each time."""
    picked = []
    for q in queries:
        for r in search_resume(rindex, q, per_query):
            if r["id"] not in picked:
                picked.append(r["id"])
    if spotlight:
        projects = [i for i, c in enumerate(rindex["chunks"]) if "PROJECT" in c["section"]]
        if projects:
            choice = random.choice(projects)
            if choice not in picked:
                picked.insert(0, choice)
    return "\n---\n".join(rindex["chunks"][i]["text"] for i in picked[:max_chunks])


# =====================================================================
# GROQ LLM
# =====================================================================
class LLMError(RuntimeError):
    """Friendly, user-facing API problems."""


def _check_response(r):
    if r.status_code == 429:
        raise LLMError(
            "The free usage limit has been reached for now. Please wait a few minutes and try again."
        )
    if r.status_code in (401, 403):
        raise LLMError("The API key was rejected. Please check GROQ_API_KEY in the app's secrets.")
    if r.status_code == 404 and "model" in r.text.lower():
        raise LLMError(
            "The AI model is no longer available on Groq. Remove any GROQ_CHAT_MODEL / "
            "GROQ_STT_MODEL secret, or set it to a current model from console.groq.com/docs/models."
        )
    if not r.ok:
        raise LLMError(f"API error {r.status_code}: {r.text[:200]}")


def _auth_headers():
    if not GROQ_API_KEY:
        raise LLMError("GROQ_API_KEY is missing. Add it in the app's Secrets settings.")
    return {"Authorization": f"Bearer {GROQ_API_KEY}"}


@st.cache_data(ttl=3600, show_spinner=False)
def list_available_models(api_key: str):
    try:
        r = requests.get(GROQ_MODELS_URL, headers={"Authorization": f"Bearer {api_key}"}, timeout=15)
        if r.ok:
            return [m.get("id", "") for m in r.json().get("data", [])]
    except Exception:
        pass
    return []


def _pick_model(preferred, override, fallback):
    if override:
        return override
    available = list_available_models(GROQ_API_KEY) if GROQ_API_KEY else []
    for name in preferred:
        if name in available:
            return name
    return fallback


def get_chat_model():
    return _pick_model(CHAT_PREFERRED, CHAT_OVERRIDE, "openai/gpt-oss-20b")


def get_stt_model():
    return _pick_model(STT_PREFERRED, STT_OVERRIDE, "whisper-large-v3")


def llm_chat(prompt, temperature=0.2, json_mode=False, seed=None):
    model = get_chat_model()
    reasoning_model = "gpt-oss" in model
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        # reasoning models spend part of this budget on hidden thinking
        "max_tokens": 4000 if reasoning_model else 1200,
    }
    if reasoning_model:
        payload["reasoning_effort"] = "low"
    if json_mode:
        payload["response_format"] = {"type": "json_object"}
    if seed is not None:
        payload["seed"] = seed

    headers = _auth_headers()
    for attempt in range(3):
        r = requests.post(GROQ_CHAT_URL, headers=headers, json=payload, timeout=60)
        if r.status_code == 400 and "reasoning_effort" in payload and "reasoning" in r.text.lower():
            payload.pop("reasoning_effort", None)  # this model doesn't accept it
            continue
        if r.status_code == 429 and attempt == 0:
            try:
                wait = float(r.headers.get("retry-after", 5))
            except ValueError:
                wait = 5
            if wait <= 20:  # short per-minute limit: wait and retry once
                time.sleep(wait)
                continue
        _check_response(r)
        break

    content = r.json()["choices"][0]["message"].get("content", "").strip()
    if not content:
        raise ValueError("The model returned an empty response.")
    return content


def parse_json(text: str) -> dict:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.S)
        if not match:
            raise
        return json.loads(match.group(0))


# =====================================================================
# QUESTION GENERATION  (different every time)
# =====================================================================
def _tokens(q: str) -> set:
    return set(re.findall(r"[a-z0-9]+", q.lower()))


def is_similar(a: str, b: str, threshold: float = 0.7) -> bool:
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return False
    return len(ta & tb) / len(ta | tb) >= threshold


def parse_numbered_questions(text: str):
    questions = []
    for line in text.splitlines():
        m = re.match(r"^\s*\d+[\.\)]\s*(.+)$", line)
        if m:
            q = m.group(1).strip().strip("*").strip()
            if len(q) > 8:
                questions.append(q)
    return questions


def generate_questions(role, round_type, resume_index, asked, n=5):
    seed = random.randint(1, 10_000_000)

    if round_type == "Technical":
        focus = random.sample(TECH_FOCUS_AREAS, 3)
        context = gather_context(
            resume_index,
            [f"{focus[0]} {role}", f"{focus[1]} projects", f"technical skills and tools for {role}"],
            per_query=2, max_chunks=6, spotlight=True,
        )
        instructions = f"""
- Ask a MIX: about 2 questions on specific projects/skills in the resume excerpts
  (mention the project or technology by name), and the rest general technical
  questions for the role "{role}".
- For the general questions, focus on these areas this time: {", ".join(focus)}.
- Suitable for a fresher / entry-level candidate.
- Each question is ONE sentence, answerable in 30-90 seconds.
- Do not combine several questions using "and".
- Do not invent projects, skills or experience that are not in the resume excerpts."""
    else:
        focus = random.sample(HR_FOCUS_AREAS, 4)
        context = gather_context(
            resume_index,
            [f"{focus[0]} {focus[1]}", "project work, teamwork and personal contribution", "education and certification"],
            per_query=2, max_chunks=4, spotlight=True,
        )
        instructions = f"""
- Mostly general HR / behavioural questions for the role "{role}".
- Cover these themes this time: {", ".join(focus)}.
- You may include ONE question about the candidate's personal contribution
  or teamwork in a project from the resume excerpts.
- Do NOT ask technical questions about algorithms, metrics or code.
- Short, natural, conversational. ONE thing per question."""

    already = "\n".join(f"- {q}" for q in asked[-30:]) or "(none yet)"

    prompt = f"""
You are an experienced, friendly interviewer conducting a realistic
{round_type} interview round for a fresher.

JOB ROLE: {role}

RELEVANT RESUME EXCERPTS (retrieved from the candidate's resume by semantic
search for this round; this is all you know about the candidate):
{context}

QUESTIONS ALREADY ASKED (never repeat or rephrase these):
{already}

TASK:
Generate exactly {n} NEW and distinct interview questions.

RULES:{instructions}
- Use simple, natural spoken English.
- No answers, explanations, headings or introductions.
- Variation code (ignore, just be creative): {seed}
- Return ONLY a numbered list (1. 2. 3. ...).
"""

    for _ in range(2):  # one retry if parsing fails
        raw = llm_chat(prompt, temperature=0.9, seed=seed)
        parsed = parse_numbered_questions(raw)
        fresh = []
        for q in parsed:
            if any(is_similar(q, old) for old in asked + fresh):
                continue
            fresh.append(q)
        if fresh:
            return fresh
        seed = random.randint(1, 10_000_000)

    raise ValueError("Could not generate new questions. Please try again.")



# =====================================================================
# EVALUATION  (adapted from your notebook, works for Technical + HR)
# =====================================================================
RUBRICS = {
    "Technical": """
1. technical_correctness (0-4): Are the statements factually and technically correct?
2. conceptual_understanding (0-3): Does the candidate show real understanding?
3. completeness (0-2): Does the answer cover the essential points of THIS question?
4. clarity (0-1): Is the answer understandable?""",
    "HR": """
1. technical_correctness (0-4): RELEVANCE & CREDIBILITY - does the answer directly
   address the question and sound honest and consistent with the resume?
2. conceptual_understanding (0-3): DEPTH & SELF-AWARENESS - does it show reflection,
   maturity and a genuine understanding of the point?
3. completeness (0-2): STRUCTURE - a concrete example or situation, the action taken
   and the result, where appropriate.
4. clarity (0-1): Is the answer clear and well communicated?""",
}

RUBRIC_LIMITS = {
    "technical_correctness": 4,
    "conceptual_understanding": 3,
    "completeness": 2,
    "clarity": 1,
}


def evaluate_answer(question, candidate_answer, round_type, role, resume_index):
    # Retrieve the resume passages most related to the question and the answer
    evidence = gather_context(
        resume_index, [question + " " + candidate_answer[:300]], per_query=3, max_chunks=3
    )

    prompt = f"""
You are a fair, supportive and knowledgeable interviewer conducting a mock
{round_type} interview for a fresher applying for the role: {role}.

Evaluate the candidate's actual answer carefully.

QUESTION:
{question}

RESUME EVIDENCE (the parts of the candidate's resume most related to this
question, retrieved by semantic search). Use it to check claims about their
projects and skills. For general concept questions, rely on your own accurate
knowledge:
{evidence}

CANDIDATE'S ANSWER (spoken, then transcribed - ignore filler words,
minor grammar slips and small transcription errors):
{candidate_answer}

SCORING RUBRIC - TOTAL 10 MARKS:
{RUBRICS[round_type]}

FAIRNESS RULES:
1. Evaluate what the candidate actually said.
2. Accept correct explanations in simple language; jargon is not compulsory.
3. A short answer can score high if it covers the essentials.
4. Do not demand examples, formulas or advanced details unless relevant.
5. Do not mark a correct statement as incorrect, and do not list a concept
   as missing if the candidate already explained it.
6. Keep strengths, missing_concepts and incorrect_statements separate.
7. Do not penalise the same omission twice.
8. The score must be consistent with the rubric and your feedback.

IMPROVED ANSWER:
Write a better version of the candidate's answer in simple, interview-appropriate
English, written in the first person as the candidate would say it. Keep their
correct points, fix genuine errors, add only essential missing information, and
keep it concise (usually 3-6 sentences). For HR questions, do not invent
facts about the candidate - stay consistent with the resume and their answer.

FOLLOW-UP:
Write exactly ONE short, natural follow-up question based on their answer
(not repeating the original question, not containing its own answer).

FEEDBACK:
Brief, specific, encouraging: what they did well and the most important
thing to improve.

RETURN ONLY VALID JSON in exactly this structure:
{{
    "score_breakdown": {{
        "technical_correctness": 0,
        "conceptual_understanding": 0,
        "completeness": 0,
        "clarity": 0
    }},
    "strengths": [],
    "missing_concepts": [],
    "incorrect_statements": [],
    "feedback": "",
    "improved_answer": "",
    "follow_up_question": ""
}}
"""

    evaluation = None
    for _ in range(2):
        try:
            raw = llm_chat(prompt, temperature=0.1, json_mode=True)
            evaluation = parse_json(raw)
            break
        except (json.JSONDecodeError, ValueError):
            continue

    if evaluation is None:
        return {
            "evaluated": False,
            "score": 0,
            "strengths": [],
            "missing_concepts": [],
            "incorrect_statements": [],
            "feedback": "Sorry, I couldn't evaluate this answer this time. It won't count toward your score.",
            "improved_answer": "",
            "follow_up_question": "",
        }

    breakdown = evaluation.get("score_breakdown", {}) or {}
    validated = {}
    for key, maximum in RUBRIC_LIMITS.items():
        try:
            value = float(breakdown.get(key, 0))
        except (TypeError, ValueError):
            value = 0
        validated[key] = round(max(0, min(maximum, value)), 1)

    evaluation["score_breakdown"] = validated
    evaluation["score"] = round(sum(validated.values()), 1)  # out of 10 (internal)
    evaluation["evaluated"] = True

    for key in ("strengths", "missing_concepts", "incorrect_statements"):
        val = evaluation.get(key, [])
        if isinstance(val, str):
            val = [val] if val.strip() else []
        evaluation[key] = [str(v) for v in val if str(v).strip()]
    for key in ("feedback", "improved_answer", "follow_up_question"):
        evaluation[key] = str(evaluation.get(key, "") or "").strip()

    return evaluation


# =====================================================================
# SPEECH TO TEXT (Groq Whisper)
# =====================================================================
def transcribe(audio_bytes: bytes) -> str:
    r = requests.post(
        GROQ_STT_URL,
        headers=_auth_headers(),
        files={"file": ("answer.wav", audio_bytes, "audio/wav")},
        data={
            "model": get_stt_model(),
            "language": "en",
            "response_format": "json",
            "temperature": "0",
            "prompt": (
                "An interview answer about machine learning, Python, NLP, deep learning, "
                "Random Forest, TensorFlow, projects and teamwork."
            ),
        },
        timeout=60,
    )
    _check_response(r)
    return r.json().get("text", "").strip()


# =====================================================================
# SESSION STATE
# =====================================================================
DEFAULTS = {
    "stage": "welcome",   # welcome -> upload -> role -> instructions -> interview -> result -> goodbye
    "resume_text": "",
    "resume_name": "",
    "resume_index": None,    # FAISS index + chunks built from the uploaded resume
    "role": "",
    "round_type": None,
    "completed_rounds": [],
    "question_queue": [],
    "asked_questions": [],   # kept across rounds so questions don't repeat
    "current_question": "",
    "q_counter": 0,
    "records": [],           # answered questions in the current round
    "feedback": None,        # evaluation of the question just answered
    "followup": "",
    "final_score": None,
}
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

s = st.session_state


def go(stage: str):
    s.stage = stage
    st.rerun()


# =====================================================================
# INTERVIEW LOGIC
# =====================================================================
def refill_queue():
    new_qs = generate_questions(s.role, s.round_type, s.resume_index, s.asked_questions, n=5)
    s.question_queue.extend(new_qs)


def pick_next_question():
    """Every second question is the interviewer's adaptive follow-up."""
    follow = s.followup.strip()
    s.followup = ""

    use_follow = (
        follow
        and len(s.records) % 2 == 1
        and not any(is_similar(follow, old) for old in s.asked_questions)
    )

    if use_follow:
        question = follow
    else:
        if not s.question_queue:
            refill_queue()
        question = s.question_queue.pop(0)

    s.current_question = question
    s.asked_questions.append(question)
    s.q_counter += 1
    s.feedback = None


def start_round(round_type: str):
    s.round_type = round_type
    s.records = []
    s.question_queue = []
    s.feedback = None
    s.followup = ""
    s.final_score = None
    try:
        with st.spinner("Preparing your interview questions..."):
            pick_next_question()
    except Exception as e:
        st.error(f"Could not start the interview: {e}")
        st.stop()
    s.stage = "interview"
    st.rerun()


def finish_round():
    evaluated = [r["evaluation"]["score"] for r in s.records if r["evaluation"].get("evaluated")]
    if evaluated:
        avg_out_of_10 = sum(evaluated) / len(evaluated)
        s.final_score = round(avg_out_of_10 / 10 * 5, 1)
    else:
        s.final_score = None
    if s.round_type not in s.completed_rounds:
        s.completed_rounds.append(s.round_type)
    s.stage = "result"
    st.rerun()


# =====================================================================
# PASSWORD GATE (optional)
# =====================================================================
def password_gate():
    if not APP_PASSWORD or st.session_state.get("authed"):
        return
    st.title("🔒 AI Interview Coach")
    pw = st.text_input("Enter the access password", type="password")
    if st.button("Enter"):
        if pw == APP_PASSWORD:
            st.session_state["authed"] = True
            st.rerun()
        else:
            st.error("Incorrect password.")
    st.stop()


password_gate()


# =====================================================================
# SIDEBAR
# =====================================================================
with st.sidebar:
    st.header("🎤 AI Interview Coach")
    if not GROQ_API_KEY:
        st.error("GROQ_API_KEY is not set.")
    else:
        st.success(f"AI service connected\n\nModel: {get_chat_model()}")

    if s.stage not in ("welcome", "upload"):
        st.markdown("---")
        if s.resume_name:
            st.caption(f"📄 Resume: {s.resume_name}")
        if s.role:
            st.caption(f"💼 Role: {s.role}")
        if s.round_type and s.stage == "interview":
            st.caption(f"🎯 Round: {s.round_type}")
            st.caption(f"✅ Answered: {len(s.records)}")

    if s.stage != "welcome":
        st.markdown("---")
        if st.button("🔄 Start over", use_container_width=True):
            for k, v in DEFAULTS.items():
                st.session_state[k] = v if not isinstance(v, list) else []
            st.rerun()


# =====================================================================
# SCREENS
# =====================================================================
def screen_welcome():
    st.title("🎤 Welcome to your AI Interview Coach")
    st.markdown(
        """
Hello, and welcome! 👋

I'll conduct a **mock interview** based on **your resume** and the **role you're applying for**.
After every answer you'll get **feedback** and an **improved sample answer**, and at the end
you'll receive an **overall score out of 5**.

Ready? Let's begin.
"""
    )
    if st.button("Start 🚀", type="primary"):
        go("upload")


def screen_upload():
    st.header("Step 1 · Upload your resume")
    st.write("Please upload your resume as a **PDF**.")
    st.caption(
        "🔒 Your resume is used only during this session and is not saved by this app. "
        "Phone numbers, emails and links are removed before processing. The remaining text "
        "is turned into embeddings on this server, and the relevant parts are sent to the "
        "Groq AI service to generate questions and feedback."
    )
    file = st.file_uploader("Resume (PDF)", type=["pdf"])

    if file is not None:
        try:
            raw = extract_pdf_text(io.BytesIO(file.getvalue()))
        except Exception as e:
            st.error(f"Couldn't read this PDF: {e}")
            return
        text = preprocess_resume(raw)
        if len(text) < 100:
            st.warning(
                "I could read very little text from this PDF. It may be a scanned image. "
                "Please upload a text-based PDF."
            )
            return
        st.success(f"Resume read successfully ({len(text)} characters after cleaning).")
        with st.expander("Preview the text that will be used"):
            st.text(text)
        if st.button("Continue ➜", type="primary"):
            with st.spinner("Processing your resume (the first time can take a minute)..."):
                try:
                    rindex = build_resume_index(text)
                except Exception as e:
                    st.error(f"Couldn't process the resume: {e}")
                    return
            s.resume_text = text
            s.resume_index = rindex
            s.resume_name = file.name
            go("role")


def screen_role():
    st.header("Step 2 · Which role are you applying for?")
    choice = st.selectbox("Select a role", ROLE_OPTIONS, index=0)
    role = choice
    if choice.startswith("Other"):
        role = st.text_input("Type the role", placeholder="e.g. Computer Vision Engineer").strip()

    if st.button("Continue ➜", type="primary", disabled=not role):
        s.role = role
        go("instructions")


def screen_instructions():
    st.header("Step 3 · Instructions & round selection")
    st.markdown(
        """
**How the interview works**

1. Choose a round below: **Technical** or **HR**.
2. I'll ask you one question at a time, based on your **role** and your **resume projects**.
3. Click the 🎙️ microphone, **speak your answer**, and click stop.
4. Your speech appears as **text**. **Edit it** if anything was misheard.
5. Click **Submit answer** to get **feedback** and an **improved answer**.
6. Click **Next question** to continue. There's no fixed limit on questions.
7. Click **Exit interview** at any time to end the round and see your **overall score out of 5**.

**Tips**
- Allow microphone access when your browser asks.
- Speak clearly in a quiet place, in 30-90 seconds per answer.
- Be honest and give examples from your own projects.
"""
    )
    round_type = st.radio(
        "Which round would you like to take?",
        ["Technical", "HR"],
        horizontal=True,
        captions=["Role concepts + your projects", "Behavioural & general questions"],
    )
    if st.button("Begin interview 🎬", type="primary"):
        start_round(round_type)


def exit_button(key):
    if st.button("⏹ Exit interview", key=key):
        finish_round()


def screen_interview():
    qid = s.q_counter
    number = len(s.records) if s.feedback else len(s.records) + 1

    st.caption(f"{s.round_type} round · Question {number}")
    with st.chat_message("assistant"):
        st.markdown(f"**{s.current_question}**")

    # ---------------- Feedback view ----------------
    if s.feedback is not None:
        ev = s.feedback
        last = s.records[-1]

        st.markdown("##### 🗣️ Your answer")
        st.write(last["answer"])

        st.markdown("##### 💬 Feedback")
        st.write(ev["feedback"] or "—")

        c1, c2 = st.columns(2)
        with c1:
            if ev["strengths"]:
                st.markdown("**✅ What you did well**")
                for item in ev["strengths"]:
                    st.markdown(f"- {item}")
        with c2:
            improve = ev["missing_concepts"] + [f"Correction: {x}" for x in ev["incorrect_statements"]]
            if improve:
                st.markdown("**📈 To improve**")
                for item in improve:
                    st.markdown(f"- {item}")

        if ev["improved_answer"]:
            st.markdown("##### ✨ Suggested improved answer")
            st.success(ev["improved_answer"])

        st.markdown("---")
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("Next question ➜", type="primary", key=f"next_{qid}"):
                try:
                    with st.spinner("Thinking of the next question..."):
                        pick_next_question()
                except Exception as e:
                    st.error(f"Couldn't generate the next question: {e}")
                    st.stop()
                st.rerun()
        with col_b:
            exit_button(f"exit_fb_{qid}")
        return

    # ---------------- Answer view ----------------
    audio = st.audio_input(
        "🎙️ Click the mic, speak your answer, then click stop",
        key=f"audio_{qid}",
    )

    ans_key = f"answer_{qid}"
    hash_key = f"audio_hash_{qid}"

    if audio is not None:
        data = audio.getvalue()
        h = hash(data)
        if s.get(hash_key) != h:
            text = ""
            with st.spinner("Converting your speech to text..."):
                try:
                    text = transcribe(data)
                except Exception as e:
                    st.error(f"Speech recognition failed: {e}")
            s[hash_key] = h
            if text:
                s[ans_key] = text  # must be set before the text_area is created
            else:
                st.warning("I couldn't hear anything clearly. Please try recording again, or type your answer.")

    answer = st.text_area(
        "📝 Your answer (edit the text if anything was misheard)",
        key=ans_key,
        height=180,
        placeholder="Your spoken answer will appear here. You can also type.",
    )

    col_a, col_b = st.columns(2)
    with col_a:
        submit = st.button("Submit answer ✔", type="primary", key=f"submit_{qid}")
    with col_b:
        exit_button(f"exit_q_{qid}")

    if submit:
        if not answer.strip():
            st.warning("Please record or type an answer first.")
            return
        try:
            with st.spinner("Evaluating your answer..."):
                ev = evaluate_answer(
                    s.current_question, answer.strip(), s.round_type, s.role, s.resume_index
                )
        except Exception as e:
            st.error(f"Evaluation failed: {e}")
            return

        s.records.append({"question": s.current_question, "answer": answer.strip(), "evaluation": ev})
        s.feedback = ev
        s.followup = ev.get("follow_up_question", "")
        st.rerun()


def screen_result():
    st.header(f"🏁 {s.round_type} round completed")

    if s.final_score is None:
        st.info("You didn't submit any answers in this round, so there's no score yet.")
    else:
        score = s.final_score
        st.markdown("### Your overall score")
        st.markdown(f"# {score} / 5")
        st.progress(min(score / 5, 1.0))
        if score >= 4.5:
            st.success("Outstanding performance! You're interview-ready. 🌟")
        elif score >= 3.5:
            st.success("Good performance. A little polishing and you'll be great. 👍")
        elif score >= 2.5:
            st.warning("Decent start. Review the improved answers and practise again. 💪")
        else:
            st.warning("Keep practising. Study the suggested answers and try again. 📚")
        st.caption(f"Based on {len([r for r in s.records if r['evaluation'].get('evaluated')])} answered question(s).")

    if s.records:
        with st.expander("Review your questions and improved answers"):
            for i, r in enumerate(s.records, 1):
                st.markdown(f"**Q{i}. {r['question']}**")
                st.markdown(f"*Your answer:* {r['answer']}")
                if r["evaluation"].get("improved_answer"):
                    st.markdown(f"*Improved answer:* {r['evaluation']['improved_answer']}")
                st.markdown("---")

    st.markdown("---")
    remaining = [r for r in ("Technical", "HR") if r not in s.completed_rounds]

    if remaining:
        nxt = remaining[0]
        st.subheader(f"Would you like to go to the next round ({nxt})?")
        c1, c2 = st.columns(2)
        with c1:
            if st.button(f"Yes, start {nxt} round", type="primary"):
                start_round(nxt)
        with c2:
            if st.button("No, exit"):
                go("goodbye")
    else:
        st.subheader("You've completed both rounds. 🎉")
        st.write("Would you like to practise a round again?")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Yes, choose a round", type="primary"):
                go("instructions")
        with c2:
            if st.button("No, exit"):
                go("goodbye")


def screen_goodbye():
    st.title("Thank you! 🙏")
    st.markdown(
        "Thanks for practising with the AI Interview Coach. "
        "Keep practising and **good luck with your interviews!** 🍀"
    )
    if st.button("Start a new interview"):
        for k, v in DEFAULTS.items():
            st.session_state[k] = v if not isinstance(v, list) else []
        st.rerun()


# =====================================================================
# ROUTER
# =====================================================================
SCREENS = {
    "welcome": screen_welcome,
    "upload": screen_upload,
    "role": screen_role,
    "instructions": screen_instructions,
    "interview": screen_interview,
    "result": screen_result,
    "goodbye": screen_goodbye,
}
SCREENS[s.stage]()
