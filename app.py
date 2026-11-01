"""
AI Interview Coach  -  Streamlit app
------------------------------------
Resume-based mock interview using:
  * Ollama + Qwen 2.5 7B   (question generation + answer evaluation)
  * MiniLM + FAISS         (RAG over your knowledge_base PDFs)
  * faster-whisper         (speech-to-text for spoken answers, runs locally)

Run:   streamlit run app.py
"""

import io
import json
import os
import random
import re
from pathlib import Path

import requests
import streamlit as st
from pypdf import PdfReader

# =====================================================================
# CONFIG  (edit these if needed)
# =====================================================================
OLLAMA_BASE = "http://localhost:11434"
OLLAMA_URL = f"{OLLAMA_BASE}/api/chat"
MODEL_NAME = "qwen2.5:7b"

# Speech recognition model. "base.en" is light (~150 MB). If your machine
# can handle it, "small.en" is noticeably more accurate.
WHISPER_MODEL_SIZE = "base.en"

# Same project folder as your notebook. You can override it by setting the
# environment variable INTERVIEW_COACH_HOME.
DEFAULT_PROJECT = Path(r"C:\Users\jiffy\OneDrive\Documents\AI_Interview_Coach")
PROJECT_FOLDER = Path(os.environ.get("INTERVIEW_COACH_HOME", DEFAULT_PROJECT))


def find_knowledge_folder() -> Path:
    candidates = [
        PROJECT_FOLDER / "knowledge_base",
        Path(__file__).resolve().parent / "knowledge_base",
        Path(__file__).resolve().parent.parent / "knowledge_base",
    ]
    for c in candidates:
        if c.exists():
            return c
    return candidates[0]


KNOWLEDGE_FOLDER = find_knowledge_folder()

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

st.set_page_config(page_title="AI Interview Coach", page_icon="🎤", layout="centered")


# =====================================================================
# PDF + RAG  (from your notebook)
# =====================================================================
def extract_pdf_text(source) -> str:
    """`source` can be a file path or a file-like object."""
    reader = PdfReader(source)
    text = "\n".join(page.extract_text() or "" for page in reader.pages)
    return text.strip()


def split_text(text, chunk_size=500, overlap=100):
    chunks = []
    if not text:
        return chunks
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == len(text):
            break
        start = end - overlap
    return chunks


@st.cache_resource(show_spinner="Loading knowledge base (first time only)...")
def load_knowledge_base(folder_str: str):
    from sentence_transformers import SentenceTransformer
    import faiss

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    chunks = []
    folder = Path(folder_str)
    if folder.exists():
        for pdf in sorted(folder.glob("*.pdf")):
            try:
                text = extract_pdf_text(str(pdf))
            except Exception:
                continue
            for c in split_text(text):
                chunks.append({"source": pdf.name, "text": c})

    if not chunks:
        return model, None, []

    embeddings = model.encode(
        [c["text"] for c in chunks],
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return model, index, chunks


def retrieve_context(query: str, top_k: int = 3):
    model, index, chunks = load_knowledge_base(str(KNOWLEDGE_FOLDER))
    if index is None:
        return []
    q = model.encode(
        [query], convert_to_numpy=True, normalize_embeddings=True
    ).astype("float32")
    scores, indices = index.search(q, top_k)
    results = []
    for score, i in zip(scores[0], indices[0]):
        if i != -1:
            results.append(
                {"score": float(score), "source": chunks[i]["source"], "text": chunks[i]["text"]}
            )
    return results


def retrieved_text(query: str, top_k: int) -> str:
    parts = [r["text"] for r in retrieve_context(query, top_k) if r["text"].strip()]
    return "\n\n".join(parts) if parts else "(No knowledge-base material available.)"


# =====================================================================
# OLLAMA
# =====================================================================
def ollama_status():
    try:
        r = requests.get(f"{OLLAMA_BASE}/api/tags", timeout=3)
        r.raise_for_status()
        names = [m.get("name", "") for m in r.json().get("models", [])]
        return True, names
    except Exception:
        return False, []


def ollama_chat(prompt, temperature=0.2, json_mode=False, seed=None):
    options = {"temperature": temperature, "num_ctx": 4096}
    if seed is not None:
        options["seed"] = seed

    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": options,
    }
    if json_mode:
        payload["format"] = "json"

    response = requests.post(OLLAMA_URL, json=payload, timeout=300)
    response.raise_for_status()
    content = response.json().get("message", {}).get("content", "").strip()
    if not content:
        raise ValueError("Qwen returned an empty response.")
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


def generate_questions(role, round_type, resume_text, asked, n=5):
    resume_context = resume_text[:4000]
    seed = random.randint(1, 10_000_000)

    if round_type == "Technical":
        focus = random.sample(TECH_FOCUS_AREAS, 3)
        query = f"{focus[0]} interview questions for {role}"
        knowledge = retrieved_text(query, top_k=3)
        instructions = f"""
- Ask a MIX: about 2 questions on specific projects/skills mentioned in the
  resume (mention the project or technology by name), and the rest general
  technical questions for the role "{role}".
- For the general questions, focus on these areas this time: {", ".join(focus)}.
- Suitable for a fresher / entry-level candidate.
- Each question is ONE sentence, answerable in 30-90 seconds.
- Do not combine several questions using "and".
- Do not invent projects, skills or experience that are not in the resume."""
    else:
        focus = random.sample(HR_FOCUS_AREAS, 4)
        query = f"HR interview question about {focus[0]}"
        knowledge = retrieved_text(query, top_k=2)
        instructions = f"""
- Mostly general HR / behavioural questions for the role "{role}".
- Cover these themes this time: {", ".join(focus)}.
- You may include ONE question about the candidate's personal contribution
  or teamwork in a project from the resume.
- Do NOT ask technical questions about algorithms, metrics or code.
- Short, natural, conversational. ONE thing per question."""

    already = "\n".join(f"- {q}" for q in asked[-30:]) or "(none yet)"

    prompt = f"""
You are an experienced, friendly interviewer conducting a realistic
{round_type} interview round for a fresher.

JOB ROLE: {role}

CANDIDATE RESUME:
{resume_context}

INTERVIEW KNOWLEDGE (for inspiration):
{knowledge}

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
        raw = ollama_chat(prompt, temperature=0.9, seed=seed)
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


def evaluate_answer(question, candidate_answer, round_type, role, resume_text):
    if round_type == "Technical":
        knowledge = retrieved_text(question, top_k=3)
    else:
        knowledge = retrieved_text("HR interview best practice answers " + question, top_k=2)

    prompt = f"""
You are a fair, supportive and knowledgeable interviewer conducting a mock
{round_type} interview for a fresher applying for the role: {role}.

Evaluate the candidate's actual answer carefully.

QUESTION:
{question}

REFERENCE KNOWLEDGE:
{knowledge}

CANDIDATE RESUME (use to check project-related answers):
{resume_text[:2500]}

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
            raw = ollama_chat(prompt, temperature=0.1, json_mode=True)
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
# SPEECH TO TEXT
# =====================================================================
@st.cache_resource(show_spinner="Loading speech-recognition model (first time only)...")
def load_whisper():
    from faster_whisper import WhisperModel

    return WhisperModel(WHISPER_MODEL_SIZE, device="cpu", compute_type="int8")


def transcribe(audio_bytes: bytes) -> str:
    model = load_whisper()
    segments, _ = model.transcribe(
        io.BytesIO(audio_bytes),
        language="en",
        beam_size=5,
        vad_filter=True,
        initial_prompt=(
            "An interview answer about machine learning, Python, NLP, deep learning, "
            "Random Forest, TensorFlow, projects and teamwork."
        ),
    )
    return " ".join(seg.text.strip() for seg in segments).strip()


# =====================================================================
# SESSION STATE
# =====================================================================
DEFAULTS = {
    "stage": "welcome",   # welcome -> upload -> role -> instructions -> interview -> result -> goodbye
    "resume_text": "",
    "resume_name": "",
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
    new_qs = generate_questions(s.role, s.round_type, s.resume_text, s.asked_questions, n=5)
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
            load_knowledge_base(str(KNOWLEDGE_FOLDER))
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
# SIDEBAR
# =====================================================================
with st.sidebar:
    st.header("🎤 AI Interview Coach")
    ok, models = ollama_status()
    if not ok:
        st.error("Ollama is not running.\n\nStart it, then refresh this page.")
    elif not any(m.startswith(MODEL_NAME) for m in models):
        st.warning(f"Ollama is running, but `{MODEL_NAME}` was not found.\n\nRun: `ollama pull {MODEL_NAME}`")
    else:
        st.success(f"Ollama ready · {MODEL_NAME}")

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
    file = st.file_uploader("Resume (PDF)", type=["pdf"])

    if file is not None:
        try:
            text = extract_pdf_text(io.BytesIO(file.getvalue()))
        except Exception as e:
            st.error(f"Couldn't read this PDF: {e}")
            return
        if len(text) < 100:
            st.warning(
                "I could read very little text from this PDF. It may be a scanned image. "
                "Please upload a text-based PDF."
            )
            return
        st.success(f"Resume read successfully ({len(text)} characters).")
        if st.button("Continue ➜", type="primary"):
            s.resume_text = text
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
                    s.current_question, answer.strip(), s.round_type, s.role, s.resume_text
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
