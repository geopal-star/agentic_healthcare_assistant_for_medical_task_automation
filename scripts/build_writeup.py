"""Build the capstone writeup PDF (docs/Capstone_Writeup.pdf).

Metrics are read from the evaluation reports so the document always matches
the code:  python scripts/build_writeup.py [--author "Name"]
"""
from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from reportlab.graphics.shapes import Drawing, Line, Polygon, Rect, String  # noqa: E402
from reportlab.lib import colors  # noqa: E402
from reportlab.lib.enums import TA_CENTER  # noqa: E402
from reportlab.lib.pagesizes import A4  # noqa: E402
from reportlab.lib.styles import ParagraphStyle  # noqa: E402
from reportlab.lib.units import mm  # noqa: E402
from reportlab.pdfbase import pdfmetrics  # noqa: E402
from reportlab.pdfbase.ttfonts import TTFont  # noqa: E402
from reportlab.platypus import (CondPageBreak, Image, KeepTogether, PageBreak, Paragraph,  # noqa: E402
                                SimpleDocTemplate, Spacer, Table, TableStyle)

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "docs" / "figures"
OUT = ROOT / "docs" / "Capstone_Writeup.pdf"
OFFLINE_REPORT = ROOT / "reports" / "eval-20260927-222036.json"
LIVE_REPORT = ROOT / "reports" / "eval-20260928-075859.json"

# Palette (validated categorical order + neutral ink), shared with the dashboard.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, HAIR, PANEL = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#f4f3ef"

# ---------------------------------------------------------------------------
# Fonts & styles
# ---------------------------------------------------------------------------
FONT_DIR = Path("C:/Windows/Fonts")
try:
    pdfmetrics.registerFont(TTFont("Body", str(FONT_DIR / "segoeui.ttf")))
    pdfmetrics.registerFont(TTFont("Body-Bold", str(FONT_DIR / "segoeuib.ttf")))
    pdfmetrics.registerFont(TTFont("Body-Italic", str(FONT_DIR / "segoeuii.ttf")))
    pdfmetrics.registerFont(TTFont("Mono", str(FONT_DIR / "consola.ttf")))
    from reportlab.pdfbase.pdfmetrics import registerFontFamily
    registerFontFamily("Body", normal="Body", bold="Body-Bold", italic="Body-Italic", boldItalic="Body-Bold")
    BODY, BOLD, MONO = "Body", "Body-Bold", "Mono"
except Exception:                                   # non-Windows fallback
    BODY, BOLD, MONO = "Helvetica", "Helvetica-Bold", "Courier"

S = {
    "title": ParagraphStyle("title", fontName=BOLD, fontSize=26, leading=31, textColor=INK, spaceAfter=6),
    "subtitle": ParagraphStyle("subtitle", fontName=BODY, fontSize=13, leading=18, textColor=INK2),
    "h1": ParagraphStyle("h1", fontName=BOLD, fontSize=16, leading=20, textColor=INK, spaceBefore=10, spaceAfter=6),
    "h2": ParagraphStyle("h2", fontName=BOLD, fontSize=12, leading=16, textColor=INK, spaceBefore=8, spaceAfter=3),
    "body": ParagraphStyle("body", fontName=BODY, fontSize=9.6, leading=14, textColor=INK, spaceAfter=5),
    "bullet": ParagraphStyle("bullet", fontName=BODY, fontSize=9.6, leading=13.5, textColor=INK,
                             leftIndent=12, bulletIndent=2, spaceAfter=2.5),
    "small": ParagraphStyle("small", fontName=BODY, fontSize=8.2, leading=11, textColor=INK2),
    "caption": ParagraphStyle("caption", fontName=BODY, fontSize=8.2, leading=11, textColor=INK2,
                              alignment=TA_CENTER, spaceBefore=3, spaceAfter=10),
    "cell": ParagraphStyle("cell", fontName=BODY, fontSize=8.4, leading=11, textColor=INK),
    "cellb": ParagraphStyle("cellb", fontName=BOLD, fontSize=8.4, leading=11, textColor=INK),
    "code": ParagraphStyle("code", fontName=MONO, fontSize=7.8, leading=10.5, textColor=INK,
                           backColor=PANEL, borderPadding=6, spaceBefore=4, spaceAfter=8),
    "kpi_v": ParagraphStyle("kpi_v", fontName=BOLD, fontSize=19, leading=22, textColor=INK),
    "kpi_l": ParagraphStyle("kpi_l", fontName=BODY, fontSize=8, leading=10.5, textColor=INK2),
}
PAGE_W, PAGE_H = A4
MARGIN = 18 * mm
CONTENT_W = PAGE_W - 2 * MARGIN


def P(text, style="body"):
    return Paragraph(text, S[style])


def bullets(items):
    return [Paragraph(t, S["bullet"], bulletText="•") for t in items]


def table(rows, widths, header=True, zebra=True):
    data = [[c if not isinstance(c, str) else Paragraph(c, S["cellb" if header and i == 0 else "cell"])
             for c in row] for i, row in enumerate(rows)]
    t = Table(data, colWidths=widths, repeatRows=1 if header else 0)
    style = [("VALIGN", (0, 0), (-1, -1), "TOP"),
             ("TOPPADDING", (0, 0), (-1, -1), 3.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 3.5),
             ("LEFTPADDING", (0, 0), (-1, -1), 5), ("RIGHTPADDING", (0, 0), (-1, -1), 5),
             ("LINEBELOW", (0, 0), (-1, -1), 0.4, colors.HexColor(HAIR))]
    if header:
        style += [("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(PANEL)),
                  ("LINEBELOW", (0, 0), (-1, 0), 0.8, colors.HexColor(MUTED))]
    t.setStyle(TableStyle(style))
    return t


def figure(path: Path, caption: str, width=CONTENT_W, max_h=205 * mm):
    from PIL import Image as PILImage
    w, h = PILImage.open(path).size
    scale = min(width / w, max_h / h)
    img = Image(str(path), width=w * scale, height=h * scale)
    img.hAlign = "CENTER"
    frame = Table([[img]], colWidths=[w * scale + 2])
    frame.setStyle(TableStyle([("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor(HAIR)),
                               ("LEFTPADDING", (0, 0), (-1, -1), 0), ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                               ("TOPPADDING", (0, 0), (-1, -1), 0), ("BOTTOMPADDING", (0, 0), (-1, -1), 0)]))
    return KeepTogether([frame, P(caption, "caption")])


def callout(text, color=BLUE):
    t = Table([[Paragraph(text, S["body"])]], colWidths=[CONTENT_W])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), colors.HexColor(PANEL)),
                           ("LINEBEFORE", (0, 0), (0, -1), 3, colors.HexColor(color)),
                           ("LEFTPADDING", (0, 0), (-1, -1), 10), ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                           ("TOPPADDING", (0, 0), (-1, -1), 7), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]))
    return t


# ---------------------------------------------------------------------------
# Architecture diagram (vector)
# ---------------------------------------------------------------------------

def arrow(d: Drawing, x1, y1, x2, y2, color=MUTED, head=4.5):
    d.add(Line(x1, y1, x2, y2, strokeColor=colors.HexColor(color), strokeWidth=1))
    import math
    ang = math.atan2(y2 - y1, x2 - x1)
    left = (x2 - head * math.cos(ang - 0.45), y2 - head * math.sin(ang - 0.45))
    right = (x2 - head * math.cos(ang + 0.45), y2 - head * math.sin(ang + 0.45))
    d.add(Polygon([x2, y2, *left, *right], fillColor=colors.HexColor(color), strokeColor=None))


def box(d: Drawing, x, y, w, h, title, sub=None, fill="#ffffff", stroke=INK2, bold=True, title_color=INK):
    d.add(Rect(x, y, w, h, rx=4, ry=4, fillColor=colors.HexColor(fill), strokeColor=colors.HexColor(stroke),
               strokeWidth=0.8))
    ty = y + h / 2 + (3 if sub else -3)
    d.add(String(x + w / 2, ty, title, fontName=BOLD if bold else BODY, fontSize=8.2,
                 fillColor=colors.HexColor(title_color), textAnchor="middle"))
    if sub:
        for i, line in enumerate(sub.split("\n")):
            d.add(String(x + w / 2, y + h / 2 - 8 - i * 8.5, line, fontName=BODY, fontSize=6.6,
                         fillColor=colors.HexColor(INK2), textAnchor="middle"))


def architecture() -> Drawing:
    W, H = CONTENT_W, 300
    d = Drawing(W, H)
    # user + UI
    box(d, 0, H - 34, 150, 28, "Users", "patient · caregiver · attendant · doctor")
    box(d, W - 170, H - 34, 170, 28, "Streamlit dashboard", "8 pages · chat · views · LLMOps")
    arrow(d, 150, H - 20, W - 170, H - 20)
    # agent container
    top = H - 58
    d.add(Rect(0, top - 92, W, 88, rx=6, ry=6, fillColor=colors.HexColor("#f3f7fd"),
               strokeColor=colors.HexColor(BLUE), strokeWidth=0.8, strokeDashArray=[3, 2]))
    d.add(String(8, top - 16, "LangGraph agent (StateGraph + checkpointer)", fontName=BOLD, fontSize=8,
                 fillColor=colors.HexColor(BLUE)))
    arrow(d, W - 85, H - 34, W - 85, top - 4, color=BLUE)
    nodes = [("safety", "red-flag\nscreen"), ("context", "user, dependents,\nmemory recall"),
             ("planner", "LLM → typed Plan\n+ validate/repair"), ("executor", "one tool per step\n(loops)"),
             ("synthesizer", "grounded reply\nwith citations"), ("memory", "interaction +\nfacts → FAISS")]
    n, gap = len(nodes), 9
    bw = (W - 16 - gap * (n - 1)) / n
    xs = [8 + i * (bw + gap) for i in range(n)]
    by = top - 82
    for (name, sub), x in zip(nodes, xs):
        box(d, x, by, bw, 50, name, sub, fill="#ffffff", stroke=BLUE)
    for i in range(n - 1):
        arrow(d, xs[i] + bw, by + 25, xs[i + 1], by + 25, color=BLUE)
    # executor loop marker
    ex = xs[3]
    d.add(String(ex + bw / 2, by + 53, "repeats per plan step", fontName=BODY, fontSize=6.5,
                 fillColor=colors.HexColor(BLUE), textAnchor="middle"))
    # tools row
    ty = by - 62
    tools = [("Doctor Schedule API", "find · check · book (atomic)\ncancel · reschedule · list"),
             ("EHR tools", "identify · history · notes\nregister · update · add note*"),
             ("Medical search (RAG)", "MedlinePlus API + WHO\n→ FAISS → cited answer")]
    tw = (W - 2 * 20) / 3
    for i, (t, sub) in enumerate(tools):
        x = i * (tw + 20)
        box(d, x, ty, tw, 38, t, sub, fill="#fff6f1", stroke=ORANGE)
        arrow(d, ex + bw / 2, by, x + tw / 2, ty + 38, color=ORANGE)
    # stores row
    sy = ty - 64
    stores = [("SQLite EHR", "patients, encounters,\nlabs, slots, appts"),
              ("FAISS", "patient_memory"),
              ("FAISS", "medical_knowledge"),
              ("Checkpointer", "short-term thread\nmemory"),
              ("Observability", "tool / LLM / memory\nlogs, agent runs")]
    k = len(stores)
    sw = (W - 8 * (k - 1)) / k
    for i, (t, sub) in enumerate(stores):
        box(d, i * (sw + 8), sy, sw, 38, t, sub, fill="#eefaf5", stroke=AQUA)
    arrow(d, tw / 2, ty, sw / 2, sy + 38, color=MUTED)                            # schedule/EHR -> SQLite
    arrow(d, tw + 20 + tw / 2, ty, sw / 2 + 6, sy + 38, color=MUTED)
    arrow(d, tw + 20 + tw / 2 + 8, ty, (sw + 8) + sw / 2, sy + 38, color=MUTED)   # EHR -> patient_memory
    arrow(d, 2 * (tw + 20) + tw / 2, ty, 2 * (sw + 8) + sw / 2, sy + 38, color=MUTED)  # search -> knowledge
    # evaluation box
    box(d, W - sw, sy - 46, sw, 30, "Evaluation", "QAEvalChain + metrics", fill="#ffffff", stroke=INK2)
    arrow(d, 4 * (sw + 8) + sw / 2, sy, W - sw / 2, sy - 16, color=MUTED)
    d.add(String(0, sy - 30, "* staff-only tools (role guard)", fontName=BODY, fontSize=6.5,
                 fillColor=colors.HexColor(MUTED)))
    return d


# ---------------------------------------------------------------------------
# Evaluation chart
# ---------------------------------------------------------------------------

def eval_chart(offline: dict, live: dict, path: Path) -> None:
    metrics = [
        ("Planner · tool F1", "planner", "tool_f1"),
        ("Planner · exact plan match", "planner", "exact_match"),
        ("Summary · QAEvalChain correct", "summary", "qa_eval_correct"),
        ("Summary · semantic similarity", "summary", "semantic_similarity"),
        ("Medical QA · QAEvalChain correct", "medical_qa", "qa_eval_correct"),
        ("Medical QA · key-fact coverage", "medical_qa", "keyword_coverage"),
        ("Medical QA · groundedness", "medical_qa", "groundedness"),
        ("Medical QA · context recall", "medical_qa", "context_recall"),
        ("Retrieval · patient hit@1", "retrieval", "patient_hit@1"),
        ("Booking · outcome as expected", "booking", "outcome_as_expected"),
    ]
    plt.rcParams.update({"font.family": "Segoe UI", "font.size": 8.5})
    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=220)
    y = range(len(metrics))
    h = 0.36
    for j, (label, src, color) in enumerate([("LLM (Llama 3.3 70B)", live, BLUE), ("Offline heuristic", offline, ORANGE)]):
        vals = [src.get(m, {}).get(k) for _, m, k in metrics]
        pos = [i + (j - 0.5) * h for i in y]
        ax.barh(pos, [v if v is not None else 0 for v in vals], height=h - 0.04, color=color, label=label)
        for p_, v in zip(pos, vals):
            ax.text((v or 0) + 0.01, p_, "n/a (needs LLM)" if v is None else f"{v:.2f}", va="center",
                    fontsize=7, color=INK2 if v is not None else MUTED)
    ax.set_yticks(list(y), [m[0] for m in metrics], color=INK)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.18)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.tick_params(axis="x", colors=MUTED, labelsize=7.5)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color=HAIR, linewidth=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(HAIR)
    ax.legend(loc="lower center", bbox_to_anchor=(0.35, 1.0), ncol=2, frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Page decoration
# ---------------------------------------------------------------------------

def on_page(canvas, doc):
    canvas.saveState()
    if doc.page > 1:
        canvas.setStrokeColor(colors.HexColor(HAIR))
        canvas.setLineWidth(0.5)
        canvas.line(MARGIN, PAGE_H - 12 * mm, PAGE_W - MARGIN, PAGE_H - 12 * mm)
        canvas.setFont(BODY, 7.5)
        canvas.setFillColor(colors.HexColor(MUTED))
        canvas.drawString(MARGIN, PAGE_H - 10.5 * mm, "Agentic Healthcare Assistant for Medical Task Automation")
        canvas.drawRightString(PAGE_W - MARGIN, PAGE_H - 10.5 * mm, "Applied Generative AI Specialisation · Capstone")
        canvas.drawRightString(PAGE_W - MARGIN, 10 * mm, f"{doc.page}")
    canvas.restoreState()


# ---------------------------------------------------------------------------
# Content
# ---------------------------------------------------------------------------

def kpi_strip(items):
    cells = [[Paragraph(v, S["kpi_v"]) for v, _ in items], [Paragraph(label, S["kpi_l"]) for _, label in items]]
    t = Table(cells, colWidths=[CONTENT_W / len(items)] * len(items))
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), colors.HexColor(PANEL)),
                           ("LINEBEFORE", (1, 0), (-1, -1), 0.5, colors.HexColor(HAIR)),
                           ("LEFTPADDING", (0, 0), (-1, -1), 9), ("TOPPADDING", (0, 0), (-1, 0), 8),
                           ("BOTTOMPADDING", (0, 1), (-1, 1), 8), ("VALIGN", (0, 0), (-1, -1), "TOP")]))
    return t


def fmt(v):
    return "–" if v is None else f"{v:.2f}"


def build(author: str) -> None:
    off = json.loads(OFFLINE_REPORT.read_text(encoding="utf-8"))["summary"]
    live_rep = json.loads(LIVE_REPORT.read_text(encoding="utf-8"))
    live = live_rep["summary"]
    chart = ROOT / "docs" / "figures" / "eval_comparison.png"
    eval_chart(off, live, chart)

    story = []
    # ---------------- Cover ----------------
    story += [Spacer(1, 42 * mm),
              P("APPLIED GENERATIVE AI SPECIALISATION · CAPSTONE PROJECT", "small"), Spacer(1, 4 * mm),
              P("Agentic Healthcare Assistant for Medical Task Automation", "title"),
              P("A LangGraph plan-and-execute agent that books appointments, manages medical records, summarises "
                "patient histories and answers disease questions from MedlinePlus and WHO, with RAG, long-term "
                "memory, guard-rails, an evaluation harness and a Streamlit dashboard.", "subtitle"),
              Spacer(1, 12 * mm),
              kpi_strip([(fmt(live["medical_qa"]["qa_eval_correct"]), "Medical answers graded correct<br/>(QAEvalChain)"),
                         (fmt(live["medical_qa"]["groundedness"]), "Answer groundedness<br/>in retrieved sources"),
                         (fmt(live["planner"]["tool_f1"]), "Planner tool-selection F1<br/>(15 multi-step queries)"),
                         (fmt(live["booking"]["outcome_as_expected"]), "Booking reliability<br/>(incl. race & idempotency)")]),
              Spacer(1, 3 * mm),
              P(f"Live-LLM evaluation run <font name='{MONO}'>{live_rep['eval_run_id']}</font> · model "
                f"{live_rep['llm'].split(':', 1)[1]} via OpenRouter.", "small"),
              Spacer(1, 38 * mm),
              P(f"<b>Submitted by:</b> {author}", "body"),
              P(f"<b>Date:</b> {date.today():%B %Y}", "body"),
              P("<b>Deliverables:</b> source code (<font name='Mono'>healthcare_agent/</font>), Streamlit app "
                "(<font name='Mono'>app.py</font>), executed notebook, 27-test pytest suite, evaluation reports.", "body"),
              PageBreak()]

    # ---------------- Contents ----------------
    toc = ["1. Executive summary", "2. Problem and objectives", "3. Solution architecture", "4. Data and ingestion",
           "5. Tools, memory and RAG", "6. Planning, prompt engineering and task chaining",
           "7. Agent execution: the reference scenario", "8. Guard-rails", "9. LLMOps: evaluation and monitoring",
           "10. Streamlit dashboard", "11. Engineering issues found and fixed", "12. Limitations and future work",
           "Appendix: running the project"]
    story += [P("Contents", "h1")] + [P(t, "body") for t in toc] + [Spacer(1, 6 * mm)]

    # ---------------- 1. Executive summary ----------------
    story += [P("1. Executive summary", "h1"),
              P("Patient-care administration is split across scheduling systems, electronic health records and "
                "reference websites. This project builds a <b>virtual medical assistant</b> that coordinates them from "
                "a single natural-language request. Given the capstone's reference request, <i>“My 70-year-old father "
                "has chronic kidney disease. I want to book a nephrologist for him. Also, can you summarize latest "
                "treatment methods?”</i>, the agent performs four steps in order:"),
              *bullets(["Identifies the father through the caregiver link.",
                        "Retrieves and summarises his CKD history, with rule-based lab alerts.",
                        "Books the earliest nephrology slot.",
                        "Returns a grounded, cited treatment summary from MedlinePlus and WHO."]),
              P("The whole run takes about 15–20 seconds, mostly the live web fetch. The system is built on "
                "<b>LangGraph</b> with a structured-output planner, 13 tools, FAISS long-term memory and a RAG "
                "pipeline. It also has deterministic safety, privacy and role guard-rails, and an LLMOps layer "
                "covering QAEvalChain grading, offline metrics and per-tool logging."),
              P("Key results (live LLM mode)", "h2"),
              *bullets([f"<b>Medical QA:</b> QAEvalChain rated {live['medical_qa']['qa_eval_correct']:.0%} of answers correct, "
                        f"with {live['medical_qa']['keyword_coverage']:.0%} key-fact coverage and "
                        f"{live['medical_qa']['groundedness']:.2f} groundedness. Every answer cites its sources.",
                        f"<b>Planner:</b> tool-selection F1 of {live['planner']['tool_f1']:.2f} and exact plan match of "
                        f"{live['planner']['exact_match']:.2f}. Specialty routing and patient identification were 100% "
                        "correct; the plan mismatches are defensible judgment calls (Section 9).",
                        "<b>Booking:</b> 100% expected outcomes, constraint satisfaction, idempotency and race safety. "
                        "Impossible requests return alternatives instead of silently booking something else.",
                        f"<b>History summaries:</b> QAEvalChain rated {live['summary']['qa_eval_correct']:.0%} correct "
                        "against the reference summaries in <font name='Mono'>records.xlsx</font>.",
                        "<b>Graceful degradation:</b> the whole system runs without an API key in a deterministic "
                        "offline mode, and a hard deadline turns a stalled provider into a logged fallback instead of "
                        "a hang."]),
              PageBreak()]

    # ---------------- 2. Problem ----------------
    story += [P("2. Problem and objectives", "h1"),
              P("The problem statement asks for an agentic system that uses LLMs, RAG and memory modules to:"),
              *bullets(["<b>Book medical appointments</b>: automate slot discovery and scheduling from patient intent "
                        "and doctor availability.",
                        "<b>Manage medical records</b>: let attendants add or update structured and unstructured history.",
                        "<b>Retrieve medical histories</b>: summarise past diagnoses, treatments and relevant alerts.",
                        "<b>Search medical information</b>: fetch up-to-date disease information from trusted sources "
                        "(MedlinePlus, WHO)."]),
              P("Part 1 covers the agent design: planning and goal decomposition, tools and memory, prompt engineering "
                "and task chaining, and the sample execution flow. Part 2 covers LLMOps: evaluation with QAEvalChain, "
                "per-module performance logging, and a Streamlit dashboard with patient/doctor views, real-time "
                "appointment tracking, retrieved medical information, metrics, memory traces and interactive scenario "
                "testing."),
              P("Requirement coverage", "h2"),
              table([["Requirement", "Implementation"],
                     ["Planner decomposes multi-step queries", "LLM structured-output <i>Plan</i> (sub-goals, tool, args, "
                      "dependencies) + validation/repair layer"],
                     ["Appointment booking API", "Doctor Schedule API over SQLite slots: atomic, idempotent booking; "
                      "cancel; reschedule"],
                     ["EHR / patient DB", "SQLite EHR built from the dataset; staff-only add/update/free-text note tools"],
                     ["Disease search (web / Medline)", "MedlinePlus Web Service + WHO fact sheets → FAISS → cited answer"],
                     ["FAISS store; long-term memory", "patient_memory and medical_knowledge indexes; memories recalled "
                      "into prompts"],
                     ["Structured prompts & chains", "Planner → history summary / medical RAG → synthesis → memory extraction"],
                     ["QAEvalChain; per-module metrics", "Evaluation harness with six modules; results in SQLite and JSON"],
                     ["Streamlit views, tracking, logs", "8-page dashboard incl. memory traces, planning breakdowns, "
                      "scenario tester"]],
                    [58 * mm, CONTENT_W - 58 * mm]),
              PageBreak()]

    # ---------------- 3. Architecture ----------------
    story += [P("3. Solution architecture", "h1"),
              P("The assistant is a <b>plan-and-execute</b> agent implemented as a LangGraph state graph. Each request "
                "passes through six nodes; the executor loops once per planned step. A checkpointer persists state per "
                "chat thread, which gives short-term memory across turns."),
              architecture(), P("Figure 1. System architecture. Blue: agent graph; orange: tools; green: stores.", "caption"),
              table([["Node", "Responsibility"],
                     ["safety", "Deterministic red-flag screen (heart attack, stroke, breathing difficulty, self-harm…). "
                                "Emergency guidance is always shown first; background information search is deferred."],
                     ["context", "Resolves the acting user and linked dependents, the active patient from the thread, "
                                 "and recalls long-term memories from FAISS."],
                     ["planner", "LLM produces a typed Plan. The validator drops unknown/duplicate tools, removes "
                                 "staff-only steps for patients, inserts a missing identify step and wires dependencies."],
                     ["executor", "Runs one tool per graph step, injects the resolved patient id and patient context, "
                                  "and skips steps whose dependencies failed."],
                     ["synthesizer", "Writes the reply from tool results only: patient context first, citations kept, "
                                     "failures stated plainly."],
                     ["memory", "Stores an interaction summary and durable facts (conditions, preferences) against the patient."]],
                    [24 * mm, CONTENT_W - 24 * mm]),
              P("Design decisions", "h2"),
              *bullets(["<b>Plan-and-execute over a free ReAct loop.</b> Requests are compositional and have side effects "
                        "(a booking must not repeat). An explicit plan is visible in the UI, measurable in evaluation and "
                        "bounded to one tool call per step.",
                        "<b>Validate every plan.</b> LLM plans are proposals; privacy and role rules are enforced in code, "
                        "so they hold even if the model ignores the prompt.",
                        "<b>Deterministic where correctness matters.</b> Clinical alerts, access control, the emergency "
                        "screen and booking logic are rules. The LLM handles language: planning, summarising, extraction "
                        "and grounded answering.",
                        "<b>Provider-agnostic, with fallbacks.</b> The app needs only structured() and text() calls. Groq, "
                        "OpenAI, Anthropic, OpenRouter and Ollama are supported, and every call site has an offline "
                        "fallback."]),
              PageBreak()]

    # ---------------- 4. Data ----------------
    story += [P("4. Data and ingestion", "h1"),
              P("The provided dataset contains <font name='Mono'>records.xlsx</font> and four clinical PDFs in two "
                "different layouts. Custom parsers convert them into a relational EHR."),
              table([["Source", "Processing", "Result"],
                     ["records.xlsx (7 rows)", "One row per encounter; de-duplicated by name and phone. The Summary "
                      "column is kept as evaluation ground truth.", "5 patients"],
                     ["3 × History & Physical PDFs", "Labelled-field parser: demographics, SOAP sections, ICD-10 "
                      "diagnoses, medications from plan text, vitals.", "3 encounters"],
                     ["sample_patient.pdf (12 pages)", "Pages grouped by encounter footer: progress notes, past medical "
                      "history, and CCD sections with lab results and reference ranges.",
                      "2 encounters, 45 labs with H/L flags, meds, vitals"],
                     ["Synthetic (tagged synthetic_demo)", "15 doctors in 13 specialties with working hours; rolling "
                      "21-day slot calendar (~30% held by other channels); Suresh Negi, 70, CKD 3b, father of Rahul Negi.",
                      "Needed by the reference scenario"]],
                    [38 * mm, CONTENT_W - 38 * mm - 38 * mm, 38 * mm]),
              Spacer(1, 3 * mm),
              callout("<b>Why synthetic data was added.</b> The dataset has no doctor directory or calendar, and the "
                      "reference scenario's father is not in it. Rather than hard-code the scenario, a realistic "
                      "schedule and one demo patient were generated, and every synthetic row is labelled so it is never "
                      "confused with the provided records."),
              P("Lab results are flagged against their reference ranges (for example ALT 52 U/L vs 6–29 → HIGH; HDL "
                "35 mg/dL vs ≥ 50 → LOW). These flags drive deterministic clinical alerts, together with allergies, "
                "elevated blood pressure, polypharmacy, age-related risk and overdue follow-ups."),
              figure(FIG / "patient.png", "Figure 2. Patient view for the reference patient: rule-based clinical alerts "
                     "computed from the EHR.", max_h=105 * mm),
              PageBreak()]

    # ---------------- 5. Tools, memory, RAG ----------------
    story += [P("5. Tools, memory and RAG", "h1"),
              table([["Group", "Tools", "Notes"],
                     ["Doctor Schedule API", "find_doctors, check_availability, book_appointment, cancel_appointment, "
                      "reschedule_appointment, list_appointments",
                      "Atomic conditional UPDATE on the slot; idempotent per specialty; reschedule books the new slot "
                      "before releasing the old one; natural-language dates (“next Monday morning”)."],
                     ["EHR", "identify_patient, get_patient_history, search_patient_notes, register_patient*, "
                      "update_patient_record*, add_clinical_note*",
                      "Fuzzy name match, relation lookup (“my dad”), LLM extraction of structure from free-text notes. "
                      "* staff only."],
                     ["Medical search", "medical_info_search",
                      "MedlinePlus Web Service (XML) + WHO site-restricted search and fact-sheet extraction; cached "
                      "43-document seed corpus for offline use."]],
                    [30 * mm, 62 * mm, CONTENT_W - 92 * mm]),
              P("Every tool is wrapped by a tracing decorator that logs arguments, result, success and latency. A tool "
                "reports a business failure (no slot, not authorised) as <font name='Mono'>{ok: false, error}</font> so "
                "the executor and synthesizer can react honestly.", "small"),
              P("Memory", "h2"),
              table([["Type", "Mechanism", "Used for"],
                     ["Short-term", "LangGraph checkpointer keyed by chat thread (messages, active patient)",
                      "Follow-ups such as “also book a dietitian for him”"],
                     ["Long-term", "SQLite memories + FAISS patient_memory (profiles, encounter chunks, learned facts)",
                      "Recalled into the planner prompt and history summaries"],
                     ["Knowledge", "FAISS medical_knowledge (chunked MedlinePlus/WHO documents)", "RAG context"]],
                    [24 * mm, 80 * mm, CONTENT_W - 104 * mm]),
              P("RAG pipeline", "h2"),
              P("Query → live fetch from MedlinePlus and WHO → cache in SQLite → paragraph-aware chunking → "
                "<font name='Mono'>bge-small-en-v1.5</font> embeddings (normalised, cosine via IndexFlatIP) → top-k "
                "retrieval with at most two passages per document → numbered-source prompt → answer with inline [n] "
                "citations and a patient-specific note. If the network is unavailable, retrieval runs over the cached "
                "corpus; with no LLM, a Maximal Marginal Relevance extractive summary is used."),
              PageBreak()]

    # ---------------- 6. Planning & prompts ----------------
    story += [P("6. Planning, prompt engineering and task chaining", "h1"),
              P("Each sub-task has its own structured prompt (role, inputs, rules, output contract). The prompts form a "
                "chain in which the output of one stage becomes context for the next:"),
              table([["Stage", "Prompt design"],
                     ["Planner", "Tool catalogue; seven planning rules (identify first, history before condition-driven "
                      "bookings, specialty routing, minimal plans, staff-only record changes, clarification when "
                      "ambiguous); injected date, user, dependents, active patient, recalled memories and recent turns."],
                     ["History summary", "Fixed sections (overview, diagnoses and treatments, medications, alerts, focus); "
                      "“use only the record”; keep ICD-10 codes and dates; alerts supplied by rules, not generated."],
                     ["Medical RAG", "Numbered excerpts; inline citations; personalisation note using the patient "
                      "context produced by the history stage; say what is missing instead of guessing."],
                     ["Synthesis", "Patient-context section first; one heading per task in request order; exact booking "
                      "details; honest failures; keep citations; under 350 words."],
                     ["Extraction", "Typed schemas for free-text clinical notes and for durable memory facts."]],
                    [30 * mm, CONTENT_W - 30 * mm]),
              P("Plan produced by the LLM for the reference request", "h2"),
              Paragraph("1. identify_patient(name=“Suresh Negi”, relation=“father”, age=70)<br/>"
                        "2. get_patient_history(focus=“chronic kidney disease”)  ← depends on 1<br/>"
                        "3. book_appointment(specialty=“Nephrology”, condition=“chronic kidney disease”)  ← depends on 1, 2<br/>"
                        "4. medical_info_search(query=“latest treatment methods for chronic kidney disease”)", S["code"]),
              P("The patient's name was resolved from the injected dependents list, a memory lookup in the prompt. The "
                "validator confirms dependencies and role permissions before any tool runs."),
              PageBreak()]

    # ---------------- 7. Reference scenario ----------------
    story += [P("7. Agent execution: the reference scenario", "h1"),
              P("The problem statement's expected workflow was (1) identify patient and context, (2) retrieve the "
                "father's history, (3) query the doctor calendar and book, (4) search and summarise treatments via "
                "RAG. The live run below follows it exactly: 4/4 steps succeeded."),
              figure(FIG / "chat.png", "Figure 3. Live run of the reference scenario: patient context, booking "
                     "confirmation, cited treatment summary and the planner's decomposition.", max_h=215 * mm),
              PageBreak(),
              P("Follow-up turns use short-term memory: <i>“Also book a dietitian for him next week in the morning”</i> "
                "resolves “him” to the active patient and books a Nutrition &amp; Dietetics slot the following week "
                "before noon. The trace of any run can be replayed, including graph path timings and memory "
                "operations:"),
              figure(FIG / "memory_logs.png", "Figure 4. Memory & logs: planning breakdown, execution status, graph "
                     "path with per-node timings, and memory operations for one run.", max_h=200 * mm),
              PageBreak()]

    # ---------------- 8. Guard-rails ----------------
    story += [P("8. Guard-rails", "h1"),
              table([["Scenario", "Behaviour"],
                     ["“My father has chest pain and is sweating and short of breath”",
                      "Emergency notice first (call 112 / 911); the information search is deferred."],
                     ["Patient asks for another patient's history", "identify_patient refuses: not self or a linked "
                      "dependent; dependent steps are skipped and explained."],
                     ["Patient asks to update their own record", "Plan repaired: staff-only tool removed, clarification "
                      "returned (attendant or doctor required)."],
                     ["Same booking requested twice", "Existing appointment returned; no double booking."],
                     ["Two claims on one slot", "Conditional UPDATE: exactly one wins."],
                     ["No slot in the requested window", "Booking fails gracefully with the nearest alternatives."],
                     ["Provider stalls or errors", "Hard deadline → logged fallback to the heuristic path; the user "
                      "still gets an answer."]],
                    [62 * mm, CONTENT_W - 62 * mm]),
              Spacer(1, 4 * mm),
              figure(FIG / "scenarios.png", "Figure 5. Scenario tester: curated scenarios (including guard-rails) run "
                     "end to end in fresh threads.", max_h=110 * mm),
              PageBreak()]

    # ---------------- 9. Evaluation ----------------
    story += [P("9. LLMOps: evaluation and monitoring", "h1"),
              P("<b>Method.</b> Golden sets cover 15 planner queries across patient, caregiver and attendant roles; 5 "
                "reference summaries (3 taken verbatim from records.xlsx); 8 medical QA pairs with listed key facts; 12 "
                "retrieval queries; and 8 booking scenarios plus idempotency and race checks. LangChain's "
                "<b>QAEvalChain</b> grades summaries and answers when an LLM is configured. Reference-based metrics "
                "always run: embedding similarity, ROUGE-L, token F1, key-fact coverage, context recall (key facts "
                "present in the retrieved passages) and groundedness (share of answer sentences supported by a "
                "retrieved passage at cosine ≥ 0.6). Evaluation bookings are rolled back."),
              Image(str(chart), width=CONTENT_W, height=CONTENT_W * 4.6 / 7.2),
              P("Figure 6. Evaluation results: live LLM vs offline heuristic mode (same golden sets).", "caption")]
    rows = [["Module", "Metric", "Offline", "LLM"]]
    for mod, met, label in [("planner", "tool_f1", "Tool-selection F1"), ("planner", "exact_match", "Exact plan match"),
                            ("planner", "specialty_accuracy", "Specialty accuracy"),
                            ("summary", "qa_eval_correct", "QAEvalChain correct"),
                            ("summary", "semantic_similarity", "Semantic similarity"),
                            ("summary", "rouge_l_recall", "ROUGE-L recall"),
                            ("medical_qa", "qa_eval_correct", "QAEvalChain correct"),
                            ("medical_qa", "keyword_coverage", "Key-fact coverage"),
                            ("medical_qa", "context_recall", "Context recall"),
                            ("medical_qa", "groundedness", "Groundedness"),
                            ("retrieval", "patient_mrr", "Patient memory MRR"),
                            ("retrieval", "knowledge_hit@3", "Knowledge hit@3"),
                            ("booking", "constraint_satisfaction", "Constraints satisfied"),
                            ("booking", "race_safe", "Race safe")]:
        rows.append([mod.replace("_", " "), label, fmt(off.get(mod, {}).get(met)), fmt(live.get(mod, {}).get(met))])
    story += [KeepTogether([P("Detailed results", "h2"),
                            table(rows, [30 * mm, CONTENT_W - 30 * mm - 44 * mm, 22 * mm, 22 * mm])]),
              P("What the numbers say", "h2"),
              *bullets(["<b>Retrieval is not the bottleneck; generation is.</b> Context recall is 1.00 in both modes: "
                        "every key fact was in the retrieved passages. The offline extractive summariser surfaced 67% "
                        "of them; the LLM surfaced all of them, with groundedness rising from 0.88 to 0.99.",
                        "<b>Offline summary scores are inflated.</b> The offline template echoes the recorded Summary "
                        "field, which is also the reference, so its ROUGE-L is high by construction. The LLM paraphrases "
                        "into structured sections, which lexical metrics penalise; QAEvalChain (0.80 correct) is the "
                        "meaningful measure. The one miss was Rebeca Nagle, the most complex record.",
                        "<b>Planner disagreements are judgment calls.</b> The heuristic planner's 1.00 is optimistic "
                        "because it was written alongside the golden set. The LLM's three mismatches: it skipped the "
                        "history lookup for two bookings where only a specialist (no condition) was named, and added a "
                        "harmless list step before a reschedule.",
                        "<b>MMR helped the offline path.</b> Replacing top-similarity sentence selection with Maximal "
                        "Marginal Relevance (λ = 0.35) raised offline key-fact coverage from 0.56 to 0.69 at unchanged "
                        "groundedness."]),
              P("Monitoring", "h2"),
              P("Every tool call, LLM call (live / fallback / offline), memory operation and agent run is logged to "
                "SQLite. The dashboard aggregates tool success rate and p50/p95 latency, per-module success, booking "
                "success and the LLM live/fallback mix, and can run the evaluation suite on demand."),
              figure(FIG / "eval_live.png", "Figure 7. Live operations dashboard: run and tool success, latency, "
                     "per-module success and LLM call modes.", max_h=112 * mm),
              CondPageBreak(60 * mm)]

    # ---------------- 10. Dashboard ----------------
    story += [P("10. Streamlit dashboard", "h1"),
              table([["Page", "Purpose"],
                     ["Chat with the assistant", "Live plan and execution trace per turn; example prompts per role"],
                     ["Scenario tester", "Run curated or custom scenarios end to end (incl. guard-rails)"],
                     ["Patient view", "Record, alerts, lab trends, appointments, memory; staff can update fields, add "
                      "notes, upload PDFs and register patients"],
                     ["Doctor view", "Calendar utilisation and upcoming patients with clinical briefing"],
                     ["Appointment tracker", "Real-time (auto-refreshing) tracking, filters, cancellation"],
                     ["Medical information", "Trusted-source search, latest retrieved answers, knowledge base"],
                     ["Evaluation & metrics", "Tool/booking success, latency, LLM call modes, evaluation runs"],
                     ["Memory & logs", "Memory traces, planning breakdowns, tool and LLM logs"]],
                    [42 * mm, CONTENT_W - 42 * mm]),
              Spacer(1, 4 * mm),
              figure(FIG / "doctor.png", "Figure 8. Doctor view: slot utilisation and the booked CKD patient.",
                     max_h=88 * mm),
              figure(FIG / "appointments.png", "Figure 9. Appointment tracker with live refresh.", max_h=62 * mm),
              PageBreak()]

    # ---------------- 11. Engineering issues ----------------
    story += [P("11. Engineering issues found and fixed", "h1"),
              P("Testing against a live model surfaced problems that offline tests could not. Each was diagnosed from "
                "the logs and fixed in code:"),
              table([["Issue", "Diagnosis", "Fix"],
                     ["A follow-up turn hung for 14 minutes", "The gateway kept a slow connection alive, so the "
                      "client's 60 s read timeout never fired", "Hard wall-clock deadline on every LLM call; timeouts "
                      "fall back and are logged"],
                     ["Slow, variable LLM latency", "Default provider routing: 1–5 s per call", "Throughput-sorted "
                      "provider routing: ~0.5 s per call"],
                     ["Redundant plan steps", "Planner added find_doctors and check_availability before booking",
                      "Planner rule: booking already picks the earliest slot"],
                     ["Reply omitted patient context", "Synthesis prompt did not ask for it", "Patient-context section "
                      "when history was retrieved"],
                     ["Doctor view showed 100% booked", "Empty IDs load as NaN, which is truthy in Python",
                      "Explicit not-null check"],
                     ["Tests overwrote evaluation reports", "Report folder not isolated in tests",
                      "Configurable REPORTS_DIR, pointed at a temp dir in tests"]],
                    [40 * mm, (CONTENT_W - 40 * mm) / 2, (CONTENT_W - 40 * mm) / 2]),
              P("12. Limitations and future work", "h1"),
              *bullets(["<b>Synthetic scheduling data.</b> A production deployment would put a real calendar/EHR behind "
                        "the same tool interfaces (e.g. FHIR Appointment, Slot and Patient resources).",
                        "<b>Simulated authentication.</b> The acting-user selector stands in for real identity, consent "
                        "records and audit logging (HIPAA / DPDP).",
                        "<b>Source coverage.</b> WHO content arrives via web search; a curated guideline corpus (KDIGO, "
                        "ADA, NICE) with freshness metadata would strengthen “latest treatment” answers.",
                        "<b>Evaluation scale.</b> Golden sets are small; next steps are paraphrase expansion, an LLM-judge "
                        "faithfulness score and tracking metrics across model versions in CI.",
                        "<b>Human in the loop.</b> Caregiver-initiated bookings should require confirmation, and "
                        "clinician sign-off should precede committing extracted notes.",
                        "<b>Stale memories.</b> Interaction memories can outlive the state they describe (e.g. a later "
                        "cancelled appointment); memories should be reconciled against the EHR."]),
              CondPageBreak(70 * mm),
              P("Appendix: running the project", "h1"),
              Paragraph("python -m venv .venv &amp;&amp; .venv\\Scripts\\activate<br/>"
                        "pip install -r requirements.txt<br/>"
                        "copy .env.example .env      # add GROQ_API_KEY / OPENAI_API_KEY / OPENROUTER_API_KEY<br/>"
                        "streamlit run app.py        # builds the EHR and indexes on first launch<br/>"
                        "python -m healthcare_agent.evaluation.runner   # full evaluation suite<br/>"
                        "python -m pytest -q         # 27 tests, isolated and offline", S["code"]),
              P("Repository: <font name='Mono'>healthcare_agent/</font> (agent, tools, memory, evaluation), "
                "<font name='Mono'>app.py</font> + <font name='Mono'>app_pages/</font> (dashboard), "
                "<font name='Mono'>notebooks/</font> (executed walkthrough), <font name='Mono'>tests/</font>, "
                "<font name='Mono'>docs/PROJECT_REPORT.md</font>."),
              Spacer(1, 6 * mm),
              P("<i>Educational prototype on sample and synthetic data. It does not diagnose or provide medical "
                "advice.</i>", "small")]

    doc = SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=MARGIN, rightMargin=MARGIN, topMargin=18 * mm,
                            bottomMargin=16 * mm, title="Agentic Healthcare Assistant – Capstone Writeup",
                            author=author, subject="Applied Generative AI Specialisation capstone")
    doc.build(story, onFirstPage=on_page, onLaterPages=on_page)
    print("wrote", OUT)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--author", default="Narayanan")
    build(ap.parse_args().author)
